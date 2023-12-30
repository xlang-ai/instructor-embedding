# This script is based on the modification from https://github.com/huggingface/transformers
import logging
import os
import torch
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on

import transformers
from filelock import FileLock
from InstructorEmbedding import INSTRUCTOR
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.utils.versions import require_version
from datasets import Dataset,DatasetDict


check_min_version("4.20.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False

class InstructorTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self) :
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        for task_id in inputs['task_id']:
            assert task_id==inputs['task_id'][0],f"Examples in the same batch should come from the same task, " \
                                                 f"but task {task_id} and task {inputs['task_id'][0]} are found"
        cur_results = {}
        for k in ['query', 'pos', 'neg']:
            cur_inputs = {
                'input_ids': inputs[f'{k}_input_ids'],
                'attention_mask': inputs[f'{k}_attention_mask'],
                'context_masks': inputs[f'{k}_context_masks'],
            }
            cur_results[k] = model(cur_inputs)['sentence_embedding']
        embeddings_query = cur_results['query']
        embeddings_pos = cur_results['pos']
        embeddings_neg = cur_results['neg']

        num = len(embeddings_query)
        all_scores = None
        from torch import nn
        similarity_fct = nn.CosineSimilarity(dim=-1)
        for i in range(0, num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                one_neg_emb = embeddings_neg[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_scores is None:
                all_scores = cur_score.unsqueeze(0)
            else:
                all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss = nn.CrossEntropyLoss()(all_scores, labels)

        all_another_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_another_scores is None:
                all_another_scores = cur_score.unsqueeze(0)
            else:
                all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

        if return_outputs:
            return loss, all_scores
        return loss 

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    processed_data_dir: Optional[str] = field(
        default=None, metadata={"help": "directory to the processed data"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    sample_selection_train_file_path: Optional[str] = field(
        default=None, metadata={"help": "sample_selection_train_file_path"}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    def_only: bool = field(
        default=False, metadata={"help": "def_only"}
    )
    add_prompt_to_document: bool = field(
        default=True, metadata={"help": "add_prompt_to_document"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    debug_mode: Optional[int] = field(
        default=None,
        metadata={"help": "debug mode"},
    )
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "debug mode"},
    )
    cl_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "temperature"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    sub_sample_ratio: Optional[float] = field(
        default=2.0,
        metadata={
            "help": (
                "sub_sample_ratio"
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    def __post_init__(self):
        pass


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.output_dir = training_args.output_dir
    real_name_or_path = model_args.model_name_or_path
    data_args.model_name_or_path = model_args.model_name_or_path
    data_args.tokenizer_name_or_path = model_args.model_name_or_path
    training_args.cl_temperature = data_args.cl_temperature
    training_args.remove_unused_columns = False
    if not os.path.isdir(data_args.output_dir):
        os.makedirs(data_args.output_dir,exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.ERROR
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    set_seed(training_args.seed)
    with open(os.path.join(model_args.cache_dir, 'medi-data.json')) as f:
        train_examples_raw = json.load(f)

    if data_args.debug_mode:
        train_examples_raw = train_examples_raw[:data_args.debug_mode]
    old_train_examples_raw = train_examples_raw
    total_train_n = len(old_train_examples_raw)

    real_batch_size = max(training_args.per_device_train_batch_size,
                          training_args.per_device_train_batch_size * torch.cuda.device_count())
    # print('real_batch_size: ', real_batch_size,training_args.per_device_train_batch_size,torch.cuda.device_count())
    def get_examples_raw(old_examples_raw, total_n, real_batch_size):
        examples_raw = []
        for idx in range(0, total_n, real_batch_size):
            local_task_name = old_examples_raw[idx]['task_id']
            cur_batch = []
            include_batch = True
            for idx1 in range(idx, min(idx + real_batch_size, total_n)):
                if not old_examples_raw[idx1]['task_id'] == local_task_name:
                    print(f'one batch in task {old_examples_raw[idx1]["task_id"]} is skipped')
                    include_batch = False
                    break
                else:
                    cur_batch.append(old_examples_raw[idx1])
            if include_batch and len(cur_batch) == real_batch_size:
                examples_raw.append(cur_batch)
        return examples_raw

    train_examples_raw = get_examples_raw(old_train_examples_raw, total_train_n, real_batch_size)
    random.shuffle(train_examples_raw)
    
    if data_args.max_examples is not None and len(train_examples_raw*real_batch_size)>data_args.max_examples:
        train_examples_raw = train_examples_raw[:int(data_args.max_examples/real_batch_size)]

    train_examples_raw_batch = train_examples_raw
    train_examples_raw = []

    for b in train_examples_raw_batch:
        train_examples_raw += b
    print(f'There are {len(train_examples_raw)} pairs to train in total.')

    if data_args.debug_mode:
        train_examples_raw = train_examples_raw[:int(data_args.debug_mode)]

    def get_dataset(examples_raw):
        examples = {'query':[],'pos':[],'neg':[],'task_id':[]}
        task_name_map = {}
        total_num = len(examples_raw)
        task_count = 0
        for i in range(total_num):
            cur_e = examples_raw[i]
            for k in ['query','pos','neg']:
                for s in cur_e[k][:-1]:
                    assert not '!@#$%^&**!@#$%^&**' in s
                cur_e[k][-1] = str(cur_e[k][-1])
                if not data_args.add_prompt_to_document:
                    cur_e[k][0] = ''
                assert cur_e[k][0].startswith('Represent ') or cur_e[k][0]==''
                examples[k].append('!@#$%^&**!@#$%^&**'.join(cur_e[k]))
            if not cur_e['task_id'] in task_name_map:
                task_name_map[cur_e['task_id']] = task_count
                task_count += 1
            examples['task_id'].append(task_name_map[cur_e['task_id']])
        return examples

    train_raw_datasets = DatasetDict({'train':Dataset.from_dict(get_dataset(train_examples_raw))})

    model = INSTRUCTOR(real_name_or_path, cache_folder=model_args.cache_dir)
    column_names = train_raw_datasets["train"].column_names

    def preprocess_function(examples):
        all_tokenized = None
        for key in ['query','pos','neg']:
            num = len(examples[key])
            contexts = []
            concatenated_input_texts = []
            for local_idx in range(num):
                splits = examples[key][local_idx].split('!@#$%^&**!@#$%^&**')
                assert len(splits) == 2
                contexts.append(splits[0])
                concatenated_input_texts.append(''.join(splits))
                assert isinstance(contexts[-1], str)
                assert isinstance(concatenated_input_texts[-1], str)
            tokenized = tokenizer(concatenated_input_texts,padding='max_length', truncation='longest_first', return_tensors="pt", max_length=data_args.max_source_length)
            context_tok = tokenizer(contexts,padding='max_length', truncation='longest_first', return_tensors="pt", max_length=data_args.max_source_length)
            tokenized['context_masks'] = torch.sum(context_tok['attention_mask'], dim=1)
            tokenized['context_masks'] = tokenized['context_masks'] - 1
            for my_idx in range(len(tokenized['context_masks'])):
                if tokenized['context_masks'][my_idx] <= 1:
                    tokenized['context_masks'][my_idx] = 0
            keys = tokenized.keys()
            if all_tokenized is None:
                all_tokenized = tokenized.copy()
                for k in keys:
                    all_tokenized[k] = all_tokenized[k].tolist()
            for k in keys:
                all_tokenized[f'{key}_{k}'] = tokenized[k].tolist()
        all_tokenized['task_id'] = examples['task_id']
        return all_tokenized

    train_dataset = train_raw_datasets["train"]
    val_dataset = None
    if data_args.validation_file:
        with open(data_args.validation_file) as f:
            val_examples_raw = json.load(f)

        old_val_examples_raw = val_examples_raw
        total_val_n = len(old_val_examples_raw)

        val_examples_raw = get_examples_raw(old_val_examples_raw, total_val_n, real_batch_size)
        random.shuffle(val_examples_raw)

        val_examples_raw_batch = val_examples_raw
        val_examples_raw = []
        for b in val_examples_raw_batch:
            val_examples_raw += b
        print(f'There are {len(val_examples_raw)} pairs in val dataset.')
        val_raw_datasets = DatasetDict({'val':Dataset.from_dict(get_dataset(val_examples_raw))})
        val_dataset = val_raw_datasets["val"]

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            val_dataset = val_dataset.map( 
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on val dataset",
            )

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    trainer = InstructorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.model.save(training_args.output_dir)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
