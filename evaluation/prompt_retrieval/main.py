import argparse
import random
import os
import torch
import numpy as np
import json
import nltk
import shutil
import time
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer,GPTJForCausalLM
from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel
from get_task import get_task
from utils import calculate_sentence_transformer_embedding,codex_execution,expand_to_aliases
from two_steps import prompt_retrieval

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True,type=str)
parser.add_argument('--model_cache_dir', required=True,type=str)
parser.add_argument('--data_cache_dir', required=False,type=str)
parser.add_argument('--output_dir', required=True,type=str)
parser.add_argument('--embedding_model', required=True,type=str)
parser.add_argument('--model_key', type=str)
parser.add_argument('--prompt_retrieval_method', default='similar',type=str)
parser.add_argument('--model_name', default='EleutherAI/gpt-j-6B',type=str)
parser.add_argument('--seed', default=0,type=int)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--add_prompt', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocess_text(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

if __name__=='__main__':
    args.data_cache_dir = args.model_cache_dir
    set_seed(args.seed)
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir,exist_ok=True)
    train_examples,eval_examples,train_text_to_encode,eval_text_to_encode,format_example,label_map = get_task(args=args)
    total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=train_text_to_encode,
                                                                  args=args)
    total_eval_embeds = calculate_sentence_transformer_embedding(text_to_encode=eval_text_to_encode,
                                                                  args=args)

    if args.task_name in ['rte','sst5','mrpc','dbpedia_14','hellaswag']:
        data_module = MetaICLData(method="direct", max_length=1024, max_length_per_example=256)
        inference_model = MetaICLModel(args=args)
        inference_model.load()
        inference_model.cuda()
        inference_model.eval()
        tokenizer_gpt = None
        return_string = False
        single_input_len = 250
        maximum_input_len = 1000


        first_phase_selected_indices = range(len(train_examples))
        processed_train_examples = [train_examples[idx] for idx in first_phase_selected_indices]
        processed_eval_examples = eval_examples

        prompt_retrieval(train_embs=total_train_embeds,test_embs=total_eval_embeds,train_examples=train_examples,
                         eval_examples=eval_examples,return_string=return_string,format_example=format_example,
                         maximum_input_len=maximum_input_len,single_context_example_len=single_input_len,label_map=label_map,args=args)

        prompt_cache_dir = os.path.join(args.output_dir, 'prompts')
        candidate_prompt_files = os.listdir(prompt_cache_dir)
        prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
        assert len(prompt_files) == len(processed_eval_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                                  f"len(processed_eval_examples)={len(processed_eval_examples)}"
        output_dir = os.path.join(args.output_dir,'results')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        count = 0
        running_flag = True
        golds = []
        preds = []
        if not args.task_name in ['hellaswag']:
            all_labels = []
            label_to_digit = {}
            for k, v in label_map.items():
                all_labels.append(v)
                label_to_digit[v] = k
        execution_count = 0
        while running_flag:
            running_flag = False
            count += 1
            bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
            for file in prompt_files:
                bar.update(1)
                load_flag = True
                try:
                    with open(f"{output_dir}/{file}") as f:
                        components = json.load(f)
                except:
                    load_flag = False
                if not load_flag:
                    running_flag = True
                    if args.task_name == 'hellaswag':
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        cur_input = {'input': format_example(one_test_example[2], label_map=label_map, args=args)[0],
                                     'options': one_test_example[2]['endings']}
                        data_module.k = len(cur_train_data)
                        data_module.tensorize(cur_train_data, [cur_input])
                        prediction = inference_model.do_predict(data_module)[0]
                        assert prediction in one_test_example[2]['endings']
                        with open(f"{output_dir}/{file}", 'w') as f:
                            json.dump([prediction, one_test_example[2]['endings'][one_test_example[2]['label']]], f)
                        preds.append(prediction)
                        golds.append(one_test_example[2]['endings'][one_test_example[2]['label']])
                    else:
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]['options'] = all_labels
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]['options'] = all_labels
                        cur_input = format_example(one_test_example[2], label_map=label_map, args=args)[0]
                        data_module.k = len(cur_train_data)
                        data_module.tensorize(cur_train_data, [cur_input], options=all_labels)
                        prediction = inference_model.do_predict(data_module)[0]
                        with open(os.path.join(output_dir, file), 'w') as f:
                            json.dump([prediction, one_test_example[2]['label']], f)
                        preds.append(label_to_digit[prediction])
                        golds.append(one_test_example[2]['label'])
                else:
                    with open(f"{output_dir}/{file}") as f:
                        components = json.load(f)
                    if args.task_name == 'hellaswag':
                        preds.append(components[0])
                    else:
                        preds.append(label_to_digit[components[0]])
                    golds.append(components[1])

        assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
        preds, golds = postprocess_text(preds, golds)
        metric = load_metric("rouge")
        result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        result = result['rougeL']
        with open(os.path.join(args.output_dir,'result_summary.json'), 'w') as f:
            json.dump(result, f)
        print(result)
