# 1 Embedder, 70 Tasks: Instruction-Finetuned Text Embeddings

Code for paper [One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://github.com/HKUNLP/instructor-embedding)

We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions: every text input is embedded together with instructions explaining the use case (e.g., task and domain descriptions). Unlike encoders from prior work that are more specialized, INSTRUCTOR is a single embedder that can generate text embeddings tailored to different downstream tasks and domains, without any further training. We first annotate instructions for 330 diverse tasks and train INSTRUCTOR on this multitask mixture with a contrastive loss. We evaluate INSTRUCTOR on 70 embedding evaluation tasks (66 of which are unseen during training), ranging from classification and information retrieval to semantic textual similarity and text generation evaluation. INSTRUCTOR, while having an order of magnitude fewer parameters than the previous best model, achieves state-of-the-art performance, with an average improvement of 3.4% compared to the previous best results on the 70 diverse datasets. Our analysis suggests that INSTRUCTOR is robust to changes in instructions, and that instruction finetuning mitigates the challenge of training a single model on diverse datasets.

## Dependencies
To establish the environment, run this code in the shell:
```
conda env create -f instructor_embedding.yml
conda activate instructor
cd transformers
pip install -e .
cd sentence-transformers
pip install -e .
```
That will create the environment INSTRUCTOR we used.

### Environment setup

Activate the environment by running
```
conda activate instructor
```

## Getting Started

First download a pretrained model

```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('hku-nlp/instructor-large')
```

Then provide the sentence and customized instruction to the model.
```
# prepare texts with instructions
text_instruction_pairs = 
[
    {
        "instruction": "Represent the Science title; Input:",
        "text": "3D ActionSLAM: wearable person tracking in multi-floor environments", 
    },
    {
        "instruction": "Represent the Medicien sentence for retrieving a duplicate sentence; Input:",
        "text": "Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear.",
    }
]

# postprocess
texts_with_instructions = []
for pair in text_instruction_pairs:
    texts_with_instructions.append([pair["instruction"], pair["text"], 0])

# calculate embeddings
customized_embeddings = model.encode(texts_with_instructions)
```
The instruction is expected to follow the unified template: Represent the `domain` `text_type` for `task_objective`; Input:
* `domain` is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
* `text_type` is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
* `task_objective` is optional, and it specifies the objective of emebdding, e.g., retrieve a document, classify the sentence, etc.

And that's it already. We now have a list of numpy arrays with the emebddings.

```
for pair, embedding in zip(text_instruction_pairs, customized_embeddings):
    print("Instruction: ", pair["instruction"])
    print("text: ", pair["text"])
    print("Embedding: ", embedding)
    print("")
```

You can use INSTRUCTOR to compute similarities between two groups of sentences
```
from sklearn.metrics.pairwise import cosine_similarity
sentences_a = [['Represent the Science sentence; Input: ','Parton energy loss in QCD matter',0], 
               ['Represent the Financial statement; Input: ','The Federal Reserve on Wednesday raised its benchmark interest rate.',0]
sentences_b = [['Represent the Science sentence; Input: ','The Chiral Phase Transition in Dissipative Dynamics', 0],
               ['Represent the Financial statement; Input: ','The funds rose less than 0.5 per cent on Friday',0]
embeddings_a = model.encode(sentences_a)
embeddings_b = model.encode(sentences_b)
similarities = cosine_similarity(embeddings_a,embeddings_b)
```

### Multitask Embedding Data with Instructions (MEDI)

We construct a collection of 330 datasets from [Super-NI](https://arxiv.org/abs/2204.07705)(Super-NaturalInstructions), [sentence-transformer embedding training data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), and [KILT](https://arxiv.org/abs/2009.02252). They span a wide range of domains (e.g., news, reviews, Wikipedia, etc.), text types (sentence, paragraph, document, etc.) and task objectives (e.g., classification, summarization, fact verification, etc.). For Super-NI, we build positive pairs by mining examples that have both similar input texts and similar labels, and build negative pairs that have only similar input texts but different labels. For other datasets, we use the provided positive and negative pairs (or random ones if not provided). We follow the unified template to annotate customized instructions for each dataset. 

You can directly download the processed MEDI data [here](https://drive.google.com/file/d/1dvmDBp095CY5hwIJxcRNaLH7sIwym4ql/view).

### Train INSTRUCTOR
We provide the example script for training INSTRUCTOR. It calls `train.py` and automatically download MEDI data for training (if not downloaded yet).
```
python train.py --model_name_or_path sentence-transformers/gtr-t5-large --output_dir {output_directory} --cache_dir {cache_directory} --max_source_length 512 --num_train_epochs 10 --save_steps 500 --cl_temperature 0.01 --warmup_ratio 0.1 --learning_rate 2e-5 --overwrite_output_dir
```
We explain the arguments in the following:
* `--model_name_or_path`: Pretrained checkpoints to start with. We support both model id (e.g., `sentence-transformers/gtr-t5-large`, `sentence-transformers/sentence-t5-large`) and checkpoint path (e.g., checkpoint saved by transformers trainer).
* `--cl_temperature`: Temperature for contrastive loss
* `--cache_dir`: The directory to cache downloaded models and data. If you download the training data manually, you should put the data under `--cache_dir`.
* `--output_dir`: The directory to store the trained models(checkpoints) for evaluation. 

All the other arguments are standard `Huggingface's transformers` training arguments. Some of the often-used arguments are `--overwrite_output_dir`, `--num_train_epochs`, `--learning_rate`, which are self-explained. 

### Evalution
We evalute INSTRUCTOR massively on 70 diverse tasks to avoid bias. The 70 tasks include three benchmarks, [MTEB](https://huggingface.co/spaces/mteb/leaderboard), [Billboard](https://arxiv.org/abs/2112.04139), and [Prompt Retrieval](https://arxiv.org/abs/2209.01975). 
* MTEB is a comprehensive embedding evaluation benchmark that aims to provide a holistic view of embedding models.  It combines several conventional benchmarks (e.g., BEIR and STS) and spans a wide range of domain-specific datasets, including science, biology, and medicine. 
* Prompt Retrieval tasks aim to retrieve a few in-context learning (i.e., demonstration) examples from annotated examples given a test instance. The embedding model is used to encode all annotated examples and to find the few most similar examples to the test instance based on the cosine similarity. We evalute emebddings by measuring the average performance on the downstream tasks. 
* Billboard applies INSTRUCTOR to automatic evaluations for text generation tasks. Following [Kasai et al. (2022a)](https://arxiv.org/abs/2112.04139), we measure the cosine similarity between the generated text and each reference text and take the maximum similarity score over all references available. We evaluate all embedding models by the Pearson correlation with the human judgments.

For results in the paper, we use 40GB A100 with CUDA 11. Using different types of devices or different versions of CUDA/other software may lead to slightly different performance

### MTEB
To evaluate the model performance on MTEB benchmark dataset, run the following command, where 
* `--model_name` is a model id or trained checkpoint.
* `--task_name` is a task name in MTEB benchmark.
* `--result_file` is a directory to stored the evaluation results.
```
cd evalution/MTEB
python examples/evaluate_model.py --model_name hku-nlp/instructor-large --output_dir outputs --task_name ArguAna --result_file results
```

### Billboard
To evaluate the model performance on Billboard, run the following command, where 
* `--model_name` is a model id or trained checkpoint.
* `--task` is a task name on Billboard (mscoco, mt or cnndm).
```
cd evaluation/text_evaluation
python main.py --model_name hku-nlp/instructor-large --task mscoco --add_prompt
```

### Prompt Retrieval
```
cd evaluation/prompt_retrieval
python main.py --embedding_model hku-nlp/instructor-large --task rte --model_cache_dir {cache_dir} --output_dir {output_dir}
```

### Encode sentence with customized instruction
```
from sentence_transformers import SentenceTransformer
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title; Input:"
model = SentenceTransformer('hku-nlp/instructor-large')
embeddings = model.encode([[instruction,sentence,0]])
print(embeddings)
```

### Computing sentence similarities
```
from sklearn.metrics.pairwise import cosine_similarity
sentences_a = [['Represent the Science sentence; Input: ','Parton energy loss in QCD matter',0], 
               ['Represent the Financial statement; Input: ','The Federal Reserve on Wednesday raised its benchmark interest rate.',0]
sentences_b = [['Represent the Science sentence; Input: ','The Chiral Phase Transition in Dissipative Dynamics', 0],
               ['Represent the Financial statement; Input: ','The funds rose less than 0.5 per cent on Friday',0]
embeddings_a = model.encode(sentences_a)
embeddings_b = model.encode(sentences_b)
similarities = cosine_similarity(embeddings_a,embeddings_b)
```

## Citation
If you find our work helpful, please cite us
