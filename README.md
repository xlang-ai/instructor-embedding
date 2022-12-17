# 1 Embedder, 70 Tasks: Instruction-Finetuned Text Embeddings

Code for paper [1 Embedder, 70 Tasks: Instruction-Finetuned Text Embeddings](https://github.com/HKUNLP/instructor-embedding)


## Cloning this repo
Run the following command to clone this repo
```
git clone https://github.com/HKUNLP/instructor-embedding
```

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
That will create the environment selective_annotation we used.

## Usage

### Environment setup

Activate the environment by running
```
conda activate instructor
```

### Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --model_name_or_path sentence-transformers/gtr-t5-large --output_dir {output_directory} --cache_dir {cache_directory} --max_source_length 512 --num_train_epochs 10 --save_steps 500 --cl_temperature 0.01 --warmup_ratio 0.1 --learning_rate 2e-5 --overwrite_output_dir
```

### Evalution
### MTEB
```
cd evalution/MTEB
python examples/evaluate_model.py --model_name hku-nlp/instructor-large --output_dir outputs --task_name ArguAna --result_file results
```

### Billboard
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

## Citation
If you find our work helpful, please cite us
