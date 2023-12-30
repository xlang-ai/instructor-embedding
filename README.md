# One Embedder, Any Task: Instruction-Finetuned Text Embeddings

This repository contains the code and pre-trained models for our paper [One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://arxiv.org/abs/2212.09741). Please refer to our [project page](https://instructor-embedding.github.io/) for a quick project overview.

We introduce **Instructor**👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) ***by simply providing the task instruction, without any finetuning***. Instructor👨‍ achieves sota on 70 diverse embedding tasks!

**************************** **Updates** ****************************

* 01/21: We updated the code structure, which supports easy package installation.
* 12/28: We updated the [checkpoint](https://huggingface.co/hkunlp/instructor-large) with hard negatives.
* 12/20: We released [our paper](https://arxiv.org/abs/2212.09741), [code](https://github.com/HKUNLP/instructor-embedding), [project page](https://instructor-embedding.github.io/) and [checkpoint](https://huggingface.co/hkunlp/instructor-large). Check them out!

## Quick Links

- [One Embedder, Any Task: Instruction-Finetuned Text Embeddings](#one-embedder-any-task-instruction-finetuned-text-embeddings)
  - [Quick Links](#quick-links)
  - [Installation](#installation)
    - [Environment setup](#environment-setup)
  - [Getting Started](#getting-started)
    - [The `encode` function](#the-encode-function)
  - [Model List](#model-list)
  - [Use Cases](#use-cases)
    - [Calculate embeddings for your customized texts](#calculate-embeddings-for-your-customized-texts)
    - [Compute similarities between texts](#compute-similarities-between-texts)
    - [Use customized embeddings for information retrieval](#use-customized-embeddings-for-information-retrieval)
    - [Use customized embeddings for clustering](#use-customized-embeddings-for-clustering)
  - [Training](#training)
    - [Data](#data)
    - [Train INSTRUCTOR](#train-instructor)
  - [Evaluation](#evaluation)
    - [MTEB](#mteb)
    - [Billboard](#billboard)
    - [Prompt Retrieval](#prompt-retrieval)
  - [Quantization](#quantization)
  - [Bugs or questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [INSTRUCTOR Elsewhere](#instructor-elsewhere)

## Installation
It is very easy to use INSTRUCTOR for any text embeddings. You can easily try it out in [Colab notebook](https://colab.research.google.com/drive/1P7ivNLMosHyG7XOHmoh7CoqpXryKy3Qt?usp=sharing). In your local machine, we recommend to first create a virtual environment:
```bash
conda env create -n instructor python=3.7
git clone https://github.com/HKUNLP/instructor-embedding
pip install -r requirements.txt
```
That will create the environment `instructor` we used. To use the embedding tool, first install the `InstructorEmbedding` package from PyPI
```bash
pip install InstructorEmbedding
```
or directly install it from our code
```bash
pip install -e .
```

### Environment setup

Activate the environment by running
```bash
conda activate instructor
```

## Getting Started

First download a pretrained model (See [model list](#model-list) for a full list of available models)

```python
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-large')
```

Then provide the sentence and customized instruction to the model.
```python
# prepare texts with instructions
text_instruction_pairs = [
    {"instruction": "Represent the Science title:", "text": "3D ActionSLAM: wearable person tracking in multi-floor environments"},
    {"instruction": "Represent the Medicine sentence for retrieving a duplicate sentence:", "text": "Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear."}
]

# postprocess
texts_with_instructions = []
for pair in text_instruction_pairs:
    texts_with_instructions.append([pair["instruction"], pair["text"]])

# calculate embeddings
customized_embeddings = model.encode(texts_with_instructions)
```

And that's it already. We now have a list of numpy arrays with the embeddings.

```python
for pair, embedding in zip(text_instruction_pairs, customized_embeddings):
    print("Instruction: ", pair["instruction"])
    print("text: ", pair["text"])
    print("Embedding: ", embedding)
    print("")
```

### The `encode` function

The users of the model need to use only the `encode` function:

```python
model.encode( sentences,
              batch_size: int = 32,
              show_progress_bar: bool = None,
              output_value: str = 'sentence_embedding',
              convert_to_numpy: bool = True,
              convert_to_tensor: bool = False,
              device: str = None,
              normalize_embeddings: bool = False)
```

* `sentences`: The sentences to be embedded. It should be in the format of `[["instruction prompt 0", "text to be embedded 0], ["instruction prompt 1", "text to be embedded 1], ...]`.
* `batch_size` (default: 32): The batch size used for the computation. It determines the number of sentences processed together in each batch.
* `show_progress_bar` (default: None): If set to `True`, it displays a progress bar while encoding sentences, providing a visual indication of the encoding progress.
* `output_value` (default: 'sentence\_embedding'): Specifies the desired output type. The default value 'sentence\_embedding' returns sentence embeddings. Setting it to 'token\_embeddings' returns wordpiece token embeddings. Setting it to None returns all output values.
* `convert_to_numpy` (default: `True`): If set to `True`, the output is a list of numpy vectors. If set to `False`, the output is a list of PyTorch tensors.
* `convert_to_tensor` (default: `False`): If set to `True`, the function returns a stacked tensor as a single output. This parameter overrides any setting specified by `convert_to_numpy`.
* `device` (default: None): Specifies the torch.device to use for the computation. If not specified, the function uses the default device.
* `normalize_embeddings` (default: `False`): If set to `True`, the returned vectors will have a length of 1, indicating that they are normalized. In this case, similarity search would use the faster dot-product (`util.dot_score`), instead of cosine similarity.

## Model List

We released a series of INSTRUCTOR checkpoints with different sizes. You can easily load these models with `InstructorEmbedding` package. 
|              Model              | Avg. Score |
|:-------------------------------|:--------:|
|  [hkunlp/instructor-base](https://huggingface.co/hkunlp/instructor-base) |   55.9 |
| [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large) |   58.4  |
|    [hkunlp/instructor-xl](https://huggingface.co/hkunlp/instructor-xl)    |   58.8  |

## Use Cases
We provide a few specific use cases in the following. For more examples and applications, refer to [our paper](https://arxiv.org/abs/2212.09741)
### Calculate embeddings for your customized texts
If you want to calculate customized embeddings for specific sentences, you may follow the unified template to write instructions: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Represent the `domain` `text_type` for `task_objective`:
* `domain` is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
* `text_type` is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
* `task_objective` is optional, and it specifies the objective of embedding, e.g., retrieve a document, classify the sentence, etc.

### Compute similarities between texts
You can use **INSTRUCTOR** to compute similarities between two groups of sentences, with **customized embeddings**.
```python
from sklearn.metrics.pairwise import cosine_similarity
sentences_a = [['Represent the Science sentence: ','Parton energy loss in QCD matter'], 
               ['Represent the Financial statement: ','The Federal Reserve on Wednesday raised its benchmark interest rate.']]
sentences_b = [['Represent the Science sentence: ','The Chiral Phase Transition in Dissipative Dynamics'],
               ['Represent the Financial statement: ','The funds rose less than 0.5 per cent on Friday']]
embeddings_a = model.encode(sentences_a)
embeddings_b = model.encode(sentences_b)
similarities = cosine_similarity(embeddings_a,embeddings_b)
```

### Use customized embeddings for information retrieval
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
query  = [['Represent the Wikipedia question for retrieving supporting documents: ','where is the food stored in a yam plant']]
corpus = [['Represent the Wikipedia document for retrieval: ','Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?] that the term "mixed economies" more precisely describes most contemporary economies, due to their containing both private-owned and state-owned enterprises. In capitalism, prices determine the demand-supply scale. For example, higher demand for certain goods and services lead to higher prices and lower demand for certain goods lead to lower prices.'],
          ['Represent the Wikipedia document for retrieval: ',"The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession"],
          ['Represent the Wikipedia document for retrieval: ','Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.']]
query_embeddings = model.encode(query)
corpus_embeddings = model.encode(corpus)
similarities = cosine_similarity(query_embeddings,corpus_embeddings)
retrieved_doc_id = np.argmax(similarities)
print(retrieved_doc_id)
```

### Use customized embeddings for clustering
```python
import sklearn.cluster
sentences = [['Represent the Medicine sentence for clustering: ','Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity'],
             ['Represent the Medicine sentence for clustering: ','Comparison of Atmospheric Neutrino Flux Calculations at Low Energies'],
             ['Represent the Medicine sentence for clustering: ','Fermion Bags in the Massive Gross-Neveu Model'],
             ['Represent the Medicine sentence for clustering: ',"QCD corrections to Associated t-tbar-H production at the Tevatron"],
             ['Represent the Medicine sentence for clustering: ','A New Analysis of the R Measurements: Resonance Parameters of the Higher,  Vector States of Charmonium']]
embeddings = model.encode(sentences)
clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=2)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_
print(cluster_assignment)
```
## Training
### Data
We construct Multitask Embeddings Data
with Instructions (MEDI), consisting of a collection of 330 datasets from [Super-NI](https://arxiv.org/abs/2204.07705)(Super-NaturalInstructions), [sentence-transformer embedding training data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), [KILT](https://arxiv.org/abs/2009.02252) and [MedMCQA](https://proceedings.mlr.press/v174/pal22a/pal22a.pdf), spanning a wide range of domains and tasks. We construct positive and negative pairs if they are not provided, and store them in a unified format:
```
[
    {'query': ['Represent the Wikipedia question for retrieving relevant documents;', 'big little lies season 2 how many episodes'], 'pos': ['Represent the Wikipedia document for retrieval;', 'Big Little Lies (TV series) series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley'], 'neg': ['Represent the Wikipedia document for retrieval;', 'Little People, Big World final minutes of the season two-A finale, "Farm Overload". A crowd had gathered around Jacob, who was lying on the ground near the trebuchet. The first two episodes of season two-B focus on the accident, and how the local media reacted to it. The first season of "Little People, Big World" generated solid ratings for TLC (especially in the important 18–49 demographic), leading to the show\'s renewal for a second season. Critical reviews of the series have been generally positive, citing the show\'s positive portrayal of little people. Conversely, other reviews have claimed that the show has a voyeuristic bend'], 'task_id': 1}
    {'query': ['Represent the Wikipedia question for retrieving relevant documents;', 'who sang waiting for a girl like you'], 'pos': ['Represent the Wikipedia document for retrieval;', 'Waiting for a Girl Like You Waiting for a Girl Like You "Waiting for a Girl Like You" is a 1981 power ballad by the British-American rock band Foreigner. The distinctive synthesizer theme was performed by the then-little-known Thomas Dolby, and this song also marked a major departure from their earlier singles because their previous singles were mid to upper tempo rock songs while this song was a softer love song with the energy of a power ballad. It was the second single released from the album "4" (1981) and was co-written by Lou Gramm and Mick Jones. It has become one of the band\'s most'], 'neg': ['Represent the Wikipedia document for retrieval;', 'Waiting for a Girl Like You held off the number 1 spot by Olivia Newton-John\'s single "Physical" for nine consecutive weeks, and then by Hall & Oates\' "I Can\'t Go for That (No Can Do)" for a tenth week on January 30, 1982. Because of its chart longevity, it ended up being the number 19 song on the Top 100 singles of 1982. The song was the band\'s biggest hit until "I Want to Know What Love Is" hit number 1 in 1985. The song lists at number 100 on ""Billboard"\'s Greatest Songs of All Time". Waiting for a Girl Like You "Waiting for a Girl'], 'task_id': 1}
    ...
    {'query': ['Represent the Wikipedia sentence for retrieving relevant documents;', 'i LOVE sweet martini drinks!'], 'pos': ['Represent the Wikipedia document for retrieval;', "Appletini Appletini\nAn Apple martini (Appletini for short) is a cocktail containing vodka and one or more of apple juice, apple cider, apple liqueur, or apple brandy.\nThis drink, originally called an Adam's Apple Martini because the bartender who created it was named Adam, was created in 1996 at Lola's West Hollywood restaurant.\nThe drink, Adam's Apple was advertised by Smirnoff in the July 1972 issue of Playboy Magazine to the inside front cover. The recipe called for an ounce or so of Smirnoff"], 'neg': ['Represent the Wikipedia document for retrieval;', "Aromatised wine similar beverages described in this legislation are 'aromatised wine-based drinks' (non-fortified) and 'aromatised wine-product cocktail' (blended, lower alcohol drink under 7% ABV).\nVarieties of aromatised wine.\nVarieties of aromatised wine Vermouth.\nVermouth is the most widely used aromatised wine due to its use in cocktails and famous commercial brands such as Martini and Cinzano which are commonplace around the world. Vermouth can be sweet or dry and red, white, pink or orange. It is traditionally"], 'task_id': 300}
]
```
Each instance consists of a query, a positive pair, a negative pair and the id of the task, which is used to ensure data in the same training batch are from the same task.
The MEDI data is available to be downloaded at [this link](https://drive.google.com/file/d/1vZ5c2oJNonGOvXzppNg5mHz24O6jcc52/view?usp=sharing).

### Train INSTRUCTOR
We provide the example script for training INSTRUCTOR. You may need to first download the [MEDI data](https://drive.google.com/file/d/1vZ5c2oJNonGOvXzppNg5mHz24O6jcc52/view?usp=sharing), unzip the folder and put `medi-data.json` under `--cache_dir`.
```python
python train.py --model_name_or_path sentence-transformers/gtr-t5-large --output_dir {output_directory} --cache_dir {cache_directory} --max_source_length 512 --num_train_epochs 10 --save_steps 500 --cl_temperature 0.1 --warmup_ratio 0.1 --learning_rate 2e-5 --overwrite_output_dir
```
We explain the arguments in the following:
* `--model_name_or_path`: Pretrained checkpoints to start with. We support both model id (e.g., `sentence-transformers/gtr-t5-large`, `sentence-transformers/sentence-t5-large`) or checkpoint path (e.g., checkpoint saved by transformers trainer).
* `--cl_temperature`: Temperature for contrastive loss
* `--cache_dir`: The directory to cache downloaded models and data. The downloaded MEDI data(`medi-data.json`) should be put under the directory `--cache_dir`.
* `--output_dir`: The directory to store the trained models(checkpoints) for evaluation. 

All the other arguments are standard `Huggingface's transformers` training arguments, such as `--overwrite_output_dir`, `--num_train_epochs`, `--learning_rate`. For details, refer to [Huggingface transformers](https://github.com/huggingface/transformers) 

## Evaluation
We evaluate INSTRUCTOR massively on 70 diverse tasks, spanning a wide range of tasks and domains. Specifically, we build our evaluation on three benchmarks, [MTEB](https://huggingface.co/spaces/mteb/leaderboard), [Billboard](https://arxiv.org/abs/2112.04139), and [Prompt Retrieval](https://arxiv.org/abs/2209.01975). We explain the details about running evaluation scripts in the following.
<!-- * MTEB is a comprehensive embedding evaluation benchmark that aims to provide a holistic view of embedding models.  It combines several conventional benchmarks (e.g., BEIR and STS) and spans a wide range of domain-specific datasets, including science, biology, and medicine. 
* Prompt Retrieval tasks aim to retrieve a few in-context learning (i.e., demonstration) examples from annotated examples given a test instance. The embedding model is used to encode all annotated examples and to find the few most similar examples to the test instance based on the cosine similarity. We evaluate embeddings by measuring the average performance on the downstream tasks. 
* Billboard applies INSTRUCTOR to automatic evaluations for text generation tasks. Following [Kasai et al. (2022a)](https://arxiv.org/abs/2112.04139), we measure the cosine similarity between the generated text and each reference text and take the maximum similarity score over all references available. We evaluate all embedding models by the Pearson correlation with the human judgments. -->

### MTEB
To evaluate the model performance on MTEB benchmark dataset, first install the MTEB library

```python
cd evaluation/MTEB
pip install -e .
```
Then run the following command:
```python
python examples/evaluate_model.py --model_name hkunlp/instructor-large --output_dir outputs --task_name ArguAna --result_file results
```
You can evaluate your trained model checkpoints by specifying `--model_name` and run all MTEB datasets by changing `--task_name`. Check [our paper](https://arxiv.org/abs/2212.09741) or [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) for evaluation metrics of all tasks.

### Billboard
To evaluate the model performance on Billboard, run the following command:
```python
cd evaluation/text_evaluation
python main.py --model_name hkunlp/instructor-large --task mscoco --add_prompt
```
You can evaluate your trained model checkpoints by specifying `--model_name` and run all Billboard datasets by changing `--task`. In all of the three datasets in Billboard, we report the Pearson correlation.

### Prompt Retrieval
To evaluate the model performance on Prompt Retrieval, run the following command:
```python
cd evaluation/prompt_retrieval
python main.py --embedding_model hkunlp/instructor-large --task rte --model_cache_dir {cache_dir} --output_dir {output_dir} --add_prompt
```
You can evaluate your trained model checkpoints by specifying `--model_name` and run prompt retrieval datasets by changing `--task`. In order to have a consistent metric, we cast all tasks in Prompt Retrieval into a "text-to-text" format, and report the Rouge-L score.


## Quantization 
To [**Quantize**](https://pytorch.org/docs/stable/quantization.html) the Instructor embedding model, run the following code: 

```python 
# imports 
import torch
from InstructorEmbedding import INSTRUCTOR

# load the model 
model = INSTRUCTOR('hkunlp/instructor-large', device='cpu')  # you can use GPU

# quantize the model 
qmodel = torch.quantization.quantize_dynamic(
model, {torch.nn.Linear}, dtype=torch.qint8)

# Inference 
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"

embeddings = qmodel.encode([[instruction,sentence]])  
# you can also normalize the embeddings:  normalize_embeddings=True 

print(f"Quantized Embeddings:\n {embeddings}")
````

It reduces the model size by 10x and inference time will be lesser than normal model :) 


## Bugs or questions?
If you have any question related to the code or the paper, feel free to email Hongjin (`hjsu@cs.hku.hk`) and Weijia (`swj0419@cs.washington.edu`). Please try to specify the problem with details so we can help you better and quicker.

## Citation
If you find our work helpful, please cite us:

```bibtex
@inproceedings{INSTRUCTOR,
  title={One Embedder, Any Task: Instruction-Finetuned Text Embeddings},
  author={Su, Hongjin and Shi, Weijia and Kasai, Jungo and Wang, Yizhong and Hu, Yushi and  Ostendorf, Mari and Yih, Wen-tau and Smith, Noah A. and  Zettlemoyer, Luke and Yu, Tao},
  url={https://arxiv.org/abs/2212.09741},
  year={2022},
}
```

## INSTRUCTOR Elsewhere
We thank the community's efforts for extending INSTRUCTOR!
* [LangChain](https://python.langchain.com/docs/integrations/text_embedding/instruct_embeddings) supports InstructEmbeddings, which use the INSTRUCTOR model.
* [MosaicML](https://www.mosaicml.com/inference) has included [Instructor-Large](https://huggingface.co/hkunlp/instructor-large) and [Instructor-XL](https://huggingface.co/hkunlp/instructor-xl)
* [embaas](https://embaas.io/docs/models/instructor) integrated [Instructor-Large](https://huggingface.co/hkunlp/instructor-large)
* [Haystack](https://haystack.deepset.ai/integrations/instructor-embedder) includes `InstructorTextEmbedder` and `InstructorDocumentEmbedder` components.
