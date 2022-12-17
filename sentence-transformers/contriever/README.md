## Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning

This repository contains pre-trained models, code for pre-training and evaluation for our paper [Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/abs/2112.09118).

We use a simple contrastive learning framework to pre-train models for information retrieval. Contriever, trained without supervision, is competitive with BM25 for R@100 on the BEIR benchmark. After finetuning on MSMARCO, Contriever obtains strong performance, especially for the recall at 100.

We also trained a multilingual version of Contriever, mContriever, achieving strong multilingual and cross-lingual retrieval performance.

## Getting started

Pre-trained models can be loaded through the HuggingFace transformers library:

```python
from src.contriever import Contriever
from transformers import AutoTokenizer

contriever = Contriever.from_pretrained("facebook/contriever") 
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer:
```

Then embeddings for different sentences can be obtained by doing the following:

```python

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings = model(**inputs)
```

Then similarity scores between the different sentences are obtained with a dot product between the embeddings:
```python

score01 = embeddings[0] @ embeddings[1] #1.0473
score02 = embeddings[0] @ embeddings[2] #1.0095
```

## Pre-trained models

The following pre-trained models are available:
* *contriever*: pre-trained on CC-net and English Wikipedia without any supervised data,
* *contriever-msmarco*: contriever with fine-tuning on MSMARCO,
* *mcontriever*: pre-trained on 29 languages using data from CC-net,
* *mcontriever-msmarco*: mcontriever with fine-tuning on MSMARCO.


```python
from src.contriever import Contriever

contriever = Contriever.from_pretrained("facebook/contriever") 
contriever_msmarco = Contriever.from_pretrained("facebook/contriever-msmarco")
mcontriever = Contriever.from_pretrained("facebook/mcontriever")
mcontriever_msmarco = Contriever.from_pretrained("facebook/mcontriever-msmarco")
```

## Evaluation

### Question answering retrieval

NaturalQuestions and TriviaQA data can be downloaded from the FiD repository <https://github.com/facebookresearch/fid>. The NaturalQuestions data slightly differs from the data provided in the DPR repository: we use the answers provided in the original NaturalQuestions data while DPR apply a post-processing step, which affects the tokenization of words.

<details>
<summary>
Retrieval is performed on the set of Wikipeda passages used in DPR. Download passages:
</summary>

```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```
</details>

<details>
<summary>
Generate passage embeddings:
</summary>
    
```bash
python generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever \
    --output_dir contriever_embeddings  \
    --passages psgs_w100.tsv \
    --shard_id 0 --num_shards 1 \
```
</details>

<details>
<summary>
Alternatively, download passage embeddings pre-computed with Contriever or Contriever-msmarco:
</summary>
    
```bash
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever/wikipedia_embeddings.tar
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
```
</details>

<details>
<summary>
Retrieve top-100 passages:
</summary>
    
```python
python passage_retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages psgs_w100.tsv \
    --passages_embeddings "contriever_embeddings/*" \
    --data nq_dir/test.json \
    --output_dir contriever_nq \
```
</details>

This leads to the following results:

<table>
  <tr>
    <td>Model</td>
    <td colspan="3">NaturalQuestions</td>
    <td colspan="3">TriviaQA</td>
  </tr>
  <tr>
      <td></td>
      <td>R@5</td>
      <td>R@20</td>
      <td>R@100</td>
      <td>R@5</td>
      <td>R@20</td>
      <td>R@100</td>
  </tr>
  <tr>
      <td>Contriever</td>
      <td>47.8</td>
      <td>67.8</td>
      <td>82.1</td>
      <td>59.4</td>
      <td>67.8</td>
      <td>83.2</td>
  </tr>
  <tr>
      <td>Contriever-msmarco</td>
      <td>65.7</td>
      <td>79.6</td>
      <td>88.0</td>
      <td>71.3</td>
      <td>80.4</td>
      <td>85.7</td>
  </tr>
</table>

### BEIR

Scores on the BEIR benchmark can be reproduced using [beireval.py](beireval.py).

```bash
python beireval.py --model_name_or_path contriever-msmarco --dataset scifact
```


The Touche-2020 dataset has been update in BEIR, thus results will differ if the current version is used.

<table>
  <tr>
    <td>nDCG@10</td>
    <td>Avg</td>
    <td>MSMARCO</td>
    <td>TREC-Covid</td>
    <td>NFCorpus</td>
    <td>NaturalQuestions</td>
    <td>HotpotQA</td>
    <td>FiQA</td>
    <td>ArguAna</td>
    <td>Tóuche-2020</td>
    <td>Quora</td>
    <td>CQAdupstack</td>
    <td>DBPedia</td>
    <td>Scidocs</td>
    <td>Fever</td>
    <td>Climate-fever</td>
    <td>Scifact</td>
  </tr>
  <tr>
    <td>Contriever</td>
    <td>37.7</td>
    <td>20.6</td>
    <td>27.4</td>
    <td>31.7</td>
    <td>25.4</td>
    <td>48.1</td>
    <td>24.5</td>
    <td>37.9</td>
    <td>19.3</td>
    <td>83.5</td>
    <td>28.4</td>
    <td>29.2</td>
    <td>14.9</td>
    <td>68.2</td>
    <td>15.5</td>
    <td>64.9</td>
  </tr>
  <tr>
    <td>Contriever-msmarco</td>
    <td>46.6</td>
    <td>40.7</td>
    <td>59.6</td>
    <td>32.8</td>
    <td>49.8</td>
    <td>63.8</td>
    <td>32.9</td>
    <td>44.6</td>
    <td>23.0</td>
    <td>86.5</td>
    <td>34.5</td>
    <td>41.3</td>
    <td>16.5</td>
    <td>75.8</td>
    <td>23.7</td>
    <td>67.7</td>
  </tr>
</table>



<table>
  <tr>
    <td>R@100</td>
    <td>Avg</td>
    <td>MSMARCO</td>
    <td>TREC-covid</td>  
    <td>NFCorpus</td>
    <td>NaturalQuestions</td>
    <td>HotpotQA</td>
    <td>FiQA</td>
    <td>ArguAna</td>
    <td>Tóuche-2020</td>
    <td>Quora</td>
    <td>CQAdupstack</td>
    <td>DBPedia</td>
    <td>Scidocs</td>
    <td>Fever</td>
    <td>Climate-fever</td>
    <td>Scifact</td>
  </tr>
  <tr>
    <td>Contriever-msmarco</td>
    <td>59.6</td>
    <td>67.2</td>
    <td>17.2</td>
    <td>29.4</td>
    <td>77.1</td>
    <td>70.4</td>
    <td>56.2</td>
    <td>90.1</td>
    <td>22.5</td>
    <td>98.7</td>
    <td>61.4</td>
    <td>45.3</td>
    <td>36.0</td>
    <td>93.6</td>
    <td>44.1</td>
    <td>92.6</td>
  </tr>
   <tr>
    <td>Contriever-msmarco</td>
    <td>67.0</td>
    <td>89.1</td>
    <td>40.7</td>
    <td>30.0</td>
    <td>92.5</td>
    <td>77.7</td>
    <td>65.6</td>
    <td>97.7</td>
    <td>29.4</td>
    <td>99.3</td>
    <td>66.3</td>
    <td>54.1</td>
    <td>37.8</td>
    <td>94.9</td>
    <td>57.4</td>
    <td>94.7</td>
  </tr>
</table>

## Multilingual evaluation

We evaluate mContriever on Mr. Tydi v1.1 and a cross-lingual retrieval setting derived from MKQA. You will find below steps to reproduce our results on these datasets.

### Mr. TyDi v1.1

For multilingual evaluation on Mr. TyDi v1.1, we download datasets from <https://github.com/castorini/mr.tydi> and convert them to the BEIR format using (data_scripts/convertmrtydi2beir.py)[data_scripts/convertmrtydi2beir]. 
Evaluation on Swahili can be performed by doing the following:

<details>
<summary>
Download data:
</summary>

```bash
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.1-swahili.tar.gz -P mrtydi
tar -xf mrtydi/mrtydi-v1.1-swahili.tar.gz -C mrtydi
gzip -d mrtydi/mrtydi-v1.1-swahili/collection/docs.jsonl.gz
```
</details>

<details>
<summary>
Convert data:
</summary>

```bash
python data_scripts/convertmrtydi2beir.py mrtydi/mrtydi-v1.1-swahili mrtydi/mrtydi-v1.1-swahili
```
</details>

<details>
<summary>
Evaluation:

</summary>

```bash
python beireval.py --model_name_or_path facebook/mcontriever --dataset mrtydi/mrtydi-v1.1-swahili --normalize_text
```
</details>


<table>
  <tr>
    <td>MRR@100</td>
    <td>ar</td>
    <td>bn</td>
    <td>en</td>
    <td>fi</td>
    <td>id</td>
    <td>ja</td>
    <td>ko</td>
    <td>ru</td>
    <td>sw</td>
    <td>te</td>
    <td>th</td>
    <td>avg</td>
  </tr>
  <tr>
    <td>mContriever</td>
    <td>27.3</td>
    <td>36.3</td>
    <td>9.2</td>
    <td>21.1</td>
    <td>23.5</td>
    <td>19.5</td>
    <td>22.3</td>
    <td>17.5</td>
    <td>38.3</td>
    <td>22.5</td>
    <td>37.2</td>
    <td>25.0</td>
  </tr>
  <tr>
    <td>mContriever-msmarco</td>
    <td>43.4</td>
    <td>42.3</td>
    <td>27.1</td>
    <td>25.1</td>
    <td>42.6</td>
    <td>32.4</td>
    <td>34.2</td>
    <td>36.1</td>
    <td>51.2</td>
    <td>37.4</td>
    <td>40.2</td>
    <td>38.4</td>
  </tr>
  <tr>
    <td>+ Mr. TyDi</td>
    <td>72.4</td>
    <td>67.2</td>
    <td>56.6</td>
    <td>60.2</td>
    <td>63.0</td>
    <td>54.9</td>
    <td>55.3</td>
    <td>59.7</td>
    <td>70.7</td>
    <td>90.3</td>
    <td>67.3</td>
    <td>65.2</td>
  </tr>
</table>


<table>
  <tr>
    <td>R@100</td>
    <td>ar</td>
    <td>bn</td>
    <td>en</td>
    <td>fi</td>
    <td>id</td>
    <td>ja</td>
    <td>ko</td>
    <td>ru</td>
    <td>sw</td>
    <td>te</td>
    <td>th</td>
    <td>avg</td>
  </tr>
  <tr>
    <td>mContriever</td>
    <td>82.0</td>
    <td>89.6</td>
    <td>48.8</td>
    <td>79.6</td>
    <td>81.4</td>
    <td>72.8</td>
    <td>66.2</td>
    <td>68.5</td>
    <td>88.7</td>
    <td>80.8</td>
    <td>90.3</td>
    <td>77.2</td>
  </tr>
  <tr>
    <td>mContriever-msmarco</td>
    <td>88.7</td>
    <td>91.4</td>
    <td>77.2</td>
    <td>88.1</td>
    <td>89.8</td>
    <td>81.7</td>
    <td>78.2</td>
    <td>83.8</td>
    <td>91.4</td>
    <td>96.6</td>
    <td>90.5</td>
    <td>87.0</td>
  </tr>
  <tr>
    <td>+ Mr. TyDi</td>
    <td>94.0</td>
    <td>98.6</td>
    <td>92.2</td>
    <td>92.7</td>
    <td>94.5</td>
    <td>88.8</td>
    <td>88.9</td>
    <td>92.4</td>
    <td>93.7</td>
    <td>98.9</td>
    <td>95.2</td>
    <td>93.6</td>
  </tr>
</table>



### Cross-lingual MKQA

Here our goal is to measure how well retrievers are to retrieve relevant documents in English Wikipedia given a query in another language.
For this we use MKQA and evaluate if the answer is in the retrieved documents based on the DPR evaluation script.

<details>
<summary>
Download data:

</summary>

```bash
wget https://raw.githubusercontent.com/apple/ml-mkqa/master/dataset/mkqa.jsonl.gz
```
</details>

<details>
<summary>
Preprocess data:

</summary>

```bash
python data_scripts/preprocess_xmkqa.py mkqa.jsonl xmkqa
```
</details>

<details>
<summary>
Generate embeddings:

</summary>

```bash
python generate_passage_embeddings.py \
    --model_name_or_path facebook/mcontriever \
    --output_dir mcontriever_embeddings  \
    --passages psgs_w100.tsv \
    --shard_id 0 --num_shards 1 \
    --lowercase --normalize_text \
```
</details>

<details>
<summary>
Alternatively, download passage embeddings pre-computed with mContriever or mContriever-msmarco:
</summary>
    
```bash
wget https://dl.fbaipublicfiles.com/contriever/embeddings/mcontriever/wikipedia_embeddings.tar
wget https://dl.fbaipublicfiles.com/contriever/embeddings/mcontriever-msmarco/wikipedia_embeddings.tar
```
</details>

 
<details>
<summary>
Retrieve passages and compute retrieval accuracy:

</summary>

```bash

python passage_retrieval.py \
    --model_name_or_path facebook/mcontriever \
    --passages psgs_w100.tsv \
    --passages_embeddings "mcontriever_embeddings/*" \
    --data "xmkqa/*.jsonl" \
    --output_dir mcontriever_xmkqa \
    --lowercase --normalize_text \
```
</details>

<table>
  <tr>
    <td>R@100</td>
    <td>avg</td>
    <td>en</td>
    <td>ar</td>
    <td>fi</td>
    <td>ja</td>
    <td>ko</td>
    <td>ru</td>
    <td>es</td>
    <td>sv</td>
    <td>he</td>
    <td>th</td>
    <td>da</td>
    <td>de</td>
    <td>fr</td>
    <td>it</td>
    <td>nl</td>
    <td>pl</td>
    <td>pt</td>
    <td>hu</td>
    <td>vi</td>
    <td>ms</td>
    <td>km</td>
    <td>no</td>
    <td>tr</td>
    <td>zh-cn</td>
    <td>zh-hk</td>
    <td>zh-tw</td>
  </tr>
    
  <tr>
    <td>mContriever</td>
    <td>49.2</td>
    <td>65.3</td>
    <td>43.0</td>
    <td>43.1</td>
    <td>47.1</td>
    <td>44.8</td>
    <td>51.8</td>
    <td>37.2</td>
    <td>54.5</td>
    <td>44.7</td>
    <td>51.4</td>
    <td>49.3</td>
    <td>49.0</td>
    <td>50.2</td>
    <td>56.7</td>
    <td>61.7</td>
    <td>44.4</td>
    <td>54.5</td>
    <td>47.7</td>
    <td>45.1</td>
    <td>56.7</td>
    <td>27.8</td>
    <td>50.2</td>
    <td>44.3</td>
    <td>54.3</td>
    <td>51.9</td>
    <td>52.5</td>
  </tr>
  <tr>
    <td>mContriever-msmarco</td>
    <td>65.6</td>
    <td>75.6</td>
    <td>53.3</td>
    <td>66.6</td>
    <td>60.4</td>
    <td>55.4</td>
    <td>64.7</td>
    <td>70.0</td>
    <td>70.8</td>
    <td>59.6</td>
    <td>63.5</td>
    <td>72.0</td>
    <td>66.6</td>
    <td>70.1</td>
    <td>70.3</td>
    <td>71.4</td>
    <td>68.8</td>
    <td>68.5</td>
    <td>66.7</td>
    <td>67.8</td>
    <td>71.6</td>
    <td>37.8</td>
    <td>71.5</td>
    <td>68.7</td>
    <td>64.1</td>
    <td>64.5</td>
    <td>64.3</td>
  </tr>
</table>

<table>
  <tr>
    <td>R@20</td>
    <td>avg</td>
    <td>en</td>
    <td>ar</td>
    <td>fi</td>
    <td>ja</td>
    <td>ko</td>
    <td>ru</td>
    <td>es</td>
    <td>sv</td>
    <td>he</td>
    <td>th</td>
    <td>da</td>
    <td>de</td>
    <td>fr</td>
    <td>it</td>
    <td>nl</td>
    <td>pl</td>
    <td>pt</td>
    <td>hu</td>
    <td>vi</td>
    <td>ms</td>
    <td>km</td>
    <td>no</td>
    <td>tr</td>
    <td>zh-cn</td>
    <td>zh-hk</td>
    <td>zh-tw</td>
  </tr>
  <tr>
    <td>mContriever</td>
    <td>31.4</td>
    <td>50.2</td>
    <td>26.6</td>
    <td>26.7</td>
    <td>29.4</td>
    <td>27.9</td>
    <td>32.7</td>
    <td>20.7</td>
    <td>37.6</td>
    <td>22.2</td>
    <td>31.1</td>
    <td>31.2</td>
    <td>31.2</td>
    <td>30.7</td>
    <td>38.6</td>
    <td>45.1</td>
    <td>25.1</td>
    <td>37.6</td>
    <td>28.3</td>
    <td>27.3</td>
    <td>39.6</td>
    <td>15.7</td>
    <td>33.2</td>
    <td>26.5</td>
    <td>35.0</td>
    <td>32.7</td>
    <td>32.5</td>
  </tr>
  <tr>
      <td>mContriever-msmarco</td>
    <td>53.9</td>
    <td>67.2</td>
    <td>40.1</td>
    <td>55.1</td>
    <td>46.2</td>
    <td>41.7</td>
    <td>52.3</td>
    <td>59.3</td>
    <td>60.0</td>
    <td>45.6</td>
    <td>52.0</td>
    <td>62.0</td>
    <td>54.8</td>
    <td>59.3</td>
    <td>59.4</td>
    <td>60.9</td>
    <td>58.1</td>
    <td>56.9</td>
    <td>55.2</td>
    <td>55.9</td>
    <td>60.9</td>
    <td>26.2</td>
    <td>61.0</td>
    <td>56.7</td>
    <td>50.9</td>
    <td>51.9</td>
    <td>51.2</td>
  </tr>
</table>


## Training


### Data pre-processing
We perform pre-training on data from CCNet and Wikipedia.
Contriever, the English monolingual model, is trained on English data from Wikipedia and CCNet.
mContriever, the multilingual model, is pre-trained on 29 languages using data from CCNet.
After converting data into a text file, we tokenize and chunk it into multiple sub-files using the [`data_scripts/tokenization_script.sh`](data_scripts/tokenization_script.sh).
The different chunks are then loaded separately by the different processes in a distributed job.
For mContriever, we use the option `--normalize_text` to preprocess data, this normalize certain common caracters that are not present in mBERT tokenizer.

### Training
[`train.py`](train.py) provides the code for the contrastive training phase of Contriever.

<details>
<summary>
For Contriever, the English monolingual model, we use the following options on 32 gpus:
</summary>


```bash
python train.py \
        --retriever_model_id bert-base-uncased --pooling average \
        --augmentation delete --prob_augmentation 0.1 \
        --train_data "data/wiki/ data/cc-net/" --loading_mode split \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.9995 --moco_queue 131072 --temperature 0.05 \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --scheduler linear --optim adamw --per_gpu_batch_size 64 \
        --output_dir /checkpoint/gizacard/contriever/xling/contriever \

```
</details>

<details>
<summary>
For mContriever, the multilingual model, we use the following options on 32 gpus:
</summary>


```bash
TDIR=encoded-data/bert-base-multilingual-cased/
TRAINDATASETS="${TDIR}fr_XX ${TDIR}en_XX ${TDIR}ar_AR ${TDIR}bn_IN ${TDIR}fi_FI ${TDIR}id_ID ${TDIR}ja_XX ${TDIR}ko_KR ${TDIR}ru_RU ${TDIR}sw_KE ${TDIR}hu_HU ${TDIR}he_IL ${TDIR}it_IT ${TDIR}km_KM ${TDIR}ms_MY ${TDIR}nl_XX ${TDIR}no_XX ${TDIR}pl_PL ${TDIR}pt_XX ${TDIR}sv_SE ${TDIR}te_IN ${TDIR}th_TH ${TDIR}tr_TR ${TDIR}vi_VN ${TDIR}zh_CN ${TDIR}zh_TW ${TDIR}es_XX ${TDIR}de_DE ${TDIR}da_DK"

python train.py \
        --retriever_model_id bert-base-multilingual-cased --pooling average \
        --train_data ${TRAINDATASETS} --loading_mode split \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.999 --moco_queue 32768 --temperature 0.05 \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --scheduler linear --optim adamw --per_gpu_batch_size 64 \
        --output_dir /checkpoint/gizacard/contriever/xling/mcontriever \
```

</details>

The full training script used on our slurm cluster are available in the [`example_scripts`](example_scripts) folder.


## References

If you find this repository useful, please consider giving a star and citing this work:

[1] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, E. Grave [*Unsupervised Dense Information Retrieval with Contrastive Learning*](https://arxiv.org/abs/2112.09118)

```bibtex
@misc{izacard2021contriever,
      title={Unsupervised Dense Information Retrieval with Contrastive Learning}, 
      author={Gautier Izacard and Mathilde Caron and Lucas Hosseini and Sebastian Riedel and Piotr Bojanowski and Armand Joulin and Edouard Grave},
      year={2021},
      url = {https://arxiv.org/abs/2112.09118},
      doi = {10.48550/ARXIV.2112.09118},
}
```

## License

See the [LICENSE](LICENSE) file for more details.
