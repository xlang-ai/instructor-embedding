import torch
import os
import json
import time
import random
import numpy as np
import argparse
from transformers import AutoTokenizer,AutoModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from filelock import FileLock

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--add_prompt", action="store_true",help="whether to add prompt")
parser.add_argument("--m", default=None,type=str)
args, _ = parser.parse_known_args()

if args.add_prompt:
    identifier = 'with_prompt'
else:
    identifier = 'without_prompt'
# with open(f'../../../text_eval/mscoco/vecs/mscoco_references_emb_{identifier}.json','w') as f:
#     json.dump(['check'],f,indent=4)

if args.add_prompt:
    model_name = '/scratch/acd13578qu/metatrain_models/1115_1817/checkpoint-16375'
    write_file_identifier = 'sentence_t5_large_trained_with_prompt'
else:
    model_name = '/scratch/acd13578qu/outputs/sentence-t5-large/cl_without_prompt/train_bi-encoder-standard_cl-sentence-transformers-sentence-t5-large_prompt0/batch_size_8-2022-10-19_21-44-02/80000'
    write_file_identifier = 'sentence_t5_large_trained_without_prompt'
if args.m is not None:
    model_name = args.m
    write_file_identifier = model_name.replace('/','_')

emb_tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=os.path.join('/scratch/acd13578qu','huggingface_models'))
definition = 'Represent a sentence; Input: '

with open('cnndm_references.jsonl') as f:
    lines = f.readlines()
all_texts_to_encode = []
all_ref_nums = []
num_count = 0
for l in lines:
    d = json.loads(l)
    cur_num = len(d["refs"])
    all_ref_nums.append(num_count)
    num_count += cur_num
    for idx in range(cur_num):
        input_text = d["refs"][idx]
        if args.add_prompt:
            if len(emb_tokenizer(definition+input_text)['input_ids'])<=256:
                input_text = [definition,input_text,0]
        all_texts_to_encode.append(input_text)
all_ref_nums.append(num_count)
total_line_num = len(lines)
assert len(all_ref_nums)==total_line_num+1

with FileLock('model.lock'):
    emb_model = SentenceTransformer(model_name,cache_folder='/scratch/acd13578qu/huggingface')
    emb_model.cuda()
embeddings = np.asarray(emb_model.encode(all_texts_to_encode, batch_size=32))

# mean_emb = np.mean(embeddings, axis=0)
# embeddings -= mean_emb
embeddings = embeddings.tolist()
reshaped_embeddings = []
for idx in range(total_line_num):
    reshaped_embeddings.append(embeddings[all_ref_nums[idx]:all_ref_nums[idx+1]])
with open(f'../../../text_eval/vecs/cnndm_references_emb_{write_file_identifier}_{identifier}.json','w') as f:
    json.dump(reshaped_embeddings,f,indent=4)
print(np.array(reshaped_embeddings).shape)
print(f'cnndm_references_emb_{write_file_identifier}_{identifier}.json')









