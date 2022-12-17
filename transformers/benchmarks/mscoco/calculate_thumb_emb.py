import torch
import os
import json
import time
import random
import argparse
from transformers import AutoTokenizer,AutoModel
from tqdm import tqdm
from filelock import FileLock

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def calculate_emb(emb_tokenizer, all_batch, max_input_length, device, emb_model):
    n = len(all_batch)
    embeddings = []
    for i in tqdm(range(n), desc='calculate embeddings with context'):
        batch = all_batch[i:i + 1]
        if args.add_prompt:
            inner_ids = []
            context_mask = []
            contexts = emb_tokenizer([pair[0] for pair in batch]).input_ids
            input_texts = emb_tokenizer([''.join(pair) for pair in batch]).input_ids
            assert len(contexts) == len(input_texts)
            for one_context, one_input_text in zip(contexts, input_texts):
                cur_len = len(one_context + one_input_text)
                if cur_len >= max_input_length:
                    inner_ids.append((one_context + one_input_text)[:max_input_length])
                else:
                    inner_ids.append(one_context + one_input_text + [emb_tokenizer.pad_token_id] * (
                            max_input_length - len(one_context + one_input_text)))
                context_mask.append(
                    [-1] * min(len(one_context), max_input_length) +
                    [0] * (max(max_input_length - len(one_context), 0)))
        else:
            inner_ids = emb_tokenizer([s for s in batch]).input_ids

        inner_ids = torch.tensor(inner_ids)
        inner_attention_mask = inner_ids != emb_tokenizer.pad_token_id
        inner_attention_mask = inner_attention_mask.type_as(inner_ids)
        inner_ids = inner_ids.to(device)
        inner_attention_mask = inner_attention_mask.to(device)
        encoder_inner_representations = \
            emb_model(input_ids=inner_ids, attention_mask=inner_attention_mask)[0]
        if torch.cuda.is_available():
            inner_ids = inner_ids.to('cpu')
            inner_attention_mask = inner_attention_mask.to('cpu')
            encoder_inner_representations = encoder_inner_representations.to('cpu')
            torch.cuda.empty_cache()

        if args.add_prompt:
            inner_attention_mask += torch.tensor(context_mask)
        embeddings += mean_pooling(encoder_inner_representations, inner_attention_mask).tolist()

    embeddings = torch.tensor(embeddings)
    mean_embeddings = torch.mean(embeddings, 0, True)
    embeddings = embeddings - mean_embeddings
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.tolist()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--add_prompt", action="store_true",help="whether to add prompt")
args, _ = parser.parse_known_args()

emb_tokenizer = AutoTokenizer.from_pretrained('allenai/tk-instruct-large-def-pos',cache_dir=os.path.join('/scratch/acd13578qu','huggingface_models'))
definition = 'Task: Retrieve a duplicate caption that is semantically similar to the query caption. Encode: query \n\n'

with open('mscoco_THumB-1.0.jsonl') as f:
    lines = f.readlines()
all_texts_to_encode = []
for l in lines:
    d = json.loads(l)
    input_text = d['hyp']
    if args.add_prompt:
        if len(emb_tokenizer(definition+input_text)['input_ids'])<=512:
            input_text = [definition,input_text]
        else:
            input_text = ['', input_text]
    all_texts_to_encode.append(input_text)

with FileLock('model.lock'):
    if args.add_prompt:
        model_root = '/scratch/acd13578qu/metatrain_models/save_task_300_1000_seed0_template1_hard_tmp0025_bsz32'
    else:
        model_root = '/scratch/acd13578qu/metatrain_models/save_task_300_1000_seed0_template1_hard_tmp0025_bsz32_no_context'
    state_dict = torch.load(os.path.join(model_root,"pytorch_model.bin"),map_location="cpu")
    keys = list(state_dict.keys())
    for key in keys:
        if not key.startswith('encoder.'):
            state_dict[f'encoder.{key}'] = state_dict.pop(key)
    torch.save(state_dict, os.path.join(model_root,"pytorch_model.bin"))
    emb_model = AutoModel.from_pretrained(model_root,cache_dir=f'/scratch/acd13578qu/huggingface_models').get_encoder()
    emb_model.cuda()
embeddings = calculate_emb(emb_tokenizer=emb_tokenizer,all_batch=all_texts_to_encode,max_input_length=512,
                           device=torch.device('cuda'),emb_model=emb_model)

if args.add_prompt:
    identifier = 'with_prompt'
else:
    identifier = 'without_prompt'
with open(f'THumB-1.0_emb_{identifier}.json','w') as f:
    json.dump(embeddings,f,indent=4)









