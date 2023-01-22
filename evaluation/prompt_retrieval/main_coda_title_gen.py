import argparse
import random
import os
import copy
import torch
import numpy as np
import json
import nltk
import shutil
import string
from nltk.corpus import stopwords
import time
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datasets import load_metric
from transformers import GPTJForCausalLM,AutoModelForSeq2SeqLM
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from InstructorEmbedding import INSTRUCTOR
from datasets import load_dataset
from sklearn.metrics import f1_score
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_model', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--model_cache_dir', required=True,type=str)
parser.add_argument('--selection_2', type=str,default='similar')
parser.add_argument('--tag',default='title_gen', type=str)
parser.add_argument('--repeat', type=int,default=1)
parser.add_argument('--seed', type=int,default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--add_prompt', action='store_true')
args = parser.parse_args()

tokenizer_gpt = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
if not os.path.isdir('cache'):
    os.makedirs('cache',exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def calculate_sentence_transformer_embedding(examples,embedding_model,description,mean_normal=False):
    if args.add_prompt:
        text_to_encode = [['Represent the Scientific sentence; Input: ',raw_item['text'],0] for raw_item in examples]
    else:
        text_to_encode = [raw_item['text'] for raw_item in examples]
    num = len(text_to_encode)
    emb_model = INSTRUCTOR(embedding_model,cache_folder=args.model_cache_dir)
    embeddings = []
    bar = tqdm(range(0,num,20),desc=description)
    for i in range(0,num,20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    if mean_normal:
        mean_embeddings = torch.mean(embeddings, 0, True)
        embeddings = embeddings - mean_embeddings
    return embeddings

def find_indices_from_embeddings(embeddings,select_num,mean_normal=False):
    if mean_normal:
        embeddings = torch.tensor(embeddings, dtype=torch.float)
        embeddings_mean = torch.mean(embeddings, 0, True)
        embeddings = embeddings - embeddings_mean
    selected_indices = []
    first_id = random.choice(range(len(embeddings)))
    selected_indices.append(first_id)
    selected_representations = embeddings[first_id].reshape(1, -1)
    for count in range(select_num - 1):
        scores = np.sum(cosine_similarity(embeddings, selected_representations), axis=1)
        for i in selected_indices:
            scores[i] = float('inf')
        min_idx = np.argmin(scores)
        selected_representations = torch.cat((selected_representations,
                                              embeddings[min_idx].reshape(1, -1)), 0)
        selected_indices.append(min_idx.item())
    return selected_indices

def get_instance_length(idx,local_examples):
    return len(tokenizer_gpt(f"write a title\n{local_examples[idx]['text']}\n"
                             f"title:{local_examples[idx]['label']}\n\n")['input_ids'])

def get_instance(idx,local_examples):
    return f"write a title\n{local_examples[idx]['text']}\ntitle:{local_examples[idx]['label']}\n\n"

def select_2(train_embs,test_embs,downstream_train_examples,downstream_test_examples,tag,phase2_selection):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    bar = tqdm(range(len(downstream_test_examples)),desc="phase 2 similar select")
    if not os.path.isdir(f"{args.output_dir}/{tag}/prompts"):
        os.makedirs(f"{args.output_dir}/{tag}/prompts",exist_ok=True)
    for test_id,one_test_instance in enumerate(downstream_test_examples):
        prev_prompt_string_len = 0
        prompt_str = ''
        if phase2_selection in ['similar']:
            test_e_reshape = test_embs[test_id].reshape(1, -1)
            scores = cos(test_e_reshape, train_embs).numpy()
            sorted_indices = np.argsort(scores)
        elif phase2_selection in ['random']:
            sorted_indices = np.random.permutation(range(len(downstream_train_examples)))

        sorted_indices = sorted_indices[-16:]

        selected_indices = []
        num_indices = len(sorted_indices)
        for idx in range(num_indices-1,-1,-1):
            cur_len = get_instance_length(sorted_indices[idx],downstream_train_examples)
            prev_prompt_string_len += cur_len
            # print('prev_prompt_string_len: ',prev_prompt_string_len)
            cur_prompt_string_len = prev_prompt_string_len + \
                                    len(tokenizer_gpt(f"write a title\n{downstream_test_examples[test_id]['text']}\n"
                                                      f"title:")['input_ids'])
            # print('cur_prompt_string_len ',cur_prompt_string_len)
            if cur_prompt_string_len>1900:
                break
            selected_indices.append(idx)

        one_test_emb = test_embs[test_id]
        indices_scores = []
        for idx in selected_indices:
            indices_scores.append([idx, cos(train_embs[sorted_indices[idx]].reshape(1, -1), one_test_emb.reshape(1, -1)).item()])
        indices_scores = sorted(indices_scores, key=lambda x: x[1], reverse=True)
        new_selected_indices = [x[0] for x in indices_scores]
        if phase2_selection in ['similar']:
            assert new_selected_indices == selected_indices, f"new_selected_indices={new_selected_indices}, " \
                                                             f"selected_indices={selected_indices}"
        selected_indices = new_selected_indices

        select_num = len(selected_indices)
        second_phase_selected_indices = []
        # cur_train_data = []
        for idx in range(select_num-1,-1,-1):
            prompt_str += get_instance(sorted_indices[selected_indices[idx]],downstream_train_examples)
            # print("len(tokenizer_gpt(prompt_str)['input_ids']): ",len(tokenizer_gpt(prompt_str)['input_ids']))
            # cur_train_data.append({
            #     'input':f"The topic is {downstream_train_examples[sorted_indices[selected_indices[idx]]]['activity_label']}. {downstream_train_examples[sorted_indices[selected_indices[idx]]]['ctx_a']} "
            #             f"{downstream_train_examples[sorted_indices[selected_indices[idx]]]['ctx_b']} ",
            #     'options':downstream_train_examples[sorted_indices[selected_indices[idx]]]['endings'],
            #     'output':downstream_train_examples[sorted_indices[selected_indices[idx]]]['endings'][downstream_train_examples[sorted_indices[selected_indices[idx]]]['label']]
            # })
            # prompt_str += get_instance(sorted_indices[selected_indices[idx]],downstream_train_examples)
            second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item(),
                                                  downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']
                                                  ])
        prompt_str += f"write a title\n{downstream_test_examples[test_id]['text']}\ntitle:"
        # assert len(tokenizer_gpt(prompt_str)['input_ids'])<1900,f"len(tokenizer_gpt(prompt_str)['input_ids'])=" \
        #                                                         f"{len(tokenizer_gpt(prompt_str)['input_ids'])}"
        with open(f"{args.output_dir}/{tag}/prompts/{downstream_test_examples[test_id]['id']}.json",'w') as f:
            json.dump([[test_id,second_phase_selected_indices],
                       prompt_str,
                       downstream_test_examples[test_id]
                       ],f,indent=4)
        bar.update(1)

def get_repeat_num(prefix=None):
    all_subdirs = os.listdir('cache')
    m = 0
    for d in all_subdirs:
        if d.split('_')[-1].isdigit():
            if prefix is not None:
                if d.startswith(prefix):
                    m = max(m,int(d.split('_')[-1]))
            else:
                m = max(m, int(d.split('_')[-1]))
    return m+1

def process_examples(examples):
    processed_examples = []
    for i,e in enumerate(examples):
        processed_examples.append({
            'id':i,
            'text':e["input"],
            'label':e["output"][0]
        })
    return processed_examples

def get_examples(path):
    with open(path) as f:
        raw_dataset = json.load(f)
    positive_examples = raw_dataset["Positive Examples"]
    negative_examples = raw_dataset["Negative Examples"]
    all_examples = raw_dataset["Instances"]
    n = len(all_examples)
    eval_sub = random.sample(range(n),256)
    left = list(set(range(n))-set(eval_sub))
    train_sub = random.sample(left,3000)
    train_examples = process_examples([all_examples[i] for i in train_sub])
    eval_examples = process_examples([all_examples[i] for i in eval_sub])
    return process_examples(positive_examples),process_examples(negative_examples),\
           train_examples,eval_examples,raw_dataset["Definition"][0]

set_seed(args.seed)
positive_examples,negative_examples,coda_title_train_examples,coda_title_eval_examples,definition = get_examples('data/coda_title_gen.json')
if args.debug:
    coda_title_train_examples = coda_title_train_examples[:50]
    coda_title_eval_examples = coda_title_eval_examples[:5]
total_train_embeds = calculate_sentence_transformer_embedding(coda_title_train_examples, args.embedding_model,
                                                                  description='calculate train embeddings',
                                                                  mean_normal=True)

cur_repeat_num = 1

inference_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=args.model_cache_dir)
inference_model.cuda()
inference_model.eval()
device = torch.device('cuda')

for i in range(cur_repeat_num,cur_repeat_num+args.repeat):
    tag = f"{args.tag}_{args.selection_2}_{args.seed}_{i}"
    os.makedirs(f"{args.output_dir}/{tag}",exist_ok=True)
    total_train_examples_num = len(coda_title_train_examples)

    first_phase_selected_indices = range(total_train_examples_num)
    first_phase_selected_indices_to_cache = []
    processed_train_examples = []
    for selected_idx in first_phase_selected_indices:
        processed_train_examples.append(coda_title_train_examples[selected_idx])

    processed_eval_examples = coda_title_eval_examples
    total_test_embeds = calculate_sentence_transformer_embedding(processed_eval_examples,
                                                                      args.embedding_model,
                                                                      description='calculate eval embeddings',
                                                                      mean_normal=True)

    select_2(total_train_embeds[first_phase_selected_indices],total_test_embeds,
                     processed_train_examples,processed_eval_examples,tag,phase2_selection=args.selection_2)

    candidate_prompt_files = os.listdir(f"{args.output_dir}/{tag}/prompts")
    prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
    assert len(prompt_files)==len(processed_eval_examples),f"len(prompt_files)={len(prompt_files)}," \
                                                          f"len(processed_eval_examples)={len(processed_eval_examples)}"
    output_dir = f"{args.output_dir}/{tag}/results"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    count = 0
    # f = True
    golds = []
    preds = []
    correct = 0
    total = 0
    # while f:
    #     f = False
    #     count += 1
    bar = tqdm(range(len(prompt_files)), desc=f"  {tag}, {count} time")
    for file in prompt_files:
        bar.update(1)
        with open(f'{args.output_dir}/{tag}/prompts/{file}') as f:
            one_test_example = json.load(f)
        if not os.path.isfile(f"{args.output_dir}/{tag}/results/{file}"):
            context = one_test_example[1]
            input_ids = tokenizer_gpt(context, return_tensors="pt").input_ids
            input_ids = input_ids[:,:1900]
            input_len = input_ids.shape[1]
            input_ids = input_ids.to(device)
            gen_tokens = inference_model.generate(input_ids, do_sample=False, temperature=0.7, max_length=input_len+64,
                                        output_scores=True, return_dict_in_generate=True)
            generated_text = tokenizer_gpt.batch_decode(gen_tokens.sequences.view(-1,1), skip_special_tokens=True)
            stop = ['--', '\n', ';', '#']
            stop_index = len(generated_text)
            for i, c in enumerate(generated_text):
                if i > input_len and c.strip(' ') in stop:
                    stop_index = i
                    break
            prediction = ' '.join(generated_text[input_len:stop_index])
            golds.append(one_test_example[2]['label'])
            preds.append(prediction)
            with open(f"{output_dir}/{file}",'w') as f:
                json.dump([' '.join(generated_text[input_len:]),' '.join(generated_text[input_len:stop_index]),
                           one_test_example[2]['label'],input_len,stop_index],f,indent=4)
        else:
            with open(f"{output_dir}/{file}") as f:
                r = json.load(f)
            golds.append(one_test_example[2]['label'])
            preds.append(r[1])

    assert len(golds)==len(preds),f"len(golds)={len(golds)}, len(preds)={len(preds)}"
    preds,golds = postprocess_text(preds,golds)
    metric = load_metric("rouge")
    result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    result = result['rougeL']
    with open(f"{args.output_dir}/{tag}/result_summary.json",'w') as f:
        json.dump(result,f)
    print(result)



