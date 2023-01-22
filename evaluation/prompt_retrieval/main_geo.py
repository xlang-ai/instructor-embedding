import sqlite3
import sqlparse
import random
import numpy as np
import torch
import os
import argparse
import json
import time
import pandas as pd
import pickle as pkl
import sys
from os.path import dirname, abspath
parent_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(1,parent_dir)
from instructor_new import INSTRUCTOR
from bridge_content_encoder import get_database_matches
from transformers import AutoTokenizer
from tqdm import tqdm
from torch import nn
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from utils import codex_execution
from datasets import load_metric

parser = argparse.ArgumentParser()
parser.add_argument('--model_key', type=str)
parser.add_argument('--prompt_retrieval_method',  default='similar',type=str)
parser.add_argument('--output_dir', required=True,type=str)
parser.add_argument('--seed', default=0,type=int)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--embedding_model', required=True,type=str)
parser.add_argument('--add_prompt', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

tokenizer_for_length = AutoTokenizer.from_pretrained('gpt2')
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir,exist_ok=True)

model_keys = args.model_key.split('##')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_sentence_transformer_embedding(examples,embedding_model,mean_normal=False):
    if not args.add_prompt:
        text_to_encode = [raw_item["seq_in"] for raw_item in examples]
    else:
        print('add prompt')
        text_to_encode = [['Represent a Geography example; Input: ',raw_item["seq_in"],0] for raw_item in examples]
    num = len(text_to_encode)
    emb_model = INSTRUCTOR(embedding_model)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    if mean_normal:
        mean_embeddings = torch.mean(embeddings, 0, True)
        embeddings = embeddings - mean_embeddings
    return embeddings

def maybe_add_quotes(val):
    if isinstance(val, str):
        return "'" + val + "'"
    return str(val)

def get_db_schemas():
    with sqlite3.connect(f'data/geoquery.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            schemas[table[0]] = cursor.fetchone()[0]
        return schemas

def get_db_rows(*, rows=5, db_content_matching=True, question=None):
    db_path = f'data/geoquery.sqlite'
    results = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            cursor.execute("PRAGMA table_info({})".format(table[0]))
            results[table[0]] = pd.read_sql_query(f"SELECT * FROM {table[0]} LIMIT {rows}", conn)
        if db_content_matching:
            for table in results.keys():
                where_clauses = list()
                for col in results[table].keys():
                    matches = get_database_matches(question, table, col, db_path)
                    for match in matches:
                        where_clause = f'{col} = {maybe_add_quotes(match)}'
                        where_clauses.append(where_clause)
                if len(where_clauses) > 0:
                    table_matches = pd.read_sql_query(
                        f"SELECT DISTINCT * FROM {table} WHERE {' OR '.join(where_clauses)} LIMIT {rows}", conn)
                    results[table] = table_matches
    for k, v in results.items():
        results[k] = v.to_string(index=False)
    return results

def get_db_prompt(*, schema=True, rows=0, db_content_matching=True,question=None, reindent_aligned=True):
    schemas = get_db_schemas()
    examples = get_db_rows(rows=rows, db_content_matching=db_content_matching, question=question)
    prompt = ''
    if schema or (rows > 0):
        for table in schemas.keys():
            if schema:
                prompt += sqlparse.format(schemas[table], reindent_aligned=reindent_aligned)
                prompt += '\n'
            if rows > 0:
                prompt += '/*\n'
                # prompt += f'{rows} example rows from table {table}:\n'
                # prompt += f'SELECT * FROM {table} LIMIT {rows};\n'
                if not schema:
                    prompt += f'Table: {table}\n'
                prompt += examples[table]
                prompt += '\n*/\n'
            prompt += '\n'
    return prompt

def get_prompt_instructions():
    return "-- Using valid SQLite, answer the following questions for the tables provided above.\n"

def construct_prompt(db_prompt, instructions, question):
    return f"{db_prompt}{instructions}\n-- {question}\nSELECT"

def get_instance_length(idx,local_examples):
    return len(tokenizer_for_length(f"-- {local_examples[idx]['question']}\n{local_examples[idx]['query']}\n\n")['input_ids'])

def select_2(train_embs,test_embs,downstream_train_examples,downstream_test_examples,phase2_selection):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    bar = tqdm(range(len(downstream_test_examples)),desc="phase 2 similar select")
    if not os.path.isdir(os.path.join(args.output_dir,'prompts')):
        os.makedirs(os.path.join(args.output_dir,'prompts'),exist_ok=True)
    instruction = get_prompt_instructions()
    prompt_dir = os.path.join(args.output_dir,'prompts')
    for test_id,one_test_instance in enumerate(downstream_test_examples):
        cur_prompt = get_db_prompt(rows=3,question=one_test_instance['question'])+instruction
        prompt_str = cur_prompt
        prev_prompt_string_len = len(tokenizer_for_length(cur_prompt)['input_ids'])
        if phase2_selection in ['similar']:
            # print('similar selection')
            test_e_reshape = test_embs[test_id].reshape(1, -1)
            scores = cos(test_e_reshape, train_embs).numpy()
            sorted_indices = np.argsort(scores)
        elif phase2_selection in ['random']:
            sorted_indices = np.random.permutation(range(len(downstream_train_examples)))
        selected_indices = []
        # sorted_indices = sorted_indices[-16:]
        num_indices = len(sorted_indices)
        for idx in range(num_indices-1,-1,-1):
            prev_prompt_string_len += get_instance_length(sorted_indices[idx],downstream_train_examples)
            cur_prompt_string_len = prev_prompt_string_len + \
                    len(tokenizer_for_length(f"-- {downstream_test_examples[test_id]['question']}\nSELECT")['input_ids'])
            if cur_prompt_string_len>3800:
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
        for idx in range(select_num-1,-1,-1):
            prompt_str += f"-- {downstream_train_examples[sorted_indices[selected_indices[idx]]]['question']}\n" \
                          f"{downstream_train_examples[sorted_indices[selected_indices[idx]]]['query']}\n\n"
            second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item(),
                                                  downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']
                                                  ])
        assert one_test_instance['question']==downstream_test_examples[test_id]['question'],\
            f"one_test_instance['question']={one_test_instance['question']}, " \
            f"downstream_test_examples[test_id]['question']={downstream_test_examples[test_id]['question']}"

        prompt_str += f"-- {one_test_instance['question']}\nSELECT"
        with open(os.path.join(prompt_dir,f"{downstream_test_examples[test_id]['id']}.json"),'w') as f:
            json.dump([[test_id,second_phase_selected_indices,],
                       prompt_str,downstream_test_examples[test_id]
                       ],f,indent=4)
        bar.update(1)

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

def vote_k_select(embeddings,select_num,k,overlap_threshold,vote_file=None):
    n = len(embeddings)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n),desc=f'vote {k} selection')
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
            for idx in sorted_indices:
                if idx!=i:
                    vote_stat[idx].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file,'w') as f:
                json.dump(vote_stat,f)
    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    j = 0
    selected_indices = []
    while len(selected_indices)<select_num and j<len(votes):
        candidate_set = set(votes[j][1])
        flag = True
        for pre in range(j):
            cur_set = set(votes[pre][1])
            if len(candidate_set.intersection(cur_set))>=overlap_threshold*len(candidate_set):
                flag = False
                break
        if not flag:
            j += 1
            continue
        selected_indices.append(int(votes[j][0]))
        j += 1
    if len(selected_indices)<select_num:
        unselected_indices = []
        cur_num = len(selected_indices)
        for i in range(n):
            if not i in selected_indices:
                unselected_indices.append(i)
        selected_indices += random.sample(unselected_indices,select_num-cur_num)
    return selected_indices

def v2_vote_k_select(embeddings,select_num,k,vote_file=None):
    n = len(embeddings)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n),desc=f'v2 vote {k} selection')
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
            for idx in sorted_indices:
                if idx!=i:
                    vote_stat[idx].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file,'w') as f:
                json.dump(vote_stat,f)
    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices

def select_iterative(train_embs,test_embs,downstream_train_examples,downstream_test_examples,phase2_selection,identifier=''):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    bar = tqdm(range(len(downstream_test_examples)), desc="prepare prompts for probability selection")
    cur_prompt_dir = os.path.join(args.output_dir,f'prompts_iterative_{identifier}')
    if not os.path.isdir(cur_prompt_dir):
        os.makedirs(cur_prompt_dir, exist_ok=True)
    instruction = get_prompt_instructions()
    for test_id, one_test_instance in enumerate(downstream_test_examples):
        cur_prompt = get_db_prompt(rows=3, question=one_test_instance['question']) + instruction
        prompt_str = cur_prompt
        prev_prompt_string_len = len(tokenizer_for_length(cur_prompt)['input_ids'])

        test_e_reshape = test_embs[test_id].reshape(1, -1)
        scores = cos(test_e_reshape, train_embs).numpy()
        sorted_indices = np.argsort(scores).tolist()
        while scores[sorted_indices[-1]]==1:
            sorted_indices.pop()
            if len(sorted_indices)==0:
                print('sorted indices: ',scores,len(sorted_indices))
            if sorted_indices[-1]>=len(scores):
                print(sorted_indices[-1],len(scores))
        sorted_indices = np.array(sorted_indices)

        selected_indices = []
        num_indices = len(sorted_indices)
        for idx in range(num_indices - 1, -1, -1):
            prev_prompt_string_len += get_instance_length(sorted_indices[idx], downstream_train_examples)
            cur_prompt_string_len = prev_prompt_string_len + \
                                    len(tokenizer_for_length(
                                        f"-- {downstream_test_examples[test_id]['question']}\nSELECT")['input_ids'])
            if cur_prompt_string_len > 3800:
                break
            selected_indices.append(idx)

        one_test_emb = test_embs[test_id]
        indices_scores = []
        for idx in selected_indices:
            indices_scores.append(
                [idx, cos(train_embs[sorted_indices[idx]].reshape(1, -1), one_test_emb.reshape(1, -1)).item()])
        indices_scores = sorted(indices_scores, key=lambda x: x[1], reverse=True)
        new_selected_indices = [x[0] for x in indices_scores]
        if phase2_selection in ['similar']:
            assert new_selected_indices == selected_indices, f"new_selected_indices={new_selected_indices}, " \
                                                             f"selected_indices={selected_indices}"
        selected_indices = new_selected_indices

        selected_indices = new_selected_indices

        select_num = len(selected_indices)
        second_phase_selected_indices = []
        for idx in range(select_num - 1, -1, -1):
            prompt_str += f"-- {downstream_train_examples[sorted_indices[selected_indices[idx]]]['question']}\n" \
                          f"{downstream_train_examples[sorted_indices[selected_indices[idx]]]['query']}\n\n"
            second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item(),
                                                  downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']
                                                  ])
        assert one_test_instance['question'] == downstream_test_examples[test_id]['question'], \
            f"one_test_instance['question']={one_test_instance['question']}, " \
            f"downstream_test_examples[test_id]['question']={downstream_test_examples[test_id]['question']}"

        prompt_str += f"-- {one_test_instance['question']}\nSELECT"
        with open(os.path.join(cur_prompt_dir,f"{downstream_test_examples[test_id]['id']}.json"), 'w') as f:
            json.dump([[test_id, second_phase_selected_indices, ],
                       prompt_str,downstream_test_examples[test_id]
                       ], f, indent=4)
        bar.update(1)

def v2_vote_k_prob(train_embs,downstream_train_examples,
                        phase2_selection,):
    knn = 150
    selected_indices = v2_vote_k_select(embeddings=train_embs,
                                         select_num=args.batch_size,
                                         k=knn,
                                         vote_file=os.path.join(args.output_dir,f"v2_vote_{args.selective_annotation_method}.json"))

    cur_annotated_examples = [downstream_train_examples[idx] for idx in selected_indices]
    select_iterative(train_embs[selected_indices], train_embs, cur_annotated_examples, downstream_train_examples,
                     phase2_selection, identifier='0')

    prompt_cache_dir = os.path.join(args.output_dir, f"prompts_iterative_0")
    candidate_prompt_files = os.listdir(prompt_cache_dir)
    prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
    assert len(prompt_files) == len(downstream_train_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                               f"len(downstream_train_examples)={len(downstream_train_examples)}"
    output_dir = os.path.join(args.output_dir,'results_iterative_0')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    count = 0
    execution_count = 0
    f = True
    while f:
        f = False
        count += 1
        bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
        for file in prompt_files:
            bar.update(1)
            if not os.path.isfile(os.path.join(output_dir,file)):
                f = True
                cur_key = model_keys[execution_count % len(model_keys)]
                execution_count += 1
                try:
                    codex_execution(key=cur_key, output_path=os.path.join(output_dir, file),
                                    prompt_path=os.path.join(prompt_cache_dir, file))
                except Exception as e:
                    print(e)
                    time.sleep(3)
    idx_scores = {}
    n = len(downstream_train_examples)
    for idx in range(n):
        if idx in selected_indices:
            idx_scores[idx] = float('inf')
            continue
        with open(f"{output_dir}/{idx}.json") as f:
            cur_result = json.load(f)
            idx_scores[idx] = sum(cur_result['choices'][0]["logprobs"]["token_logprobs"]) / len(
                cur_result['choices'][0]["logprobs"]["token_logprobs"])
    sorted_scores = sorted(idx_scores.items(), key=lambda x: x[1])

    with open(os.path.join(args.output_dir,f'v2_vote_{args.selective_annotation_method}.json')) as f:
        vote_stat = json.load(f)
    votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
    selected_times = defaultdict(int)
    select_num_1 = args.annotation_size-len(selected_indices)
    inter = int(len(downstream_train_examples)*0.9/select_num_1)
    for prev_idx in selected_indices:
        for idx_support in vote_stat[str(prev_idx)]:
            selected_times[idx_support] += 1
    count_t = 0
    while len(selected_indices)<args.annotation_size and count_t*inter<len(votes):
        cur_scores = defaultdict(int)
        for idx, _ in sorted_scores[count_t*inter:(count_t+1)*inter]:
            if not str(idx) in vote_stat:
                cur_scores[idx] = 0
                continue
            candidates = vote_stat[str(idx)]
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        if cur_selected_idx in vote_stat:
            for idx_support in vote_stat[cur_selected_idx]:
                selected_times[idx_support] += 1
        count_t += 1
    if len(selected_indices)<args.annotation_size:
        unselected_indices = []
        for unselected_i in range(len(downstream_train_examples)):
            if not unselected_i in selected_indices:
                unselected_indices.append(unselected_i)
        selected_indices += random.sample(unselected_indices,args.annotation_size-len(selected_indices))

    return selected_indices

def process_examples():
    with open('data/geoquery_train.json') as f:
        prepared_train_examples = json.load(f)
    with open('data/geoquery_eval.json') as f:
        prepared_val_examples = json.load(f)
    return prepared_train_examples,prepared_val_examples

set_seed(args.seed)
total_train_examples,total_eval_examples = process_examples()
if args.debug:
    total_train_examples = total_train_examples[:50]
    args.annotation_size = 10
    args.batch_size = 3
if not args.debug:
    eval_phase_selected_indices = random.sample(range(len(total_eval_examples)), 256)
else:
    eval_phase_selected_indices = random.sample(range(len(total_eval_examples)), 5)
eval_examples = [total_eval_examples[idx] for idx in eval_phase_selected_indices]
processed_eval_examples = eval_examples

total_train_embeds = calculate_sentence_transformer_embedding(total_train_examples,args.embedding_model,mean_normal=True)

os.makedirs(args.output_dir,exist_ok=True)
total_train_examples_num = len(total_train_examples)
first_phase_selected_indices = range(len(total_train_examples))

first_phase_selected_indices_to_cache = []
processed_train_examples = []
first_phase_selected_indices = sorted(first_phase_selected_indices)
for selected_idx in first_phase_selected_indices:
    processed_train_examples.append(total_train_examples[selected_idx])
    first_phase_selected_indices_to_cache.append([selected_idx, total_train_examples[selected_idx]['id']])
with open(os.path.join(args.output_dir,'example_pool.json'),'w') as f:
    json.dump(processed_train_examples,f,indent=4)
with open(os.path.join(args.output_dir,'example_pool.json'),'w') as f:
    json.dump(first_phase_selected_indices_to_cache,f,indent=4)
with open(os.path.join(args.output_dir,'eval_phase_selected_indices.json'),'w') as f:
    json.dump(eval_phase_selected_indices,f,indent=4)

if args.prompt_retrieval_method in ['similar']:
    select_2(total_train_embeds[first_phase_selected_indices],
                     calculate_sentence_transformer_embedding(processed_eval_examples,args.embedding_model),
                     processed_train_examples,processed_eval_examples,phase2_selection='similar')
elif args.prompt_retrieval_method in ['random']:
    select_2(total_train_embeds[first_phase_selected_indices],
                     calculate_sentence_transformer_embedding(processed_eval_examples,args.embedding_model),
             processed_train_examples,processed_eval_examples,phase2_selection='random')

candidate_prompt_files = os.listdir(os.path.join(args.output_dir,'prompts'))
prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
prompt_cache_dir = os.path.join(args.output_dir,'prompts')
assert len(prompt_files) == len(processed_eval_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                         f"len(processed_dev_examples)={len(processed_eval_examples)}"

output_dir = os.path.join(args.output_dir,'results')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
count = 0
f = True
execution_count = 0
while f:
    f = False
    count += 1
    bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
    for file in prompt_files:
        bar.update(1)
        if not os.path.isfile(os.path.join(output_dir, file)):
            f = True
            cur_key = model_keys[execution_count % len(model_keys)]
            execution_count += 1
            try:
                codex_execution(key=cur_key, output_path=os.path.join(output_dir, file),
                                prompt_path=os.path.join(prompt_cache_dir, file))
            except Exception as e:
                print(e)
                time.sleep(3)
preds = []
for i in eval_phase_selected_indices:
    with open(os.path.join(output_dir,f'{i}.json')) as f:
        r = json.load(f)
        preds.append(r['choices'][0]['text'].replace('\n', ' '))
with open(os.path.join(output_dir,'preds_geogrpahy.txt'), 'w') as f:
    for p in preds:
        f.write("SELECT"+p + '\n')
gold_dicts = pkl.load(open('test_suite_database_gold_sql/gold_pkls/geography_gold.pickle', 'rb'))
with open(os.path.join(args.output_dir,'eval_phase_selected_indices.json')) as f:
    selected_evaluation_indices = json.load(f)
old_gold_dicts = gold_dicts
gold_dicts = []
for idx in selected_evaluation_indices:
    gold_dicts.append(old_gold_dicts[idx])
golds = [d['query'] for d in gold_dicts]
assert len(golds)==len(preds)
metric = load_metric("rouge")
result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
result = {k: round(v, 4) for k, v in result.items()}
result = result['rougeL']
with open(os.path.join(args.output_dir,'result_summary.json'), 'w') as f:
    json.dump(result, f)
print(result)

