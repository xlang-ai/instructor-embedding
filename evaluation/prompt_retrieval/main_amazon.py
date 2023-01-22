import argparse
import random
import os
import copy
import torch
import numpy as np
import json
import time
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from InstructorEmbedding import INSTRUCTOR
from datasets import load_dataset
from sklearn.metrics import f1_score
from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel
from collections import defaultdict
from datasets import load_metric

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_model', type=str)
parser.add_argument('--selection_2', type=str,default='similar')
parser.add_argument('--model_name', default='EleutherAI/gpt-j-6B',type=str)
parser.add_argument('--repeat', type=int,default=1)
parser.add_argument('--seed', type=int,default=0)
parser.add_argument('--tag', default='',type=str)
parser.add_argument('--output_dir', default=None,type=str)
parser.add_argument('--model_cache_dir',type=str)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--add_prompt', action='store_true')
args = parser.parse_args()

label_map = {
            0:"very negative",
            1:"negative",
            2:"neutral",
            3:"positive",
            4:"very positive"
}
all_labels = []
label_to_digit = {}
for k,v in label_map.items():
    all_labels.append(v)
    label_to_digit[v] = k

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if not os.path.isdir('cache'):
    os.makedirs('cache',exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_sentence_transformer_embedding(examples,embedding_model,mean_normal=False):
    if not args.add_prompt:
        text_to_encode = [raw_item['text'] for raw_item in examples]
    else:
        text_to_encode = [['Represent the amazon review; Input: ',raw_item['text'],0] for raw_item in examples]
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
    return len(tokenizer(f"How do you feel about the following sentence?\n"
                         f"{local_examples[idx]['text']}.\n"
                         f"answer:")['input_ids'])+\
           len(tokenizer(f"{label_map[local_examples[idx]['label']]}")['input_ids'])

def select_2(train_embs,test_embs,downstream_train_examples,downstream_test_examples,tag,phase2_selection):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    bar = tqdm(range(len(downstream_test_examples)),desc="phase 2 similar select")
    if not os.path.isdir(f"{args.output_dir}/{tag}/prompts"):
        os.makedirs(f"{args.output_dir}/{tag}/prompts",exist_ok=True)
    for test_id,one_test_instance in enumerate(downstream_test_examples):
        prev_prompt_string_len = 0
        if phase2_selection in ['similar']:
            test_e_reshape = test_embs[test_id].reshape(1, -1)
            scores = cos(test_e_reshape, train_embs).numpy()
            sorted_indices = np.argsort(scores)
        elif phase2_selection in ['random']:
            sorted_indices = np.random.permutation(range(len(downstream_train_examples)))
        # test_e_reshape = one_test_emb.reshape(1, -1)
        # scores = cos(test_e_reshape, train_embs).numpy()
        # sorted_indices = np.argsort(scores)

        sorted_indices = sorted_indices[-2:]


        selected_indices = []
        num_indices = len(sorted_indices)
        for idx in range(num_indices-1,-1,-1):
            cur_len = get_instance_length(sorted_indices[idx],downstream_train_examples)
            if cur_len>250:
                continue
            prev_prompt_string_len += cur_len
            cur_prompt_string_len = prev_prompt_string_len + \
                    len(tokenizer(f"How do you feel about the following sentence?\n"
                                 f"{downstream_test_examples[test_id]['text']}.\n"
                                 f"answer:"
                        )['input_ids'])
            if cur_prompt_string_len>1000:
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
        cur_train_data = []
        for idx in range(select_num-1,-1,-1):

            cur_train_data.append({
                'input':f"How do you feel about the following sentence?\n"
                         f"{downstream_train_examples[sorted_indices[selected_indices[idx]]]['text']}.\n"
                         f"answer:",
                'output':label_map[downstream_train_examples[sorted_indices[selected_indices[idx]]]['label']]
            })
            second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item(),
                                                  downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']
                                                  ])
        with open(f"{args.output_dir}/{tag}/prompts/{downstream_test_examples[test_id]['id']}.json",'w') as f:
            json.dump([[test_id,second_phase_selected_indices,downstream_test_examples[test_id]['label']],
                       cur_train_data,
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
    for idx,e in enumerate(examples):
        e['text'] = e['text'].replace('\n','')
        e['id'] = idx
        processed_examples.append(e)
    return processed_examples

set_seed(args.seed)
with open('data/amazon_train.json') as f:
    total_train_examples = json.load(f)
total_train_examples = random.sample(total_train_examples,3000)
with open('data/amazon_test.json') as f:
    all_total_eval_examples = json.load(f)
total_eval_examples = []
total_eval_examples_in_reviewer_id = defaultdict(list)
reviewer_ids = set()
for e in all_total_eval_examples:
    reviewer_ids.add(e['meta'][0])
    total_eval_examples_in_reviewer_id[e['meta'][0]].append(e)
if args.debug:
    my_reviewer_ids = random.sample(list(reviewer_ids),5)
else:
    my_reviewer_ids = random.sample(list(reviewer_ids), 20)
for rid in my_reviewer_ids:
    if args.debug:
        total_eval_examples += random.sample(total_eval_examples_in_reviewer_id[rid],5)
    else:
        total_eval_examples += random.sample(total_eval_examples_in_reviewer_id[rid], 50)

if args.debug:
    total_train_examples = total_train_examples[:50]
processed_total_train_examples = process_examples(total_train_examples)
total_train_embeds = calculate_sentence_transformer_embedding(processed_total_train_examples, args.embedding_model,
                                                                  mean_normal=True)
cur_repeat_num = get_repeat_num(f"{args.tag}template1_{args.selection_2}_{args.seed}")

data_module = MetaICLData(method="direct", max_length=1024, max_length_per_example=256)
inference_model = MetaICLModel(args=args)
inference_model.load()
inference_model.cuda()
inference_model.eval()

for i in range(cur_repeat_num,cur_repeat_num+args.repeat):
    tag = f"{args.tag}template1_{args.selection_2}_{args.seed}_{i}"
    os.makedirs(f"{args.output_dir}/{tag}",exist_ok=True)
    total_train_examples_num = len(total_train_examples)
    first_phase_selected_indices = range(len(processed_total_train_examples))

    first_phase_selected_indices_to_cache = []
    processed_train_examples = []
    for selected_idx in first_phase_selected_indices:
        processed_train_examples.append(processed_total_train_examples[selected_idx])
        first_phase_selected_indices_to_cache.append([selected_idx, processed_total_train_examples[selected_idx]['id']])
    with open(f"{args.output_dir}/{tag}/first_phase_selected_indices.json",'w') as f:
        json.dump(first_phase_selected_indices_to_cache,f,indent=4)
    processed_eval_examples = process_examples(total_eval_examples)

    select_2(total_train_embeds[first_phase_selected_indices],
                     calculate_sentence_transformer_embedding(processed_eval_examples,args.embedding_model),
                     processed_train_examples,processed_eval_examples,tag,phase2_selection='similar')

    candidate_prompt_files = os.listdir(f"{args.output_dir}/{tag}/prompts")
    prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
    assert len(prompt_files)==len(processed_eval_examples),f"len(prompt_files)={len(prompt_files)}," \
                                                          f"len(processed_eval_examples)={len(processed_eval_examples)}"
    output_dir = f"{args.output_dir}/{tag}/results"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    count = 0
    f = True
    group_acc = {}
    for e in total_eval_examples:
        if not e['meta'][0] in group_acc:
            group_acc[e['meta'][0]] = [0,1]
        else:
            group_acc[e['meta'][0]][1] += 1
    if not args.debug:
        assert len(group_acc)==20
    else:
        assert len(group_acc) == 5
    for k,v in group_acc.items():
        if not args.debug:
            assert v[1]==50
        else:
            assert v[1]==5
    golds = defaultdict(list)
    preds = defaultdict(list)
    total_correct = 0
    total_count = 0
    while f:
        f = False
        count += 1
        bar = tqdm(range(len(prompt_files)), desc=f"  {tag}, {count} time")
        for file in prompt_files:
            bar.update(1)
            if not os.path.isfile(f"{args.output_dir}/{tag}/results/{file}"):
                with open(f'{args.output_dir}/{tag}/prompts/{file}') as f:
                    one_test_example = json.load(f)
                cur_train_data = one_test_example[1]
                for idx in range(len(cur_train_data)):
                    cur_train_data[idx]['options'] = all_labels
                cur_input = f"How do you feel about the following sentence?\n" \
                            f"{one_test_example[2]['text']}.\n"\
                             f"answer:"
                data_module.k = len(cur_train_data)
                data_module.tensorize(cur_train_data, [cur_input], options=all_labels)
                prediction = inference_model.do_predict(data_module)[0]
                with open(f"{output_dir}/{file}",'w') as f:
                    json.dump([prediction,one_test_example[2]['label']],f)
                preds[one_test_example[2]['meta'][0]].append(str(label_to_digit[prediction]))
                golds[one_test_example[2]['meta'][0]].append(str(one_test_example[2]['label']))
                if label_to_digit[prediction]==one_test_example[2]['label']:
                    group_acc[one_test_example[2]['meta'][0]][0] += 1
                    total_correct += 1
                total_count += 1
    sorted_acc = sorted(group_acc.items(),key=lambda x:x[1][0]/x[1][1])
    print('acc: ',sorted_acc[2][1][0]/sorted_acc[2][1][1],'\n')
    preds = preds[sorted_acc[2][0]]
    golds = golds[sorted_acc[2][0]]

    metric = load_metric("rouge")
    result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    result = result['rougeL']
    with open(os.path.join(args.output_dir, 'result_summary.json'), 'w') as f:
        json.dump(result, f)
    print(result)



