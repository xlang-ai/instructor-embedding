import torch
import argparse
import random
import json, os,copy
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer
from collections import defaultdict
from utils import slot_values_to_seq_sql,table_prompt,PreviousStateRecorder,codex_completion,sql_pred_parse,typo_fix
from utils import evaluate
from datasets import load_metric
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_model', type=str)
parser.add_argument('--model_key', type=str)
parser.add_argument('--selection_1', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--selection_2', type=str,default='similar')
parser.add_argument('--annotation_size', default=100,type=int)
parser.add_argument('--repeat', type=int,default=1)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--seed', type=int,default=0)
args = parser.parse_args()

model_keys = args.model_key.split('##')

tokenizer_for_length = AutoTokenizer.from_pretrained('gpt2')
if not os.path.isdir('cache'):
    os.makedirs('cache',exist_ok=True)

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir,exist_ok=True)
if os.path.isfile(os.path.join(args.output_dir,'predictions.txt')):
    os.remove(os.path.join(args.output_dir,'predictions.txt'))
if os.path.isfile(os.path.join(args.output_dir,'golds.txt')):
    os.remove(os.path.join(args.output_dir,'golds.txt'))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def conversion(prompt, reverse=False):
    conversion_dict = {"leaveat": "depart_time", "arriveby": "arrive_by_time",
                       "book_stay": "book_number_of_days",
                       "food": "food_type"}
    reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
    used_dict = reverse_conversion_dict if reverse else conversion_dict

    for k, v in used_dict.items():
        prompt = prompt.replace(k, v)
    return prompt

def v2_vote_k_select(embeddings,examples,select_num,k,vote_file=None):
    n = len(embeddings)
    assert len(examples)==n,f"len(examples)={len(examples)},n={n}"
    dial_id_indices = defaultdict(list)
    for i,e in enumerate(examples):
        dial_id_indices[e['dialogue_ID']].append(i)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n),desc=f'v2 vote {k} selection')
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            all_sorted_indices = np.argsort(cur_scores).tolist()
            sorted_indices = []
            sorted_indices_idx = len(all_sorted_indices) - 1
            while sorted_indices_idx>0 and len(sorted_indices)<k:
                if examples[all_sorted_indices[sorted_indices_idx]]['dialogue_ID']!=examples[i]['dialogue_ID']:
                    sorted_indices.append(all_sorted_indices[sorted_indices_idx])
                sorted_indices_idx -= 1
            for idx in sorted_indices:
                if not i in vote_stat[examples[idx]['dialogue_ID']]:
                    vote_stat[examples[idx]['dialogue_ID']].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file,'w') as f:
                json.dump(vote_stat,f)
    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    selected_dial = []
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_dial)<select_num:
        cur_scores = defaultdict(int)
        for dial,candidates in votes:
            if dial in selected_dial:
                cur_scores[dial] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[dial] += 10 ** (-selected_times[one_support])
        cur_selected_dial = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_dial.append(cur_selected_dial)
        selected_indices += dial_id_indices[cur_selected_dial]
        for idx_support in vote_stat[cur_selected_dial]:
            selected_times[idx_support] += 1
    return selected_indices,selected_dial

def calculate_sentence_transformer_embedding(examples,embedding_model,mean_normal=False):
    text_to_encode = [e['history'] for e in examples]
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

def calculate_one_embed(example,embedding_model):
    text_to_encode = example['history']
    emb_model = INSTRUCTOR(embedding_model)
    emb = emb_model.encode(text_to_encode)
    return emb

def select_iterative(train_embs,one_test_emb,downstream_train_examples,one_test_example,tag,given_context,phase2_selection,identifier=''):
    assert len(train_embs)==len(downstream_train_examples)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if not os.path.isdir(f"{args.output_dir}/{tag}/prompts_iterative_{identifier}"):
        os.makedirs(f"{args.output_dir}/{tag}/prompts_iterative_{identifier}",exist_ok=True)
    prompt_string = f"{conversion(table_prompt)}\n"
    prev_prompt_string = f"{conversion(table_prompt)}\n"
    if phase2_selection in ['similar']:
        test_e_reshape = one_test_emb.reshape(1, -1)
        scores = cos(test_e_reshape, train_embs).numpy()
        sorted_indices = np.argsort(scores)
    elif phase2_selection in ['random']:
        sorted_indices = np.random.permutation(range(len(downstream_train_examples)))
    selected_indices = []
    num_indices = len(sorted_indices)
    count = 1
    for idx in range(num_indices-1,-1,-1):
        prev_prompt_string += get_instance(count,downstream_train_examples[sorted_indices[idx]])

        cur_prompt_string = prev_prompt_string + f"Example #{count + 1}\n"
        last_slot_values = given_context
        cur_prompt_string += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"
        last_sys_utt = one_test_example['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        cur_prompt_string += f"[system] {last_sys_utt}\n"
        cur_prompt_string += f"Q: [user] {one_test_example['dialog']['usr'][-1]}\n"
        cur_prompt_string += "SQL: SELECT * FROM"

        length = len(tokenizer_for_length(cur_prompt_string)['input_ids'])
        if length > 3800:
            break
        selected_indices.append(idx)
        count += 1

    indices_scores = []
    for idx in selected_indices:
        indices_scores.append([idx,cos(train_embs[sorted_indices[idx]].reshape(1, -1),one_test_emb.reshape(1, -1)).item()])
    indices_scores = sorted(indices_scores,key=lambda x:x[1],reverse=True)
    new_selected_indices = [x[0] for x in indices_scores]
    if phase2_selection in ['similar']:
        assert new_selected_indices==selected_indices,f"new_selected_indices={new_selected_indices}, " \
                                                      f"selected_indices={selected_indices}"
    selected_indices = new_selected_indices

    select_num = len(selected_indices)
    count = 0
    second_phase_selected_indices = []
    for idx in range(select_num - 1, -1, -1):
        prompt_string += get_instance(count, downstream_train_examples[sorted_indices[selected_indices[idx]]])
        # print(type(sorted_indices[selected_indices[idx]].item()),type(downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']))
        second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item(),
                                              downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']
                                              ])
        count += 1

    prompt_string += f"Example #{count}\n"
    last_slot_values = given_context
    prompt_string += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"
    last_sys_utt = one_test_example['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_string += f"[system] {last_sys_utt}\n"
    prompt_string += f"Q: [user] {one_test_example['dialog']['usr'][-1]}\n"
    prompt_string += "SQL: SELECT * FROM"

    assert len(tokenizer_for_length(prompt_string)['input_ids']) <= 3800
    # print('prompt example num: ',len(second_phase_selected_indices))
    with open(f"{args.output_dir}/{tag}/prompts_iterative_{identifier}/{one_test_example['name'].replace('.','')}_{one_test_example['id']}.json", 'w') as f:
        json.dump([[one_test_example['name'].replace('.',''), one_test_example['id'],second_phase_selected_indices], prompt_string], f, indent=4)

    return prompt_string

def get_instance(example_id,example):
    prompt_text = f"Example #{example_id}\n"
    last_slot_values = {s: v.split(
        '|')[0] for s, v in example['last_slot_values'].items()}
    prompt_text += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"
    last_sys_utt = example['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"Q: [user] {example['dialog']['usr'][-1]}\n"
    prompt_text += f"SQL: {conversion(slot_values_to_seq_sql(example['turn_slot_values']))};\n"
    prompt_text += "\n\n"
    return prompt_text

def select_2(train_embs,one_test_emb,downstream_train_examples,one_test_example,tag,given_context,phase2_selection):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if not os.path.isdir(f"{args.output_dir}/{tag}/prompts"):
        os.makedirs(f"{args.output_dir}/{tag}/prompts",exist_ok=True)
    prompt_string = f"{conversion(table_prompt)}\n"
    prev_prompt_string = f"{conversion(table_prompt)}\n"
    if phase2_selection in ['similar']:
        test_e_reshape = one_test_emb.reshape(1, -1)
        scores = cos(test_e_reshape, train_embs).numpy()
        sorted_indices = np.argsort(scores)
    elif phase2_selection in ['random']:
        sorted_indices = np.random.permutation(range(len(downstream_train_examples)))
    selected_indices = []
    num_indices = len(sorted_indices)
    count = 1
    for idx in range(num_indices-1,-1,-1):
        prev_prompt_string += get_instance(count,downstream_train_examples[sorted_indices[idx]])

        cur_prompt_string = prev_prompt_string + f"Example #{count + 1}\n"
        last_slot_values = given_context
        cur_prompt_string += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"
        last_sys_utt = one_test_example['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        cur_prompt_string += f"[system] {last_sys_utt}\n"
        cur_prompt_string += f"Q: [user] {one_test_example['dialog']['usr'][-1]}\n"
        cur_prompt_string += "SQL: SELECT * FROM"

        length = len(tokenizer_for_length(cur_prompt_string)['input_ids'])
        if length > 3800:
            break
        selected_indices.append(idx)
        count += 1

    indices_scores = []
    for idx in selected_indices:
        indices_scores.append([idx,cos(train_embs[sorted_indices[idx]].reshape(1, -1),one_test_emb.reshape(1, -1)).item()])
    indices_scores = sorted(indices_scores,key=lambda x:x[1],reverse=True)
    new_selected_indices = [x[0] for x in indices_scores]
    if phase2_selection in ['similar']:
        assert new_selected_indices==selected_indices,f"new_selected_indices={new_selected_indices}, " \
                                                      f"selected_indices={selected_indices}"
    selected_indices = new_selected_indices

    select_num = len(selected_indices)
    count = 0
    second_phase_selected_indices = []
    for idx in range(select_num - 1, -1, -1):
        prompt_string += get_instance(count, downstream_train_examples[sorted_indices[selected_indices[idx]]])
        # print(type(sorted_indices[selected_indices[idx]].item()),type(downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']))
        second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item(),
                                              downstream_train_examples[sorted_indices[selected_indices[idx]]]['id']
                                              ])
        count += 1

    prompt_string += f"Example #{count}\n"
    last_slot_values = given_context
    prompt_string += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"
    last_sys_utt = one_test_example['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_string += f"[system] {last_sys_utt}\n"
    prompt_string += f"Q: [user] {one_test_example['dialog']['usr'][-1]}\n"
    prompt_string += "SQL: SELECT * FROM"

    assert len(tokenizer_for_length(prompt_string)['input_ids']) <= 3800
    print('select_2, prompt example num: ',len(second_phase_selected_indices))
    with open(f"{args.output_dir}/{tag}/prompts/{one_test_example['name'].replace('.','')}_{one_test_example['id']}.json", 'w') as f:
        json.dump([[one_test_example['name'].replace('.',''), one_test_example['id'],second_phase_selected_indices], prompt_string], f, indent=4)

    return prompt_string

def read_MW_dataset(mw_json_fn):
    DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']
    with open(mw_json_fn, 'r') as f:
        data = json.load(f)
    dial_dict = {}
    examples = defaultdict(list)
    idx = 0
    for turn in data:
        if not set(turn["domains"]).issubset(set(DOMAINS)):
            continue
        sys_utt = turn["dialog"]['sys'][-1]
        usr_utt = turn["dialog"]['usr'][-1]
        if sys_utt == 'none':
            sys_utt = ''
        if usr_utt == 'none':
            usr_utt = ''
        history = f"[system] {sys_utt} [user] {usr_utt}"
        name = f"{turn['ID']}_turn_{turn['turn_id']}"
        assert not name in dial_dict
        dial_dict[name] = [idx,history]
        one_example = copy.deepcopy(turn)
        one_example['id'] = idx
        one_example['name'] = name
        one_example['dialogue_ID'] = turn['ID']
        one_example['history'] = history
        examples[one_example['dialogue_ID']].append(one_example)
        # examples.append(one_example)
        idx += 1
    return dial_dict,examples

set_seed(args.seed)
mw_train,raw_total_train_examples = read_MW_dataset("data/mw24_100p_train.json")
dialog_ids = list(raw_total_train_examples.keys())
selected_dialog_ids = random.sample(dialog_ids,1000)
total_train_examples = []
for did in selected_dialog_ids:
    total_train_examples += raw_total_train_examples[did]
# mw_dev,total_dev_examples = read_MW_dataset("data/mw24_100p_dev.json")
mw_test,raw_total_test_examples = read_MW_dataset("data/mw24_100p_test.json")
original_all_test_dialogues = []
for k,v in raw_total_test_examples.items():
    original_all_test_dialogues.append(v)
if args.debug:
    args.batch_size = 5
    args.annotation_size = 10
    processed_total_train_examples = []
    for did in selected_dialog_ids[:30]:
        processed_total_train_examples.append(raw_total_train_examples[did][0])
else:
    processed_total_train_examples = total_train_examples
# total_train_embeds = []
for e in processed_total_train_examples:
    text1 = f"{e['name']}_{e['history']}"
    text1 = text1.replace('\'', '%%%%%').replace('\"', '$$$$$')
    cur_key = text1
    # total_train_embeds.append(train_embs_cache[cur_key])
# total_train_embeds = torch.tensor(total_train_embeds)

total_train_embeds = calculate_sentence_transformer_embedding(processed_total_train_examples,args.embedding_model,mean_normal=True)
cur_repeat_num = get_repeat_num(f"{args.annotation_size}_{args.selection_1}_{args.selection_2}_{args.seed}")

with open('data/mw24_ontology.json') as f:
    ontology = json.load(f)

for i in range(cur_repeat_num,cur_repeat_num+args.repeat):
    tag = f"{args.annotation_size}_{args.selection_1}_{args.selection_2}_{args.seed}_{i}"
    result_dict = defaultdict(list)
    if os.path.isdir(f"{args.output_dir}/{tag}"):
        raise ValueError(f"{tag} has been calculated")
    os.makedirs(f"{args.output_dir}/{tag}",exist_ok=True)
    with open(f'{args.output_dir}/{tag}/total_selected_dials.json','w') as f:
        json.dump(selected_dialog_ids,f)
    total_train_examples_num = len(processed_total_train_examples)
    first_phase_selected_indices = range(total_train_examples_num)

    first_phase_selected_indices_to_cache = []
    processed_train_examples = []
    for selected_idx in first_phase_selected_indices:
        processed_train_examples.append(processed_total_train_examples[selected_idx])
        first_phase_selected_indices_to_cache.append([selected_idx,processed_total_train_examples[selected_idx]['id']])
    with open(f"{args.output_dir}/{tag}/first_phase_selected_indices.json",'w') as f:
        json.dump(first_phase_selected_indices_to_cache,f,indent=4)
    random.shuffle(original_all_test_dialogues)
    all_test_dialogues = []
    for l in original_all_test_dialogues:
        all_test_dialogues += l
    if args.debug:
        processed_test_examples = all_test_dialogues[:5]
    else:
        processed_test_examples = all_test_dialogues[:256]

    prediction_recorder = PreviousStateRecorder()
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0

    # cur_idx = -1
    output_dir = f"{args.output_dir}/{tag}/results"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    execution_count = 0
    for data_item in tqdm(processed_test_examples,desc=f'{tag} prediction'):
        n_total += 1
        # cur_idx += 1
        # assert cur_idx==data_item['id']
        predicted_context = prediction_recorder.state_retrieval(data_item)
        modified_item = copy.deepcopy(data_item)
        modified_item['last_slot_values'] = predicted_context
        one_example_text = f"{data_item['name']}_{data_item['history']}"
        one_example_text = one_example_text.replace('\'', '%%%%%').replace('\"', '$$$$$')
        cur_prompt = select_2(
            train_embs=total_train_embeds[first_phase_selected_indices],
            one_test_emb=torch.tensor(calculate_one_embed(data_item,args.embedding_model)),
            downstream_train_examples=processed_train_examples,
            one_test_example=modified_item,
            tag=tag,
            given_context=predicted_context,
            phase2_selection=args.selection_2
        )
        data_item['prompt'] = cur_prompt

        complete_flag = False
        parse_error_count = 0
        while not complete_flag:
            execution_count += 1
            try:
                cur_model_key = model_keys[execution_count % len(model_keys)]
                completion = codex_completion(
                    prompt_text=cur_prompt,
                    key=cur_model_key,
                    output_path=f"{output_dir}/{modified_item['name'].replace('.','')}_{modified_item['id']}.json"
                )
                completion = conversion(completion, reverse=True)
            except Exception as e:
                print(e)
                time.sleep(5)
            else:
                try:
                    temp_parse = sql_pred_parse(completion)
                except:
                    parse_error_count += 1
                    if parse_error_count >= 5:
                        complete_flag = True
                else:
                    complete_flag = True

        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)
        except Exception as e:
            print(e)
            print(f"Invalid: {completion}")
            data_item['not_valid'] = 1
        predicted_slot_values = typo_fix(predicted_slot_values, ontology=ontology, version=2.4)

        context_slot_values = data_item['last_slot_values']  # a dictionary

        all_slot_values = prediction_recorder.state_retrieval(
                data_item).copy()

        for s, v in predicted_slot_values.items():

            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            elif v != "[DELETE]":
                all_slot_values[s] = v

        # some slots may contain multiple values
        all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}

        # record current turn prediction
        prediction_recorder.add_state(data_item, all_slot_values)

        # record the predictions
        data_item['pred'] = all_slot_values
        data_item['ontology_path'] = 'data/mw24_ontology.json'
        data_item['completion'] = completion
        all_result.append(data_item)

        # print the result
        # print(completion)
        # print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        # print(f"pred turn change: {sv_dict_to_string(predicted_slot_values, sep='-')}")
        # print(f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}")
        # print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        # print(f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(all_slot_values, data_item['slot_values'],args)
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
            # print("\n=====================correct!=======================")
        else:
            result_dict[data_item['turn_id']].append(0)
            # print("\n=====================wrong!=======================")

    with open(os.path.join(args.output_dir, 'predictions.txt')) as f:
        preds = f.readlines()
    with open(os.path.join(args.output_dir, 'golds.txt')) as f:
        golds = f.readlines()
    metric = load_metric("rouge")
    result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    result = result['rougeL']
    with open(os.path.join(args.output_dir, 'result_summary.json'), 'w') as f:
        json.dump(result, f)
    print(result)











