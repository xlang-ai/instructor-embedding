import os
import random
import json
import torch
import time
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from utils import codex_execution

def prompt_retrieval(train_embs,test_embs,train_examples,eval_examples,return_string,format_example,
                     maximum_input_len,args, label_map,prompt_identifier='prompts',single_context_example_len=None):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    eval_example_num = len(eval_examples)
    bar = tqdm(range(eval_example_num), desc="Retrieve examples from annotated pool")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_cache_dir = os.path.join(args.output_dir,prompt_identifier)
    if not os.path.isdir(prompt_cache_dir):
        os.makedirs(prompt_cache_dir, exist_ok=True)
    for test_id, one_test_instance in enumerate(eval_examples):
        one_test_instance_input_text,one_test_instance_output_text = format_example(example=one_test_instance,args=args,
                                                                                    label_map=label_map)
        cur_prompt_string_len = get_instance_length(one_test_instance_input_text,one_test_instance_output_text,tokenizer)[0]
        if args.prompt_retrieval_method=='similar':
            test_e_reshape = test_embs[test_id].reshape(1, -1)
            scores = cos(test_e_reshape, train_embs).numpy()
            sorted_indices = np.argsort(scores)
        elif args.prompt_retrieval_method=='random':
            sorted_indices = np.random.permutation(range(eval_example_num))
        else:
            raise ValueError(f"The prompt retrieval method {args.prompt_retrieval_method} is not supported")
        selected_indices = []
        num_indices = len(sorted_indices)
        for idx in range(num_indices - 1, -1, -1):
            if args.prompt_retrieval_method=='similar' and scores[sorted_indices[idx]]==1:
                continue
            cur_example_input_text,cur_example_output_text = format_example(example=train_examples[sorted_indices[idx]],
                                                                            args=args,label_map=label_map)
            cur_len = sum(get_instance_length(cur_example_input_text, cur_example_output_text,tokenizer=tokenizer))
            if single_context_example_len is not None and cur_len>single_context_example_len:
                continue
            cur_prompt_string_len += cur_len
            if cur_prompt_string_len > maximum_input_len:
                break
            selected_indices.append(idx)

        one_test_emb = test_embs[test_id]
        indices_scores = []
        for idx in selected_indices:
            indices_scores.append(
                [idx, cos(train_embs[sorted_indices[idx]].reshape(1, -1), one_test_emb.reshape(1, -1)).item()])
        indices_scores = sorted(indices_scores, key=lambda x: x[1], reverse=True)
        new_selected_indices = [x[0] for x in indices_scores]
        if args.prompt_retrieval_method in ['similar']:
            assert new_selected_indices == selected_indices, f"new_selected_indices={new_selected_indices}, " \
                                                             f"selected_indices={selected_indices}"
        selected_indices = new_selected_indices

        select_num = len(selected_indices)
        second_phase_selected_indices = []
        if return_string:
            cur_train_data = ''
        else:
            cur_train_data = []
        for idx in range(select_num - 1, -1, -1):
            cur_input_text, cur_output_text = format_example(
                example=train_examples[sorted_indices[selected_indices[idx]]],
                args=args, label_map=label_map)
            if return_string:
                cur_train_data += f'{cur_input_text}{cur_output_text}\n\n'
            else:
                if args.task_name=='hellaswag':
                    cur_train_data.append({
                        'input': cur_input_text,
                        'output': cur_output_text,
                        'options': train_examples[sorted_indices[selected_indices[idx]]]['endings']
                    })
                else:
                    cur_train_data.append({
                        'input': cur_input_text,
                        'output': cur_output_text
                    })
            second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item()])
        if return_string:
            cur_train_data += format_example(
                example=one_test_instance,
                args=args, label_map=label_map)[0]
        # print(f'{len(second_phase_selected_indices)} examples in context')
        with open(os.path.join(prompt_cache_dir,f"{one_test_instance['id']}.json"),'w') as f:
            json.dump([[test_id, second_phase_selected_indices, one_test_instance['label']],
                       cur_train_data,
                       one_test_instance
                       ], f, indent=4)
        bar.update(1)

def fast_votek(embeddings,select_num,k,vote_file=None):
    n = len(embeddings)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n),desc=f'voting')
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

def iterative_selection(train_embs,test_embs,train_examples,test_examples,return_string,format_example,maximum_input_len,
                        label_map,single_context_example_len,inference_model,inference_data_module,tokenizer_gpt,args):
    if args.selective_annotation_method=='least_confidence':
        selected_indices = random.sample(range(len(train_examples)),args.batch_size)
    elif args.selective_annotation_method=='votek':
        selected_indices = fast_votek(embeddings=train_embs,select_num=args.batch_size,k=150,
                                      vote_file=os.path.join(args.output_dir,'votek_cache.json'))
    else:
        raise ValueError(f'iterative selection does not support {args.selective_annotation_method}')
    if not args.task_name in ['hellaswag', 'xsum','nq']:
        all_labels = []
        label_to_digit = {}
        for k, v in label_map.items():
            all_labels.append(v)
            label_to_digit[v] = k
    batch_count = 0
    device = torch.device('cuda')
    while len(selected_indices)<args.annotation_size:
        batch_count += 1
        cur_annotated_examples = [train_examples[idx] for idx in selected_indices]
        prompt_retrieval(train_embs=train_embs[selected_indices],
                         test_embs=test_embs,
                         train_examples=cur_annotated_examples,
                         eval_examples=test_examples,
                         return_string=return_string,
                         format_example=format_example,
                         maximum_input_len=maximum_input_len,
                         args=args,label_map=label_map,
                         prompt_identifier=f'prompts_{batch_count}',
                         single_context_example_len=single_context_example_len)

        candidate_prompt_files = os.listdir(os.path.join(args.output_dir,f'prompts_{batch_count}'))
        prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
        assert len(prompt_files) == len(test_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                                  f"len(processed_eval_examples)={len(test_examples)}"
        output_dir = os.path.join(args.output_dir,f'results_iteration_{batch_count}')
        prompt_dir = os.path.join(args.output_dir,f'prompts_{batch_count}')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        count = 0
        execution_count = 0
        model_keys = args.model_key.split('##')
        running_flag = True
        while running_flag:
            running_flag = False
            count += 1
            bar = tqdm(range(len(prompt_files)), desc=f"  prediction iteration {batch_count}")
            for file in prompt_files:
                bar.update(1)
                if not os.path.isfile(os.path.join(output_dir,file)):
                    running_flag = True

                    if args.task_name=='hellaswag':
                        with open(os.path.join(prompt_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        cur_input = {'input': format_example(one_test_example[2], label_map=label_map, args=args)[0],
                                     'options': one_test_example[2]['endings']}
                        inference_data_module.k = len(cur_train_data)
                        inference_data_module.tensorize(cur_train_data, [cur_input])
                        prediction = inference_model.do_predict(inference_data_module, require_loss=True)[0]
                        with open(f"{output_dir}/{file}", 'w') as f:
                            json.dump(prediction, f)
                    elif args.task_name=='xsum':
                        with open(os.path.join(prompt_dir, file)) as f:
                            one_test_example = json.load(f)
                        context = one_test_example[1]
                        input_ids = tokenizer_gpt(context, return_tensors="pt").input_ids
                        input_ids = input_ids[:, :1900]
                        input_len = input_ids.shape[1]
                        input_ids = input_ids.to(device)
                        # print(input_ids.shape)
                        # print(os.path.join(prompt_dir,file))
                        gen_tokens = inference_model.generate(input_ids, do_sample=False, temperature=0.7,
                                                              max_length=input_len + 64,
                                                              output_scores=True, return_dict_in_generate=True)
                        generated_text = tokenizer_gpt.batch_decode(gen_tokens.sequences.view(-1, 1))  #
                        stop = ['--', '\n', ';', '#']
                        stop_index = len(generated_text)
                        for i, c in enumerate(generated_text):
                            if i > input_len and c.strip(' ') in stop:
                                stop_index = i
                                break
                        prediction = [' '.join(generated_text[input_len:stop_index]),
                                      sum(gen_tokens.probs[:stop_index - input_len])]
                        with open(f"{output_dir}/{file}", 'w') as f:
                            json.dump(prediction, f)
                    elif args.task_name=='nq':
                        cur_key = model_keys[execution_count % len(model_keys)]
                        execution_count += 1
                        try:
                            codex_execution(key=cur_key,output_path=os.path.join(output_dir,file),
                                            prompt_path=os.path.join(prompt_dir, file))
                        except Exception as e:
                            print(e)
                            time.sleep(3)
                    else:
                        with open(os.path.join(prompt_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]['options'] = all_labels
                        cur_input = format_example(one_test_example[2],label_map=label_map,args=args)[0]
                        inference_data_module.k = len(cur_train_data)
                        inference_data_module.tensorize(cur_train_data, [cur_input], options=all_labels)
                        prediction = inference_model.do_predict(inference_data_module, require_loss=True)[0]
                        with open(f"{output_dir}/{file}", 'w') as f:
                            json.dump(prediction, f)


        idx_scores = {}
        n = len(test_examples)
        for idx in range(n):
            if idx in selected_indices:
                if args.task_name in ['xsum','nq']:
                    idx_scores[idx] = float('inf')
                else:
                    idx_scores[idx] = float('-inf')
                continue
            with open(f"{output_dir}/{idx}.json") as f:
                one_pred = json.load(f)
                if args.task_name in ['nq']:
                    idx_scores[idx] = sum(one_pred['choices'][0]["logprobs"]["token_logprobs"]) / len(
                        one_pred['choices'][0]["logprobs"]["token_logprobs"])
                else:
                    idx_scores[idx] = one_pred[1]
        if args.task_name in ['xsum','nq']:
            sorted_scores = sorted(idx_scores.items(), key=lambda x: x[1])
        else:
            sorted_scores = sorted(idx_scores.items(), key=lambda x:x[1],reverse=True)
        sorted_scores_len = len(sorted_scores)
        if args.selective_annotation_method=='least_confidence':
            cur_selected = []
            cur_select_num = min(args.batch_size, args.annotation_size - len(selected_indices))
            for sorted_scores_iter in range(sorted_scores_len):
                if len(cur_selected)>=cur_select_num:
                    break
                if not sorted_scores[sorted_scores_iter][0] in selected_indices:
                    cur_selected.append(sorted_scores[sorted_scores_iter][0])
            selected_indices += cur_selected
        else:
            with open(os.path.join(args.output_dir,'votek_cache.json')) as f:
                vote_stat = json.load(f)
            selected_times = defaultdict(int)
            select_num_1 = args.annotation_size - len(selected_indices)
            inter = int(len(train_examples) * 0.9 / select_num_1)
            for prev_idx in selected_indices:
                for idx_support in vote_stat[str(prev_idx)]:
                    selected_times[idx_support] += 1
            count_t = 0
            while len(selected_indices) < args.annotation_size and count_t * inter < sorted_scores_len:
                cur_scores = defaultdict(int)
                for idx, _ in sorted_scores[count_t * inter:(count_t + 1) * inter]:
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
                selected_indices.append(cur_selected_idx)
                if cur_selected_idx in vote_stat:
                    for idx_support in vote_stat[cur_selected_idx]:
                        selected_times[idx_support] += 1
                count_t += 1
            if len(selected_indices) < args.annotation_size:
                unselected_indices = []
                for unselected_i in range(len(train_examples)):
                    if not unselected_i in selected_indices:
                        unselected_indices.append(unselected_i)
                selected_indices += random.sample(unselected_indices, args.annotation_size - len(selected_indices))
                print(f"{args.annotation_size - len(selected_indices)} examples are randomly selected")
    return selected_indices

def selective_annotation(args,**kwargs):
    if args.selective_annotation_method=='random':
        train_examples = kwargs['train_examples']
        selected_indices = random.sample(range(len(train_examples)),args.annotation_size)
    elif args.selective_annotation_method=='diversity':
        embeddings = kwargs['embeddings']
        selected_indices = []
        first_id = random.choice(range(len(embeddings)))
        selected_indices.append(first_id)
        selected_representations = embeddings[first_id].reshape(1, -1)
        for count in range(args.annotation_size - 1):
            scores = np.sum(cosine_similarity(embeddings, selected_representations), axis=1)
            for i in selected_indices:
                scores[i] = float('inf')
            min_idx = np.argmin(scores)
            selected_representations = torch.cat((selected_representations,
                                                  embeddings[min_idx].reshape(1, -1)), 0)
            selected_indices.append(min_idx.item())
    elif args.selective_annotation_method=='fast_votek':
        selected_indices = fast_votek(embeddings=kwargs['embeddings'],select_num=args.annotation_size,k=150,
                                      vote_file=os.path.join(args.output_dir,'nearest_neighbors.json'))
    elif args.selective_annotation_method=='mfl':
        embeds = kwargs['embeddings']
        N, D = embeds.shape
        norm_embeds = embeds / embeds.norm(dim=1, keepdim=True)
        cosine = torch.einsum('nd,md->nm', norm_embeds, norm_embeds)
        selected = torch.zeros(N, dtype=torch.bool)
        max_similarity = torch.zeros(N) - 1
        for k in tqdm(range(args.annotation_size)):
            marginal_gain = torch.relu(cosine - max_similarity).sum(dim=1) * (1 - selected.float())
            node = torch.argmax(marginal_gain)
            selected[node] = True
            max_similarity = torch.max(max_similarity, cosine[node])
        selected_indices = torch.nonzero(selected).squeeze().tolist()
    elif args.selective_annotation_method in ['votek','least_confidence']:
        selected_indices = iterative_selection(train_embs=kwargs['embeddings'],
                                               test_embs=kwargs['embeddings'],
                                               train_examples=kwargs['train_examples'],
                                               test_examples=kwargs['train_examples'],
                                               return_string=kwargs['return_string'],
                                               format_example=kwargs['format_example'],
                                               maximum_input_len=kwargs['maximum_input_len'],
                                               label_map=kwargs['label_map'],
                                               single_context_example_len=kwargs['single_context_example_len'],
                                               inference_model=kwargs['inference_model'],
                                               inference_data_module=kwargs['inference_data_module'],
                                               tokenizer_gpt=kwargs['tokenizer_gpt'],
                                               args=args)
    else:
        raise ValueError(f'The selective annotation method {args.selective_annotation_method} is not supported')
    return selected_indices

def get_instance_length(input_text,output_text,tokenizer):
    return len(tokenizer(input_text)['input_ids']),len(tokenizer(output_text)['input_ids'])

