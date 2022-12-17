# Metric submission
import argparse, glob
import numpy as np
import os, json
import torch
from scipy.stats.stats import pearsonr
from filelock import FileLock
from sentence_transformers import SentenceTransformer

np.random.seed(0)


def compute_sims(hyps, refs, cos=False):
    # each image compute inner products
    hyps = torch.Tensor(hyps)
    refs = torch.Tensor(refs)
    if cos:
        # normalize
        hyps = torch.nn.functional.normalize(hyps, p=2.0, dim=-1)
        refs = torch.nn.functional.normalize(refs, p=2.0, dim=-1)
    sims = torch.bmm(hyps, refs.transpose(2, 1))
    # [nb_images, nb_systems, nb_refs]
    # [500, 5, 4]
    sims = sims.max(dim=-1)[0]

    return sims


def read_json(file_name):
    with open(file_name) as fin:
        data = json.load(fin)
    data = np.array(data)
    return data


def read_jsonl(infile, extract_key=None):
    f = open(infile, 'r')
    if extract_key is None:
        out = [json.loads(line.strip()) for line in f]
    else:
        out = [json.loads(line.strip())[extract_key] for line in f]
    f.close()
    return out


def compute_corr(hyp, ref, human, use_clip=False):
    hyps = read_json(hyp)
    refs = read_json(ref)
    dim = hyps.shape[-1]
    nb_images = refs.shape[0]
    hyps = hyps.reshape(nb_images, -1, dim)
    refs = refs.reshape(nb_images, -1, dim)
    sims = np.array(compute_sims(hyps, refs, cos=True)).reshape(-1)
    if use_clip:
        clip_scores = np.array(read_jsonl(human, "clip_score"))
        sims = sims + clip_scores / 2.5

    human_scores = np.array(read_jsonl(human, "human_score"))
    corr = pearsonr(human_scores, sims)
    print(corr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--cache', type=str, default='/scratch/acd13578qu/huggingface', help='cache folder path')
    parser.add_argument("--add_prompt", action="store_true", help="whether to add prompt")
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--task", default=None, choices=['mscoco','cnndm','mt'],type=str)
    args = parser.parse_args()

    if args.add_prompt:
        identifier = 'with_prompt'
    else:
        identifier = 'without_prompt'
    identifier = args.model_name.replace('/','_')+'_'+args.task+'_'+identifier
    definitions = {
        'mscoco': 'Represent the caption for retrieving duplicate captions; Input: ',
        'cnndm': 'Represent a sentence; Input: ',
        'mt': 'Represent the statement for retrieving duplicate statements. Input: '
    }
    reference_files = {
        'mscoco': 'data/mscoco/mscoco_references.jsonl',
        'cnndm': 'data/cnndm/cnndm_references.jsonl',
        'mt': 'data/wmt20-zh-en/wmt20-zh-en_references.jsonl',
    }
    definition = definitions[args.task]

    with open(reference_files[args.task]) as f:
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
            assert isinstance(input_text, str), f'{type(input_text)}'
            if args.add_prompt:
                input_text = [definition, input_text, 0]
            all_texts_to_encode.append(input_text)
    all_ref_nums.append(num_count)
    total_line_num = len(lines)
    assert len(all_ref_nums) == total_line_num + 1

    # with FileLock('model.lock'):
    emb_model = SentenceTransformer(args.model_name, cache_folder=args.cache)
    emb_model.cuda()
    embeddings = np.asarray(emb_model.encode(all_texts_to_encode, batch_size=32))
    embeddings = embeddings.tolist()

    reshaped_embeddings = []
    for idx in range(total_line_num):
        reshaped_embeddings.append(embeddings[all_ref_nums[idx]:all_ref_nums[idx + 1]])
    with open(os.path.join(args.cache,f'references_emb_{identifier}.json'), 'w') as f:
        json.dump(reshaped_embeddings, f, indent=4)

    thumb_files = {
        'mscoco': 'data/mscoco/mscoco_THumB-1.0.jsonl',
        'cnndm': 'data/cnndm/cnndm_THumB-1.0.jsonl',
        'mt': 'data/wmt20-zh-en/wmt20-zh-en_THumB-1.0.jsonl',
    }
    with open(thumb_files[args.task]) as f:
        lines = f.readlines()
    all_texts_to_encode = []
    for l in lines:
        d = json.loads(l)
        input_text = d['hyp']
        if args.add_prompt:
            assert isinstance(input_text, str), f'{type(input_text)}'
            input_text = [definition, input_text, 0]
        all_texts_to_encode.append(input_text)

    embeddings = np.asarray(emb_model.encode(all_texts_to_encode, batch_size=32))

    with open(os.path.join(args.cache,f'THumB-1.0_emb_{identifier}.json'), 'w') as f:
        json.dump(embeddings.tolist(), f, indent=4)

    compute_corr(os.path.join(args.cache,f'THumB-1.0_emb_{identifier}.json'),
                 os.path.join(args.cache,f'references_emb_{identifier}.json'), thumb_files[args.task])