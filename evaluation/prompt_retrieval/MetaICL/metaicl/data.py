# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class MetaICLData(object):

    def __init__(self, logger=None, tokenizer=None, method="channel", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1):

        self.logger = logger
        self.tokenizer = tokenizer
        self.method = method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank

        self.tensorized_inputs = None
        self.metadata = None

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        # self.logger.info(shape)
        for v in inputs.values():
            assert v.shape==shape
        if "labels" in inputs:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        else:
            dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification):
        assert len(predictions)==len(self.metadata)
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)
            if is_classification:
                recalls[groundtruth].append(is_correct)
                precisions[prediction].append(is_correct)

        if not is_classification:
            return np.mean(accs)

        f1s = []
        for key in recalls:
            precision = np.mean(precisions[key]) if key in precisions else 1.0
            recall = np.mean(recalls[key])
            if precision+recall==0:
                f1s.append(0)
            else:
                f1s.append(2*precision*recall / (precision+recall))

        return np.mean(f1s)

    def _prepro_each_datapoint(self, dp, is_first=True, is_training=False, for_demonstrations=False,
                               add_newlines=True):
        dp = dp.copy()
        if add_newlines:
            no_label = np.all([option=="" for option in dp["options"]])
            no_input = dp["input"]==""
            if self.method=="direct":
                if not is_first:
                    if no_input:
                        dp["input"] = "\n\n" + dp["input"]
                    else:
                        dp["input"] = "\n\n\n" + dp["input"]
                if not no_label:
                    dp["output"] = "\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n" + opt for opt in dp["options"]]
            elif self.method=="channel":
                if not is_first:
                    dp["output"] = "\n\n\n" + dp["output"]
                    if "options" in dp:
                        dp["options"] = ["\n\n\n" + opt for opt in dp["options"]]
                if not no_input:
                    if not no_label:
                        dp["input"] = "\n" + dp["input"]
            else:
                raise NotImplementedError()
        else:
            if not is_first:
                if self.method=="direct":
                    dp["input"] = " " + dp["input"]
                elif self.method=="channel":
                    dp["output"] = " " + dp["output"]
                    if "options" in dp:
                        dp["options"] = [" "+opt for opt in dp["options"]]
                else:
                    raise NotImplementedError()
            if self.method=="direct":
                dp["output"] = " " + dp["output"]
                if "options" in dp:
                    dp["options"] = [" " + opt for opt in dp["options"]]
            elif self.method=="channel":
                dp["input"] = " " + dp["input"]
            else:
                raise NotImplementedError()

        input_tokens = self.tokenizer(dp["input"])["input_ids"]

        if is_training or for_demonstrations:
            output_tokens = self.tokenizer(dp["output"])["input_ids"]

            if "task" in dp:
                if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                        len(input_tokens)+len(output_tokens)+2>self.max_length_per_example:
                    input_tokens = input_tokens[:self.max_length_per_example // 2]
                    output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

                elif len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                    if dp["task"].startswith("inst:") and len(input_tokens)<len(output_tokens):
                        output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                    else:
                        input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

            if self.method=="direct":
                return input_tokens, output_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens
            else:
                raise NotImplementedError()

        else:
            assert len(dp["options"])>=2, dp
            assert dp["output"] in dp["options"]
            option_tokens = [self.tokenizer(option)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

            input_tokens = [input_tokens for _ in option_tokens]
            output_tokens = option_tokens
            option_tokens = [dp["options"].index(dp["output"])]

            if self.method=="direct":
                return input_tokens, output_tokens, option_tokens
            elif self.method=="channel":
                return output_tokens, input_tokens, option_tokens
            else:
                raise NotImplementedError()

    def _tensorize_for_training(self, train_data):
        for dp in train_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        n_answers = []

        if self.use_demonstrations:
            first_tokenized = []
            nonfirst_tokenized = []

            for dp in train_data:
                first_tokenized.append(self._prepro_each_datapoint(
                    dp, is_first=True, is_training=True))
                nonfirst_tokenized.append(self._prepro_each_datapoint(
                    dp, is_first=False, is_training=True))

            N=1

            def _draw_random(tot, n, exclude_indices):
                r = np.random.choice([i for i in range(tot) if i not in exclude_indices])
                if n==1:
                    return [r]
                return [r] + _draw_random(tot, n-1, exclude_indices | set([r]))

            for dp_idx, dp in enumerate(train_data):
                for _ in range(N):
                    demo_indices = _draw_random(len(train_data), self.k, set([dp_idx]))
                    inputs = []
                    for demo_idx, index in enumerate(demo_indices):
                        if demo_idx==0:
                            inputs += first_tokenized[index][0] + first_tokenized[index][1]
                        else:
                            inputs += nonfirst_tokenized[index][0] + nonfirst_tokenized[index][1]
                        assert index!=dp_idx
                    inputs += nonfirst_tokenized[dp_idx][0]
                    outputs = nonfirst_tokenized[dp_idx][1]

                    encoded = prepro_sentence_pair_single(
                        inputs, outputs, self.max_length, bos_token_id, eos_token_id,
                        allow_truncation=True)

                    input_ids.append(encoded[0])
                    attention_mask.append(encoded[1])
                    token_type_ids.append(encoded[2])

        else:
            for dp in train_data:
                inputs, outputs = self._prepro_each_datapoint(
                    dp, is_first=True, is_training=True)

                encoded = prepro_sentence_pair_single(
                    inputs, outputs, self.max_length, bos_token_id, eos_token_id)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))


    def _tensorize_for_training_with_random_english_words(self, train_data):
        for dp in train_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        n_answers = []

        from english_words import english_words_set
        english_words_set = sorted(english_words_set)

        if self.use_demonstrations:
            N=1
            def _draw_random(tot, n, exclude_indices):
                r = np.random.choice([i for i in range(tot) if i not in exclude_indices])
                if n==1:
                    return [r]
                return [r] + _draw_random(tot, n-1, exclude_indices | set([r]))

            for dp_idx, dp in enumerate(train_data):
                for _ in range(N):
                    demo_indices = _draw_random(len(train_data), self.k, set([dp_idx]))
                    inputs = []

                    # create a mapping
                    mapping = {option: np.random.choice(english_words_set) for option in dp["options"]}

                    for demo_idx, index in enumerate(demo_indices):
                        curr_demo_dp = train_data[index].copy()
                        curr_demo_dp["output"] = mapping[curr_demo_dp["output"]]
                        curr_demo_dp["options"] = [mapping[o] for o in curr_demo_dp["options"]]
                        inputs_, outputs_ = self._prepro_each_datapoint(curr_demo_dp, is_first=demo_idx==0, is_training=True)
                        inputs += inputs_ + outputs_
                        assert index!=dp_idx

                    curr_dp = dp.copy()
                    curr_dp["output"] = mapping[curr_dp["output"]]
                    curr_dp["options"] = [mapping[o] for o in curr_dp["options"]]
                    inputs_, outputs = self._prepro_each_datapoint(curr_dp, is_first=False, is_training=True)
                    inputs += inputs_
                    encoded = prepro_sentence_pair_single(
                        inputs, outputs, self.max_length, bos_token_id, eos_token_id,
                        allow_truncation=True)
                    input_ids.append(encoded[0])
                    attention_mask.append(encoded[1])
                    token_type_ids.append(encoded[2])
        else:
            for dp in train_data:
                # create a mapping
                mapping = {option: np.random.choice(english_words_set) for option in dp["options"]}
                dp["output"] = mapping[dp["output"]]
                dp["options"] = [mapping[o] for o in dp["options"]]
                inputs, outputs = self._prepro_each_datapoint(
                    dp, is_first=True, is_training=True)
                encoded = prepro_sentence_pair_single(
                    inputs, outputs, self.max_length, bos_token_id, eos_token_id)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        return dict(input_ids=torch.LongTensor(input_ids),
                    attention_mask=torch.LongTensor(attention_mask),
                    token_type_ids=torch.LongTensor(token_type_ids))

    def tensorize(self, _train_data, _test_data, options=None,
                  add_newlines=True):

        # print("start debug message")
        # print(options is not None)
        # for i, dp in enumerate(_test_data):
        #     print(i,"options" not in dp)
        # print("end debug message\n\n\n")
        if options is not None:
            assert np.all([dp["output"] in options for dp in _train_data])
            for i, dp in enumerate(_test_data):
                # assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data = [], []
        if self.use_demonstrations:
            for dp in _train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                train_data.append(dp.copy())
        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        if self.use_demonstrations:
            assert len(train_data)==self.k
            demonstrations = []
            for i, dp in enumerate(train_data):
                input_, output_ = self._prepro_each_datapoint(
                    dp, is_first=i==0, for_demonstrations=True,
                    add_newlines=add_newlines)
                demonstrations += input_ + output_

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            indices = [[i] for i in range(len(input_ids), len(input_ids)+len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_

                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, bos_token_id, eos_token_id,
                    allow_truncation=self.use_demonstrations)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata

    def tensorize_for_training(self, train_data, keyword, seed, use_random_english_words=False):
        assert self.tensorize_dir is not None

        if not os.path.exists(self.tensorize_dir):
            os.makedirs(self.tensorize_dir)

        method_name = self.method + "-demon" if self.use_demonstrations else self.method
        k_name = "%d-%d" % (len(train_data), self.k) if self.use_demonstrations else len(train_data)
        length_name = "%d-%d" % (self.max_length, self.max_length_per_example) if self.use_demonstrations else self.max_length
        postfix = "-randomEnglish" if use_random_english_words else ""

        tensorize_path = os.path.join(self.tensorize_dir,
                                      "{}_{}_k={}_seed={}_length={}{}-rank=%d.pkl".format(
                                          keyword, method_name, k_name, seed, length_name,
                                          postfix))

        # if self.local_rank==-1:
        #     self.logger.info(tensorize_path)
        # else:
        #     self.logger.info(tensorize_path % self.local_rank)
        all_tensorize_paths = [tensorize_path % i for i in range(self.n_gpu)]

        if not self.do_tensorize:
            if not np.all([os.path.exists(_path) for _path in all_tensorize_paths]):
                print("Tensorization was not done. Run with `--do_tensorize` without distributed mode"
                            "and then run training command again")
                raise NotImplementedError()

            if self.local_rank==-1:
                inputs = defaultdict(list)
                for i in range(self.n_gpu):
                    with open(tensorize_path % i, "rb") as f:
                        curr_inputs = pkl.load(f)
                    for k, v in curr_inputs.items():
                        inputs[k] += v
            else:
                assert 0<=self.local_rank<self.n_gpu
                with open(tensorize_path % self.local_rank, "rb") as f:
                    inputs = pkl.load(f)

            self.tensorized_inputs = inputs
            return

        assert self.local_rank==-1
        if any([os.path.exists(_path) for _path in all_tensorize_paths]):
            print("tensorize file already exists...")
            return

        unique_task_names = set([dp["task"] for dp in train_data])
        sharded_inputs = []
        if self.use_demonstrations or (len(unique_task_names)>200 and len(train_data)>=1638400):
            tot = 0
            for i, curr_train_task in enumerate(unique_task_names):
                curr_train_data = [dp for dp in train_data if dp["task"]==curr_train_task]
                tot += len(curr_train_data)
                if self.use_demonstrations and len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 10% of the data
                    print("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//10)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//10]
                    curr_train_data = [curr_train_data[i] for i in indices]
                elif len(unique_task_names)>200 and len(train_data)>=1638400:
                    # data is too huge; sampling 50% of the data
                    print("Sampling training data from %d to %d", len(curr_train_data), len(curr_train_data)//2)
                    indices = np.random.permutation(range(len(curr_train_data)))[:len(curr_train_data)//2]
                    curr_train_data = [curr_train_data[i] for i in indices]
                sharded_inputs.append(curr_train_data)
            assert len(train_data)==tot
        else:
            n_per_shard = math.ceil(len(train_data) / self.n_process)
            for i in range(self.n_process):
                sharded_inputs.append(train_data[i*n_per_shard:(i+1)*n_per_shard])

        inputs = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        _tensorize_for_training = self._tensorize_for_training_with_random_english_words \
            if use_random_english_words else self._tensorize_for_training
        if self.n_process==1:
            for in_ in sharded_inputs:
                out = _tensorize_for_training(in_)
                for key in ["input_ids", "attention_mask", "token_type_ids"]:
                    inputs[key] += out[key].numpy().tolist()
        else:
            with Pool(self.n_process) as p:
                for out in p.imap_unordered(_tensorize_for_training, sharded_inputs):
                    for key in ["input_ids", "attention_mask", "token_type_ids"]:
                        inputs[key] += out[key].numpy().tolist()

        N = len(inputs["input_ids"])
        indices = np.random.permutation(range(N))
        for k, v in inputs.items():
            inputs[k] = np.array(v)[indices]
        n_per_shard = math.ceil(N / self.n_gpu)

        for i, _path in enumerate(all_tensorize_paths):
            start = i*n_per_shard
            end = (i+1)*n_per_shard
            curr_inputs = {k:v[start:end].tolist() for k, v in inputs.items()}
            with open(_path, "wb") as f:
                pkl.dump(curr_inputs, f)
            # self.logger.info("Preprocessing done for i=%d" % i)

        # self.logger.info("Finish saving preprocessed data ...")

    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        idx = 0
        text = "Checking the first example..."
        input_ids = self.tensorized_inputs["input_ids"][idx]
        token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        text += "\nInput:\n"
        text += self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
        text += "\nOutput:\n"
        text += self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

        if return_string:
            return text

        # if self.local_rank<=0:
        #     self.logger.info(text)

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id,
                                allow_truncation=False):

    #if bos_token_id is not None:
    #    ids1 = [bos_token_id] + ids1
    #if eos_token_id is not None:
    #    ids2 = ids2 + [eos_token_id]
    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # len = max_length-len(ids2)
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    return input_ids, attention_mask, token_type_ids

def prepro_sentence_pair(train_inputs, test_inputs, max_length,
                         bos_token_id, eos_token_id,
                         allow_truncation=False):
    input_ids, attention_mask, token_type_ids = [], [], []
    for test_input in test_inputs:
        for train_input in train_inputs:
            _input_ids, _attention_mask, _token_type_ids = \
                prepro_sentence_pair_single(train_input, test_input, max_length,
                                            bos_token_id, eos_token_id,
                                            allow_truncation=allow_truncation)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
            token_type_ids.append(_token_type_ids)

    return {"input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "token_type_ids": torch.LongTensor(token_type_ids)}

