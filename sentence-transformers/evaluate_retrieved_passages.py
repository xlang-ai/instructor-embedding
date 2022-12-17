# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import glob

import numpy as np
import torch

import src.utils

from src.evaluation import calculate_matches

logger = logging.getLogger(__name__)

def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    #logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    #logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return top_k_hits


def main(opt):
    logger = src.utils.init_logger(opt, stdout_only=True)
    datapaths = glob.glob(args.data)
    r20, r100 = [], []
    for path in datapaths:
        data = []
        with open(path, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
            #data = json.load(fin)
        answers = [ex['answers'] for ex in data]
        top_k_hits = validate(data, args.validation_workers)
        message = f"Evaluate results from {path}:"
        for k in [5, 10, 20, 100]:
            if k <= len(top_k_hits):
                recall = 100 * top_k_hits[k-1]
                if k == 20:
                    r20.append(f"{recall:.1f}")
                if k == 100:
                    r100.append(f"{recall:.1f}")
                message += f' R@{k}: {recall:.1f}'
        logger.info(message)
    print(datapaths)
    print('\t'.join(r20))
    print('\t'.join(r100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None)
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")

    args = parser.parse_args()
    main(args)
