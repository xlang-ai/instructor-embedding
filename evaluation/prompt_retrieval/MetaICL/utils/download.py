# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
script for downloading preprocessed data and trained checkpoints
'''
import os
import json
import argparse
import subprocess

from .utils import all_settings, all_methods
from .utils import download_file, get_checkpoint_id

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default=False, action="store_true")
    parser.add_argument("--demo_data", default=False, action="store_true")

    parser.add_argument("--target_only", default=False, action="store_true")
    parser.add_argument("--inst", default=False, action="store_true")

    parser.add_argument("--setting", default="all", type=str,
                        choices=["all"]+all_settings)
    parser.add_argument("--method", default="all", type=str,
                        choices=["all"]+all_methods)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    return args

def main(args):
    if args.demo_data:
        download_file("15grQwt3B1tALtUCGtaDI_rwC28LL8wSj",
                      os.path.join(args.data_dir, "financial_phrasebank", "financial_phrasebank_16_100_train.jsonl"))

    if args.checkpoints:
        if args.setting=="all":
            settings = all_settings
        else:
            settings = [args.setting]
        if args.method=="all":
            methods = all_methods
        else:
            methods = [args.method]
        for method in methods:
            for setting in settings:
                _, _, _id = get_checkpoint_id(method + "/" + setting)
                download_file(_id, os.path.join(args.checkpoint_dir, method, setting, "model.pt"))

if __name__=='__main__':
    args = parse_args()
    main(args)
