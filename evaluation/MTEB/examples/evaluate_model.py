import os
import sys
import logging
import argparse
from mteb import MTEB
from InstructorEmbedding import INSTRUCTOR
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None,type=str)
    parser.add_argument('--output_dir', default=None,type=str)
    parser.add_argument('--task_name', default=None,type=str)
    parser.add_argument('--cache_dir', default=None,type=str)
    parser.add_argument('--result_file', default=None,type=str)
    parser.add_argument('--prompt', default=None,type=str)
    parser.add_argument('--split', default='test',type=str)
    parser.add_argument('--batch_size', default=128,type=int)
    args = parser.parse_args()

    if not args.result_file.endswith('.txt') and not os.path.isdir(args.result_file):
        os.makedirs(args.result_file,exist_ok=True)

    # from tqdm import tqdm
    # from functools import partialmethod
    #
    # tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    model = INSTRUCTOR(args.model_name,cache_folder=args.cache_dir)
    evaluation = MTEB(tasks=[args.task_name],task_langs=["en"])
    evaluation.run(model, output_folder=args.output_dir, eval_splits=[args.split],args=args,)

    print("--DONE--")
