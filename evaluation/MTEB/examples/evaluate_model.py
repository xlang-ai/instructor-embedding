import os
import sys
import logging
import argparse
from evaluation.MTEB.mteb import MTEB
from InstructorEmbedding import INSTRUCTOR
import time

logging.basicConfig(level=logging.INFO)
start_time = time.perf_counter()
print("--start--Outside")
if __name__ == '__main__':
    print("--start--inside")
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger...")
    debugpy.wait_for_client()
    print("Debugger Attached.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None,type=str)
    parser.add_argument('--output_dir', default=None,type=str)
    parser.add_argument('--task_names',nargs="+", default= ['HotpotQA',  'MSMARCO', 'DBPedia', 'Touche2020',  'NQ','CQADupstackTexRetrieval', 'CQADupstackWebmastersRetrieval', 'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackMathematicaRetrieval', 'CQADupstackStatsRetrieval', 'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackAndroidRetrieval', 'CQADupstackWordpressRetrieval'])
    # 'FEVER', 'QuoraRetrieval', 'NFCorpus',
    parser.add_argument('--cache_dir', default=None,type=str)
    parser.add_argument('--save_suffix', default="",type=str)
    parser.add_argument('--result_file', default=None,type=str)
    parser.add_argument('--prompt', default=None,type=str)
    parser.add_argument('--split', default='test',type=str)
    parser.add_argument('--batch_size', default=128,type=int)
    parser.add_argument('--dont_corpus_prepend',action="store_true")
    parser.add_argument('--overlap_percentage',default=0,type=int)
    args = parser.parse_args()

    if not args.result_file.endswith('.txt') and not os.path.isdir(args.result_file):
        os.makedirs(args.result_file,exist_ok=True)

    # from tqdm import tqdm
    # from functools import partialmethod
    #
    # tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    model = INSTRUCTOR(args.model_name,cache_folder=args.cache_dir)
    evaluation = MTEB(tasks=args.task_names,task_langs=["en"])
    evaluation.run(model, output_folder=args.output_dir, eval_splits=[args.split],args=args,)
    print("--DONE--inside")
end_time = time.perf_counter()
print("--DONE-- outside")
print(f"total time taken: {end_time-start_time}")
