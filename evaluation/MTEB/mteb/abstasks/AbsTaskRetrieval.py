import logging
from time import time
from typing import Dict, List

import torch.multiprocessing as mp

from sentence_transformers import SentenceTransformer

from .AbsTask import AbsTask
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]
DRPES_METHODS = ["start_multi_process_pool", "stop_multi_process_pool", "encode_queries", "encode_corpus", "encode_corpus_parallel"]

DEFINITIONS = {
    'hku-nlp/instructor-xl': {
        'ClimateFEVER':
            {
                'query': 'Represent the climate question for retrieving documents; Input: ',
                'corpus': 'Represent the climate document for retrieval; Input: ',
            },
        'HotpotQA':
            {
                'query': 'Represent a Wikipedia question for retrieving supporting documents; Input: ',
                'corpus': 'Represent the Wikipedia document for retrieval; Input: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence; Input: ',
                'corpus': 'Represent the evidence; Input: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving documents:\n',
                'corpus': 'Represent the document for retrieval:\n',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia sentence for retrieving documents:\n',
                'corpus': 'Represent the Wikipedia document for retrieval:\n',
            },
        'NQ':
            {
                'query': 'Represent a Wikipedia question for retrieving supporting documents:\n',
                'corpus': 'Represent the document for retrieval; Input: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions; Input: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions; Input: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent the Scientific question for retrieving supporting documents:\n',
                'corpus': 'Represent the Scientific paper:\n',
            },
        'TRECCOVID':
            {
                'query': 'Represent a Coronavirus question for retrieving supporting documents:\n',
                'corpus': 'Represent the Coronavirus document:\n',
            },
        'Touche2020':
            {
                'query': 'Represent a question for retrieving supporting documents:\n',
                'corpus': 'Represent the document for retrieval; Input: ',
            },
        'SciFact':
            {
                'query': 'Represent a Scientific question for retrieving a supporting document;\n',
                'corpus': 'Represent the science document:\n',
            },
        'NFCorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document:\n',
                'corpus': 'Represent the medical document for retrieval:\n',
            },
        'ArguAna':
            {
                'query': 'Represent the Debating discourse for retrieving a counter-discourse:\n',
                'corpus': 'Represent the Counter-discourse:\n',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers:\n',
                'corpus': 'Represent the answer for retrieval:\n',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Website question for retrieving answers:\n',
                'corpus': 'Represent the Website answer:\n',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents:\n',
                'corpus': 'Represent the English answer for retrieval:\n',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers:\n',
                'corpus': 'Represent the Gaming answer for retrieval:\n',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers:\n',
                'corpus': 'Represent the Gis answer for retrieval:\n',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers:\n',
                'corpus': 'Represent the Unix answer for retrieval:\n',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers:\n',
                'corpus': 'Represent the Mathematical answer for retrieval:\n',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers; Input: ',
                'corpus': 'Represent the Statistical answer for retrieval; Input: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers; Input: ',
                'corpus': 'Represent the Physics answer for retrieval; Input: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers:\n',
                'corpus': 'Represent the Programming answer for retrieval:\n',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers; Input: ',
                'corpus': 'Represent the Android answer for retrieval; Input: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers; Input: ',
                'corpus': 'Represent the Wordpress answer for retrieval; Input: ',
            },
        'FiQA2018':
            {
                'query': 'Represent a financial question for retrieving the supporting answers:\n',
                'corpus': 'Represent the finance answer for retrieval:\n',
            },
    },
    'hku-nlp/instructor-large':{
        'ClimateFEVER':
            {
                'query': 'Represent the climate question for retrieving documents; Input: ',
                'corpus': 'Represent the climate document for retrieval; Input: ',
            },
        'HotpotQA':
            {
                'query': 'Represent a Wikipedia question for retrieving supporting documents:\n',
                'corpus': 'Represent the wikipedia document:\n',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence; Input: ',
                'corpus': 'Represent the evidence; Input: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving documents:\n',
                'corpus': 'Represent the document for retrieval:\n',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia sentence for retrieving documents:\n',
                'corpus': 'Represent the Wikipedia document for retrieval:\n',
            },
        'NQ':
            {
                'query': 'Represent a Wikipedia question for retrieving supporting documents:\n',
                'corpus': 'Represent the document for retrieval; Input: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions; Input: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions; Input: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent the Scientific question for retrieving supporting documents:\n',
                'corpus': 'Represent the Scientific paper:\n',
            },
        'TRECCOVID':
            {
                'query': 'Represent a Coronavirus question for retrieving supporting documents:\n',
                'corpus': 'Represent the Coronavirus document:\n',
            },
        'Touche2020':
            {
                'query': 'Represent a question; Input: ',
                'corpus': 'Represent an argument; Input: ',
            },
        'SciFact':
            {
                'query': 'Represent a Scientific question for retrieving a supporting document;\n',
                'corpus': 'Represent the science document:\n',
            },
        'NFCorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document:\n',
                'corpus': 'Represent the medical document for retrieval:\n',
            },
        'ArguAna':
            {
                'query': 'Represent the Debating discourse for retrieving a counter-discourse:\n',
                'corpus': 'Represent the Counter-discourse:\n',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers:\n',
                'corpus': 'Represent the answer for retrieval:\n',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Website question for retrieving answers:\n',
                'corpus': 'Represent the Website answer:\n',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents:\n',
                'corpus': 'Represent the English answer for retrieval:\n',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers:\n',
                'corpus': 'Represent the Gaming answer for retrieval:\n',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers:\n',
                'corpus': 'Represent the Gis answer for retrieval:\n',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers:\n',
                'corpus': 'Represent the Unix answer for retrieval:\n',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers:\n',
                'corpus': 'Represent the Mathematical answer for retrieval:\n',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers; Input: ',
                'corpus': 'Represent the Statistical answer for retrieval; Input: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers; Input: ',
                'corpus': 'Represent the Physics answer for retrieval; Input: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers:\n',
                'corpus': 'Represent the Programming answer for retrieval:\n',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers; Input: ',
                'corpus': 'Represent the Android answer for retrieval; Input: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers; Input: ',
                'corpus': 'Represent the Wordpress answer for retrieval; Input: ',
            },
        'FiQA2018':
            {
                'query': 'Represent a financial question for retrieving the supporting answers:\n',
                'corpus': 'Represent the finance answer for retrieval:\n',
            },
    },
}

class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_dres_compatible(model, is_parallel=True):
        methods = DRPES_METHODS if is_parallel else DRES_METHODS
        for method in methods:
            op = getattr(model, method, None)
            if not(callable(op)):
                return False
        return True

    def evaluate(
        self,
        model,
        split="test",
        batch_size=128,
        corpus_chunk_size=None,
        target_devices=None,
        score_function="cos_sim",
        **kwargs
    ):
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]

        try:
            if self.description["beir_name"].startswith("cqadupstack"):
                raise ImportError("CQADupstack is incompatible with latest BEIR")
            from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
            model = model if self.is_dres_compatible(model, is_parallel=True) else DRESModel(model,**kwargs)
            model = DRPES(
                model,
                batch_size=batch_size,
                target_devices=target_devices,
                corpus_chunk_size=corpus_chunk_size,
                **kwargs,
            )
        except ImportError:
            if target_devices is not None:
                logger.warning(
                    "DenseRetrievalParallelExactSearch could not be imported from beir. Using DenseRetrievalExactSearch instead."
                )
                logger.warning("The parameter target_devices is ignored.")

            from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

            model = model if self.is_dres_compatible(model, is_parallel=False) else DRESModel(model,**kwargs)

            model = DRES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
                **kwargs,
            )

        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values)
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }

        return scores


class DRESModel:
    """
    Dense Retrieval Exact Search (DRES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """

    def __init__(self, model, sep=" ", **kwargs):
        self.model = model
        self.sep = sep
        self.args = kwargs['args']
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(
                target=SentenceTransformer._encode_multi_process_worker,
                args=(process_id, device_name, self.model, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool["output"]
        [output_queue.get() for _ in range(len(pool["processes"]))]
        return self.model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        new_sentences = []
        if self.args.prompt and isinstance(DEFINITIONS[self.args.prompt][self.args.task_name],str):
            instruction = DEFINITIONS[self.args.prompt][self.args.task_name]
        elif self.args.prompt:
            instruction = DEFINITIONS[self.args.prompt][self.args.task_name]['query']
        if self.args.prompt:
            print('with prompt')
            for s in queries:
                if len(self.tokenizer(instruction + s)['input_ids']) <= 512:
                    new_sentences.append([instruction, s, 0])
                else:
                    new_sentences.append(['', s, 0])
        else:
            new_sentences = queries

        return self.model.encode(new_sentences, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if self.args.task_name in ['SciFact','Touche2020']:
            if type(corpus) is dict:
                sentences = [
                    (corpus["title"][i] + ' ' + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                    for i in range(len(corpus["text"]))
                ]
            else:
                sentences = [
                    (doc["title"] + ' ' + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                    for doc in corpus
                ]
            new_sentences = []
            instruction = DEFINITIONS[self.args.prompt][self.args.task_name]['corpus']
            for s in sentences:
                new_sentences.append([instruction, s, 0])
            return self.model.encode(sentences, batch_size=128, **kwargs)
        else:
            if type(corpus) is dict:
                sentences = [
                    (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                    for i in range(len(corpus["text"]))
                ]
            else:
                sentences = [
                    (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                    for doc in corpus
                ]

            if self.args.prompt:
                new_sentences = []
                if isinstance(DEFINITIONS[self.args.prompt][self.args.task_name], str):
                    instruction = DEFINITIONS[self.args.prompt][self.args.task_name]
                else:
                    instruction = DEFINITIONS[self.args.prompt][self.args.task_name]['corpus']
                for s in sentences:
                    if len(self.tokenizer(instruction + s)['input_ids']) <= 512:
                        new_sentences.append([instruction, s, 0])
                    else:
                        new_sentences.append(['', s, 0])
                    # new_sentences.append([instruction, s, 0])
            else:
                new_sentences = sentences
            return self.model.encode(new_sentences, batch_size=batch_size, **kwargs)

    def encode_corpus_parallel(
        self, corpus: List[Dict[str, str]], pool: Dict[str, object], batch_size: int, chunk_id: int, **kwargs
    ):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]

        if chunk_id is not None and chunk_id >= len(pool["processes"]):
            output_queue = pool["output"]
            output_queue.get()

        input_queue = pool["input"]
        input_queue.put([chunk_id, batch_size, sentences])