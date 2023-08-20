import logging
import queue
from time import time
from typing import Dict, List

import torch.multiprocessing as mp

from .AbsTask import AbsTask
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]
DRPES_METHODS = ["start_multi_process_pool", "stop_multi_process_pool", "encode_queries", "encode_corpus", "encode_corpus_parallel"]

DEFINITIONS = {
    # 'hkunlp/instructor-xl update': {
    #     'ClimateFEVER':
    #         {
    #             'query': 'Represent the Climate questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the Climate documents for retrieval: ',
    #         },
    #     'HotpotQA':
    #         {
    #             'query': 'Represent the Wikipedia questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the Wikipedia documents for retrieval: ',
    #         },
    #     'FEVER':
    #         {
    #             'query': 'Represent the Wikipedia facts to retrieve a supporting document: ',
    #             'corpus': 'Represent the Wikipedia documents for retrieval: ',
    #         },
    #     'MSMARCO':
    #         {
    #             'query': 'Represent the questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the documents for retrieval: ',
    #         },
    #     'DBPedia 40.24':
    #         {
    #             'query': 'Represent the Wikipedia questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the Wikipedia documents for retrieval: ',
    #         },
    #     'DBPedia':
    #         {
    #             'query': 'Represent the Wikipedia questions for retrieving a supporting document: ',
    #             'corpus': 'Represent the Wikipedia documents for retrieval: ',
    #         },
    #     'NQ 57.32':
    #         {
    #             'query': 'Represent the Wikipedia questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the Wikipedia documents for retrieval: ',
    #         },
    #     'NQ':
    #         {
    #             'query': 'Represent the Wikipedia questions for retrieving a supporting document: ',
    #             'corpus': 'Represent the Wikipedia documents for retrieval: ',
    #         },
    #     'QuoraRetrieval':
    #         {
    #             'query': 'Represent the Quora question to retrieve question: ',
    #             'corpus': 'Represent the Quora question to retrieve question: ',
    #         },
    #     'SCIDOCS':
    #         {
    #             'query': 'Represent the Science question: ',
    #             'corpus': 'Represent the Science document: '
    #         },
    #     'TRECCOVID 71.4':
    #         {
    #             'query': 'Represent the Coronavirus questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the Coronavirus documents for retrieval: ',
    #         },
    #     'TRECCOVID':
    #         {
    #             'query': 'Represent the Coronavirus questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the Coronavirus documents: ',
    #         },
    #     'Touche2020 23.44':
    #         {
    #             'query': 'Represent questions: ',
    #             'corpus': 'Represent arguments: ',
    #         },
    #     'Touche2020':
    #         {
    #             'query': 'Represent conversation questions: ',
    #             'corpus': 'Represent conversation arguments: ',
    #         },
    #     'SciFact':
    #         {
    #             'query': 'Represent the Scientific queries for retrieving a supporting passage: ',
    #             'corpus': 'represent the scientific paragraph for retrieval: ',
    #         },
    #     'NFCorpus':
    #         {
    #             'query': 'Represent the nutrition facts to retrieve Public medical articles: ',
    #             'corpus': 'Represent the Public medical articles for retrieval: ',
    #         },
    #     'ArguAna 55.65':
    #         {
    #             'query': 'Represent Debating conversations to retrieve a counter-argument: ',
    #             'corpus': 'Represent counter-arguments: ',
    #         },
    #     'ArguAna':
    #         {
    #             'query': 'Represent Debating conversation discourses to retrieve a counter-argument: ',
    #             'corpus': 'Represent counter-arguments in debating: ',
    #         },
    #     'CQADupstackTexRetrieval':
    #         {
    #             'query': 'Represent the questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the answers for retrieval: ',
    #         },
    #     'CQADupstackWebmastersRetrieval':
    #         {
    #             'query': 'Represent the Webmaster questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Webmaster answers for retrieval: ',
    #         },
    #     'CQADupstackEnglishRetrieval':
    #         {
    #             'query': 'Represent the English questions to retrieve a supporting document: ',
    #             'corpus': 'Represent the English answers for retrieval: ',
    #         },
    #     'CQADupstackGamingRetrieval':
    #         {
    #             'query': 'Represent the Gaming questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Gaming answers for retrieval: ',
    #         },
    #     'CQADupstackGisRetrieval':
    #         {
    #             'query': 'Represent the Gis questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Gis answers for retrieval: ',
    #         },
    #     'CQADupstackUnixRetrieval':
    #         {
    #             'query': 'Represent the Unix questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Unix answers for retrieval: ',
    #         },
    #     'CQADupstackMathematicaRetrieval':
    #         {
    #             'query': 'Represent the Mathematical questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Mathematical answers for retrieval: ',
    #         },
    #     'CQADupstackStatsRetrieval':
    #         {
    #             'query': 'Represent the Statistical questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Statistical answers for retrieval: ',
    #         },
    #     'CQADupstackPhysicsRetrieval':
    #         {
    #             'query': 'Represent the Physics questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Physics answers for retrieval: ',
    #         },
    #     'CQADupstackProgrammersRetrieval':
    #         {
    #             'query': 'Represent the Programming questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Programming answers for retrieval: ',
    #         },
    #     'CQADupstackAndroidRetrieval':
    #         {
    #             'query': 'Represent the Android questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Android answers for retrieval: ',
    #         },
    #     'CQADupstackWordpressRetrieval':
    #         {
    #             'query': 'Represent the Wordpress questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the Wordpress answers for retrieval: ',
    #         },
    #     'FiQA2018':
    #         {
    #             'query': 'Represent the finance questions to retrieve a supporting answer: ',
    #             'corpus': 'Represent the finance answers for retrieval: ',
    #         },
    # },
    'hkunlp/instructor-xl': {
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'HotpotQA':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia questions to retrieve a supporting document: ',
                'corpus': 'Represent the Wikipedia documents for retrieval: ',
            },
        'NQ':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question to retrieve question: ',
                'corpus': 'Represent the Quora question to retrieve question: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'TRECCOVID':
            {
                'query': 'Represent the Coronavirus questions to retrieve a supporting document: ',
                'corpus': 'Represent the Coronavirus documents for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent questions: ',
                'corpus': 'Represent arguments: ',
            },
        'SciFact':
            {
                'query': 'Represent the Scientific queries for retrieving a supporting passage: ',
                'corpus': 'represent the scientific paragraph for retrieval: ',
            },
        'NFCorpus':
            {
                'query': 'Represent the nutrition facts to retrieve Public medical articles: ',
                'corpus': 'Represent the Public medical articles for retrieval: ',
            },
        'ArguAna':
            {
                'query': 'Represent Debating conversations to retrieve a counter-argument: ',
                'corpus': 'Represent counter-arguments: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix questions to retrieve a supporting answer: ',
                'corpus': 'Represent the Unix answers for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'FiQA2018':
            {
                'query': 'Represent the finance questions to retrieve a supporting answer: ',
                'corpus': 'Represent the finance answers for retrieval: ',
            },
    },
    'hkunlp/instructor-large':{
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'HotpotQA':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'NQ':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'TRECCOVID':
            {
                'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
                'corpus': 'Represent the Coronavirus document for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent a question: ',
                'corpus': 'Represent an argument: ',
            },
        'SciFact':
            {
                'query': 'Represent a Scientific query for retrieving a supporting passage; ',
                'corpus': 'represent the Scientific passage for retrieval; ',
            },
        'NFCorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document: ',
                'corpus': 'Represent the medical document for retrieval: ',
            },
        'ArguAna':
            {
                'query': 'Represent a Debate argument for retrieving a counter-argument: ',
                'corpus': 'Represent a Counter-argument: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers: ',
                'corpus': 'Represent the Unix answer for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'FiQA2018':
            {
                'query': 'Represent the finance question for retrieving the supporting answers: ',
                'corpus': 'Represent the finance answer for retrieval: ',
            },
    },
    'hkunlp/instructor-base': {
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'HotpotQA':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'MSMARCO':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'DBPedia':
            {
                'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'NQ':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'QuoraRetrieval':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
            },
        'SCIDOCS':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'TRECCOVID':
            {
                'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
                'corpus': 'Represent the Coronavirus document for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent a question: ',
                'corpus': 'Represent an argument: ',
            },
        'SciFact':
            {
                'query': 'Represent a Scientific query for retrieving a supporting passage; ',
                'corpus': 'represent the Scientific passage for retrieval; ',
            },
        'NFCorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document: ',
                'corpus': 'Represent the medical document for retrieval: ',
            },
        'ArguAna':
            {
                'query': 'Represent the Debate argument for retrieving a counter-argument: ',
                'corpus': 'Represent the Counter debate argument: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers: ',
                'corpus': 'Represent the Unix answer for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'FiQA2018':
            {
                'query': 'Represent the finance question for retrieving the supporting answers: ',
                'corpus': 'Represent the finance answer for retrieval: ',
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
        self.count = 0

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
        print('encode queries')
        if self.args.prompt and isinstance(DEFINITIONS[self.args.prompt][self.args.task_name],str):
            instruction = DEFINITIONS[self.args.prompt][self.args.task_name]
        elif self.args.prompt:
            instruction = DEFINITIONS[self.args.prompt][self.args.task_name]['query']
        if self.args.prompt:
            for s in queries:
                new_sentences.append([instruction, s, 0])
        else:
            new_sentences = queries

        kwargs['show_progress_bar'] = False
        return self.model.encode(new_sentences, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        self.count += 1
        # print('count: ',self.count)
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
        # kwargs['show_progress_bar'] = False
        return self.model.encode(sentences, batch_size=128, **kwargs)

    def encode_corpus_parallel(
        self, corpus: List[Dict[str, str]], pool: Dict[str, object], batch_size: int, chunk_id: int, **kwargs
    ):
        instruction = DEFINITIONS[self.args.prompt][self.args.task_name]['corpus']
        if type(corpus) is dict:
            sentences = [
                [instruction, (corpus["title"][i] + self.sep + corpus["text"][i]).strip()]
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                [instruction, (doc["title"] + self.sep + doc["text"]).strip()]
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]

        if chunk_id is not None and chunk_id >= len(pool["processes"]):
            output_queue = pool["output"]
            output_queue.get()

        input_queue = pool["input"]
        input_queue.put([chunk_id, batch_size, sentences])