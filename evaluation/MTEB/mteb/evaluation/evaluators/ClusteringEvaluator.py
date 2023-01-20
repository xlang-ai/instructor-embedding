import logging

import numpy as np
import sklearn
import sklearn.cluster

from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

from .Evaluator import Evaluator

DEFINITIONS = {
    'hkunlp/instructor-xl': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for clustering; ',
        'BiorxivClusteringS2S': 'Represent the biological statement for retrieval; ',
        'MedrxivClusteringS2S': 'Represent the Biological statement for clustering biological statements: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the Science statements for retrieval: ',
        # 'ArxivClusteringS2S': 'Represent the Scientific statements for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Biological passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the Biological paragraph for retrieval: ',
        # 'MedrxivClusteringP2P': 'Represent the Biological document for retrieval: ',
        'RedditClustering': 'represent the Reddit community title: ',
        # 'RedditClustering': 'represent the Reddit community sentence: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent a question for retrieval: ',
        # 'StackExchangeClustering': 'Represent the questions for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer passage for retrieving relevant question and answer passages: ',
        # 'StackExchangeClusteringP2P': 'Represent the question and answer passage for retrieving relevant question and answer passages: ',
    },
    # 'hkunlp/instructor-xl original': {
    #     'TwentyNewsgroupsClustering': 'Represent the news comment for clustering; ',
    #     'BiorxivClusteringS2S': 'Represent the biological statement for retrieval; ',
    #     'MedrxivClusteringS2S': 'Represent the Medical statement for retrieving duplicate sentence: ',
    #     'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
    #     # 'ArxivClusteringS2S 32.05': 'Represent the science statements for retrieval: ',
    #     'ArxivClusteringS2S': 'Represent the Science statements for retrieval: ',
    #     'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
    #     'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
    #     'RedditClustering': 'represent the Reddit community title: ',
    #     'RedditClusteringP2P': 'represent a Reddit community passage: ',
    #     'StackExchangeClustering': 'Represent a question for retrieval: ',
    #     'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
    # },
    'hkunlp/instructor-large': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieving duplicate sentences: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
        'RedditClustering': 'represent a reddit community title: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent the question for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
    },
    'hkunlp/instructor-base': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieving duplicate sentences: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
        'RedditClustering': 'represent a reddit community title: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent the question for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
    },
}

class ClusteringEvaluator(Evaluator):
    def __init__(self, sentences, labels, clustering_batch_size=500, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.args = kwargs['args']
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        new_sentences = []
        if self.args.prompt:
            print('with prompt')
            for s in self.sentences:
                if len(self.tokenizer(DEFINITIONS[self.args.prompt][self.args.task_name]+s)['input_ids']) <= 256:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                else:
                    new_sentences.append(['', s, 0])
        else:
            new_sentences = self.sentences
        corpus_embeddings = np.asarray(model.encode(new_sentences))
        # mean_emb = np.mean(corpus_embeddings,axis=0)
        # corpus_embeddings -= mean_emb

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)), batch_size=self.clustering_batch_size
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}
