import logging

import numpy as np
import sklearn
import sklearn.cluster

from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

from .Evaluator import Evaluator

DEFINITIONS = {
    'hku-nlp/instructor-xl': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval; Input: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval; Input: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieval; Input: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval; Input: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval; Input: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval; Input: ',
        'MedrxivClusteringP2P': 'Represent the medical claim for retrieval; Input: ',
        'RedditClustering': 'represent a reddit community title: Input, ',
        'RedditClusteringP2P': 'represent a Reddit passage:\n',
        'StackExchangeClustering': 'Represent the question for retrieval; Input: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers; Input: ',
    },
    'hku-nlp/instructor-large': {
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval; Input: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval; Input: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieval; Input: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval; Input: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval; Input: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval; Input: ',
        'MedrxivClusteringP2P': 'Represent the medical claim for retrieval; Input: ',
        'RedditClustering': 'represent a reddit community title: Input, ',
        'RedditClusteringP2P': 'represent a Reddit passage:\n',
        'StackExchangeClustering': 'Represent the question for retrieval; Input: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers; Input: ',
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
