import logging

import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from scipy.stats import pearsonr, spearmanr


logger = logging.getLogger(__name__)

from .Evaluator import Evaluator

DEFINITIONS = {
    'hku-nlp/instructor-large': {
        'STS12': 'Represent a statement: ',
        'STS13': 'represent the sentence: ',
        'STS14': 'represent the sentence: ',
        'STS15': 'represent the sentence: ',
        'STS16': 'Represent a sentence for classification: ',
        'STS17': 'represent a sentence: ',
        'STS22': 'represent the sentence for classification: ',
        'BIOSSES': 'Represent the bio-medical statement: ',
        'SICK-R': 'Represent a sentence: ',
        'STSBenchmark': 'represent the statement:\n',
    },
    'hku-nlp/instructor-xl': {
        'STS12': 'Represent the sentence for retrieving duplicate sentences; Input: ',
        'STS13': 'Represent a sentence for retrieving duplicate sentence; Input: ',
        'STS14': 'Represent a sentence for retrieving duplicate sentence; Input: ',
        'STS15': 'Represent a sentence for retrieving duplicate sentence; Input: ',
        'STS16': 'Represent a sentence for retrieving duplicate sentences; Input: ',
        'STS17': 'represent a sentence: ',
        'STS22': 'Represent a sentence for retrieving duplicate sentence; Input: ',
        'BIOSSES': 'Represent a Bio-medicine sentence for retrieving duplicate sentences; Input: ',
        'SICK-R': 'Represent a sentence for retrieving duplicate sentence; Input: ',
        'STSBenchmark': 'Represent a sentence for retrieving duplicate sentences; Input: ',
    }
}

class STSEvaluator(Evaluator):
    def __init__(self, sentences1, sentences2, gold_scores, batch_size=64, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences1 = sentences1[:limit]
            sentences2 = sentences2[:limit]
            gold_scores = gold_scores[:limit]
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.gold_scores = gold_scores
        self.batch_size = batch_size
        self.args = kwargs['args']

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences1)} sentences1...")

        new_sentences = []
        if self.args.prompt:
            print('with prompt')
            for s in self.sentences1:
                new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
            self.sentences1 = new_sentences

        embeddings1 = np.asarray(model.encode(self.sentences1, batch_size=self.batch_size))

        logger.info(f"Encoding {len(self.sentences2)} sentences2...")

        new_sentences = []
        if self.args.prompt:
            print('with prompt')
            for s in self.sentences2:
                new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
            self.sentences2 = new_sentences

        embeddings2 = np.asarray(model.encode(self.sentences2, batch_size=self.batch_size))

        logger.info("Evaluating...")
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        # manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        # euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        # cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        cosine_spearman, _ = spearmanr(self.gold_scores, cosine_scores)

        # manhatten_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        # manhatten_spearman, _ = spearmanr(self.gold_scores, manhattan_distances)
        #
        # euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)
        # euclidean_spearman, _ = spearmanr(self.gold_scores, euclidean_distances)

        return {
            "cos_sim": {
                # "pearson": cosine_pearson,
                "spearman": cosine_spearman,
            },
            # "manhattan": {
            #     "pearson": manhatten_pearson,
            #     "spearman": manhatten_spearman,
            # },
            # "euclidean": {
            #     "pearson": euclidean_pearson,
            #     "spearman": euclidean_spearman,
            # },
        }
