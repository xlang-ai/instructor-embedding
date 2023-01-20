import logging

import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from scipy.stats import pearsonr, spearmanr


logger = logging.getLogger(__name__)

from .Evaluator import Evaluator

DEFINITIONS = {
    'hkunlp/instructor-base': {
        'STS12': 'Represent the statement, ',
        'STS13': 'represent the statement, ',
        'STS14': 'Represent the statement, ',
        'STS15': 'Represent the post, ',
        'STS16': 'Represent the post: ',
        'STS17': 'Represent the sentence for classification: ',
        'STS22': 'represent the statement: ',
        'BIOSSES': 'Represent the Bio-medical statement: ',
        'SICK-R': 'Represent the statement: ',
        'STSBenchmark': 'represent the statement: ',
    },
    'hkunlp/instructor-large': {
        'STS12': 'Represent the statement: ',
        'STS13': 'represent the statement: ',
        'STS14': 'Represent the statement: ',
        'STS15': 'Represent the statement: ',
        'STS16': 'Represent the statement: ',
        'STS17': 'Represent the sentence for classification: ',
        'STS22': 'represent the statement: ',
        'BIOSSES': 'Represent the Bio-medical statement: ',
        'SICK-R': 'Represent the statement: ',
        'STSBenchmark': 'represent the statement: ',
    },
    'hkunlp/instructor-xl': {
        'STS12': 'represent texts, ',
        'STS13': 'represent a casual post, ',
        'STS14': 'Represent a post; ',
        'STS15': 'Represent a posts,,, ',
        # 'STS15': 'Represent a post for classification, ',
        'STS16': 'Represent posts: ',
        'STS17': 'Represent a statement, ',
        'STS22': 'represent the statement: ',
        'BIOSSES': 'represent the Biological statement: ',
        'SICK-R': 'Represent a post: ',
        'STSBenchmark': 'represent posts, ',
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
            print('with prompt: ', DEFINITIONS[self.args.prompt][self.args.task_name])
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
