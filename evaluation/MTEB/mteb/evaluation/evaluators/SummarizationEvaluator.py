import logging

import numpy as np
import torch
from tqdm import trange

from scipy.stats import pearsonr, spearmanr

from .utils import cos_sim, dot_score


logger = logging.getLogger(__name__)

from .Evaluator import Evaluator

DEFINITIONS = {
    'hkunlp/instructor-xl': {
        'SummEval': 'Represent the news statement for retrieval: ',
    },
    'hkunlp/instructor-large': {
        'SummEval': 'Represent the news sentence for retrieval: ',
    },
    'hkunlp/instructor-base': {
        'SummEval': 'Represent the news sentence for retrieval: ',
    }

}

class SummarizationEvaluator(Evaluator):
    def __init__(
        self, human_summaries=None, machine_summaries=None, texts=None, gold_scores=None, limit=None, **kwargs
    ):
        # human_summaries shape: (None, num_human_summaries)
        # machine_summaries shape: (None, num_machine_summaries)
        # gold scores shape: (None, num_machine_summaries)
        # texts: (None,)
        super().__init__(**kwargs)
        if limit is not None:
            human_summaries = human_summaries[:limit]
            machine_summaries = machine_summaries[:limit]
            gold_scores = gold_scores[:limit]
            texts = texts[:limit]
        self.human_summaries = human_summaries
        self.machine_summaries = machine_summaries
        self.texts = texts
        self.gold_scores = gold_scores
        self.args = kwargs['args']

    def __call__(self, model):

        cosine_spearman_scores = []
        cosine_pearson_scores = []
        dot_spearman_scores = []
        dot_pearson_scores = []

        for i in trange(len(self.texts), desc="Texts"):  # iterate over all original texts
            human_summaries = self.human_summaries[i]  # Get the human summaries for the text

            new_sentences = []
            if self.args.prompt:
                # print('with prompt')
                for s in human_summaries:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
            else:
                new_sentences = human_summaries

            embs_human_summaries = model.encode(new_sentences)
            cosine_pred_scores = []  # Predicted quality score for a summary
            dot_pred_scores = []  # Predicted quality score for a summary
            human_scores = []  # Human score for a summary
            for machine_summary, human_eval_score in zip(
                self.machine_summaries[i], self.gold_scores[i]
            ):  # Get all machine summaries + scores for this text
                if self.args.prompt:
                    new_machine_summary = [DEFINITIONS[self.args.prompt][self.args.task_name], machine_summary, 0]
                else:
                    new_machine_summary = machine_summary
                emb_machine_summary = model.encode(
                    new_machine_summary, show_progress_bar=False
                )  # 1 embedding for the summary
                cosine_scores = cos_sim(emb_machine_summary, embs_human_summaries)
                dot_scores = dot_score(emb_machine_summary, embs_human_summaries)

                cosine_max_score = torch.max(cosine_scores).item()
                cosine_pred_scores.append(cosine_max_score)
                dot_max_score = torch.max(dot_scores).item()
                dot_pred_scores.append(dot_max_score)
                human_scores.append(human_eval_score)

            if (len(set(human_scores)) == 1) or (len(set(dot_pred_scores)) == 1) or (len(set(cosine_pred_scores)) == 1):
                logger.info(f"Skipping sample {i} due to equal scores")
                continue

            cosine_spearman_scores.append(spearmanr(human_scores, cosine_pred_scores))
            cosine_pearson_scores.append(pearsonr(human_scores, cosine_pred_scores))
            dot_spearman_scores.append(spearmanr(human_scores, dot_pred_scores))
            dot_pearson_scores.append(pearsonr(human_scores, dot_pred_scores))

        cosine_spearman = np.mean(cosine_spearman_scores)
        dot_spearman = np.mean(dot_spearman_scores)
        cosine_pearson = np.mean(cosine_pearson_scores)
        dot_pearson = np.mean(dot_pearson_scores)

        return {
            "cos_sim": {
                "spearman": cosine_spearman,
                "pearson": cosine_pearson,
            },
            "dot": {
                "spearman": dot_spearman,
                "pearson": dot_pearson,
            },
        }
