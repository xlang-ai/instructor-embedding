# from .. import util
import torch
import os
from torch import nn, Tensor
from typing import Iterable, Dict

class StandardCLLoss(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|.
    By default, sim() is the dot-product.
    For more details, please refer to https://arxiv.org/abs/2010.02666.
    """
    def __init__(self, model,args=None):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use.
        """
        super(StandardCLLoss, self).__init__()
        self.model = model
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        self.count = 0
        self.args = args

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        num = len(embeddings_query)
        all_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_query[i].unsqueeze(0)
            pos_emb = embeddings_pos[i].unsqueeze(0)
            cur_score = self.similarity_fct(anchor_emb, pos_emb) / self.args.cl_temperature

            one_neg_emb = embeddings_neg[i].unsqueeze(0)
            one_neg_score = self.similarity_fct(anchor_emb, one_neg_emb) / self.args.cl_temperature
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            # for j in range(0,num):
            #     other_pos_emb = embeddings_pos[j].unsqueeze(0)
            #     other_pos_score = self.similarity_fct(anchor_emb, other_pos_emb) / self.args.cl_temperature
            #     cur_score = torch.cat([other_pos_score,cur_score], dim=-1)
            if all_scores is None:
                all_scores = cur_score.unsqueeze(0)
            else:
                all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

        labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
        loss_v = self.loss_fct(all_scores, labels)
        with open(os.path.join(self.args.model_save_path, 'loss_log.txt'), 'a') as f:
            f.write(f'loss {self.count}: {loss_v}\n')

        self.count += 1
        return loss_v
