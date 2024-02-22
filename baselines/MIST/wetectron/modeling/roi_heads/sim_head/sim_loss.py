from torch.nn import functional as F
from wetectron.utils.utils import cos_sim
import torch.nn as nn
import torch
from itertools import combinations

class Supcon_Loss(nn.Module):
    def __init__(self, temp):
        super(Supcon_Loss, self).__init__()
        self.temperature = temp
    def forward(self, overlaps_enc, device):
        features = torch.zeros((0), dtype=torch.float, device=device)
        labels = torch.zeros((0), dtype=torch.float, device=device)
        #score_weight = score_col.detach()
        #mask = torch.eq(labels.view(-1,1), labels.view(-1,1).T).float().cuda()
        #import IPython; IPython.embed()
        ### custom supervision from overlaps_encoding ###
        for i, embedding_v in enumerate(overlaps_enc):
            if embedding_v.shape[0] != 0:
                features = torch.cat((features, embedding_v))
                labels = torch.cat((labels, torch.ones(embedding_v.shape[0]).to(device)*(i+1)))
        ### same as supcon loss ###

        similarity = torch.div(torch.mm(features, features.T), self.temperature)
        #similarity = torch.mul(torch.mm(features, features.T), score_weight)                  ### temp_score

        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        label_mask = torch.eq(labels.view(-1,1), labels.view(-1,1).T).float().cuda()
        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)


        loss = - per_label_log_prob
        #loss = - per_label_log_prob * score_weight                                             ### contral weight
        return loss.mean()

class SupConLossV2(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, overlaps_enc, score_col, device):
        features = torch.zeros((0), dtype=torch.float, device=device)
        labels = torch.zeros((0), dtype=torch.float, device=device)
        score_weight = score_col.detach()
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label

        for i, embedding_v in enumerate(overlaps_enc):
            if embedding_v.shape[0] != 0:
                features = torch.cat((features, embedding_v))
                labels = torch.cat((labels, torch.ones(embedding_v.shape[0]).to(device)*(i)))

        similarity = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)


        exp_sim = torch.exp(similarity)
        label_mask = torch.eq(labels.view(-1,1), labels.view(-1,1).T).float().cuda()

        mask = logits_mask * label_mask

        log_prob = torch.log( (exp_sim * mask).sum(1) / (exp_sim * logits_mask).sum(1) )

        loss = -log_prob * score_weight

        return loss.mean()

