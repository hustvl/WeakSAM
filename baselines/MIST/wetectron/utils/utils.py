import torch
import collections
import torch.nn as nn
import random
import numpy as np
from torch.nn import functional as F
from itertools import combinations
from wetectron.layers import smooth_l1_loss
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async, boxlist_nms_index, boxlist_nms
from wetectron.structures.bounding_box import BoxList
from wetectron.modeling.matcher import Matcher
from wetectron.data.datasets.evaluation.voc.voc_eval import calc_detection_voc_prec_rec

@torch.no_grad()
def to_boxlist(proposal, index):
    boxlist = BoxList(proposal.bbox[index], proposal.size, proposal.mode)
    return boxlist

@torch.no_grad()
def cal_iou(proposal, target_index, iou_thres=1e-5):
    iou_index = torch.nonzero(torch.ge(boxlist_iou(proposal, to_boxlist(proposal, target_index.view(-1))), iou_thres).max(dim=1)[0]).view(-1)
    iou_score = boxlist_iou(proposal, to_boxlist(proposal, target_index.view(-1)))[iou_index]
    return iou_index, iou_score

@torch.no_grad()
def easy_nms(proposals, cluster, source_score, nms_iou=0.1):
    cluster_box = BoxList(proposals.bbox[cluster], proposals.size, proposals.mode)
    cluster_box.add_field('scores', source_score[cluster])
    cluster_nms = cluster[boxlist_nms_index(cluster_box, nms_iou)[1]]
    return cluster_nms

@torch.no_grad()
def cos_sim(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t())#.clamp(min=eps)

@torch.no_grad()
def get_share_class(targets):
    b1_labels = targets[0].get_field('labels').unique().type(torch.long)
    b2_labels = targets[1].get_field('labels').unique().type(torch.long)
    target_cat = torch.cat((b1_labels, b2_labels))
    duplicate = target_cat.unique()[np.random.choice(torch.where(target_cat.unique(return_counts=True)[1] > 1)[0].detach().cpu().numpy())]
    b1_neg = b1_labels[b1_labels != duplicate]
    b2_neg = b2_labels[b2_labels != duplicate]
    return b1_neg, b2_neg, duplicate

@torch.no_grad()
def generate_img_label(num_classes, labels, device):
    img_label = torch.zeros(num_classes)
    img_label[labels.long()] = 1
    img_label[0] = 0
    return img_label.to(device)

def temp_softmax(logits, dim=0, T=1):
    m = torch.max(logits, dim, keepdim=True)[0]
    logits = logits/T
    x_exp = torch.exp(logits-m)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return x_exp/x_exp_sum

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

@torch.no_grad()
def cal_precision_recall(proposals_per_image, max_index, sim_close, pos_c, target, iter_dict, sim_thresh):

    for iou_threshold in np.arange(0.25, 1.0, 0.25):
        pgt_base = BoxList(proposals_per_image.bbox[max_index.view(-1)], proposals_per_image.size, mode=proposals_per_image.mode)
        pgt_base.add_field('labels', (pos_c+1).expand(len(pgt_base)).detach().cpu().expand(len(pgt_base)))
        pgt_base.add_field('scores', torch.Tensor([1.]).expand(len(pgt_base)))
        #prec_base, rec_base = calc_detection_voc_prec_rec([target], [pgt_base], iou_threshold)

        tp_base = torch.nonzero(boxlist_iou(target, pgt_base) > iou_threshold).shape[0]
        fn_base = (torch.cat((torch.arange(len(target)).to("cuda"),torch.nonzero(boxlist_iou(target, pgt_base) > iou_threshold)[:,0].unique())).unique(return_counts=True)[1] == 1).nonzero(as_tuple=False).shape[0]
        prec_base2 = tp_base / len(pgt_base)
        rec_base2 = tp_base / (tp_base + fn_base)#len(target)

        pgt_ours = BoxList(proposals_per_image.bbox[torch.cat((sim_close, max_index.view(-1)))], proposals_per_image.size, mode=proposals_per_image.mode) ### torch.cat((sim_close, max_index.view(-1))) add top-scoring proposal to calculate our pgts
        pgt_ours.add_field('labels', (pos_c+1).expand(len(pgt_ours)).detach().cpu().expand(len(pgt_ours)))
        pgt_ours.add_field('scores', torch.Tensor([1.]).expand(len(pgt_ours)))
        #prec_ours, rec_ours = calc_detection_voc_prec_rec([target], [pgt_ours], iou_threshold)

        tp_ours = torch.nonzero(boxlist_iou(target, pgt_ours) > iou_threshold).shape[0]
        fn_ours = (torch.cat((torch.arange(len(target)).to("cuda"),torch.nonzero(boxlist_iou(target, pgt_ours) > iou_threshold)[:,0].unique())).unique(return_counts=True)[1] == 1).nonzero(as_tuple=False).shape[0]
        prec_ours2 = tp_ours / len(pgt_ours)
        rec_ours2 = tp_ours / (tp_ours + fn_ours)#len(target)

        #prec_ours2 = torch.nonzero(boxlist_iou(target, pgt_ours) > iou_threshold).shape[0] / len(pgt_ours)
        #rec_ours2 = torch.nonzero(boxlist_iou(target, pgt_ours) > iou_threshold).shape[0] / len(target)

        #iter_dict['prec_base_%.2f'%iou_threshold][pos_c] += prec_base2[pos_c+1][-1]
        #iter_dict['rec_base_%.2f'%iou_threshold][pos_c] += rec_base2[pos_c+1][-1]
        #iter_dict['prec_ours_%.2f'%iou_threshold][pos_c] += prec_ours2[pos_c+1][-1]
        #iter_dict['rec_ours_%.2f'%iou_threshold][pos_c] += rec_ours2[pos_c+1][-1]

        iter_dict['prec_base_%.2f'%iou_threshold][pos_c] += prec_base2
        iter_dict['rec_base_%.2f'%iou_threshold][pos_c] += rec_base2
        iter_dict['prec_ours_%.2f'%iou_threshold][pos_c] += prec_ours2
        iter_dict['rec_ours_%.2f'%iou_threshold][pos_c] += rec_ours2


    if type(sim_thresh) == float:
        iter_dict['update_threshold'] += sim_thresh
        iter_dict['tau_c'][pos_c] += sim_thresh
    else:
        iter_dict['update_threshold'] += round(sim_thresh.item(),4)
        iter_dict['tau_c'][pos_c] += round(sim_thresh.item(),4)
    iter_dict['update_c_instance'][pos_c] += sim_close.shape[0]
    iter_dict['update_instance'] += sim_close.shape[0]
    iter_dict['no_of_index'][pos_c] += 1
    return iter_dict
