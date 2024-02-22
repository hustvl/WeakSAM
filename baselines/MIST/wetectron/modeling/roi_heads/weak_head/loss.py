# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
import collections
import torch.nn as nn
import random
import os
import numpy as np
from torch.nn import functional as F

from wetectron.layers import smooth_l1_loss
from wetectron.modeling import registry
from wetectron.modeling.utils import cat
from wetectron.config import cfg
from wetectron.structures.boxlist_ops import boxlist_iou, boxlist_iou_async, boxlist_nms_index
from wetectron.structures.bounding_box import BoxList
from wetectron.modeling.matcher import Matcher
from wetectron.utils.utils import to_boxlist, cal_iou, easy_nms, cos_sim, get_share_class, generate_img_label
from .pseudo_label_generator import oicr_layer, mist_layer, od_layer
from wetectron.modeling.roi_heads.sim_head.sim_loss import Supcon_Loss, SupConLossV2
from wetectron.modeling.roi_heads.sim_head.sim_net import Sim_Net

def compute_avg_img_accuracy(labels_per_im, score_per_im, num_classes):
    """
       the accuracy of top-k prediction
       where the k is the number of gt classes
    """
    num_pos_cls = max(labels_per_im.sum().int().item(), 1)
    cls_preds = score_per_im.topk(num_pos_cls)[1]
    accuracy_img = labels_per_im[cls_preds].mean()
    return accuracy_img


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

@registry.ROI_WEAK_LOSS.register("WSDDNLoss")
class WSDDNLossComputation(object):
    """ Computes the loss for WSDDN."""
    def __init__(self, cfg):
        self.type = "WSDDN"

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-10):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])
        Returns:
            img_loss (Tensor)
            accuracy_img (Tensor): the accuracy of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        total_loss = 0
        accuracy_img = 0
        for final_score_per_im, targets_per_im in zip(final_score_list, targets):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            total_loss += F.binary_cross_entropy(img_score_per_im, labels_per_im)
            accuracy_img += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)

        total_loss = total_loss / len(final_score_list)
        accuracy_img = accuracy_img / len(final_score_list)
        return dict(loss_img=total_loss), dict(accuracy_img=accuracy_img)


@registry.ROI_WEAK_LOSS.register("RoILoss")
class RoILossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        self.type = "RoI_loss"
        if refine_p == 0:
            self.roi_layer = oicr_layer()
        elif refine_p > 0 and refine_p < 1:
            self.roi_layer = mist_layer(refine_p)
        else:
            raise ValueError('please use propoer ratio P.')

    def __call__(self, class_score, det_score, ref_scores, proposals, targets, epsilon=1e-8):
        """
        Arguments:
            class_score (list[Tensor])
            det_score (list[Tensor])
            ref_scores
            proposals
            targets
        Returns:
            return_loss_dict (dictionary): all the losses
            return_acc_dict (dictionary): all the accuracies of image-level classification
        """
        class_score = cat(class_score, dim=0)
        class_score = F.softmax(class_score, dim=1)

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)

        device = class_score.device
        num_classes = class_score.shape[1]

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])
        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)
        for i in range(num_refs):
            return_loss_dict['loss_ref%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            # Region loss
            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                lmda = 3 if i == 0 else 1
                pseudo_labels, loss_weights = self.roi_layer(proposals_per_image, source_score, labels_per_im, device)
                return_loss_dict['loss_ref%d'%i] += lmda * torch.mean(F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights)

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0
        for l, a in zip(return_loss_dict.keys(), return_acc_dict.keys()):
            return_loss_dict[l] /= len(final_score_list)
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict


@registry.ROI_WEAK_LOSS.register("RoIRegLoss")
class RoIRegLossComputation(object):
    """ Generic roi-level loss """
    def __init__(self, cfg):
        self.refine_p = cfg.MODEL.ROI_WEAK_HEAD.OICR_P
        self.contra = cfg.SOLVER.CONTRA

        if self.refine_p > 0 and self.refine_p < 1 and not self.contra:
            self.mist_layer = mist_layer(self.refine_p)
        self.oicr_layer = oicr_layer()
        self.od_layer = od_layer()

        # for regression
        self.cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        # for partial labels
        self.roi_refine = cfg.MODEL.ROI_WEAK_HEAD.ROI_LOSS_REFINE
        self.partial_label = cfg.MODEL.ROI_WEAK_HEAD.PARTIAL_LABELS
        assert self.partial_label in ['none', 'point', 'scribble']
        self.proposal_scribble_matcher = Matcher(
            0.5, 0.5, allow_low_quality_matches=False,
        )

        self.nms = cfg.nms
        self.sim_lmda = cfg.lmda
        self.pos_update = cfg.pos_update
        self.p_thres = cfg.thres
        self.p_iou = cfg.iou
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.temp = cfg.temp
        if cfg.loss == 'supcon':
            self.sim_loss = Supcon_Loss(self.temp)
        elif cfg.loss == 'supconv2':
            self.sim_loss = SupConLossV2(self.temp)
        self.output_dir = cfg.OUTPUT_DIR

    def filter_pseudo_labels(self, pseudo_labels, proposal, target):
        """ refine pseudo labels according to partial labels """
        if 'scribble' in target.fields() and self.partial_label=='scribble':
            scribble = target.get_field('scribble')
            match_quality_matrix_async = boxlist_iou_async(scribble, proposal)
            _matched_idxs = self.proposal_scribble_matcher(match_quality_matrix_async)
            pseudo_labels[_matched_idxs < 0] = 0
            matched_idxs = _matched_idxs.clone().clamp(0)
            _labels = target.get_field('labels')[matched_idxs]
            pseudo_labels[pseudo_labels != _labels.long()] = 0

        elif 'click' in target.fields() and self.partial_label=='point':
            clicks = target.get_field('click').keypoints
            clicks_tiled = torch.unsqueeze(torch.cat((clicks, clicks), dim=1), dim=1)
            num_obj = clicks.shape[0]
            box_repeat = torch.cat([proposal.bbox.unsqueeze(0) for _ in range(num_obj)], dim=0)
            diff = clicks_tiled - box_repeat
            matched_ids = (diff[:,:,0] > 0) * (diff[:,:,1] > 0) * (diff[:,:,2] < 0) * (diff[:,:,3] < 0)
            matched_cls = matched_ids.float() * target.get_field('labels').view(-1, 1)
            pseudo_labels_repeat = torch.cat([pseudo_labels.unsqueeze(0) for _ in range(matched_ids.shape[0])])
            correct_idx = (matched_cls == pseudo_labels_repeat.float()).sum(0)
            pseudo_labels[correct_idx==0] = 0

        return pseudo_labels

    def __call__(self, class_score, det_score, ref_scores, ref_bbox_preds, proposals, targets, sim_feature = None, clean_pooled_feats = None, feature_extractor = None, model_sim = None, epsilon=1e-8):
        # print(class_score)
        class_score = F.softmax(cat(class_score, dim=0), dim=1)
        class_score_list = class_score.split([len(p) for p in proposals])

        det_score = cat(det_score, dim=0)
        det_score_list = det_score.split([len(p) for p in proposals])
        final_det_score = []
        for det_score_per_image in det_score_list:
            det_score_per_image = F.softmax(det_score_per_image, dim=0)
            final_det_score.append(det_score_per_image)
        final_det_score = cat(final_det_score, dim=0)
        detection_score_list = final_det_score.split([len(p) for p in proposals])

        final_score = class_score * final_det_score
        final_score_list = final_score.split([len(p) for p in proposals])

        device = class_score.device
        num_classes = class_score.shape[1]

        ref_score = ref_scores.copy()
        for r, r_score in enumerate(ref_scores):
            ref_score[r] = F.softmax(r_score, dim=1)
        avg_score = torch.stack(ref_score).mean(0).detach()
        avg_score_split = avg_score.split([len(p) for p in proposals])

        ref_scores = [rs.split([len(p) for p in proposals]) for rs in ref_scores]
        ref_bbox_preds = [rbp.split([len(p) for p in proposals]) for rbp in ref_bbox_preds]

        return_loss_dict = dict(loss_img=0)
        return_acc_dict = dict(acc_img=0)
        num_refs = len(ref_scores)

        for i in range(num_refs):
            return_loss_dict['loss_ref_cls%d'%i] = 0
            return_loss_dict['loss_ref_reg%d'%i] = 0
            return_acc_dict['acc_ref%d'%i] = 0

        pos_classes = [generate_img_label(num_classes, target.get_field('labels').unique(), device)[1:].eq(1).nonzero(as_tuple=False)[:,0] for target in targets]
        if self.contra:
            return_loss_dict['loss_sim'] = 0
            sim_feature = sim_feature.split([len(p) for p in proposals])
            clean_pooled_feat = clean_pooled_feats.split([len(p) for p in proposals])

            pgt_index = [[torch.zeros((0), dtype=torch.long, device=device) for x in range(num_classes-1)] for y in range(len(targets))]
            pgt_collection = [torch.zeros((0), dtype=torch.float, device=device) for x in range(num_classes-1)]
            pgt_update = [torch.zeros((0), dtype=torch.float, device=device) for x in range(num_classes-1)]
            instance_diff = torch.zeros((0), dtype=torch.float, device=device)

            for idx, (final_score_per_im, pos_classes_per_im, proposals_per_image) in enumerate(zip(final_score_list, pos_classes, proposals)):
                for i in range(num_refs):
                    source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                    proposal_score = source_score[:, 1:].clone()
                    for pos_c in pos_classes_per_im:
                        max_index = torch.argmax(proposal_score[:,pos_c])
                        overlaps, _ = cal_iou(proposals_per_image, max_index, self.p_thres)
                        pgt_index[idx][pos_c] = torch.cat((pgt_index[idx][pos_c], overlaps)).unique() ###

                for pos_c in pos_classes_per_im:
                    iou_samples = pgt_index[idx][pos_c]
                    pgt_update[pos_c] = torch.cat((pgt_update[pos_c], sim_feature[idx][iou_samples])) ### iou sampling

                    hardness = final_score_list[idx][iou_samples, pos_c+1] / final_score_list[idx][:,pos_c+1].sum()
                    #hardness = avg_score_split[idx][iou_samples, pos_c+1] / avg_score_split[idx][:, pos_c+1].sum()

                    instance_diff = torch.cat((instance_diff, hardness))
                    drop_logit = feature_extractor.forward_neck(feature_extractor.drop_pool(clean_pooled_feat[idx][iou_samples]))
                    pgt_update[pos_c] = torch.cat((pgt_update[pos_c], model_sim(drop_logit) ))

                    instance_diff = torch.cat((instance_diff, hardness))

                    noise_logit = feature_extractor.forward_neck(feature_extractor.noise_pool(clean_pooled_feat[idx][iou_samples]))
                    pgt_update[pos_c] = torch.cat((pgt_update[pos_c], model_sim(noise_logit) ))
                    instance_diff = torch.cat((instance_diff, hardness))

                    pgt_collection[pos_c] = pgt_update[pos_c].clone()

            pgt_instance = [[[torch.zeros((0), dtype=torch.long, device=device) for x in range(num_classes-1)] for z in range(num_refs)] for y in range(len(targets))]

            for idx, (final_score_per_im, pos_classes_per_im, proposals_per_image) in enumerate(zip(final_score_list, pos_classes, proposals)):
                for i in range(num_refs):
                    source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                    proposal_score = source_score[:, 1:].clone()

                    for pos_c in pos_classes_per_im:
                        max_index = torch.argmax(proposal_score[:,pos_c])

                        sim_mat = torch.mm(sim_feature[idx], sim_feature[idx].T)
                        sim_thresh = torch.mm(sim_feature[idx][max_index].view(1,-1), pgt_collection[pos_c].T).mean()

                        if pos_classes_per_im.shape[0] > 1:
                            neg_classes = pos_classes_per_im[(pos_classes_per_im != pos_c)]
                            sim_close = torch.ge(sim_mat[max_index], sim_thresh)
                            for neg_c in neg_classes:
                                neg_max_index = torch.argmax(proposal_score[:,neg_c])
                                sim_close = torch.ge(sim_close, sim_mat[neg_max_index])
                            sim_close = sim_close.nonzero(as_tuple=False).view(-1)
                        else:
                            sim_close = torch.ge(sim_mat[max_index], sim_thresh).nonzero(as_tuple=False).view(-1)

                        sim_close = easy_nms(proposals_per_image, sim_close, proposal_score[:,pos_c], nms_iou=self.nms)      ### operate nms
                        sim_close = torch.cat((sim_close, max_index.view(-1))) if sim_close.nelement() == 0 else sim_close   ### avoid none
                        pgt_instance[idx][i][pos_c] = torch.cat((pgt_instance[idx][i][pos_c], sim_close))

                        dup = torch.cat((sim_close, pgt_index[idx][pos_c])).unique()[torch.where(torch.cat((sim_close, pgt_index[idx][pos_c])).unique(return_counts=True)[1]>1)]
                        sim_close = torch.cat((sim_close,dup)).unique()[torch.where(torch.cat((sim_close,dup)).unique(return_counts=True)[1]==1)]
                        sim_close = torch.cat((sim_close, max_index.view(-1))) if sim_close.nelement() == 0 else sim_close

                        pgt_update[pos_c] = torch.cat((pgt_update[pos_c], sim_feature[idx][sim_close]))
                        pgt_index[idx][pos_c] = torch.cat((pgt_index[idx][pos_c], sim_close)).unique()

                        sim_hardness = final_score_list[idx][sim_close, pos_c+1] / final_score_list[idx][:,pos_c+1].sum()
                        #sim_hardness = avg_score_split[idx][sim_close, pos_c+1] / avg_score_split[idx][:, pos_c+1].sum()
                        instance_diff = torch.cat((instance_diff, sim_hardness.view(-1)))

            return_loss_dict['loss_sim'] = self.sim_lmda * self.sim_loss(pgt_update, instance_diff, device)

        for idx, (final_score_per_im, targets_per_im, proposals_per_image) in enumerate(zip(final_score_list, targets, proposals)):
            labels_per_im = targets_per_im.get_field('labels').unique()
            # print(targets)
            # print(class_score)
            labels_per_im = generate_img_label(class_score.shape[1], labels_per_im, device)
            # MIL loss
            img_score_per_im = torch.clamp(torch.sum(final_score_per_im, dim=0), min=epsilon, max=1-epsilon)
            return_loss_dict['loss_img'] += F.binary_cross_entropy(img_score_per_im, labels_per_im.clamp(0, 1))
            # Region loss
            for i in range(num_refs):
                source_score = final_score_per_im if i == 0 else F.softmax(ref_scores[i-1][idx], dim=1)
                if not self.contra and self.refine_p == 0:           ### oicr_layer ###
                    pseudo_labels, loss_weights, regression_targets = self.oicr_layer(
                        proposals_per_image, source_score, labels_per_im, device, return_targets=True
                        )
                elif not self.contra and self.refine_p > 0:          ### mist layer ###
                    pseudo_labels, loss_weights, regression_targets = self.mist_layer(
                        proposals_per_image, source_score, labels_per_im, device, return_targets=True
                        )
                elif self.contra and self.refine_p == 0:                ### od layer ###
                    pseudo_labels, loss_weights, regression_targets = self.od_layer(
                    proposals_per_image, source_score, labels_per_im, device, pgt_instance[idx][i], return_targets=True
                    )
                if self.roi_refine:
                    pseudo_labels = self.filter_pseudo_labels(pseudo_labels, proposals_per_image, targets_per_im)

                lmda = 3 if i == 0 else 1

                return_loss_dict['loss_ref_cls%d'%i] += lmda * torch.mean(
                    F.cross_entropy(ref_scores[i][idx], pseudo_labels, reduction='none') * loss_weights
                )

                # regression
                sampled_pos_inds_subset = torch.nonzero(pseudo_labels>0, as_tuple=False).squeeze(1)
                labels_pos = pseudo_labels[sampled_pos_inds_subset]
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([4, 5, 6, 7], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

                box_regression = ref_bbox_preds[i][idx]
                reg_loss = lmda * torch.sum(smooth_l1_loss(
                    box_regression[sampled_pos_inds_subset[:, None], map_inds],
                    regression_targets[sampled_pos_inds_subset],
                    beta=1, reduction=False) * loss_weights[sampled_pos_inds_subset, None]
                )
                reg_loss /= pseudo_labels.numel()
                return_loss_dict['loss_ref_reg%d'%i] += reg_loss

            with torch.no_grad():
                return_acc_dict['acc_img'] += compute_avg_img_accuracy(labels_per_im, img_score_per_im, num_classes)
                for i in range(num_refs):
                    ref_score_per_im = torch.sum(ref_scores[i][idx], dim=0)
                    return_acc_dict['acc_ref%d'%i] += compute_avg_img_accuracy(labels_per_im[1:], ref_score_per_im[1:], num_classes)

        assert len(final_score_list) != 0
        for l in return_loss_dict.keys():
            if 'sim' in l:
                continue
            return_loss_dict[l] /= len(final_score_list)

        for a in return_acc_dict.keys():
            return_acc_dict[a] /= len(final_score_list)

        return return_loss_dict, return_acc_dict


def make_roi_weak_loss_evaluator(cfg):
    func = registry.ROI_WEAK_LOSS[cfg.MODEL.ROI_WEAK_HEAD.LOSS]
    return func(cfg)
