import torch
import torch.nn as nn
import numpy as np
import os
import time

from wetectron.config import cfg
from wetectron.structures.bounding_box import BoxList, BatchBoxList
from wetectron.structures.boxlist_ops import boxlist_iou, batch_boxlist_iou
from wetectron.modeling.box_coder import BoxCoder
from wetectron.utils.utils import to_boxlist, cal_iou
from wetectron.data.datasets.evaluation.voc.voc_eval import calc_detection_voc_prec_rec

class mist_layer(object):
    def __init__(self, p, iou=0.2):
        self.portion = p
        self.iou_th = iou
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)

    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, return_targets=False):
        num_rois = len(proposals)
        
        k = int(num_rois * self.portion)  # when num_rois = 4, portion = 0.15 the int of k being 0?
        if num_rois < 10:
            k = 1
        num_gt_cls = labels[1:].sum()
        if num_gt_cls != 0 and num_rois != 0:
            cls_prob = source_score[:, 1:]
            gt_cls_inds = labels[1:].nonzero(as_tuple=False)[:, 0]
            sorted_scores, max_inds = cls_prob[:, gt_cls_inds].sort(dim=0, descending=True)
            sorted_scores = sorted_scores[:k]
            max_inds = max_inds[:k]

            _boxes = proposals.bbox[max_inds.t().contiguous().view(-1)].view(num_gt_cls.int(), -1, 4)
            _boxes = BatchBoxList(_boxes, proposals.size, mode=proposals.mode)
            ious = batch_boxlist_iou(_boxes, _boxes)
            k_ind = torch.zeros(num_gt_cls.int(), k, dtype=torch.bool, device=device)
            if k_ind.shape[1] == 0:
                # k_ind = torch.zeros(num_gt_cls.int(), 1, dtype=torch.bool, device=device)
                print(num_rois, k)
                # print(k_ind)
                # print(k, num_rois)
            k_ind[:, 0] = 1 # always take the one with max score
            # bug occurrence
            
            for ii in range(1, k):
                max_iou, _ = torch.max(ious[:,ii:ii+1, :ii], dim=2)
                k_ind[:, ii] = (max_iou < self.iou_th).byte().squeeze(-1)

            gt_boxes = _boxes.bbox[k_ind]
            gt_cls_id = gt_cls_inds + 1
            temp_cls = torch.ones((_boxes.bbox.shape[:2]), device=device) * gt_cls_id.view(-1, 1).float()
            gt_classes = temp_cls[k_ind].view(-1, 1).long()
            gt_scores = sorted_scores.t().contiguous()[k_ind].view(-1, 1)

            if gt_boxes.shape[0] != 0:
                gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
                overlaps = boxlist_iou(proposals, gt_boxes)

                # TODO: pytorch and numpy argmax perform differently
                # max_overlaps, gt_assignment = overlaps.max(dim=1)
                max_overlaps  = torch.tensor(overlaps.cpu().numpy().max(axis=1), device=device)
                gt_assignment = torch.tensor(overlaps.cpu().numpy().argmax(axis=1), device=device)

                pseudo_labels = gt_classes[gt_assignment, 0]
                loss_weights = gt_scores[gt_assignment, 0]

                # fg_inds = max_overlaps.ge(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
                # Select background RoIs as those with <= FG_IOU_THRESHOLD
                bg_inds = max_overlaps.lt(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
                pseudo_labels[bg_inds] = 0

                # compute regression targets
                if return_targets:
                    matched_targets = gt_boxes[gt_assignment]
                    regression_targets = self.box_coder.encode(
                        matched_targets.bbox, proposals.bbox
                    )
                    return pseudo_labels, loss_weights, regression_targets

                return pseudo_labels, loss_weights

        # corner case
        pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
        loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        if return_targets:
            regression_targets = torch.zeros(num_rois, 4, dtype=torch.float, device=device)
            return pseudo_labels, loss_weights, regression_targets
        return pseudo_labels, loss_weights

class oicr_layer(object):
    """ OICR. Tang et al. 2017 (https://arxiv.org/abs/1704.00138) """
    def __init__(self):
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)
    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, return_targets=False):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=device)
        gt_classes = torch.zeros((0, 1), dtype=torch.long, device=device)
        gt_scores = torch.zeros((0, 1), dtype=torch.float, device=device)

        # not using the background class
        _prob = source_score[:, 1:].clone()
        _labels = labels[1:]
        positive_classes = _labels.eq(1).nonzero(as_tuple=False)[:, 0]
        for c in positive_classes:
            cls_prob = _prob[:, c]
            max_index = torch.argmax(cls_prob)
            gt_boxes = torch.cat((gt_boxes, proposals.bbox[max_index].view(1, -1)), dim=0)
            gt_classes = torch.cat((gt_classes, c.add(1).view(1, 1)), dim=0)
            gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1, 1)), dim=0)
            _prob[max_index].fill_(0)

        if gt_boxes.shape[0]  == 0:
            num_rois = len(source_score)
            pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
            loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        else:
            gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
            overlaps = boxlist_iou(proposals, gt_boxes)
            max_overlaps, gt_assignment = overlaps.max(dim=1)
            pseudo_labels = gt_classes[gt_assignment, 0]
            loss_weights = gt_scores[gt_assignment, 0]

            # Select background RoIs as those with <= FG_IOU_THRESHOLD
            bg_inds = max_overlaps.le(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0]
            pseudo_labels[bg_inds] = 0

            if return_targets:
                matched_targets = gt_boxes[gt_assignment]
                regression_targets = self.box_coder.encode(
                     matched_targets.bbox, proposals.bbox
                )
                return pseudo_labels, loss_weights, regression_targets

            # PCL_TRICK:
            # ignore_thres = 0.1
            # ignore_inds = max_overlaps.le(ignore_thres).nonzero(as_tuple=False)[:,0]
            # loss_weights[ignore_inds] = 0

        return pseudo_labels, loss_weights

class od_layer(object):
    def __init__(self):
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        self.box_coder = BoxCoder(weights=bbox_reg_weights)
    @torch.no_grad()
    def __call__(self, proposals, source_score, labels, device, pgt_instance, return_targets=False):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=device)
        gt_classes = torch.zeros((0, 1), dtype=torch.long, device=device)
        gt_scores = torch.zeros((0, 1), dtype=torch.float, device=device)

        # not using the background class
        _prob = source_score[:, 1:].clone()
        # print(labels)
        _labels = labels[1:]
        # print(_labels)
        positive_classes = _labels.eq(1).nonzero(as_tuple=False)[:, 0]
        for c in positive_classes:
            cls_prob = _prob[:, c]
            max_index = torch.argmax(cls_prob)

            sim_box = pgt_instance[c]

            if sim_box.nelement() == 0:
                gt_boxes = torch.cat((gt_boxes, proposals.bbox[max_index].view(1, -1)), dim=0)
                gt_classes = torch.cat((gt_classes, c.add(1).view(1, 1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1, 1)), dim=0)
                _prob[max_index].fill_(0)
            else:
                gt_boxes = torch.cat((gt_boxes, proposals.bbox[sim_box]), dim=0)
                tmp_cls = torch.ones((sim_box.shape), dtype=torch.long, device=device) * c.add(1).item()
                gt_classes = torch.cat((gt_classes, tmp_cls.unsqueeze(1)), dim=0)
                gt_scores = torch.cat((gt_scores, cls_prob[sim_box].unsqueeze(1)), dim=0)
                _prob[max_index].fill_(0)

        if gt_boxes.shape[0]  == 0:
            num_rois = len(source_score)
            pseudo_labels = torch.zeros(num_rois, dtype=torch.long, device=device)
            loss_weights = torch.zeros(num_rois, dtype=torch.float, device=device)
        else:
            gt_boxes = BoxList(gt_boxes, proposals.size, mode=proposals.mode)
            overlaps = boxlist_iou(proposals, gt_boxes)
            #max_overlaps, gt_assignment = overlaps.max(dim=1)

            max_overlaps  = torch.tensor(overlaps.cpu().numpy().max(axis=1), device=device)
            gt_assignment = torch.tensor(overlaps.cpu().numpy().argmax(axis=1), device=device)

            pseudo_labels = gt_classes[gt_assignment, 0]
            loss_weights = gt_scores[gt_assignment, 0]

            # Select background RoIs as those with <= FG_IOU_THRESHOLD
            bg_inds = max_overlaps.le(cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD).nonzero(as_tuple=False)[:,0] ### IoU for instance supervision
            pseudo_labels[bg_inds] = 0

            if return_targets:
                matched_targets = gt_boxes[gt_assignment]
                regression_targets = self.box_coder.encode(
                     matched_targets.bbox, proposals.bbox
                )
                return pseudo_labels, loss_weights, regression_targets

            # PCL_TRICK:
            # ignore_thres = 0.1
            # ignore_inds = max_overlaps.le(ignore_thres).nonzero(as_tuple=False)[:,0]
            # loss_weights[ignore_inds] = 0
        return pseudo_labels, loss_weights
