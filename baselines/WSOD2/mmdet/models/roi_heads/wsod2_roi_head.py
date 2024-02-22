import torch

from mmdet.core import (bbox2result, bbox2roi, build_assigner, 
                        build_sampler, bbox_overlaps, bbox_mapping, 
                        merge_aug_bboxes, multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .oicr_roi_head  import OICRRoIHead

from torchvision.ops import nms

@HEADS.register_module()
class WSOD2RoIHead(OICRRoIHead):

    def __init__(self, steps, **kwargs):
        super(WSOD2RoIHead, self).__init__(**kwargs)
        self.alpha = 0.0
        self.steps = steps

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_labels,
                      ss):
        losses = dict()
        # bbox head forward and loss
        bbox_results = self._bbox_forward_train(x, ss, proposal_list, gt_labels, img_metas)
        losses.update(
            loss_wsddn=bbox_results['loss_wsddn'],
            loss_oicr_1=bbox_results['loss_oicr_1'],
            loss_oicr_2=bbox_results['loss_oicr_2'],
            loss_oicr_3=bbox_results['loss_oicr_3']
        )
        if 'loss_bbox' in bbox_results:
            losses.update(loss_bbox=bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        cls, refine1, refine2, refine3, bbox = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls=cls,
            refine1=refine1,
            refine2=refine2,
            refine3=refine3,
            bbox=bbox
        )
        return bbox_results

    def _get_pseudo_gt(self, boxes, cls_prob, gt_labels):
        num_classes = gt_labels.shape[0]
        device = boxes.device

        gt_bboxes = []
        gt_classes = []
        gt_scores = []

        for i in range(num_classes):
            if gt_labels[i] == 1:
                max_prob = cls_prob[:, i].max()
                candidate_idx = (cls_prob[:, i] >= max_prob * 0.9).nonzero().squeeze(1)
                gt_bboxes.append(boxes[candidate_idx])
                gt_classes += [i] * candidate_idx.shape[0]
                gt_scores.append(cls_prob[candidate_idx, i].detach())


        gt_bboxes = torch.cat(gt_bboxes, dim=0)
        gt_classes = torch.tensor(gt_classes).to(device=device)
        gt_scores = torch.cat(gt_scores, dim=0)

        keep = nms(gt_bboxes[:, 1:], gt_scores, 0.5)
        gt_bboxes = gt_bboxes[keep]
        gt_classes = gt_classes[keep]
        gt_scores = gt_scores[keep]

        pseudo_gt = dict(
            gt_bboxes=gt_bboxes,
            gt_classes=gt_classes,
            gt_scores=gt_scores
        )
        return pseudo_gt
        
    def _sample_rois(self, rois, pseudo_gt):
        
        gt_bboxes = pseudo_gt['gt_bboxes']
        gt_classes = pseudo_gt['gt_classes']
        gt_scores = pseudo_gt['gt_scores']

        overlaps = bbox_overlaps(rois[:, 1:], gt_bboxes[:, 1:])
        max_overlaps, gt_assignment = overlaps.max(dim=1)
        labels = gt_classes[gt_assignment]
        weights = gt_scores[gt_assignment]
        weights = torch.ones_like(weights)
        targets = gt_bboxes[gt_assignment]

        fg_inds = (max_overlaps >= 0.5).nonzero().squeeze(1)
        bg_inds = (max_overlaps < 0.5).nonzero().squeeze(1)

        labels[bg_inds] = -1

        return labels, weights, targets

    def _cal_bu(self, ss, rois, scale_factor):
        device = rois.device

        rois = rois[:, 1:]
        rois = rois / torch.tensor(scale_factor).to(device=device).view(-1, 4)
        area = (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1)
        rois = rois.long()

        ss = ss.squeeze(0).long()
        h, w = ss.shape[:2]

        ss = ss[:, :,  0] * 256 * 256 + ss[:, :, 1] * 256 + ss[:, :, 2]
        ss = ss.view(-1)
        ss_unique, ss_idx = torch.unique(ss, return_inverse=True)
        tot = ss_idx.max() + 1
        ss_bin = torch.bincount(ss_idx, minlength=tot)

        ss_idx = ss_idx.view((h, w))

        bu = []
        for i in range(rois.shape[0]):
            crop_ss = ss_idx[rois[i, 1]:rois[i, 3]+1, rois[i, 0]:rois[i, 2]+1].contiguous().view(-1)
            crop_bin = torch.bincount(crop_ss, minlength=tot)
            crop_unique = torch.unique(crop_ss)

            ss_select = ss_bin[crop_unique]
            crop_select = crop_bin[crop_unique]

            diff = ss_select - crop_select
            diff = torch.min(diff, crop_select).sum().float()
            bu.append(1. - diff / area[i])

        return torch.tensor(bu, dtype=torch.float32).to(device=device)
        
    def _bbox_forward_train(self, x, ss, proposal_list, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        assert x[0].shape[0] == 1
        rois = bbox2roi(proposal_list)

        gt_labels = torch.cat(gt_labels, dim=0)
        bbox_results = self._bbox_forward(x, rois)

        pseudo_gt_1 = self._get_pseudo_gt(rois, bbox_results['cls'].clone(), gt_labels)
        pseudo_gt_2 = self._get_pseudo_gt(rois, bbox_results['refine1'][:, 1:].clone(), gt_labels)
        pseudo_gt_3 = self._get_pseudo_gt(rois, bbox_results['refine2'][:, 1:].clone(), gt_labels)

        refine_labels_1, refine_weights_1, _ = self._sample_rois(rois, pseudo_gt_1)
        refine_labels_2, refine_weights_2, _ = self._sample_rois(rois, pseudo_gt_2)
        refine_labels_3, refine_weights_3, bbox_targets = self._sample_rois(rois, pseudo_gt_3)

        pos_idx = (refine_labels_1 != -1).nonzero().squeeze(1)
        bu = self._cal_bu(ss, rois[pos_idx], img_metas[0]['scale_factor'])
        refine_weights_1[pos_idx] = self.alpha * refine_weights_1[pos_idx] + (1. - self.alpha) * bu

        pos_idx = (refine_labels_2 != -1).nonzero().squeeze(1)
        bu = self._cal_bu(ss, rois[pos_idx], img_metas[0]['scale_factor'])
        refine_weights_2[pos_idx] = self.alpha * refine_weights_2[pos_idx] + (1. - self.alpha) * bu

        pos_idx = (refine_labels_3 != -1).nonzero().squeeze(1)
        bu = self._cal_bu(ss, rois[pos_idx], img_metas[0]['scale_factor'])
        refine_weights_3[pos_idx] = self.alpha * refine_weights_3[pos_idx] + (1. - self.alpha) * bu

        self.alpha += 1. / self.steps
        self.alpha = min(1., self.alpha)

        loss_wsddn = self.bbox_head.loss_wsddn(bbox_results['cls'], gt_labels)
        loss_oicr_1 = self.bbox_head.loss_oicr(bbox_results['refine1'], refine_labels_1, refine_weights_1)
        loss_oicr_2 = self.bbox_head.loss_oicr(bbox_results['refine2'], refine_labels_2, refine_weights_2)
        loss_oicr_3 = self.bbox_head.loss_oicr(bbox_results['refine3'], refine_labels_3, refine_weights_3)

        bbox_results.update(
            loss_wsddn=loss_wsddn,
            loss_oicr_1=loss_oicr_1,
            loss_oicr_2=loss_oicr_2,
            loss_oicr_3=loss_oicr_3
        )

        if bbox_results['bbox'] is not None:
            bbox_targets = self.bbox_head.get_targets(rois[:, 1:], bbox_targets[:, 1:])
            loss_bbox = self.bbox_head.loss_bbox(bbox_results['bbox'], bbox_targets, refine_labels_3, refine_weights_3)
            bbox_results.update(
                loss_bbox=loss_bbox
            )
        return bbox_results
