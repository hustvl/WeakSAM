import torch

from mmdet.core import (bbox2result, bbox2roi, build_assigner, 
                        build_sampler, bbox_overlaps, bbox_mapping, 
                        merge_aug_bboxes, multiclass_nms, bbox_mapping_back)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from torchvision.ops import nms

@HEADS.register_module()
class OICRRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):

    def init_assigner_sampler(self):
        pass

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        pass

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_labels):
        losses = dict()
        # bbox head forward and loss
        bbox_results = self._bbox_forward_train(x, proposal_list, gt_labels, img_metas)
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

        if len(gt_bboxes) == 0:
            print(gt_labels)
            print(boxes)
            return None
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
        targets = gt_bboxes[gt_assignment]

        fg_inds = (max_overlaps >= 0.5).nonzero().squeeze(1)
        bg_inds = (max_overlaps < 0.5).nonzero().squeeze(1)

        labels[bg_inds] = -1

        return labels, weights, targets
        
    def _bbox_forward_train(self, x, proposal_list, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi(proposal_list)

        gt_labels = torch.cat(gt_labels, dim=0)
        bbox_results = self._bbox_forward(x, rois)

        pseudo_gt_1 = self._get_pseudo_gt(rois, bbox_results['cls'].clone(), gt_labels)
        if pseudo_gt_1 is None:
            for item in img_metas:
                print(item)
        pseudo_gt_2 = self._get_pseudo_gt(rois, bbox_results['refine1'][:, 1:].clone(), gt_labels)
        pseudo_gt_3 = self._get_pseudo_gt(rois, bbox_results['refine2'][:, 1:].clone(), gt_labels)
        refine_labels_1, refine_weights_1, _ = self._sample_rois(rois, pseudo_gt_1)
        refine_labels_2, refine_weights_2, _ = self._sample_rois(rois, pseudo_gt_2)
        refine_labels_3, refine_weights_3, bbox_targets = self._sample_rois(rois, pseudo_gt_3)

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

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results


    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)  # Getting those proposals' rois for feature computing.
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls1 = bbox_results['refine1']
        cls2 = bbox_results['refine2']
        cls3 = bbox_results['refine3']
        cls = (cls1 + cls2 + cls3) / 3 # prediction probs of each class?
        cls = cls[:, 1:]
        #cls = bbox_results['cls']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls = cls.split(num_proposals_per_img, 0)

        bbox_pred = (None, ) * len(proposals)  # all of them are none, in this split.
        if bbox_results['bbox'] is not None:
            bbox_pred = bbox_results['bbox'].split(num_proposals_per_img, 0)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls[i],  # here the cls item represents the scores for those rois.
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

# simple test for OICR head.
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, proposal, img_meta in zip(feats, proposal_list, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            #proposals = bbox_mapping(proposal[0], img_shape,
            #                         scale_factor, flip, flip_direction)
            rois = bbox2roi([proposal[0]])
            bbox_results = self._bbox_forward(x, rois)
            cls1 = bbox_results['refine1']
            cls2 = bbox_results['refine2']
            cls3 = bbox_results['refine3']
            cls = (cls1 + cls2 + cls3) / 3.
            cls = cls[:, 1:]
            bbox_pred = bbox_results['bbox']

            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                cls,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)

        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels



    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        return [bbox_results]
