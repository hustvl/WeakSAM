# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torch import nn

from wetectron.structures.bounding_box import BoxList
from wetectron.structures.boxlist_ops import boxlist_nms, boxlist_iou, cat_boxlist, remove_small_boxes
from wetectron.modeling.box_coder import BoxCoder
from wetectron.utils.utils import to_boxlist, cal_iou

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes, softmax_on=True):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression(Nx324) from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """

        class_logits, box_regression = x
        if softmax_on:
            class_prob = F.softmax(class_logits, -1)
        else:
            class_prob = class_logits
        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)

            #filtered_boxlist = self.filter_box(boxlist, num_classes)
            #new_boxlist = self.get_class_map(filtered_boxlist, num_classes, image_shape)

            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)

            results.append(boxlist)
        return results

    def get_class_map(self, boxlist, num_classes, image_shape):
        new_boxlist = BoxList(boxlist.bbox, boxlist.size, mode='xyxy')
        class_map = torch.zeros((num_classes,image_shape[0],image_shape[1]), dtype=torch.float, device=boxlist.bbox.device)
        round_boxes = torch.round(boxlist.bbox).type(torch.int)
        classes = boxlist.get_field('labels')
        scores = boxlist.get_field('scores')
        map_score = torch.zeros_like(scores)

        for box, c, score in zip(round_boxes, classes, scores):
            x1, y1, x2, y2 = box
            class_map[c][x1:x2, y1:y2] += score

        #for c in range(num_classes):
        #    class_map[c] = (class_map[c] - class_map[c].min()) / (class_map[c].max() - class_map[c].min())
        class_map = (class_map - class_map.min()) / (class_map.max() - class_map.min()) * 255

        ### try percentile normalization / some alternative ways of normalization
        for c in range(num_classes):
            if class_map[c].max() != 0:
                _, thresholded_map = cv2.threshold(class_map[c].detach().cpu().numpy(), 255*0.7, maxval=255, type=cv2.THRESH_BINARY)
                import IPython; IPython.embed()
        T = 0.5
        for i,(box, c, score) in enumerate(zip(round_boxes, classes, scores)):
            x1, y1, x2, y2 = box
            if torch.where(class_map[c][x1:x2,y1:y2] > T)[0].shape != 0:
                map_score[i] = class_map[c][x1:x2,y1:y2][torch.where(class_map[c][x1:x2,y1:y2] > T)].sum() / torch.where(class_map[c][x1:x2,y1:y2] > T)[0].shape[0]
            else:
                map_score[i] = class_map[c][x1:x2, y1:y2].mean()
        new_boxlist.add_field('scores', map_score)
        new_boxlist.add_field('labels', classes)
        return new_boxlist

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def post_process(self, boxlist, num_classes):
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        if scores[:,1:].max() <= 0.01:
            inds_all = scores >= scores[:,1:].max()
        else:
            inds_all = scores > 0.01
        #inds_all = scores > self.score_thresh

        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero(as_tuple=False).squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)

            if inds.nelement() != 0:
                #new_box = []
                new_box = torch.zeros((0), dtype=torch.float, device='cuda')
                new_box_labels = torch.zeros((0), dtype=torch.int64, device='cuda')
                new_box_scores = torch.zeros((0), dtype=torch.float, device='cuda')
                for n_th in scores_j.sort(descending=True)[1]:
                    keep_boxlist, keep_ind = remove_inside_boxlist(boxlist_for_class, boxlist_for_class[n_th.view(-1)])
                    keep_boxes = keep_boxlist.bbox
                    keep_scores = keep_boxlist.get_field('scores')

                    score_v, score_ind = keep_scores.sort(descending=True)
                    iou_v = boxlist_iou(keep_boxlist[score_ind], keep_boxlist[score_ind])[0]
                    h_iou_ind = torch.ge(iou_v, 0.8).nonzero(as_tuple=False)

                    merge_box = keep_boxes[h_iou_ind].squeeze(1).mean(0)
                    #mean_score = keep_scores[h_iou_ind].mean()
                    mean_score = scores_j[n_th]

                    new_box = torch.cat((new_box, merge_box.view(1,-1)))
                    new_box_labels = torch.cat((new_box_labels, torch.tensor(j, dtype=torch.int64, device=device).view(-1)))
                    new_box_scores = torch.cat((new_box_scores, torch.tensor(mean_score, dtype=torch.float, device=device).view(-1)))

                merge_boxlist = BoxList(new_box, boxlist.size, mode='xyxy')
                merge_boxlist.add_field('labels', new_box_labels)
                merge_boxlist.add_field('scores', new_box_scores)

                merge_boxlist = boxlist_nms(merge_boxlist, self.nms)

                result.append(merge_boxlist)

        if len(result) == 0:
            import IPython; IPython.embed()
        result = cat_boxlist(result)
        number_of_detections = len(result)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]
        #import IPython; IPython.embed()
        return result

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero(as_tuple=False).squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]

        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    )
    return postprocessor
