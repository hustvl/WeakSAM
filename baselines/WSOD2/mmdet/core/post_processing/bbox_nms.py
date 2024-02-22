import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]
    if score_factors is not None:
        scores = scores * score_factors[:, None]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    # TODO: add size check before feed into batched_nms
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs

class WeaklyMulticlassNMS(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.score_thr = [0.] * self.num_classes

    def __call__(self,
                 multi_bboxes,
                 multi_scores,
                 nms_cfg,
                 max_num=-1,
                 score_factors=None):

        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)
    
        scores = multi_scores[:, :-1]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
    
        labels = torch.arange(num_classes, dtype=torch.long)
        labels = labels.view(1, -1).expand_as(scores)
    
        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
    
        # remove low scoring boxes
        score_thr = 0.
        valid_mask = scores > score_thr
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        if inds.numel() == 0:
            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                   'as it has not been executed this time')
            return bboxes, labels

        final_bboxes = []
        final_scores = []
        final_labels = []
        for i in range(num_classes):
            idx_c = (labels == i).nonzero().squeeze(1)
            bboxes_c = bboxes[idx_c]
            scores_c = scores[idx_c]
            labels_c = labels[idx_c]

            if scores_c.shape[0] > max_num:
                _, idx_c = torch.topk(scores_c, k=max_num)
                bboxes_c = bboxes_c[idx_c]
                scores_c = scores_c[idx_c]
                labels_c = labels_c[idx_c]
    
            idx_c = (scores_c > self.score_thr[i]).nonzero().squeeze(1)
            bboxes_c = bboxes_c[idx_c]
            scores_c = scores_c[idx_c]
            labels_c = labels_c[idx_c]

    
            if scores_c.shape[0] > 40:
                _, idx_c = torch.topk(scores_c, k=40)
                bboxes_c = bboxes_c[idx_c]
                scores_c = scores_c[idx_c]
                labels_c = labels_c[idx_c]

                self.score_thr[i] = max(self.score_thr[i], scores_c.min().item())
    
            final_bboxes.append(bboxes_c)
            final_scores.append(scores_c)
            final_labels.append(labels_c)

        bboxes = torch.cat(final_bboxes, dim=0)
        scores = torch.cat(final_scores, dim=0)
        labels = torch.cat(final_labels, dim=0)

        if bboxes.shape[0] == 0:
            return bboxes, labels
    
        # TODO: add size check before feed into batched_nms
        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]
    
        return dets, labels[keep]
    
