import os
import numpy as np
import torch
import argparse
import torch.nn as nn
import cv2


def peak_extract(heat, kernel=5, K=25, device='cpu'):
    B, C, H, W = heat.size()
    # getting the size of the heat map.
    heat = heat.detach().to(device)
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    
    
    keep = (hmax == heat).float()

    # the pooling coordination
    peak = heat * keep
    # getting the peak points' value of the heat map.
    topk_scores, topk_inds = torch.topk(peak.view(B, C, -1), K)
    # sorting the top k best values of the pooled heat map.

    topk_inds = topk_inds % (H * W)
    topk_ys = (topk_inds / W).int().float()
    topk_xs = (topk_inds % W).int().float()

    topk_scores = topk_scores[0].float().detach().cpu().numpy()
    topk_ys = topk_ys[0].int().detach().cpu().numpy()
    topk_xs = topk_xs[0].int().detach().cpu().numpy()
    
# Below is for the holding distance filtering, and the holding distance is set to kernel size // 2 as default.
    filtering_indices = []
# The points are naturally in sorted list.
    for item in zip(topk_ys[0], topk_xs[0]):
        if filtering_indices == []:
            filtering_indices.append(item)
        else:
            flag = 1
            for idx in range(len(filtering_indices)):
                if abs(item[0] - filtering_indices[idx][0]) + abs(item[1] - \
                    filtering_indices[idx][1]) < (kernel - 1) // 2:
                    # filtering_indices[idx] = ((filtering_indices[idx][0] + item[0]) // 2, (filtering_indices[idx][1] + item[1]) // 2)
                    flag = 0
                    break
            if flag:
                filtering_indices.append(item)
    # getting the topk xs and topk ys for the peak point genaration
    torch.cuda.empty_cache()
    return topk_scores, filtering_indices


def smoothing(heat, kernel=3):
    pad = (kernel - 1) // 2
    heat = torch.nn.functional.avg_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    return heat



def get_bboxes(cam, cam_thr=0.2, box_thr=0.5, iou_thr=0.5, type="single"):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    # map_thr = cam_thr * np.max(cam)
    map_thr = cam_thr * 255.

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # the contours using the api of opencv's algo.
    if len(contours) == 0:
        return [[0, 0, 1, 1]]
    if type == "single":
        # if only need to find oen bounding box just getting the bbox from the found countour's maximum value.
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [[x, y, x + w, y + h]]
        return estimated_bbox

    estimated_bbox = []
    confidence = []
    max_ares = cv2.contourArea(max(contours, key=cv2.contourArea))
    for c in contours:
        # for all the found contours generate bboxes.
        x, y, w, h = cv2.boundingRect(c)
        if w * h > max_ares * box_thr:
            estimated_bbox.append([x, y, x + w, y + h])
            confidence.append(cam[y:y + h, x:x + w].max())
    confidence = np.array(confidence)
    estimated_bbox = np.array(estimated_bbox)
    sorted_ind = np.argsort(-confidence)
    estimated_bbox = estimated_bbox[sorted_ind, :]
    # all depending on the opencv found contours.

    # Filtering out bboxes with high overlap
    for ind, bb in enumerate(estimated_bbox):
        ixmin = np.maximum(estimated_bbox[:, 0], bb[0])
        iymin = np.maximum(estimated_bbox[:, 1], bb[1])
        ixmax = np.minimum(estimated_bbox[:, 2], bb[2])
        iymax = np.minimum(estimated_bbox[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)
        inters = iw * ih
        # getting the IOU of each two boxes.

        # union
        uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (estimated_bbox[:, 2] - estimated_bbox[:, 0] + 1.0) *
                (estimated_bbox[:, 3] - estimated_bbox[:, 1] + 1.0)
                - inters
        )
        overlaps = inters / uni
        drop_ind = np.argwhere(overlaps > iou_thr)[0]
        drop_ind = np.delete(drop_ind, drop_ind <= ind)
        estimated_bbox = np.delete(estimated_bbox, drop_ind, axis=0)
    return estimated_bbox  # , thr_gray_heatmap, len(contours)

