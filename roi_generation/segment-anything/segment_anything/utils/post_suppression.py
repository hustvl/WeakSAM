import numpy as np
import os 
from pathlib import Path
import torch
import pickle as pkl    
import argparse 



def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True\
    # the calculation format of recall is in the xyxy format.
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def nms(prop_source, threshold):
    pick = []
    while prop_source.shape[0] > 0:

        i = prop_source[0]
        pick.append(i)
        ovr = bbox_overlaps(np.array([i]), prop_source)
        remove_ixs = np.where(ovr > threshold)[1]
        prop_source = np.delete(prop_source, remove_ixs, axis=0)
    return np.array(pick)


def post_suppression(proposal_source, proposal_target, threshold):
    ious = bbox_overlaps(proposal_source, proposal_target)  # n * 4, k * 4 -> n * k
    idxes = np.where(ious > threshold)
    del_idx = np.unique(idxes[1])
    filtered_prop = np.delete(proposal_target, del_idx, axis=0)
    print(proposal_target.shape, filtered_prop.shape)
    return filtered_prop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-prop', default=None)
    parser.add_argument('--threshold', default=0.95, type=float)
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--save-path', default=None)
    args = parser.parse_args()
    
    filtered_target_list = []
        
    with open(args.target_prop, 'rb') as f:
        proposal_target = pkl.load(f)
        
    print(len(proposal_target))
        
    for i in range(len(proposal_target)):
        # filtered_target = post_suppression(proposal_source[i], proposal_target[i], args.threshold)
        # filtered_target = post_suppression(proposal_target[i], proposal_target[i], args.threshold)
        
        filtered_target = nms(proposal_target[i], args.threshold)
        filtered_target_list.append(filtered_target)
        
    with open(os.path.join(args.save_path, args.save_name), 'wb') as file:
        pkl.dump(filtered_target_list, file)   

    