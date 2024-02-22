import numpy as np
from terminaltables import AsciiTable
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pickle as pkl
from pycocotools.coco import COCO
np.seterr(divide='ignore',invalid='ignore')

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

def _recalls(all_ious, proposal_nums, thrs):
    if isinstance(all_ious, list):
        img_num = len(all_ious)
    else:
        img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format.
    """
    if isinstance(proposal_nums, list):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, list):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs


def eval_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 iou_thrs=None,
                 print_summary=True):
    """Calculate recalls.
    Args:
        gts(list or ndarray): a list of arrays of shape (n, 4)
        proposals(list or ndarray): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums(int or list of int or ndarray): top N proposals
        thrs(float or list or ndarray): iou thresholds
    Returns:
        ndarray: recalls of different ious and proposal nums
    """
    ## the gts is a  list containing ground truths of different images.
    ##
    img_num = len(gts)
    assert img_num == len(proposals)
 
    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)
 
    all_ious = []
    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]
        if proposal_nums is None:
            proposal_nums = np.array([img_proposal.shape[0]])
        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            # print(img_proposal.shape)
            if img_proposal.size == 0:
                all_ious.append(np.zeros((0, 1), dtype=np.float32))
                continue
            if img_proposal.size == 4:
                ious = bbox_overlaps(gts[i], img_proposal[ :4])
            else:
                ious = bbox_overlaps(gts[i], img_proposal[:prop_num, :4])
        all_ious.append(ious)
    # all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)
    if print_summary:
        print_recall_summary(recalls, proposal_nums, iou_thrs)
    return recalls 


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None):
    """Print recalls in a table.
    Args:
        recalls(ndarray): calculated from `bbox_recalls`
        proposal_nums(ndarray or list): top N proposals
        iou_thrs(ndarray or list): iou thresholds
        row_idxs(ndarray): which rows(proposal nums) to print
        col_idxs(ndarray): which cols(iou thresholds) to print
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [
            '{:.3f}'.format(val)
            for val in recalls[row_idxs[i], col_idxs].tolist()
        ]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print(table.table)
 
 
def plot_num_recall(recalls, proposal_nums):
    """Plot Proposal_num-Recalls curve.
    Args:
        recalls(ndarray or list): shape (k,)
        proposal_nums(ndarray or list): same shape as `recalls`
    """
    if isinstance(proposal_nums, np.ndarray):
        _proposal_nums = proposal_nums.tolist()
    else:
        _proposal_nums = proposal_nums
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls
 
    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot([0] + _proposal_nums, [0] + _recalls)
    plt.xlabel('Proposal num')
    plt.ylabel('Recall')
    plt.axis([0, proposal_nums.max(), 0, 1])
    f.show()


def xywh2xyxy(proposal):
    tl_x = proposal[0]
    tl_y = proposal[1]
    width = proposal[2]
    height = proposal[3]
    proposal[0] = tl_x
    proposal[1] = tl_y
    proposal[2] = tl_x + width
    proposal[3] = tl_y + height
    return proposal

 
def plot_iou_recall(recalls, iou_thrs):
    """Plot IoU-Recalls curve.
    Args:
        recalls(ndarray or list): shape (k,)
        iou_thrs(ndarray or list): same shape as `recalls`
    """
    if isinstance(iou_thrs, np.ndarray):
        _iou_thrs = iou_thrs.tolist()
    elif isinstance(iou_thrs, float):
        _iou_thrs = [iou_thrs]
    else:
        _iou_thrs = iou_thrs
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    elif isinstance(recalls, float):
        _recalls = [recalls]
    else:
        _recalls = recalls
 
    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot(_iou_thrs + [1.0], _recalls + [0.])
    plt.xlabel('IoU')
    plt.ylabel('Recall')
    plt.axis([iou_thrs.min(), 1, 0, 1])
    f.show()
    
#@ TODO: adding the code for COCO dataset recall evaluation.
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal-SAM', default='/home/junweizhou/MCTformer/MCTformer_results/proposals/mct_trainval.pkl')
    parser.add_argument('--proposal-path', default='/home/junweizhou/WSOD/OD-WSCL/proposal/SS/voc_2007_trainval.pkl', help='path for pickle file to be evaluated.')
    parser.add_argument('--iou-thresh', default=[0.5, 0.6, 0.75, 0.9], help='the threshold of iou for TP calculating, (can be a list of ints)')
    parser.add_argument('--vis-curve', default=False, help='whether or not to plot the PR curve of proposals.')
    parser.add_argument('--image-set', default='//home/junweizhou/WeakSAM/WSOD2/data/voc/VOC2007/ImageSets/Main/trainval.txt')
    parser.add_argument('--xml-path', default='/home/junweizhou/WeakSAM/WSOD2/data/voc/VOC2007/Annotations')
    parser.add_argument('--iscoco', default=False)
    parser.add_argument('--json-path', default='/home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/annotations/instances_train2014.json')
    
    args = parser.parse_args()
    
    f = open(args.image_set, encoding='utf-8')
    num_list = []
    data2write = []
    if args.iscoco:
        for line in f:
            num_list.append('COCO_val2014_' + line.strip())
    else:
        for line in f:
            num_list.append(line.strip())
    
    tot_gt = []
    if args.iscoco:
        cocoann = COCO(args.json_path)
        print('coco annotations loaded.')
        records = cocoann.getImgIds()
        records.sort()   # sorting in the sequence order for alignment of proposals.
        for i, id in enumerate(records):
            cur_gt = []
            annIds = cocoann.getAnnIds(id)
            anndisc = cocoann.loadAnns(annIds)
            for item in anndisc:
                gtbox = xywh2xyxy(item['bbox'])
                cur_gt.append(gtbox)
            cur_gt = np.array(cur_gt)
            tot_gt.append(cur_gt)
        
    else:    
        for num in num_list:
            xml_path = os.path.join(args.xml_path, num + '.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            file_name = root.find('filename').text
            file_number = file_name[0:-4]
            xml_name = file_number + '.xml'
            sing_gt = []
            for object in root.findall('object'):
                object_name = object.find('name').text
                Xmin = int(float(object.find('bndbox').find('xmin').text))
                Ymin = int(float(object.find('bndbox').find('ymin').text))
                Xmax = int(float(object.find('bndbox').find('xmax').text))
                Ymax = int(float(object.find('bndbox').find('ymax').text))
                cur_box = np.array([Xmin, Ymin, Xmax, Ymax])
                diff = int(object.find('difficult').text)
                # if diff:
                #     continue
                sing_gt.append(cur_box)     
            sing_gt = np.array(sing_gt)
            tot_gt.append(sing_gt)
            
    with open(args.proposal_SAM, 'rb') as f:  
        proposal = pkl.loads(f.read())
    print('the average recall of SAM + multi_scale peak point proposals is:')
    recalls_sam = eval_recalls(tot_gt, proposal, proposal_nums=30000, iou_thrs=args.iou_thresh)
    