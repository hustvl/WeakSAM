import numpy as np 
import os 
from pathlib import Path

from pycocotools.coco import COCO
import argparse
import cv2
# Here for default settings, we consider the COCO2014 train dataset for peak point quality evaluation.#


def batch_readtxt(folder_path, img_label, starting_layernum=1, ending_layernum=12, is_multi=False):
    whole_pth = os.path.join(folder_path, str(img_label) + '.txt')
    if os.path.getsize(whole_pth) == 0:
        return None, None
    pt_list = np.genfromtxt(whole_pth, dtype=[float, float, int, float], delimiter=' ') # the read all txt
    sorted_pts = {}
    pts_list = []
    if pt_list.size == 1:
        #point all
        x = pt_list.item()[0]
        y = pt_list.item()[1]
        pts_list.append([x, y])
        # cls sorted
        cls = pt_list.item()[2]
        if cls not in sorted_pts:
            sorted_pts[cls] = []
            sorted_pts[cls].append([x, y])
        return pts_list, sorted_pts

    for item in pt_list:
        x = item[0]
        y = item[1]
        if is_multi:
            head_idx = item[4]
            if (head_idx % 12) + 1 < starting_layernum or head_idx % 12 + 1 > ending_layernum:  # confining starting and ending visualization.
                continue
        item_ = [x, y]
        pts_list.append(item_)
        cls = item[2]
        #####sorted points
        if cls not in sorted_pts:
            sorted_pts[cls] = []
            sorted_pts[cls].append(item_)
        else:
            sorted_pts[cls].append(item_)
    return pts_list, sorted_pts


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


def calculating_ap(recall, prec):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori-img-path', default='/home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/train2014')
    parser.add_argument('--annotation-path', default='/home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/annotations/instances_train2014.json')
    parser.add_argument('--peak-path', default='/home/junweizhou/WeakTr/WeakTr/weaktr_results_coco/COCO-peak/cross/peak-pam-k129-t90-div10')
    parser.add_argument('--ovthresh', default=0.5, type=float)
    
    args = parser.parse_args()
    ori_img_path = args.ori_img_path
    annotation = args.annotation_path
    peak_point_path = args.peak_path
    ovthresh = args.ovthresh
    coco = COCO(annotation)
    print('COCO annotations loaded....')
    records = coco.getImgIds()
    ovthresh = [ovthresh, ovthresh-0.2, ovthresh-0.4]
    
    all_count = 0
    tp = [0, 0, 0]
    fp = [0, 0, 0]
    
    for i, id in enumerate(records):
        if i % 500 == 0:
            print('%d image processed' % i)
        filename = 'COCO_train2014_' + str(id).rjust(12, '0')
        all_pts, _ = batch_readtxt(peak_point_path, filename)
        if all_pts is None:
            continue
        ovmin = np.inf
        
        all_pts = np.array(all_pts)
        
        cur_gt = []
        
        record = records[i]
        annIds = coco.getAnnIds(record)
        anns_dic = coco.loadAnns(annIds)
        
        for item in anns_dic:
            gtbox = xywh2xyxy(item['bbox'])
            cur_gt.append(gtbox)
        all_count += len(cur_gt)
        if cur_gt is not []:
            for BBGT in cur_gt:
                BBGT = np.array([BBGT])
                centerx = (BBGT[:, 0] + BBGT[:, 2]) // 2
                centery = (BBGT[:, 1] + BBGT[:, 3]) // 2

                len_x = BBGT[:, 2] - BBGT[:, 0]
                len_y = BBGT[:, 3] - BBGT[:, 1]
                # dis = (bb[0] - centerx) ** 2 + (bb[1] - centery) ** 2
                dis_x = all_pts[:, 0] - centerx
                dis_y = all_pts[:, 1] - centery
                # the definition of overlaps is calculated as the diagonal distance ** 2.
                # overlaps = dis / ((BBGT[:, 2] - BBGT[:, 0]) ** 2 + (BBGT[:, 3] - BBGT[:, 1]) ** 2)
                prop_x = np.abs(dis_x / len_x)
                prop_y = np.abs(dis_y / len_y)

                idx_x = np.where(prop_x > 0.5)
                idx_y = np.where(prop_y > 0.5)
                overlaps = (prop_x + prop_y) / 2
                overlaps[idx_x] = 1
                overlaps[idx_y] = 1

                ovmin = np.min(overlaps)
        
                for i, thresh in enumerate(ovthresh):
                    if ovmin < thresh:
                        if cur_gt is []:
                            fp[i] += 1
                        else:
                            tp[i] += 1
                    else:
                        fp[i] += 1

    for i in range(len(tp)):
        recall = tp[i] / all_count
        prec = tp[i] / np.maximum(tp[i] + fp[i], np.finfo(np.float64).eps)
        # ap = calculating_ap(recall, prec)
        print('--------recall on thresh %d: %f' %(50 - 20 * i, recall))
        print('--------precision on thresh %d: %f' %(50 - 20 * i, prec))
        # print('--------ap on thresh %d: %f' %(50 - 20 * i, ap))
    