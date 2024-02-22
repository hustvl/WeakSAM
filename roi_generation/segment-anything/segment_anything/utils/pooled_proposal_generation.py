import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import argparse
import os
import PIL.Image as Image
import time
from pathlib import Path
import pickle as pkl
from amg import batched_mask_to_box
import multiprocessing
from multiprocessing import shared_memory


def show_box(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        # the bbox is in the form of tlx, tly, brx, bry
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def xywh2xyxy(raw_data):
    for proposal in raw_data:
        tl_x = proposal[0]
        tl_y = proposal[1]
        width = proposal[2]
        height = proposal[3]
        proposal[0] = tl_x
        proposal[1] = tl_y
        proposal[2] = tl_x + width
        proposal[3] = tl_y + height
    return raw_data


def transform_one_img(mask_list):
    mask_list = np.array(mask_list)
    mask_list = torch.from_numpy(mask_list)
    batched_box = batched_mask_to_box(mask_list) # the batched box: Tensor with c * 4
    return batched_box


def store_mask(masks_1cls, storage_path, folder_name, idx):
    # the output mask of one class points is just a 3*H*W mask
    # which is saved to the num_list[i] folder.
    # storing as the npy file.
    Path(storage_path + '/' + str(folder_name)).mkdir(parents=True, exist_ok=True)
    np.save(storage_path + '/' + str(folder_name) + '/' + str(idx), masks_1cls)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def batch_readtxt(folder_path, img_label, starting_layer=1, ending_layer=12, is_multi_cross=False, iscoco=False):
    # if iscoco and 'val' in folder_path:
    #     whole_pth = os.path.join(folder_path, 'COCO_val2014_' + str(img_label) + '.txt')
    # elif iscoco:
    #     whole_pth = os.path.join(folder_path, 'COCO_train2014_' + str(img_label) + '.txt')
    # else:
    whole_pth = os.path.join(folder_path, str(img_label) + '.txt')
    
    if os.path.getsize(whole_pth) == 0:
        return None, None
    # the read all txt
    sorted_pts = {}
    
    if is_multi_cross:  # for multi cross attnmap generation.
        pt_list = np.genfromtxt(whole_pth, dtype=[float, float, int, float, int], delimiter=' ')
        for item in pt_list:
            x = item[0]
            y = item[1]
            head_idx = item[4]
            if (head_idx % 12) + 1 < starting_layer or head_idx % 12 + 1 > ending_layer:  # confining starting and ending visualization.
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
        return pt_list, sorted_pts
    
    pt_list = np.genfromtxt(whole_pth, dtype=[float, float, int, float], delimiter=' ') 
    if pt_list.size == 0:
        return None, None
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

# calculating how much time each procedure cost. (If mobileSAM can be better for generating?)
def inference_single_image(img, points_list, predictor, point_input_method, mask_type=False, iscoco=False):
    # time_start = time.perf_counter()
    predictor.set_image(img)  # image encoder for feature extracting.
    # time_end = time.perf_counter()
    # elapsed = time_end - time_start
    # print('extracting features using: %f seconds.' %elapsed )
    if point_input_method == 'single':
        masks_batch = []
        scores_batch = []
        logits_batch = []
        points = points_list
        label = np.array([1])
        # time_start = time.perf_counter()
        for point in points:
            point = np.array([point])
            masks, scores, logits = predictor.predict(
                point_coords=point,
                point_labels=label,
                multimask_output=mask_type,
            )
            masks_batch.append(masks[1:])
            scores_batch.append(scores[1:])
            logits_batch.append(logits[1:])
        # time_end = time.perf_counter()
        # elapsed = time_end - time_start
        # print('getting masks using: %f seconds.' %elapsed )
        return masks_batch, scores_batch, logits_batch
    else:
        points = np.array(points_list)
        lent = len(points_list)
        label = np.full(lent, 1)
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=label,
            multimask_output=mask_type,
        )
        return masks[1:], scores[1:], logits[1:]

def sam_init(device):
    # sam model initialization
    sam_checkpoint = '../../checkpoints/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('------------------------loading sam model------------------------- ' + model_type)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def single_process_prediction(num_list, folder_path, process_num, num_gpus, iscoco=False, starting_gpu=0, is_cls_aware=False):
    device = 'cuda:' + str(process_num % num_gpus + starting_gpu)
    predictor = sam_init(device)
    start_time = time.perf_counter()
    proposals = []
    for i in range(len(num_list)):
        if i % 10 == 1:
            cur_time = time.perf_counter()
            elapsed = (cur_time - start_time) / 60 # minutes
            print('-------------------- %d image processed, %d remaining, elapsed time: %4f min' %(i, len(num_list) - i, elapsed))
        point_single, point_sorted = batch_readtxt(folder_path=folder_path, img_label=num_list[i], iscoco=iscoco)
        # image related
        imgpth = os.path.join(img_path, str(num_list[i]) + '.jpg')
        image = cv2.imread(imgpth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        point_single = np.array(point_single)
        # labels = np.full(len(point_single), 1)
        if point_sorted is None:
            # Path(mask_storage + '/' + num_list[i]).mkdir(exist_ok=True, parents=True)
            proposals.append(np.array([]))
            continue
        else :
            cls_num = point_sorted.keys()
        
    # single
        if iscoco:
            masks, scores, logits = inference_single_image(image, point_single,  predictor, 'single', args.mask_type, iscoco=True)
            batched_single_prop = transform_one_img(masks)  # A tensor for return.
        else:
            masks, scores, logits = inference_single_image(image, point_single,  predictor, 'single', args.mask_type)
            batched_single_prop = transform_one_img(masks)  # A tensor for return.
        # print(batched_single_prop.shape)
    # batch
        if is_cls_aware:
            batched_batch_prop = []
            for cls in cls_num:
                pt_s = point_sorted[cls]
                pt_s = np.array(pt_s)
                labels = np.full(len(pt_s), 1)
                masks, scores, logits = inference_single_image(image, pt_s,  predictor, 'batch',args.mask_type)
                batched_batch_prop_ = transform_one_img(masks)
                batched_batch_prop.append(np.array(batched_batch_prop_))
                # the masks are stored in the npy file with shape 3*H*W(original shape of SAM output.
            
            batched_batch_prop = torch.tensor(np.array(batched_batch_prop))
            prop_img = torch.cat([batched_single_prop, batched_batch_prop], dim=0)
        else:
            prop_img = batched_single_prop

        a, b, c = prop_img.shape    
        prop_img = prop_img.view(a * b, c)

        prop_img = np.array(prop_img)
        proposals.append(prop_img)
    return proposals
    
def single_process(p):
    return single_process_prediction(p[0], p[1], p[2], p[3], p[4], p[5], p[6])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset reading options
    parser.add_argument('--txt-folder-path', default='./WeakTr/WeakTr/weaktr_results/VOC07-peak/', type=str)
    parser.add_argument('--ori-image-path', default='./WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages/', type=str)
    # parser.add_argument('--img-set-path', default='./Datasets/VOC/VOC2012/ImageSets/Segmentation/train.txt')
    parser.add_argument('--img-set-path', default='./WeakTr/WeakTr/voc07/test.txt')

    # mask generation options(obligatory storage)
    parser.add_argument('--mask-type', default=True, type=str)
    parser.add_argument('--peakfile-name', default='fine/peak-pam-k33-t90-test/')
    parser.add_argument('--point-input-type', default='mixture', help='Whether to use one point or '
                                                                    'a batch for inference in one time.'
                                                                    '(options: single, batch, mixture)')
    parser.add_argument('--proposal-storage-path', default='./WeakSAM/segment-anything/peak_proposal/VOC07/pam-k33', help='for the storage of proposals.')
    parser.add_argument('--proposal-name', default='k33_test.pkl')
    parser.add_argument('--device', default=None, type=int)
    parser.add_argument('--starting-layer', default=1, type=int)
    parser.add_argument('--ending-layer', default=12, type=int)
    parser.add_argument('--is-multi', default=False)
    parser.add_argument('--num-gpus', default=1, type=int)
    parser.add_argument('--n-processes', default=12, type=int)
    parser.add_argument('--iscoco', default=False)
    parser.add_argument('--is-cls-aware', default=False, help='For class-aware prompting.')
    ## TODO@: creating a process pool for parallel running.
    args =parser.parse_args()

    peakfile = args.peakfile_name
    txt_folder = args.txt_folder_path
    img_path = args.ori_image_path
    n_processes = args.n_processes
    point_input_method = args.point_input_type
    img_set_txt = args.img_set_path 
    is_cls_aware = args.is_cls_aware
    
    
    f = open(img_set_txt, encoding='utf-8')
    num_list = []
    partitioned_numlist = []
    for line in f:
        if args.iscoco and 'val' in img_set_txt:
            num_list.append('COCO_val2014_' + line.strip())
        elif args.iscoco:
            num_list.append('COCO_train2014_' + line.strip())
        else:
            num_list.append(line.strip())
    folder_path = os.path.join(txt_folder, peakfile)
    pool = multiprocessing.Pool(processes=n_processes)
    single_pool_num = len(num_list) // n_processes
    param_list = []
    for i in range(n_processes):
        if i != n_processes - 1:
            partitioned_numlist.append(num_list[i * single_pool_num : (i+1) * single_pool_num])
        else: partitioned_numlist.append(num_list[i * single_pool_num :])
        param_list.append((partitioned_numlist[i], folder_path, i, args.num_gpus, args.iscoco, args.device, is_cls_aware))
    proposals = pool.map(single_process, param_list)
    pool.close()
    pool.join()
    fin_proposals = []
    for i in range(n_processes):
        fin_proposals.extend(proposals[i])
    
    if point_input_method == 'mixture':
        assert len(fin_proposals) == len(num_list), 'The number of proposals is not fit. got %d instead of %d' %(len(proposals), len(num_list))
        storage_path = args.proposal_storage_path
        Path(storage_path + '/').mkdir(exist_ok=True, parents=True)
        os.chdir(storage_path + '/')
        with open(args.proposal_name, 'wb') as f:
            pkl.dump(fin_proposals, f)        
        
