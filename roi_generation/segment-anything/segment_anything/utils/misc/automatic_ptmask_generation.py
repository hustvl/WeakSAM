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
    mask_list = torch.Tensor(mask_list)
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


def batch_readtxt(folder_path, img_label, starting_layer=1, ending_layer=12, is_multi_cross=False):
    whole_pth = folder_path + str(img_label) + '.txt'
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


def inference_single_image(img, points_list, model, point_input_method, mask_type=False):
    model.set_image(img)
    if point_input_method == 'single':
        masks_batch = []
        scores_batch = []
        logits_batch = []
        points = points_list
        label = np.array([1])
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset reading options
    parser.add_argument('--txt-folder-path', default='/home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC07-peak/', type=str)
    parser.add_argument('--ori-image-path', default='/home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages/', type=str)
    # parser.add_argument('--img-set-path', default='/home/junweizhou/Datasets/VOC/VOC2012/ImageSets/Segmentation/train.txt')
    parser.add_argument('--img-set-path', default='/home/junweizhou/WeakTr/WeakTr/voc07/test.txt')
    # sample storage options
    parser.add_argument('--sample-storage', default=False, help='whether to display the generated mask and store it in a folder')
    parser.add_argument('--store-path', default='/home/junweizhou/WeakTr/WeakTr/weaktr_results/visible/', type=str)
    parser.add_argument('--save-as', default='sample_k129_batch_coarse', help='folder for saving masks with images.(sample)')

    # mask generation options(obligatory storage)
    parser.add_argument('--mask-type', default=True, type=str)
    parser.add_argument('--peakfile-name', default='fine/peak-pam-k33-t90-test/')
    parser.add_argument('--mask-storage-path', default='/home/junweizhou/WeakSAM/segment-anything/point_mask/voc07/k129_single_coarse')
    parser.add_argument('--point-input-type', default='mixture', help='Whether to use one point or '
                                                                       'a batch for inference in one time.'
                                                                     '(options: single, batch, mixture)')
    parser.add_argument('--proposal-storage-path', default='/home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/pam-k33', help='for the storage of proposals.')
    parser.add_argument('--proposal-name', default='k33_test.pkl')
    parser.add_argument('--device', default=None)
    parser.add_argument('--starting-layer', default=1, type=int)
    parser.add_argument('--ending-layer', default=12, type=int)
    parser.add_argument('--is-multi', default=False)
    parser.add_argument('--num-gpus', default=1, type=int)
    ## TODO@: creating a process pool for parallel running.
    args =parser.parse_args()

    save_filename = args.save_as
    peakfile = args.peakfile_name
    txt_folder = args.txt_folder_path
    img_path = args.ori_image_path
    display_option = args.sample_storage
    # mask_storage = args.mask_storage_path
    mask_storage = None

    point_input_method = args.point_input_type

    if mask_storage is not None:
        Path(mask_storage).mkdir(parents=True, exist_ok=True)

    if display_option:
        assert args.store_path is not None, 'The sample path for storage is not given.'
        store_path = args.store_path
        Path(store_path + save_filename).mkdir(parents=True, exist_ok=True)
    else:
        store_path = None

    # sam model initialization
    sam_checkpoint = '../../checkpoints/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    device = args.device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    print('------------------------loading sam model------------------------- ' + model_type)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    #######
    # peak_of_singleimg = batch_readtxt(folder_path=txt_folder, img_label= '2007_000039' )
    # print(peak_of_singleimg)
    img_set_txt = args.img_set_path
    f = open(img_set_txt, encoding='utf-8')
    num_list = []
    for line in f:
        num_list.append(line.strip())
    proposals = []
    folder_path = txt_folder + peakfile

    start_time = time.perf_counter()
    for i in range(len(num_list)):
        if i % 10 == 1:
            cur_time = time.perf_counter()
            elapsed = (cur_time - start_time) / 60 # minutes
            print('-------------------- %d image processed, %d remaining, elapsed time: %4f min' %(i, len(num_list) - i, elapsed))
        # print('processing ' + str(num_list[i]))
        if display_option:
            Path(store_path + save_filename + '/' + str(num_list[i])).mkdir(parents=True, exist_ok=True)
        point_single, point_sorted = batch_readtxt(folder_path=folder_path, img_label=num_list[i])
        # image related
        imgpth = img_path + str(num_list[i]) + '.jpg'
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



        if point_input_method == 'single':
            cnt = 0
            masks, scores, logits = inference_single_image(image, point_single, predictor, point_input_method, args.mask_type)
            for mask in masks:
                cnt += 1
                store_mask(mask, mask_storage, num_list[i], str(cnt))

        elif point_input_method == 'batch':
            for cls in cls_num:
                pt_s = point_sorted[cls]
                pt_s = np.array(pt_s)
                labels = np.full(len(pt_s), 1)
                masks, scores, logits = inference_single_image(image, pt_s, predictor, args.mask_type)
                store_mask(masks, mask_storage, num_list[i], cls)
                # the masks are stored in the npy file with shape 3*H*W(original shape of SAM output)
            
        elif point_input_method == 'mixture':
            # single
            masks, scores, logits = inference_single_image(image, point_single,  predictor, 'single',args.mask_type)
            batched_single_prop = transform_one_img(masks)  # A tensor for return.
            if display_option:  
                lb = np.array([1])
                
            # batch  或许不要做这个class-aware prompting(in multihead cross attention map)?
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

            a, b, c = prop_img.shape    
            prop_img = prop_img.view(a * b, c)

            prop_img = np.array(prop_img)
            proposals.append(prop_img)
            
            
    if point_input_method == 'mixture':
        assert len(proposals) == len(num_list), 'The number of proposals is not fit. got %d instead of %d' %(len(proposals), len(num_list))
        storage_path = args.proposal_storage_path
        Path(storage_path + '/').mkdir(exist_ok=True, parents=True)
        os.chdir(storage_path + '/')
        with open(args.proposal_name, 'wb') as f:
            pkl.dump(proposals, f)        
        
