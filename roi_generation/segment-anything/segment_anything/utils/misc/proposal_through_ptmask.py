import numpy as np
import sys
import os
import torch
import pickle as pkl
import cv2
from amg import batched_mask_to_box
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

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


def transform_one_img(img_file):
    file_list = os.listdir(img_file)
    mask_list = []
    if not file_list:
        return torch.Tensor([])
    for file in file_list:
        current_mask = np.load(img_file + '/' + file)  # the mask shape: 3HW
        for mask in current_mask:
            mask_list.append(mask)
    mask_list = torch.Tensor(mask_list)
    batched_box = batched_mask_to_box(mask_list) # the batched box: Tensor with c * 4
    return batched_box

##### core: batched_mask_to_box(input a torch Tensor for batched generation of boxes around the masks)
##### the input shape: C * H * W mask, output C * 4 boxes
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--npy-path', default='/home/junweizhou/WeakSAM/segment-anything/point_mask/voc07', help='The npy root folder for mask storage')
    parser.add_argument('--npy-filename', default='k129_coarse', help='the filename of the point mask file.')
    parser.add_argument('--img-set-path', default='/home/junweizhou/WeakTr/WeakTr/voc07/trainval.txt')
    parser.add_argument('--pickle-storage-path', default='/home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/pam-k129')
    parser.add_argument('--pickle-file-name', default='k129_coarse.pkl')
    parser.add_argument('--sample-show', default=False, help='whether or not to show the boxes with the images.')
    parser.add_argument('--sample-storage', default='/home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/pam-k33/sample')
    parser.add_argument('--img-path', default='/home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages/')
    parser.add_argument('--concatenation', default=True, type=bool, help='whether to concatenate the single with the batch input outcomes.')
    parser.add_argument('--basic-sam-out', default='/home/junweizhou/WeakSAM/segment-anything/VOC07/tri1_test_32grid', help='The basic proposal output of SAMautomatic_mask_generator.')
    args = parser.parse_args()

    mixture_option = args.concatenation
    npy_path = args.npy_path
    img_set_path = args.img_set_path
    storage_path = args.pickle_storage_path
    sample_storage = args.sample_storage
    img_path = args.img_path

    f = open(img_set_path, encoding='utf-8')
    num_list = []
    data2write = []
    cnt = 0
    for line in f:
        num_list.append(line.strip())

    if mixture_option:
        for num in num_list:
            cnt += 1
            if cnt % 500 == 499:
                print('%d image processed, %d remaining.' % (cnt, len(num_list) - cnt))
            batched_box_single_img_batch = transform_one_img(npy_path + '/' + args.npy_filename + '_batch/' + str(num))
            batched_box_single_img_single = transform_one_img(npy_path + '/' + args.npy_filename + '_single/' + str(num))
            base_sam_path = args.basic_sam_out + '/' + str(num)
            metadata = np.genfromtxt(base_sam_path + '/metadata.csv', delimiter=",", usecols=[2, 3, 4, 5], dtype=np.float32)
            # in which the metadata is a numpy array.
            basic_proposals = metadata[1:, :]  # the basic proposals are also numpy arrays.
            basic_proposals_xyxy = xywh2xyxy(basic_proposals)
            basic_proposals_xyxy = torch.tensor(basic_proposals_xyxy)
            batched_box_single_img = torch.cat([batched_box_single_img_single, batched_box_single_img_batch, basic_proposals_xyxy], dim=0)
            batched_box_single_img = np.array(batched_box_single_img)
            data2write.append(batched_box_single_img)


            if args.sample_show and cnt <= 100:
                Path(sample_storage).mkdir(parents=True, exist_ok=True)
                imgpth = img_path + str(num) + '.jpg'
                image = cv2.imread(imgpth)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                # show_mask(masks[0], plt.gca())
                show_box(batched_box_single_img, plt.gca())
                plt.axis('off')
                plt.savefig(sample_storage + '/' + str(num))
                plt.close()

    else:
        for num in num_list:
            cnt += 1
            if cnt % 500 == 499:
                print('%d image processed, %d remaining.' % (cnt, len(num_list) - cnt))
            batched_box_single_img = transform_one_img(npy_path + '/' + args.npy_filename + '/' + str(num))
            # The box is in XYXY format. seeing the function show_box for more info, this box can be directly wrote into the pickle file.
            batched_box_single_img = np.array(batched_box_single_img)
            data2write.append(batched_box_single_img)
            if args.sample_show and cnt <= 100:
                Path(sample_storage).mkdir(parents=True, exist_ok=True)
                imgpth = img_path + str(num) + '.jpg'
                image = cv2.imread(imgpth)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                # show_mask(masks[0], plt.gca())
                show_box(batched_box_single_img, plt.gca())
                plt.axis('off')
                plt.savefig(sample_storage + '/' + str(num))
                plt.close()


    Path(storage_path + '/').mkdir(exist_ok=True, parents=True)
    os.chdir(storage_path + '/')
    with open(args.pickle_file_name, 'wb') as f:
        pkl.dump(data2write, f)




