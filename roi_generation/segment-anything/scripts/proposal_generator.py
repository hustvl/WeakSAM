import pickle
import pandas as pd
import os
import numpy as np
import argparse


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

def get_index_list(dataset, ann_path):
    num_list = []
    if dataset == 'voc':
        f = open(ann_path, encoding='utf-8')
# for line in f:
#     num_list.append('COCO_val2014_' + line.strip())
        for line in f:
            num_list.append(line.strip())
    elif dataset == 'coco':
        f = open(ann_path, encoding='utf-8')
        for line in f:
            if 'train' in ann_path:
                num_list.append('COCO_train2014_' + line.strip())
            elif 'val' in ann_path:
                num_list.append('COCO_val2014_' + line.strip())
                
    return num_list
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', default=None)
    parser.add_argument('--ann-path', default=None)
    parser.add_argument('--saving-path', default=None)
    parser.add_argument('--saving-name', default=None)
    parser.add_argument('--dataset', default='voc')
    args = parser.parse_args()
    
    
    source = args.source_path
    ann = args.ann_path
    saving_path = args.saving_path
    saving_name = args.saving_name
    set = args.dataset

    num_list = get_index_list(set, ann)
    
    proposals = []
    prop = 0
    cnt = 0
    for i in range(len(num_list)):
        os.chdir(source + '/' + num_list[i])
        metadata = np.genfromtxt('./metadata.csv', delimiter=",", usecols=[2, 3, 4, 5], dtype=np.float32)
    # bounding_box_coor = metadata[['bbox_x0', 'bbox_y0', 'bbox_w', 'bbox_h']]
        bounding_box = metadata[1:, :]
        bounding_box_xyxy = xywh2xyxy(bounding_box)
        prop += len(bounding_box_xyxy)

        if len(bounding_box_xyxy) <= 5:
            print(num_list[i])
            cnt += 1
    # print(metadata[1].dtype)
        proposals.append(bounding_box_xyxy)
    # print(bounding_box_xyxy)

    print(prop / len(num_list))
    print(cnt)

    os.chdir(saving_path)
    with open(os.path.join(saving_path, saving_name), 'wb') as f:
        pickle.dump(proposals, f)
    