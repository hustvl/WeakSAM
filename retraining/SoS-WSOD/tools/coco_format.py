import json
# from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
# from detectron2.config import get_cfg
import torch
import numpy as np
import argparse
import copy
import os
from tqdm import tqdm
from typing import Dict,  List
import pickle as pkl
import re
import xml.etree.ElementTree as ET

def xywh2xyxy(box):
    x_tl = box[0]
    y_tl = box[1]
    w = box[2]
    h = box[3]
    box = [x_tl, y_tl, x_tl + w, y_tl + h]
    return box

def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    # print(filename)
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(sorted(annotation_paths)):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        ## Get image info from this api and img_info being the images key.
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        ann_list = generate_instance(detection_result, img_id, 2, tot_label_pos)
        
        output_json_dict['annotations'].extend(ann_list)
        # for obj in ann_root.findall('object'):
        #     ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
        #     ann.update({'image_id': img_id, 'id': bnd_id})
        #     output_json_dict['annotations'].append(ann)
        #     bnd_id = bnd_id + 1
    for item in output_json_dict['annotations']:
        item.update({'id': bnd_id})
        bnd_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)
        
image_set =  '/home/junweizhou/WeakTr/WeakTr/voc07/trainval.txt'
f = open(image_set, encoding='utf-8')
num_list = []
data2write = []
for line in f:
    num_list.append(line.strip())
ann_route = '/home/junweizhou/WeakSAM/WSOD2/data/voc/VOC2007/Annotations'
annotation_paths = []
for num in num_list:
    num = num + '.xml'
    cur = os.path.join(ann_route, num)
    annotation_paths.append(cur)
    
det_result = '/home/junweizhou/WSOD/OD-WSCL/MIST/SAM_voc07/outputlr1e-2/inference/voc_2007_trainval/bbox.json'
detection_result = json.load(open(det_result))

label_pth = '/home/junweizhou/WeakTr/WeakTr/voc07/cls_labels.npy'
tot_label_pos = np.load(label_pth, allow_pickle=True).item()
tot_label_pos_seq = sorted(tot_label_pos)


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def generate_instance(detection_result, image_id, topk, label_list):
    det = []
    for item in detection_result:
        if item['image_id'] == str(image_id).rjust(6, '0'): 
            det.append(item)
    lab = label_list[str(image_id).rjust(6, '0')]
    mem = np.zeros(20)
    ann_list = []
    for instance in det:
        ann = {'area': 100,
        'iscrowd': 0,
        'bbox': [0, 0, 0, 0],
        'category_id': 0,
        'ignore': 0,
        'segmentation': []}  # This script is not for segmentation}
        if lab[instance['category_id'] - 1] == 1:
            mem[instance['category_id'] - 1] += 1
            bbox = instance['bbox']
            bbox = [int(item) for item in bbox]
            ann['bbox'] = bbox
            ann['category_id'] = instance['category_id']
            ann['image_id'] = image_id
            ann_list.append(ann)
            
        if mem[instance['category_id'] - 1] == topk:
            lab[instance['category_id'] - 1] = 0
    return ann_list
            
labe2id = {
	"background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}
output_ = None
convert_xmls_to_cocojson(annotation_paths, labe2id, output_)