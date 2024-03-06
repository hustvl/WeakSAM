import os.path

from chainercv.datasets import VOCInstanceSegmentationDataset
import numpy as np
from pycococreatortools import pycococreatortools
from PIL import Image
import json
import matplotlib.pyplot as plt
from pathlib import Path
from chainercv.utils import read_label

categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
              'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']
    
def generate_cocostyle_json(ins_dataset, output):
    ins_gt_ids = ins_dataset.ids
    coco_output = {"images": [], "annotations": [],
                   "categories": [{"supercategory": name, "name": name, "id": int(i+1)} for i, name in enumerate(categories)]}
    global_instance_id = 1
    for i, img_name in enumerate(ins_gt_ids):
        img_id = int(i + 1)
        gt_masks = ins_dataset.get_example_by_keys(i, (1,))[0]
        gt_labels = ins_dataset.get_example_by_keys(i, (2,))[0]
        img = ins_dataset.get_example_by_keys(i, (0,))[0]
        img_size = img.shape[1:]

        image_info = pycococreatortools.create_image_info(
            img_id, img_name + ".jpg", (img_size[1], img_size[0]))
        coco_output["images"].append(image_info)

        instance_id = 0
        for mask, class_id in zip(gt_masks, gt_labels):
            category_info = {'id': int(class_id + 1), 'is_crowd': False}
            # unique instance id
            annotation_info = pycococreatortools.create_annotation_info(
                instance_id + global_instance_id, img_id, category_info, mask, img_size[::-1], tolerance=0)

            instance_id += 1

            coco_output['annotations'].append(annotation_info)
        global_instance_id += instance_id

    with open(output, 'w') as outfile:
        json.dump(coco_output, outfile)


if __name__ == '__main__':
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    ins_train_dataset = VOCInstanceSegmentationDataset(split="train", data_dir=os.path.join(_root, "VOC2012"))
    ins_val_dataset = VOCInstanceSegmentationDataset(split="val", data_dir=os.path.join(_root, "VOC2012"))

    generate_cocostyle_json(ins_train_dataset, os.path.join(_root, "VOC2012", "voc_2012_train_cocostyle.json"))
    generate_cocostyle_json(ins_val_dataset, os.path.join(_root, "VOC2012", "voc_2012_val_cocostyle.json"))
