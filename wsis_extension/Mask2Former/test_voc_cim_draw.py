import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# You may need to restart your runtime prior to this, to let your installation take effect
# %cd /content/Mask2Former
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common libraries
import numpy as np
import cv2
import torch
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, Metadata
from detectron2.projects.deeplab import add_deeplab_config

# class mapping
# coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# for voc
coco_metadata = MetadataCatalog.get("voc_2012_val")

# import Mask2Former project
from mask2former import add_maskformer2_config

print(coco_metadata)
coco_metadata_dict = coco_metadata.as_dict()
print(coco_metadata_dict.keys())
coco_metadata_dict['stuff_classes'] = coco_metadata_dict['thing_classes']
# turn the dict into a class
coco_metadata = Metadata(**coco_metadata_dict)
print(coco_metadata)

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets.pascal_voc import CLASS_NAMES
import random
import os

VOC_DATASET_PATH = "/data/zhulianghui/yingyueli/Mask2Former/datasets/VOC2012/JPEGImages"
JSON_FILE_PATH = "/data/zhulianghui/yingyueli/Mask2Former/cim-visualization/MaskRCNN/Mask RCNN-VOC-val.json"
OUTPUT_ROOT_PATH = "/data/zhulianghui/yingyueli/Mask2Former/cim-visualization/cim-visualization"

voc_metadata = MetadataCatalog.get("voc12_val_coco")

# 获取数据集字典
dataset_dicts = load_coco_json(JSON_FILE_PATH, VOC_DATASET_PATH)
# print(len(dataset_dicts))
# print(dataset_dicts[0])

# 可视化
for d in dataset_dicts:  # 随机选择三张图片进行展示
    if "2007_002903.jpg" not in d["file_name"]:
        continue
    # print(d["file_name"])
    # break
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=voc_metadata, scale=1.2)
    # remove 'bbox' from item in d
    for item in d['annotations']:
        item.pop('bbox', None)
        item['category_id'] = item['category_id'] - 1
    out = visualizer.draw_dataset_dict(d)
    # save the image
    cv2.imwrite(os.path.join(OUTPUT_ROOT_PATH, os.path.basename(d["file_name"])), out.get_image()[:, :, ::-1])
    
    # cv2.imshow("VOC2012 Visualized", out.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    break