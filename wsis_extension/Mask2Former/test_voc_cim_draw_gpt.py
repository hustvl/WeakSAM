import os
import torch
import numpy as np
import cv2
from collections import defaultdict
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from pycocotools import mask as mask_utils

import json

# 示例预测数据
data = json.load(open("/data/zhulianghui/yingyueli/Mask2Former/cim-visualization/MaskRCNN/Mask RCNN-VOC-val.json"))

VOC_DATASET_PATH = "/data/zhulianghui/yingyueli/Mask2Former/datasets/VOC2012/JPEGImages"
OUTPUT_ROOT_PATH = "/data/zhulianghui/yingyueli/Mask2Former/cim-visualization/cim-visualization"

# 预设的类别名称
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

# 设置 MetadataCatalog
MetadataCatalog.get("my_dataset").set(thing_classes=CLASS_NAMES)
metadata = MetadataCatalog.get("my_dataset")

# 根据 image_id 分组预测数据
predictions_by_image = defaultdict(list)
for pred in data:
    predictions_by_image[pred["image_id"]].append(pred)

# 转换预测数据为 Detectron2 格式的函数
def convert_predictions_to_instances(predictions, image_height, image_width):
    instances = Instances((image_height, image_width))
    # boxes = [pred["bbox"] for pred in predictions]
    scores = [pred["score"] for pred in predictions]
    classes = [pred["category_id"] - 1 for pred in predictions]
    masks = [mask_utils.decode(pred["segmentation"]) for pred in predictions]

    # instances.pred_boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    instances.scores = torch.tensor(scores, dtype=torch.float32)
    instances.pred_classes = torch.tensor(classes, dtype=torch.int64)
    instances.pred_masks = torch.tensor(masks, dtype=torch.uint8)

    # drop the pred_classes and pred_masks with scores lower than 0.7
    instances = instances[instances.scores > 0.4]

    # delete instances.scores
    # del instances.scores
    del instances._fields['scores']

    return instances

# 可视化每张图片的预测
for image_id, predictions in predictions_by_image.items():
    # 在 str(image_id) 的第五位插入 '_'
    image_id = str(image_id)
    image_id = image_id[:4] + '_' + image_id[4:]

    if "2007_009756" not in image_id:
        continue

    # 构建图片文件路径
    image_file = os.path.join(VOC_DATASET_PATH, image_id + '.jpg')
    
    # 加载图片
    image = cv2.imread(image_file)
    height, width = image.shape[:2]

    # 转换预测数据
    instances = convert_predictions_to_instances(predictions, height, width)

    # 可视化
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    output = v.draw_instance_predictions(instances.to("cpu"))

    # 显示或保存结果
    cv2.imwrite(os.path.join(OUTPUT_ROOT_PATH, image_id + ".jpg"), output.get_image()[:, :, ::-1])
    # cv2.imshow(f"Instance Segmentation {image_id}", output.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    # break
