from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from ubteacher import add_ubteacher_config
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader
from detectron2.utils.events import EventStorage
from detectron2.data import get_detection_dataset_dicts
import json
import argparse
# python generate_base_split.py --config configs/code_release/voc_baseline.yaml --save-path ./dataseed/test.txt
def parse_args():
    parser = argparse.ArgumentParser("Perform dataset split.")
    parser.add_argument("--config", default="./configs/split/voc_split.yaml")
    parser.add_argument("--save-path", default="./dataseed/VOC07_all.txt")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config_file = args.config
    save_path = args.save_path
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()

    datasets = get_detection_dataset_dicts(cfg.DATASETS.TRAIN)
    length = len(datasets)
    target = length - 1
    low, high = 0, 100
    percent = None
    while True:
        middle = round((low + high) / 2, 7)
        val = int(middle / 100 * length)
        if val == target:
            percent = middle
            break
        elif val < target:
            low = middle
        else:
            high = middle
    
    split_dict = {str(percent):{"1": list(range(target))}}
    json.dump(split_dict, open(save_path, "w"))