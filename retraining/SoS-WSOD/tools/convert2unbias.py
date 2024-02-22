import torch
from collections import OrderedDict
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description="conver the unbiased model to detectron2 model.")
parser.add_argument("source_path", help="the path of source model")
parser.add_argument("target_path", help="the path of target model")
args = parser.parse_args()

source_path = args.source_path
target_path = args.target_path
# source_path = "/mnt/data3/suilin/wsod/wsis/output/wsis/coco_irn_b_8_lr_008/model_0034999.pth"
# target_path = "/mnt/data3/suilin/wsod/wsis/output/wsis/coco_irn_b_8_lr_008/model_0034999_unbias.pth"

source_state_dict = torch.load(source_path, map_location="cpu")["model"]
new_state_dict = OrderedDict()
for key in source_state_dict.keys():
    new_state_dict["modelStudent."+key] = deepcopy(source_state_dict[key])
    new_state_dict["modelTeacher."+key] = deepcopy(source_state_dict[key])
target_state = {
    "model": new_state_dict
}
torch.save(target_state, target_path)

