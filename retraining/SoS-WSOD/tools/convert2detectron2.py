# This function is used for converting into Faster RCNN model.

import torch
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description="conver the unbiased model to detectron2 model.")
parser.add_argument("source_path", help="the path of source model")
parser.add_argument("target_path", help="the path of target model")
parser.add_argument("--mode", "-m", choices=["teacher", "student"], default="teacher", help="choice to save teacher or student model")
args = parser.parse_args()

source_path = args.source_path
target_path = args.target_path

source_state_dict = torch.load(source_path)["model"]
new_state_dict = OrderedDict()
key_word = "modelTeacher" if args.mode == "teacher" else "modelStudent"
for key in source_state_dict.keys():
    if key_word in key:
        new_state_dict[key[13:]] = source_state_dict[key]

target_state = {
    "model": new_state_dict
}
torch.save(target_state, target_path)