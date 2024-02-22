"""
Perform dataset splitting with single process/gpu.
You'd better not use this file to split COCO dataset.
"""
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

def parse_args():
    parser = argparse.ArgumentParser("Perform dataset split.")
    parser.add_argument("--config", default="./configs/split/voc_split.yaml")
    parser.add_argument("--ckpt", default="./output/voc_baseline/model_0007999.pth")
    parser.add_argument("--save-path", default="./dataseed/VOC07_oicr_plus_split.txt")
    parser.add_argument("--k", default=2000, type=int)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_file = args.config
    ckpt_path = args.ckpt
    save_path = args.save_path
    k = args.k

    print("loading config file")
    cfg = get_cfg()
    add_ubteacher_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.freeze()

    print("loading state_dict")
    state_dict = torch.load(ckpt_path)["model"]

    state_dict_new = {}
    for key in state_dict:
        if "Student" in key:
            state_dict_new[key[13:]] = state_dict[key]

    model = build_model(cfg)
    result = model.load_state_dict(state_dict_new, strict=True)
    print(result)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS=0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    print("loading dataset")
    train_loader = build_detection_train_loader(cfg)
    dataset = train_loader.dataset.dataset
    index_list = []
    loss_list = []
    
    print("loss calculating")
    with EventStorage() as storage:
        for i in tqdm(range(len(dataset))):
            data = [dataset[i]]
            image_id = int(data[0]["image_id"])
            index_list.append(image_id)
            with torch.no_grad():
                loss_dict, _, _, _ = model(data)
            loss_list.append(
                (loss_dict["loss_cls"] + loss_dict["loss_box_reg"] + loss_dict["loss_rpn_cls"] + loss_dict["loss_rpn_loc"]).cpu().item()
            )
    
    all_losses = torch.Tensor(loss_list)
    sort_val, sort_ind = torch.sort(all_losses)
    l1 = []
    for i in sort_ind:
        l1.append(int(index_list[i]))
    
    # VOC dataset is ordered by image_ids
    imgid2id = {}
    # for i in tqdm(range(len(dataset))):
    #     img_id = int(dataset[i]["image_id"])
    #     imgid2id[img_id] = i
    for i in tqdm(range(len(index_list))):
        imgid2id[index_list[i]] = i
    
    for i in tqdm(range(len(l1))):
        l1[i] = imgid2id[l1[i]]
    
    # bisearch to find percent
    length = len(dataset)
    low = k / length
    high = (k+1) / length
    percent = -1
    while True:
        middle = round((low + high) / 2, 7)
        val = int(length * middle)
        if val == k:
            percent = middle * 100
            break
        elif val > k:
            high = middle
        else:
            begin = middle

    split_dict = {}
    
    l_dict = {}
    l_dict["1"] = l1[:k]
    split_dict[str(percent)] = l_dict
    print(f"The finded percent is: {percent}")
    json.dump(split_dict, open(save_path, "w"))

if __name__ == "__main__":
    main()



