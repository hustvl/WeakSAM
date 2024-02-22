"""
Perform dataset splitting with multiple process/gpu.
"""
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader
from detectron2.utils.events import EventStorage
from detectron2.data import get_detection_dataset_dicts
import detectron2.utils.comm as comm
import torch.distributed as torch_dist
# from multiprocessing import freeze_support, Pool, Lock, Value
# import subprocess
import os
import sys
from detectron2.engine import launch
import json
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Perform dataset split.")
    parser.add_argument("--config", default="./configs/split/voc_split.yaml")
    parser.add_argument("--ckpt", default="./output/voc_baseline/model_0007999.pth")
    parser.add_argument("--save-path", default="./dataseed/VOC07_oicr_plus_split.txt")
    parser.add_argument("--k", default=2000, type=int)
    parser.add_argument("--gpu", default=8, type=int)
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(11451))

    args = parser.parse_args()
    return args

def log_message(msg):
    if comm.get_local_rank() == 0:
        print(msg)

index_list = []
loss_list = []

def main(args):
    config_file = args.config
    ckpt_path = args.ckpt

    log_message("loading config file")

    cfg = get_cfg()
    # add_ubteacher_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.freeze()

    log_message("loading state_dict")
    state_dict = torch.load(ckpt_path, map_location="cpu")["model"]


    model = build_model(cfg)
    result = model.load_state_dict(state_dict, strict=True)
    log_message(result)

    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = args.gpu
    cfg.freeze()

    log_message("loading dataset")
    train_loader = build_detection_train_loader(cfg)
    data_loader_iter = iter(train_loader)
    dataset = train_loader.dataset.dataset

    
    log_message("loss counting...")
    global index_list
    global loss_list
    base = 0
    d = {}
    with EventStorage() as storage:
        while True:
            log_message(f"{base}/{len(dataset)}")
            data = [dataset[(base+comm.get_local_rank())%len(dataset)]]
            base += args.gpu

            image_id = int(data[0]["image_id"])
            with torch.no_grad():
                loss_dict = model(data)
            loss_local = loss_dict["loss_cnt"].cpu().numpy()
            
            # gather messages
            gathered_image_id = [0 for _ in range(torch_dist.get_world_size())]
            gathered_loss = [0.0 for _ in range(torch_dist.get_world_size())]
            torch_dist.all_gather_object(gathered_image_id, image_id)
            torch_dist.all_gather_object(gathered_loss, loss_local)

            if comm.get_local_rank() == 0:
                for i in range(args.gpu):
                    if d.get(gathered_image_id[i], False):
                        continue
                    d[gathered_image_id[i]] = True
                    index_list.append(gathered_image_id[i])
                    loss_list.append(gathered_loss[i])
            if base >= len(dataset):
                break
    
    # gpu0 perform the following step
    if comm.get_local_rank() == 0:
        split_dict = {}
        for i in len(index_list):
            split_dict[index_list[i]] = loss_dict[i]

        print('The losses and image_ids are saved to the target file.')
        json.dump(split_dict, open(args.save_path, "w"))

if __name__ == "__main__":
    args = parse_args()
    num_gpus = args.gpu
    dist_url = args.dist_url

    save_path = args.save_path

    launch(
        main,
        num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=dist_url,
        args=(args, ),
    )
    
    