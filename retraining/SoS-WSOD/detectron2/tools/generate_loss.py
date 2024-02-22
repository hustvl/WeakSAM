"""
Perform dataset splitting with multiple process/gpu.
"""
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader, build_detection_test_loader
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
from add_insdrop import add_insdrop_config

def parse_args():
    parser = argparse.ArgumentParser("Perform dataset split.")
    parser.add_argument("--config", default="WeakSAM/SoS-WSOD/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml")
    parser.add_argument("--ckpt", default="WeakSAM/SoS-WSOD/pseudo_dropvoc07/model_0001999.pth")
    parser.add_argument("--gpu", default=4, type=int)
    parser.add_argument('--save-path', default='/WeakSAM/WeakSAM_ckpts/logits/loss_dict_top3_pgt_1k.json')
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(11451))

    args = parser.parse_args()
    return args


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    # the calculation format of recall is in the xyxy format.
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def log_message(msg):
    if comm.get_local_rank() == 0:
        print(msg)

index_list = []
loss_list = []
loss_list_positive = []
loss_list_negative = []
loss_list_neg75 = []
loss_list_neg25 = []
loss_cls_list_negative = []
loss_cls_list_neg25 = []
loss_cls_list_neg75 = []
loss_cls_list = []

loss_cls_bg_neg_list = []
loss_cls_all_list = []

def main(args):
    config_file = args.config
    ckpt_path = args.ckpt

    log_message("loading config file")

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_insdrop_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.freeze()

    log_message("loading state_dict")
    state_dict = torch.load(ckpt_path, map_location="cpu")["model"]

    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = args.gpu
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.9
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 50

    model = build_model(cfg)
    result = model.load_state_dict(state_dict, strict=True)
    log_message(result)





    log_message("loading dataset")
    train_loader = build_detection_train_loader(cfg)
    
    #


    #
    wsl_train_loader = get_detection_dataset_dicts((f'voc_2007_trainval_wsl',))
    gt_train_loader = get_detection_dataset_dicts((f'voc_2007_trainval_gt',))
    data_loader_iter = iter(train_loader)
    
    
    dataset = train_loader.dataset.dataset


    log_message("loss counting...")
    global index_list
    global loss_list
    global loss_list_positive
    global loss_list_negative
    global loss_list_neg75
    global loss_list_neg25
    
    global loss_cls_list_negative
    global loss_cls_list_neg25
    global loss_cls_list_neg75
    global loss_cls_list
    
    global loss_cls_all_list
    global loss_cls_bg_neg_list
    base = 0
    d = {}
    with EventStorage() as storage:
        while True:
            log_message(f"{base}/{len(dataset)}")
            
            data = [dataset[(base+comm.get_local_rank())%len(dataset)]]
            data_gt = [gt_train_loader[(base+comm.get_local_rank())%len(dataset)]][0]
            data_wsl = [wsl_train_loader[(base+comm.get_local_rank())%len(dataset)]][0]
            # data_gt = [dataset_gt[(base+comm.get_local_rank())%len(dataset_gt)]]
            base += args.gpu


            
            
            
            image_id = int(data[0]["image_id"])
            with torch.no_grad():
                # Here the loss dict should contain loss value and its corresponding coordinations.
                # In which we should partition these into two loss dicts.
                loss_dict = model(data)
            loss_local = loss_dict["loss_cnt"].cpu().numpy()
            proposals_fg = loss_dict['proposal_fg'].cpu().numpy()
            proposals = loss_dict['proposal_all'].cpu().numpy()
            loss_cls_fg = loss_dict['loss_cls_fg'].cpu().numpy()
            loss_cls_all = loss_dict['loss_cls'].cpu().numpy()
            fg_ind = loss_dict['fg_ind'].cpu().numpy().astype(np.int8)
            
            ##------------
            pgt_boxes = [annotation['bbox'] for annotation in data_wsl['annotations']]
            pgt_boxes =  np.array(pgt_boxes)
            assignment = bbox_overlaps(pgt_boxes, proposals_fg)
            assigned_indices = np.argmax(assignment, axis=0)
            assigned_boxes = [pgt_boxes[ind] for ind in assigned_indices]
            assigned_boxes = np.array(assigned_boxes)
            
            
            
            #foreground
            gt_boxes = [annotation['bbox'] for annotation in data_gt['annotations']]
            gt_boxes = np.array(gt_boxes)
            # ious = bbox_overlaps(gt_boxes, proposals_fg)
            #
            ious = bbox_overlaps(gt_boxes, assigned_boxes)
            #
            
            
            ious_all = bbox_overlaps(gt_boxes, proposals)
            maxi_ = np.max(ious_all, axis = 0)
            maxi_[fg_ind] = 0.0
            neg_indices = np.where(maxi_ > 0.1)
            
            
            
            
            positive_indices = np.where(ious >= 0.5)
            positive_075 = np.where(ious > 0.7)
            positive_025 = np.where(ious > 0.25)
            all_indices = np.linspace(0, ious.shape[1] - 1, ious.shape[1])
            all_indices = np.delete(all_indices, positive_indices[1]).astype(np.int8)
            all_indices1 = np.linspace(0, ious.shape[1] - 1, ious.shape[1])
            all_indices1 = np.delete(all_indices1, positive_075[1]).astype(np.int8)
            all_indices2 = np.linspace(0, ious.shape[1] - 1, ious.shape[1])
            all_indices2 = np.delete(all_indices2, positive_025[1]).astype(np.int8)


            loss_negative = loss_local[all_indices] 
            loss_neg_075 = loss_local[all_indices1]
            loss_neg_025 = loss_local[all_indices2]
            
            loss_cls_negative = loss_cls_fg[all_indices] 
            loss_cls_neg_075 = loss_cls_fg[all_indices1]
            loss_cls_neg_025 = loss_cls_fg[all_indices2]
            
            loss_cls_bg_neg = loss_cls_all[neg_indices]
            ##------------
            
            
            # gather messages
            gathered_image_id = [0 for _ in range(torch_dist.get_world_size())]
            gathered_loss = [0.0 for _ in range(torch_dist.get_world_size())]
            torch_dist.all_gather_object(gathered_image_id, image_id)
            torch_dist.all_gather_object(gathered_loss, loss_local)
            gathered_loss_cls = [0.0 for _ in range(torch_dist.get_world_size())]
            torch_dist.all_gather_object(gathered_loss_cls, loss_cls_fg)
            
            ##------------
            gathered_loss_negative = [0.0 for _ in range(torch_dist.get_world_size())]
            gathered_loss_negative_075 = [0.0 for _ in range(torch_dist.get_world_size())]
            gathered_loss_negative_025 = [0.0 for _ in range(torch_dist.get_world_size())]
            
            gathered_cls_loss_negative = [0.0 for _ in range(torch_dist.get_world_size())]
            gathered_cls_loss_negative_075 = [0.0 for _ in range(torch_dist.get_world_size())]
            gathered_cls_loss_negative_025 = [0.0 for _ in range(torch_dist.get_world_size())]
            
            torch_dist.all_gather_object(gathered_loss_negative, loss_negative)
            torch_dist.all_gather_object(gathered_loss_negative_075, loss_neg_075)
            torch_dist.all_gather_object(gathered_loss_negative_025, loss_neg_025)
            
            torch_dist.all_gather_object(gathered_cls_loss_negative, loss_cls_negative)
            torch_dist.all_gather_object(gathered_cls_loss_negative_075, loss_cls_neg_075)
            torch_dist.all_gather_object(gathered_cls_loss_negative_025, loss_cls_neg_025)
            
            
            gathered_cls_loss_negative_bg = [0.0 for _ in range(torch_dist.get_world_size())]
            
            torch_dist.all_gather_object(gathered_cls_loss_negative_bg, loss_cls_bg_neg)
            
            loss_cls_all
            
            gathered_cls_loss_all = [0.0 for _ in range(torch_dist.get_world_size())]
            
            torch_dist.all_gather_object(gathered_cls_loss_all, loss_cls_all)
            
            ##------------
            
            if comm.get_local_rank() == 0:
                for i in range(args.gpu):
                    if d.get(gathered_image_id[i], False):
                        continue
                    d[gathered_image_id[i]] = True
                    index_list.append(gathered_image_id[i])
                    loss_list.append(gathered_loss[i])
                    loss_list_negative.append(gathered_loss_negative[i])
                    loss_list_neg75.append(gathered_loss_negative_075[i])
                    loss_list_neg25.append(gathered_loss_negative_025[i])
                    
                    loss_cls_list.append(gathered_loss_cls[i])
                    loss_cls_list_negative.append(gathered_cls_loss_negative[i])
                    loss_cls_list_neg75.append(gathered_cls_loss_negative_075[i])
                    loss_cls_list_neg25.append(gathered_cls_loss_negative_025[i])
                    
                    loss_cls_bg_neg_list.append(gathered_cls_loss_negative_bg[i])
                    loss_cls_all_list.append(gathered_cls_loss_all[i])
            if base >= len(dataset):
                break
    
    # gpu0 perform the following step
    if comm.get_local_rank() == 0:
        split_dict = {}
        for i, item in enumerate(index_list):
            print(str(item))
            split_dict[str(item)] = {}
            split_dict[str(item)]['all'] = loss_list[i].tolist()
            split_dict[str(item)]['negative'] = loss_list_negative[i].tolist()
            split_dict[str(item)]['negative_75'] = loss_list_neg75[i].tolist()
            split_dict[str(item)]['negative_25'] = loss_list_neg25[i].tolist()
            
            split_dict[str(item)]['all_cls'] = loss_cls_list[i].tolist()
            split_dict[str(item)]['negative_cls'] = loss_cls_list_negative[i].tolist()
            split_dict[str(item)]['negative_cls_75'] = loss_cls_list_neg75[i].tolist()
            split_dict[str(item)]['negative_cls_25'] = loss_cls_list_neg25[i].tolist()
            
            split_dict[str(item)]['negative_cls_bg'] = loss_cls_bg_neg_list[i].tolist()
            split_dict[str(item)]['total_cls'] = loss_cls_all_list[i].tolist()
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
    
    
