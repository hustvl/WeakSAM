# -*- coding:utf-8 -*-
import json
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.config import get_cfg
import torch
import numpy as np
import argparse
import copy
import os
from tqdm import tqdm

# Noting: The dataset's category id should be (0 - clsnum - 1)
# Noting: The PGT ALGO does NOT have a limitation on pseudo gt's numbers(e.g. topk)
#@ TODO: Changing the algo of pgt into finding the top1 scoring prediction of each positive class and then others as vanilla pgt.
# Maybe selecting the predictions above threshold score and then keep the top1 of each positive class.

def parse_args():
    parser = argparse.ArgumentParser("Perform Adaptive PGT.")
    parser.add_argument("--det-path", default='WeakSAM/MIST/MIST/SAM_vocjoint/Aug/inference/voc_2012_trainaug')
    parser.add_argument("--save-path", default='WeakSAM/WSOD2/data/voc/VOC2012/pseudo_ann')
    parser.add_argument("--prefix", default='voc12_top1_pgt')
    parser.add_argument("--dataset", default='voc2012', choices=('voc2007', 'voc2012', 'coco'))
    parser.add_argument("--coco-path", default=None)
    parser.add_argument("--t-con", default=0.85)
    parser.add_argument("--t-keep", default=0.5, type=float)
    parser.add_argument("--use-diff", action="store_true", default= False)
    parser.add_argument('--topk', default=1, type=int)
    args = parser.parse_args()
    return args

def pgt_voc(det_path, save_path, prefix, t_con, t_keep, use_diff, year, topk):
    print("loading voc datasets...")
    use_diff = False

    # trainvalset = get_detection_dataset_dicts((f'voc_{year}_trainval_gt',))
    
    trainvalset = get_detection_dataset_dicts(('voc_2012_trainaug',))
    
    # trainset = get_detection_dataset_dicts((f'voc_{year}_train',))
    # valset = get_detection_dataset_dicts((f'voc_{year}_val',))
    os.chdir("../")

    print("loading voc detection results...")
    trainval_detection_result = json.load(open(f"{det_path}/bbox.json"))
    print(len(trainval_detection_result))
    print(len(trainvalset))


    # image_id 2 anns
    train_gt_anns = {}
    val_gt_anns = {}
    cnt = 0
    
    
    
    for i in range(len(trainvalset)):
        message = trainvalset[i]
        cnt += len(trainvalset[i]['annotations'])
        image_id = int(message["image_id"])
        
        #
        image_id = int(os.path.basename(os.path.normpath(message['file_name'].rstrip('.jpg'))))
        #
        
        train_gt_anns[image_id] = message["annotations"]
    print(cnt)
    # for i in range(len(valset)):
    #     message = valset[i]
    #     image_id = int(message["image_id"])
    #     val_gt_anns[image_id] = message["annotations"]

    trainval_result = {}
    val_result = {}
    for i in range(len(trainval_detection_result)):
        message = trainval_detection_result[i]
        image_id = int(message["image_id"].lstrip('0'))
        message["category_id"] = message["category_id"] - 1
        ######### Added for alignment of the format of bboxes. In the detection result there should be 
        box = message['bbox']
        # print('before: box', box) # XYWH MODE
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        # print('after: box', box)
        message['bbox'] = box # XYXY MODE.
        #########
        if image_id not in train_gt_anns:
            print(image_id)
            continue
        if not trainval_result.get(image_id, False):
            trainval_result[image_id] = [message]
        else:
            trainval_result[image_id].append(message)
    # for i in range(len(val_detection_result)):
    #     message = val_detection_result[i]
    #     image_id = message["image_id"]
    #     message["category_id"] = message["category_id"] - 1
    #     if image_id not in val_gt_anns:
    #         continue
    #     if not val_result.get(image_id, False):
    #         val_result[image_id] = [message]
    #     else:
    #         val_result[image_id].append(message)

    # multi-label messages of images
    trainval_class_dict = {}
    val_class_dict = {}
    for img_id in tqdm(train_gt_anns):
        anns = train_gt_anns[img_id]
        classes = []
        for ann in anns:
            c = ann["category_id"]
            if c not in classes:
                classes.append(c)
        trainval_class_dict[img_id] = classes

    # perform pgt
    print("performing pgt...")  
    # 1. filter by class label
    class_filter(trainval_result, trainval_class_dict, "trainval")
    # class_filter(val_result, val_class_dict, "val")
    # 2. pgt

    # diff_classes = [4, 5, 6, 8, 9, 15, 16]
    diff_classes = []
    # pgt(trainval_result, "trainval", t_con, t_keep, use_diff, diff_classes)
    pgt_topk(trainval_result, "trainval", t_con, t_keep, use_diff, None, 1, trainval_class_dict)
    print(len(trainval_result))
    print("saving results...")
    json.dump(trainval_result, open(f"{save_path}/{prefix}_voc_{year}_trainval.json", "w"))


def pgt_coco(det_path, save_path, prefix, t_con, t_keep, use_diff, coco_path, topk):
    print("loading coco datasets...")
    
    #@ TODO: print out those categories for visualization of coco dataset.
    #  In guess, it is 0-79, which is invalid.
    

    # trainset = get_detection_dataset_dicts(('coco_2014_true_train',))
    trainset = get_detection_dataset_dicts(('coco_2017_train_gt',))
    # valset = get_detection_dataset_dicts(('coco_2014_valminusminival',))
    os.chdir("../")
    print("loading coco detection results...")
    train_detection_result = json.load(open(f"{det_path}/bbox_instances.json"))
    print(len(train_detection_result))
    # val_detection_result = json.load(open(f"{det_path}/{prefix}coco_2014_valminusminival.json"))

    # image_id 2 anns
    train_gt_anns = {}
    # val_gt_anns = {}
    
    id2cat = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    
    for i in range(len(trainset)):
        message = trainset[i]
        image_id = message["image_id"]
        train_gt_anns[image_id] = message["annotations"]
        
    # for i in range(len(valset)):
    #     message = valset[i]
    #     image_id = message["image_id"]
    #     val_gt_anns[image_id] = message["annotations"]

    # filter images which do not contain objects
    train_result = {}
    val_result = {}
    for i in range(len(train_detection_result)):
        message = train_detection_result[i]
        image_id = message["image_id"]
        if image_id not in train_gt_anns:
            print(image_id)
            continue
        
        train_result[image_id] = message["instances"]
        #----------------------------------------------
        # For mmdet output predictions.
        #     item['category_id'] = id2cat[item['category_id']]
        #----------------------------------------------
    # for i in range(len(val_detection_result)):
    #     message = val_detection_result[i]
    #     image_id = message["image_id"]
    #     if image_id not in val_gt_anns:
    #         continue
    #     val_result[image_id] = message["instances"]

    # multi-label messages of images
    
    
    train_class_dict = {}
    val_class_dict = {}
    for img_id in tqdm(train_gt_anns):
        anns = train_gt_anns[img_id]
        classes = []
        for ann in anns:
            c = ann["category_id"]
            #### added for alignment of wetectron data IO
            c = id2cat[c]
            #### 
            if c not in classes:
                classes.append(c)
        train_class_dict[img_id] = classes

    # for img_id in tqdm(val_gt_anns):
    #     anns = val_gt_anns[img_id]
    #     classes = []
    #     for ann in anns:
    #         c = ann["category_id"]
    #         if c not in classes:
    #             classes.append(c)
    #     val_class_dict[img_id] = classes

    # perform pgt
    print("performing pgt...")
    # 1. filter by class label
    class_filter(train_result, train_class_dict, "train")
    # class_filter(val_result, val_class_dict, "val")
    diff_cls = []
    # 2. pgt   In COCO dataset, no difficult class is attributed.
    pgt(train_result, "train", t_con, t_keep, use_diff, diff_cls) # vanilla pgt(the implementation does not include any limitation on prediction num)
    # pgt_topk(train_result, "train", t_con, t_keep, use_diff, None, topk)   # In this func we reformulated the algo and added a limitation.
    # pgt(val_result, "val", t_con, t_keep, use_diff, None)

    # load gt annotations and replace gt annotations by pseudo labels
    print("saving results...")
    coco_train_gt_path = f"{coco_path}/annotations/instances_train2017.json"
    coco_train = json.load(open(coco_train_gt_path))
    # coco_val_gt_path = f"{coco_path}/annotations/instances_valminusminival2014.json"
    # coco_val = json.load(open(coco_val_gt_path))

    new_train_annotations = gen_annotations(train_result)
    coco_train["annotations"] = new_train_annotations
    
    # new_val_annotations = gen_annotations(val_result)
    # coco_val["annotations"] = new_val_annotations
    
    # save pseudo labels
    json.dump(coco_train, open(f"{save_path}/{prefix}coco_2017_train.json", "w"))
    # json.dump(coco_val, open(f"{save_path}/{prefix}coco_2014_valminusminival2014.json", "w"))

def gen_annotations(result):
    new_annotations = []
    INDEX = 0
    id2cat = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
    for img_id in tqdm(result):
        predictions = result[img_id]
        for prediction in predictions:
            new_annotations.append(
                {
                    "image_id": img_id,
                    "bbox": prediction["bbox"],
                    # "category_id": id2cat[prediction["category_id"]],
                    "category_id": prediction["category_id"],
                    "id": INDEX
                }
            )
            INDEX += 1
    return new_annotations

def contain_cal(a_, b_):
    a = copy.deepcopy(a_)
    b = copy.deepcopy(b_)
    a[2]+=a[0]
    a[3]+=a[1]
    b[2]+=b[0]
    b[3]+=b[1]
    c = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    area_c = max(0, c[2]-c[0]) * max(0, c[3]-c[1])
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    return area_c/(area_a+1e-6)

def pgt(result, split, t_con, t_keep, use_diff, diff_classes):  # later we can change the keeping number of each gt class.
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length before pgt: {length}")
    see_list = {}
    for img_id in result:
        see_list[img_id] = []
    
    for img_id in tqdm(result):
        predictions = result[img_id]
        
        # ------------------------Added.------------------------# for Normalization of scores in different classes.
        score_dict_high = {}
        score_dict_low = {}
        for prediction in predictions:
            if prediction['category_id'] not in score_dict_high.keys():
                score_dict_high[prediction['category_id']] = prediction['score']
            if score_dict_high[prediction['category_id']] < prediction['score']:
                score_dict_high[prediction['category_id']] = prediction['score']
                
        for prediction in predictions:
            if prediction['category_id'] not in score_dict_low.keys():
                score_dict_low[prediction['category_id']] = prediction['score']
            if score_dict_low[prediction['category_id']] > prediction['score']:
                score_dict_low[prediction['category_id']] = prediction['score']
                
        for prediction in predictions:
            prediction['score'] = prediction['score'] / (score_dict_high[prediction['category_id']] - score_dict_low[prediction['category_id']] + 1e-8)
        #------------------------Added.------------------------#
                            
        drop_list = []
        for i in range(len(predictions)):
            c = predictions[i]["category_id"]
            if (c not in see_list[img_id]):
                see_list[img_id].append(c)  # getting all the classes that have appeared in the prediction set.
                continue
            # TODO@: Seek whether the t_keep can be substituded by another algo.
            if predictions[i]["score"] < t_keep:  # filtering out those annotations with low scores.
                drop_list.append(i)
        for i in drop_list[::-1]:  # dropping the predictions in the drop list.
            result[img_id].pop(i)
    
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length in middle of pgt: {length}")

    for i in tqdm(result):
        anns = result[i] # anns being the predictions in a single image.
        save = [True] * len(anns)  # storing whether the annotation should be saved or not.
        bboxes = [b["bbox"] for b in anns]  # a list storing all the annotation contents.
        cats = [b["category_id"] for b in anns]
        new_anns = []
        for b_i in range(len(save)):
            for b_j in range(len(save)):
                if b_i == b_j or (cats[b_i] != cats[b_j]): continue
                if not use_diff and cats[b_i] in diff_classes: continue
                val = contain_cal(bboxes[b_i], bboxes[b_j])  # calculating the iou between two bboxes.
                if val >= t_con:
                    save[b_i] = False
        for j in range(len(save)):
            if save[j]:  # This algo does not include a limitation on the number of pseudo gt predicitons.
                new_anns.append(copy.deepcopy(anns[j]))
        result[i] = new_anns

    length = 0
    for i in result:
        length += len(result[i])

    print(f"{split} split length after pgt: {length}")


def pgt_topk(result, split, t_con, t_keep, use_diff, diff_classes, topk, cls_dict):  # for the limitation for each positive gt class.
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length before pgt: {length}")
    see_list = {}
    for img_id in result:
        see_list[img_id] = []
        
        
    cls_cnt = 0
    for img_id in tqdm(result):
        predictions = result[img_id]
        predictions.sort(key=lambda prediction:prediction['score'], reverse=True) # From small to large.
        class_label = cls_dict[img_id]
        cls_cnt += len(class_label)
        save_list = []
        cur = 0
        counter = {}
        for label in class_label:
            counter[label] = topk
        for i in range(len(predictions)):
            if predictions[i]['score'] > cur:
                cls = predictions[i]['category_id']
                if cls in counter.keys():
                    if counter[cls] == 0:
                        continue
                    save_list.append(i)
                    counter[cls] -= 1
        
        new_res = [result[img_id][i] for i in save_list]
        result[img_id] = new_res
    
    length = 0  
    for i in result:
        length += len(result[i])
    print(f"{split} split length in middle of pgt: {length}")
    
    print('total classes:', cls_cnt)
            
    # for img_id in tqdm(result):
    #     predictions = result[img_id]
    #     drop_list = []
    #     for i in range(len(predictions)):
    #         c = predictions[i]["category_id"]
    #         if (c not in see_list[img_id]):
    #             see_list[img_id].append(c)  # getting all the classes that have appeared in the prediction set.
    #             continue
    #         # TODO@: Seek whether the t_keep can be substituded by another algo.
    #         if predictions[i]["score"] < t_keep:  # filtering out those annotations with low scores.
    #             drop_list.append(i)
    #     for i in drop_list[::-1]:  # dropping the predictions in the drop list.
    #         result[img_id].pop(i)
    
    # length = 0
    # for i in result:
    #     length += len(result[i])
    # print(f"{split} split length in middle of pgt: {length}")

    # for i in tqdm(result):
    #     anns = result[i] # anns being the predictions in a single image.
    #     save = [True] * len(anns)  # storing whether the annotation should be saved or not.
    #     bboxes = [b["bbox"] for b in anns]  # a list storing all the annotation contents.
    #     cats = [b["category_id"] for b in anns]
    #     new_anns = []
    #     for b_i in range(len(save)):
    #         for b_j in range(len(save)):
    #             if b_i == b_j or (cats[b_i] != cats[b_j]): continue
    #             if not use_diff and cats[b_i] in diff_classes: continue
    #             val = contain_cal(bboxes[b_i], bboxes[b_j])  # calculating the iou between two bboxes.
    #             if val >= t_con:
    #                 save[b_i] = False
    #     for j in range(len(save)):
    #         if save[j]:  # This algo does not include a limitation on the number of pseudo gt predicitons.
    #             new_anns.append(copy.deepcopy(anns[j]))
                
        # result[i] = new_anns

    length = 0
    for i in result:
        length += len(result[i])

    print(f"{split} split length after pgt: {length}")
    

def class_filter(result, class_dict, split):
    length = 0
    print(len(result))
    for i in result:
        length += len(result[i])
    print(f"{split} split length before multi-class filter: {length}")
    for img_id in tqdm(result):
        predictions = result[img_id]
        gt_classes = class_dict[img_id]
        drop_list = []
        for i in range(len(predictions)):
            if predictions[i]["category_id"] not in gt_classes:
                # Only filtering out those predictions that are not in the gt classes.
                drop_list.append(i)
        for i in drop_list[::-1]:
            result[img_id].pop(i)
    length = 0
    for i in result:
        length += len(result[i])
    print(f"{split} split length after multi-class filter: {length}")
    



def main():
    args = parse_args()
    det_path = args.det_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")
    prefix = args.prefix
    dataset = args.dataset
    t_con = args.t_con
    t_keep = args.t_keep
    topk = args.topk
    use_diff = args.use_diff
    if dataset == "coco":
        coco_path = args.coco_path
        pgt_coco(det_path, save_path, prefix, t_con, t_keep, use_diff, coco_path, topk)
    elif "voc" in dataset:
        year = dataset[3:]
        pgt_voc(det_path, save_path, prefix, t_con, t_keep, use_diff, year, topk)
    else:
        raise ValueError(f"{dataset} is not supported.")

if __name__ == "__main__":
    main()
