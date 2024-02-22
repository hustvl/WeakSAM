import json
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
import os
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Adding multi-label messages into pseudo label files.")
    parser.add_argument("--pgt-temp", default="unbias/datasets/VOC2007/pseudo_labels/oicr_plus_voc_2007_{}.json")
    parser.add_argument("--dataset", default="voc2007", choices=('voc2007', 'voc2012', 'coco'))
    args = parser.parse_args()
    return args

def get_multi_class_label(dataset):
    multi_class_label = {}
    for data in tqdm(dataset):
        img_id = int(data["image_id"])
        anns = data["annotations"]
        label = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in label:
                label.append(cat_id)
        multi_class_label[str(img_id)] = label
    return multi_class_label

def get_multi_class_label_coco(dataset):
    gt_anns = {}
    for i in tqdm(range(len(dataset))):
        message = dataset[i]
        image_id = message["image_id"]
        gt_anns[image_id] = message["annotations"]
    
    multi_class_label = {}
    for img_id in tqdm(gt_anns):
        anns = gt_anns[img_id]
        label = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in label:
                label.append(cat_id)
        multi_class_label[img_id] = label
    return multi_class_label

######## EDITED BY COLEZ. originally the 'trainval' in this function is 'train'.
def add_voc07(pgt_temp):
    os.chdir("unbias/")
    trainvalset = get_detection_dataset_dicts(('voc_2007_trainval_gt', ))
    # valset = get_detection_dataset_dicts(('voc_2007_val', ))
    os.chdir("../")
    trainval_pgt = json.load(open(pgt_temp.format("trainval"), "r"))
    # val_pgt = json.load(open(pgt_temp.format("val"), "r"))

    trainval_multi_class_label = get_multi_class_label(trainvalset)
    # val_multi_class_label = get_multi_class_label(valset)

    trainval_pgt["multi_label"] = trainval_multi_class_label
    # val_pgt["multi_label"] = val_multi_class_label
    json.dump(trainval_pgt, open(pgt_temp.format("train"), "w"))
    # json.dump(val_pgt, open(pgt_temp.format("val"), "w"))
########


def add_voc12(pgt_temp):
    os.chdir("unbias/")
    trainset = get_detection_dataset_dicts(('voc_2012_train', ))
    valset = get_detection_dataset_dicts(('voc_2012_val', ))
    os.chdir("../")
    train_pgt = json.load(open(pgt_temp.format("train"), "r"))
    val_pgt = json.load(open(pgt_temp.format("val"), "r"))

    train_multi_class_label = get_multi_class_label(trainset)
    val_multi_class_label = get_multi_class_label(valset)

    train_pgt["multi_label"] = train_multi_class_label
    val_pgt["multi_label"] = val_multi_class_label
    json.dump(train_pgt, open(pgt_temp.format("train"), "w"))
    json.dump(val_pgt, open(pgt_temp.format("val"), "w"))

def add_coco(pgt_temp):
    os.chdir("unbias/")
    trainset = get_detection_dataset_dicts(('coco_2014_true_train', ))
    # valset = get_detection_dataset_dicts(('coco_2014_valminusminival', ))
    os.chdir("../")
    train_pgt = json.load(open(pgt_temp.format("train"), "r"))
    # val_pgt = json.load(open(pgt_temp.format("valminusminival"), "r"))

    train_class_label = get_multi_class_label_coco(trainset)
    # val_class_label = get_multi_class_label_coco(valset)

    train_multi_class_label = {}
    val_multi_class_label = {}
    for l in trainset:
        img_id = l["image_id"]
        train_multi_class_label[img_id] = train_class_label[img_id]
    
    # for l in valset:
    #     img_id = l["image_id"]
    #     val_multi_class_label[img_id] = val_class_label[img_id]
    
    train_pgt["multi_label"] = train_multi_class_label
    # val_pgt["multi_label"] = val_multi_class_label
    json.dump(train_pgt, open(pgt_temp.format("train"), "w"))
    # json.dump(val_pgt, open(pgt_temp.format("valminusminival"), "w"))

def main():
    args = parse_args()
    pgt_temp = args.pgt_temp
    dataset = args.dataset

    if dataset == "coco":
        add_coco(pgt_temp)
    elif dataset == "voc2007":
        add_voc07(pgt_temp)
    elif dataset == "voc2012":
        add_voc12(pgt_temp)
    else:
        raise ValueError(f"{dataset} is not supported.")

if __name__ == "__main__":
    main()