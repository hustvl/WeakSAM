# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

# from detectron2.data import MetadataCatalog
# from detectron2.utils import comm
# from detectron2.utils.file_io import PathManager

import pandas as pd
import argparse

from PIL import Image
import cv2

from pathlib import Path

class PascalVOCDetectionEvaluator:
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, dirname, year, split, thing_classes, img_dir, save_dir, difficult):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        # meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        # annotation_dir_local = PathManager.get_local_path(
        #     os.path.join(dirname, "Annotations/")
        # )
        assert year in [2007, 2012], year
        self._is_2007 = year == 2007
        if self._is_2007:
            dirname = "data/voc07/VOCdevkit/VOC2007"
        annotation_dir_local = os.path.join(dirname, "Annotations/")
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(dirname, "ImageSets", "Segmentation", split + ".txt")
        if self._is_2007:
            self._image_set_path = os.path.join(dirname, "ImageSets", 'Main', split + ".txt")
        self._class_names = thing_classes
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.difficult = difficult

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self, peak_file, point_thresh):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        self.reset()
        df = pd.read_csv(self._image_set_path, names=['filename'], converters={"filename": str})
        name_list = df['filename'].values
        for name in name_list:
            with open(os.path.join(peak_file, "%s.txt" % name), 'r') as peak_txt: # THIS LINE EDITTED BY COLEZ, ACHIEVING VOC07
                lines = peak_txt.readlines()
                split = [l.split(" ") for l in lines]
                for x, y, key, score in split:
                    if float(score) < point_thresh:
                        continue
                    self._predictions[int(key)].append(
                        f"{name} {float(score):.3f} {float(x):.1f} {float(y):.1f}"
                    )
        # all_predictions = comm.gather(self._predictions, dst=0)
        # if not comm.is_main_process():
        #     return
        predictions = defaultdict(list)
        # for predictions_per_rank in all_predictions:
        for clsid, lines in self._predictions.items():
            predictions[clsid].extend(lines)
        # del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            recs = defaultdict(list)
            precs = defaultdict(list)
            f1s = defaultdict(list)
            pos = 0
            pre = 0
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                if lines == [""]:
                    continue

                for thresh in range(5, 51, 5):
                    rec, prec, ap, npos, npre = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        cls_id,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                        img_dir=self.img_dir,
                        save_dir=self.save_dir,
                        diff=self.difficult
                    )
                    aps[thresh].append(ap)
                    recs[thresh].append(rec[-1])
                    precs[thresh].append(prec[-1])
                    f1s[thresh].append(2 * rec[-1] * prec[-1] / (prec[-1] + rec[-1]))
                pos += npos
                pre += npre

        ret = {}
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        mRec = {iou: np.mean(x) for iou, x in recs.items()}
        mPrec = {iou: np.mean(x) for iou, x in precs.items()}
        mf1 = {iou: np.mean(x) for iou, x in f1s.items()}
        ret["AP"] = mAP
        ret["Rec"] = mRec
        ret["Prec"] = mPrec
        ret["F1"] = mf1
        print(ret)
        print("GT Total: %d" % pos)
        print("Pred Total: %d" % pre)
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, cls_id,
            ovthresh=0.5, use_07_metric=False, img_dir=None, save_dir=None, diff=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        if diff:
            difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)  # len R: for the number of objects in each image,
        npos = npos + sum(~difficult)  # class aware metric calculating.
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    # Peak Points
    Peak = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 2)
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    Peak = Peak[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    bad_image_id = []

    img = {}
    if img_dir is not None:
        for imagename in imagenames:
            if imagename in image_ids:
                try:
                    image = np.array(Image.open(os.path.join(img_dir, f"{imagename}_{cls_id}.png")))
                except:
                    image = np.array(Image.open(os.path.join(img_dir, f"{imagename}.jpg")))
                img[imagename] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd): # for each image.
        R = class_recs[image_ids[d]]
        bb = Peak[d, :].astype(int)
        # ovmax = -np.inf
        ovmin = np.inf
        BBGT = R["bbox"].astype(int)
        flag = 0

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            # ixmin = np.maximum(BBGT[:, 0], bb[0])
            # iymin = np.maximum(BBGT[:, 1], bb[1])
            # ixmax = np.minimum(BBGT[:, 2], bb[2])
            # iymax = np.minimum(BBGT[:, 3], bb[3])
            # iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            # ih = np.maximum(iymax - iymin + 1.0, 0.0)
            # inters = iw * ih
            #
            # # union
            # uni = (
            #         (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
            #         + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
            #         - inters
            # )

            # overlaps = inters / uni
            # ovmax = np.max(overlaps)
            # jmax = np.argmax(overlaps)
            centerx = (BBGT[:, 0] + BBGT[:, 2]) // 2
            centery = (BBGT[:, 1] + BBGT[:, 3]) // 2

            len_x = BBGT[:, 2] - BBGT[:, 0]
            len_y = BBGT[:, 3] - BBGT[:, 1]
            # dis = (bb[0] - centerx) ** 2 + (bb[1] - centery) ** 2
            dis_x = bb[0] - centerx
            dis_y = bb[1] - centery
            # the definition of overlaps is calculated as the diagonal distance ** 2.
            # overlaps = dis / ((BBGT[:, 2] - BBGT[:, 0]) ** 2 + (BBGT[:, 3] - BBGT[:, 1]) ** 2)
            prop_x = np.abs(dis_x / len_x)
            prop_y = np.abs(dis_y / len_y)

            idx_x = np.where(prop_x > 0.5)
            idx_y = np.where(prop_y > 0.5)
            overlaps = (prop_x + prop_y) / 2
            overlaps[idx_x] = 1
            overlaps[idx_y] = 1

            ovmin = np.min(overlaps)
            jmin = np.argmin(overlaps)
        else:
            continue

        # if ovmax > ovthresh:
        if ovmin < ovthresh:
            if not R["difficult"][jmin]:
                if not R["det"][jmin]:
                    tp[d] = 1.0
                    R["det"][jmin] = 1
                    if img_dir is not None:
                        cv2.rectangle(img[image_ids[d]],
                                    (bb[0] - 3, bb[1] - 3),
                                    (bb[0] + 3, bb[1] + 3),
                                    color=(0, 255, 0))
                        cv2.rectangle(img[image_ids[d]],
                                    (BBGT[jmin, 0], BBGT[jmin, 1]),
                                    (BBGT[jmin, 2], BBGT[jmin, 3]),
                                    color=(0, 255, 0))
                else:
                # if this image has been detected.
                    flag = 1
                    fp[d] = 1.0
                    if img_dir is not None:
                        cv2.rectangle(img[image_ids[d]],
                                    (bb[0] - 3, bb[1] - 3),
                                    (bb[0] + 3, bb[1] + 3),
                                    color=(0, 255, 0))
        else:
            fp[d] = 1.0
            flag = 1
            if img_dir is not None:
                cv2.rectangle(img[image_ids[d]],
                            (bb[0] - 3, bb[1] - 3),
                            (bb[0] + 3, bb[1] + 3),
                            color=(255, 0, 0))
        if flag:
            bad_image_id.append(image_ids[d])
    if save_dir is not None:
        for imagename in imagenames:
            if imagename in image_ids:
                R = class_recs[imagename]
                BBGT = R["bbox"].astype(int)
                for b in range(len(BBGT)):
                    if not R["det"][b]:
                        bad_image_id.append(imagename)
                        cv2.rectangle(img[imagename],
                                      (BBGT[b, 0], BBGT[b, 1]),
                                      (BBGT[b, 2], BBGT[b, 3]),
                                      color=(0, 0, 255))
                if imagename in bad_image_id:
                    cv2.imwrite(os.path.join(save_dir, f"{imagename}_{cls_id}.png"), img[imagename])

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, float(npos), tp[-1] + fp[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--peak-file', default='../BESTIE/PAM/Peak_Points', type=str)
    parser.add_argument('--point-thresh', default=0.7, type=float)
    parser.add_argument("--img-dir", type=str, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--difficult", action="store_true")

    args = parser.parse_args()
    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                  'tvmonitor']
    if args.save_dir is not None:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    pascal_voc_eval = PascalVOCDetectionEvaluator(dataset_name="voc_2012_train",
                                                  dirname="data/voc12",
                                                  year=2012,
                                                  split="train",
                                                  thing_classes=categories,
                                                  img_dir=args.img_dir,
                                                  save_dir=args.save_dir,
                                                  difficult=args.difficult)
    print("Point threshold: ", args.point_thresh)
    print("Peak File: ", args.peak_file)
    pascal_voc_eval.evaluate(args.peak_file, args.point_thresh)
