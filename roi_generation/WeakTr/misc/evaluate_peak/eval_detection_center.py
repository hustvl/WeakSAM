import math

import chainercv
import numpy as np

import pandas as pd
import argparse
import os

import torch
from PIL import Image

from tqdm import tqdm

import xml.dom.minidom
from pathlib import Path
import joblib

import cv2

from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

from tscam_bbox_extraction import get_bboxes

from sklearn.metrics import average_precision_score
from pascal_voc_evaluation import PascalVOCDetectionEvaluator

from peak_points_extraction import peak_extract

categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


# bus car train [5, 6, 18]
# chair sofa [8, 17]
# bicycle motorbike [1, 13]

def center_map_gen(center_map, x, y, sigma, g):
    """
    Center map generation. point to heatmap.
    Arguments:
        center_map: A Tensor of shape [H, W].
        x: A Int type value. x-coordinate for the center point.
        y: A Int type value. y-coordinate for the center point.
        sigma: A Int type value. sigma for 2D gaussian kernel.
        g: A numpy array. predefined 2D gaussian kernel to be encoded in the center_map.

    Returns:
        A numpy array of shape [H, W]. center map in which points are encoded in 2D gaussian kernel.
    """

    height, width = center_map.shape

    # outside image boundary
    if x < 0 or y < 0 or x >= width or y >= height:
        return center_map

    # upper left
    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
    # bottom right
    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

    c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
    a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

    cc, dd = max(0, ul[0]), min(br[0], width)
    aa, bb = max(0, ul[1]), min(br[1], height)

    center_map[aa:bb, cc:dd] = np.maximum(
        center_map[aa:bb, cc:dd], g[a:b, c:d])

    return center_map


def gaussian(sigma=6):
    """
    2D Gaussian Kernel Generation.
    """
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    return g


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)


def do_python_eval(args, name_list, threshold, min_distance, border, kernel):
    def compare(idx):
        name = name_list[idx]
        predict = None
        if args.pseudo_mask is not None:
            # this method is using mask for generation,  if not, looking at the npy interface below.
            predict_file = os.path.join(args.pseudo_mask, '%s.png' % name)
            predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)

        # npy file standard interface
        predict_file = os.path.join(args.predict_dir, '%s.npy' % name)
        predict_dict = np.load(predict_file, allow_pickle=True).item()
        # loading the npy content from the cam predicted files.

        cams = np.array([predict_dict[key] for key in predict_dict.keys()])
        label_key = np.array([key for key in predict_dict.keys()]).astype(np.uint8)
        # the cam's indices.

        orig_image = np.array(Image.open(os.path.join(args.ori_img_dir, name + ".jpg")))
        h, w, c = orig_image.shape
        adapt_weight = 1
        if args.adapt:
            adapt_weight = (math.sqrt(math.pow(h, 2) + math.pow(w, 2)) /
                            math.sqrt(math.pow(224, 2) + math.pow(224, 2)))
        if args.peak_file is not None:
            peak_txt = open(os.path.join(args.peak_file, "%s.txt" % name), 'w')
            # writing mode for peak file writing.
        for cam, key in zip(cams, label_key):
            center_map = np.zeros(cam.shape, dtype=np.float32)
            if predict is not None:
                # predict here is an array of the pseudo mask
                cam[predict != key + 1] = 0

            # based on box center points
            if args.point_type == "box":
                # using algo in tscam and generate bboxes.
                boxes = get_bboxes(cam, cam_thr=threshold, type=args.box_type)
                if args.img_dir is not None:
                    img = np.array(Image.open(os.path.join(args.img_dir, f"{name}_{key}.png")))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                coordinates = []
                for box in boxes:
                    coordinates.append([(box[1] + box[3]) // 2, (box[0] + box[2]) // 2])
                    # the coordinations are the center of boxes.
                    if args.img_dir is not None:
                        cv2.rectangle(img,
                                      (box[0], box[1]),
                                      (box[2], box[3]),
                                      color=(255, 0, 0))
                        cv2.rectangle(img,
                                      ((box[0] + box[2]) // 2 - 3, (box[1] + box[3]) // 2 - 3),
                                      ((box[0] + box[2]) // 2 + 3, (box[1] + box[3]) // 2 + 3),
                                      color=(255, 0, 0))

                if args.img_dir is not None and args.save_box_dir is not None:
                    cv2.imwrite(os.path.join(args.save_box_dir, f"{name}_{key}.png"), img)
                # saving the bounding boxes' relative coordinations.
            elif args.point_type == "pam":
                # generating multiple peak points from a CAM file.

                # smooth_cam = smoothing(torch.fromnumpy(cam).unsqueeze(0).unsqueeze(0))
                _, coordinate = peak_extract(torch.tensor(cam).unsqueeze(0).unsqueeze(0), kernel)
                coordinates = coordinate

            else:
                # single peak point generation.
                coordinates = peak_local_max(cam, min_distance=int(min_distance * adapt_weight),
                                             threshold_abs=threshold,
                                             exclude_border=int(border * adapt_weight), num_peaks=25)

            for x, y in coordinates:
                if cam[x][y] < threshold:
                    break
                center_map = center_map_gen(center_map, y, x, args.sigma, args.g)
                # plotting in the center map and display.
                if args.peak_file is not None:
                    peak_txt.write("%d %d %d %.3f\n" % (y, x, key, cam[x][y]))
            if args.pseudo_cam is not None:
                fname = os.path.join(args.pseudo_cam, name + '_' + str(key) + '.png')
                show_cam_on_image(orig_image, center_map, fname)

    # for j in range(len(name_list)):
    #     compare(j)
    joblib.Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(compare)(j) for j in range(len(name_list))]
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--list", default='data/voc12/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--gt-dir", default='data/voc12/SegmentationClassAug')
    parser.add_argument("--predict_dir", default=None, type=str)
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--n_jobs', default=10, type=int)
    parser.add_argument("--ori-img-dir", default='data/voc07/VOCdevkit/VOC2007/JPEGImages')

    # peak generations
    parser.add_argument("--point-type", type=str, default="ours") # pam or box
    parser.add_argument("--pseudo-mask", type=str, default=None)
    parser.add_argument("--pseudo-cam", default=None, type=str)
    parser.add_argument('--peak-file', default='weaktr_results/weaktr/crossattn-patchrefine-ms-peak', type=str)
    parser.add_argument('--point-thresh', default=0.7, type=float)  # the peak parameter to be changed.
    parser.add_argument("--adapt", action="store_true")

    # center map generation
    parser.add_argument("--sigma", type=int, default=6)

    # peak_local_max
    parser.add_argument('--min-distance', default=None, type=int)
    parser.add_argument("--border", default=0, type=int)
    parser.add_argument('--t', default=None, type=float)

    # pam peak point extraction
    parser.add_argument("--kernel", default=15, type=int)

    # box peak point
    parser.add_argument("--box-type", default="single", type=str)

    # peak evaluation
    parser.add_argument("--difficult", action="store_true")
    parser.add_argument("--img-dir", type=str, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-box-dir", default=None)

    args = parser.parse_args()
    args.g = gaussian(args.sigma)

    df = pd.read_csv(args.list, names=['filename'], converters={"filename": str})
    name_list = df['filename'].values
    if args.t is not None:
        args.t = args.t / 100.0
    if args.peak_file is not None:
        Path(args.peak_file).mkdir(parents=True, exist_ok=True)
    if args.save_dir is not None:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.save_box_dir is not None:
        Path(args.save_box_dir).mkdir(parents=True, exist_ok=True)
    if args.pseudo_cam is not None:
        Path(args.pseudo_cam).mkdir(parents=True, exist_ok=True)
    pascal_voc_eval = PascalVOCDetectionEvaluator(dataset_name="voc_2012_train",
                                                  dirname="data/voc12/VOCdevkit/VOC2012",
                                                  year=2007,
                                                  split=args.list.split("/")[-1].split(".")[0],
                                                  thing_classes=categories,
                                                  img_dir=args.ori_img_dir,
                                                  save_dir=args.save_dir,
                                                  difficult=args.difficult)

# args.curve should be set as False when single evaluation.
    if not args.curve:
        if args.predict_dir is not None:
            do_python_eval(args, name_list, args.t, args.min_distance, args.border, args.kernel)
        print("Point threshold: ", args.point_thresh)
        print("Peak File: ", args.peak_file)
        pascal_voc_eval.evaluate(args.peak_file, args.point_thresh)
        # the point thresh here is used for evalutation of generated peak points.
    else:
        l = []
        max_precision = 0.0
        max_recall = 0.0
        max_f1_score = 0.0
        max_map = 0
        best_pre_thr = 0
        best_pre_dis = 0
        best_pre_border = 0
        best_pre_kernel = 0
        best_re_thr = 0
        best_re_dis = 0
        best_re_border = 0
        best_re_kernel = 0
        best_f1_thr = 0
        best_f1_dis = 0
        best_f1_border = 0
        best_f1_kernel = 0
        best_map_thr = 0
        best_map_dis = 0
        best_map_border = 0
        best_map_kernel = 0
        min_distance = args.min_distance
        t = args.t
        border = args.border
        kernel = args.kernel
        for i in range(85, 100):
            t = i / 100.0  # getting the background threshold using a loop for curvy evaluation.
            for min_distance in range(200, 250):  # for kernel size = 2 * min_distance + 1
        # for kernel in range(181, 200, 2):
                for border in range(0, 10):  # border marking the distance between the point coordination and the contour.

                    do_python_eval(args, name_list, t, min_distance=min_distance,
                                border=border, kernel=kernel)
                    print("Point threshold: ", args.point_thresh)
                    print("Peak File: ", args.peak_file)
                    ret = pascal_voc_eval.evaluate(args.peak_file, args.point_thresh)
                    print('the ret of cur temple:', ret)
                    if ret["AP"][border * 5 + 5] > max_map:
                        max_map = ret["AP"][border * 5 + 5]
                        best_map_thr = t
                        best_map_dis = min_distance
                        best_map_border = border
                        best_map_kernel = kernel
                    if ret["Rec"][border * 5 + 5] > max_recall:
                        max_recall = ret["Rec"][border * 5 + 5]
                        best_re_thr = t
                        best_re_dis = min_distance
                        best_re_border = border
                        best_re_kernel = kernel
                    if ret["Prec"][border * 5 + 5] > max_precision:
                        max_precision = ret["Prec"][border * 5 + 5]
                        best_pre_thr = t
                        best_pre_dis = min_distance
                        best_pre_border = border
                        best_pre_kernel = kernel
                    if ret["F1"][border * 5 + 5] > max_f1_score:
                        max_f1_score = ret["F1"][border * 5 + 5]
                        best_f1_thr = t
                        best_f1_dis = min_distance
                        best_f1_border = border
                        best_f1_kernel = kernel

                    print("Background threshold: ", t)
                    # background threshold: using t to filter out those pixels with activation value larger than it.
                    print("Min distance: ", min_distance)
                    print("Border: ", border)
                    print("Kernel: ", kernel)
                    print(f"Best Precision: {max_precision}/{best_pre_thr}/{best_pre_dis}/{best_pre_border}/{best_pre_kernel}")
                    print(f"Best Recall: {max_recall}/{best_re_thr}/{best_re_dis}/{best_re_border}/{best_re_kernel}")
                    print(f"Best F1 Score: {max_f1_score}/{best_f1_thr}/{best_f1_dis}/{best_f1_border}/{best_f1_kernel}")
                    print(f"Best MAP: {max_map}/{best_map_thr}/{best_map_dis}/{best_map_border}/{best_map_kernel}")
