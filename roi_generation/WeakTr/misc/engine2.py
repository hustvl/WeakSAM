import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
from tscam_bbox_extraction import get_bboxes
from peak_points_extraction import peak_extract
from skimage.feature import peak_local_max
import joblib

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, patch_outputs, attn_outputs = model(samples)

            loss = F.multilabel_soft_margin_loss(outputs, targets)
            metric_logger.update(cls_loss=loss.item())

            ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
            metric_logger.update(pat_loss=ploss.item())
            loss = loss + ploss

            aloss = F.multilabel_soft_margin_loss(attn_outputs, targets)
            metric_logger.update(attn_loss=aloss.item())
            loss = loss + aloss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []
    patch_mAP = []
    attn_mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output, patch_output, attn_output = model(images)

            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)

            ploss = criterion(patch_output, target)
            loss += ploss
            patch_output = torch.sigmoid(patch_output)

            mAP_list = compute_mAP(target, patch_output)
            patch_mAP = patch_mAP + mAP_list
            metric_logger.meters['patch_mAP'].update(np.mean(mAP_list), n=batch_size)

            aloss = criterion(attn_output, target)
            loss += aloss
            attn_output = torch.sigmoid(attn_output)

            mAP_list = compute_mAP(target, attn_output)
            attn_mAP = attn_mAP + mAP_list
            metric_logger.meters['attn_mAP'].update(np.mean(mAP_list), n=batch_size)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        '* mAP {mAP.global_avg:.3f} patch_mAP {patch_mAP.global_avg:.3f} attn_mAP {attn_mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(mAP=metric_logger.mAP, patch_mAP=metric_logger.mAP, attn_mAP=metric_logger.mAP,
                losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP

############## Edited by Colez.
@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args, epoch=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(args.img_ms_list).readlines()  # the args.img_ms_list here is defined in the scripts.
    index = args.rank

    all_cnt = 0 # for peak search
    
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
    # of single image. s for distributed inference.
        if index >= len(img_list):
            continue
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        img_name = img_list[index].strip()
        index += args.world_size

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.cuda.amp.autocast():
            cam_list = []
            coarse_cam_list = []
            cross_attn_list = []
            multi_cross_attn_list = []
            cam_tensor = 0
            multi_tensor = 0
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[
                    3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                output, cams, patch_attn, coarse_cam, cross_attn, attn_weights = model(images, return_att=True, attention_type=args.attention_type)
                patch_attn = torch.sum(patch_attn, dim=0)
                b, head, k, c, m, n = attn_weights.shape
                if args.patch_attn_refine:

                    cams = torch.matmul(patch_attn.unsqueeze(1),
                                                cams.view(cams.shape[0], cams.shape[1],
                                                                    -1, 1)).reshape(cams.shape[0],
                                                                                    cams.shape[1],
                                                                                    w_featmap, h_featmap)
                                                                    
                    # attn_weights = torch.matmul(patch_attn.unsqueeze(1),
                    #                             attn_weights.view(b, head
                    #                                             , k, c, -1, 1)).reshape(b, head, k, c, 
                    #                                                                             w_featmap, h_featmap)
                    ## for cross attention patch refinement.
                ## Here representing the refinement branch of fine cam generation. Other attention maps neednt to be passed through. 
                cams = \
                    F.interpolate(cams, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                
                cams = cams.to(device)
                # ############## Added by Colez.
                # coarse_cam = \
                #     F.interpolate(coarse_cam, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                # cross_attn = \
                #     F.interpolate(cross_attn, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                ##############8.2 for multi cross attention generation
                # attn_weights = attn_weights.to(device)
                
               
                
                # attn_weights = attn_weights.view([b*head*k, c, m, n])
                
                # multi_cross_attn = \
                #     F.interpolate(attn_weights, size=(w_orig, h_orig), mode='bilinear', align_corners=False)
                # multi_cross_attn = multi_cross_attn.view(head * k, c, w_orig, h_orig)  # dim 6
                
                ##############
                
                # upsampling.
                cams = cams * target.clone().view(args.nb_classes, 1, 1)
                

                # coarse_cam = coarse_cam.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()
                # cross_attn = cross_attn.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()
                # multi_cross_attn = multi_cross_attn * target.clone().view(args.nb_classes, 1, 1)
                
                
                if s % 2 == 1:
                    cams = torch.flip(cams, [-1])
                    # coarse_cam = np.flip(coarse_cam, axis=-1)
                    # cross_attn = np.flip(cross_attn, axis=-1)
                    # multi_cross_attn = torch.flip(multi_cross_attn, [-1])
                    
                if s == 0:
                    cam_tensor = cams.detach()
                    # multi_tensor = multi_cross_attn.detach()
                else:
                    cam_tensor = torch.add(cam_tensor, cams)
                    # multi_tensor = torch.add(multi_tensor, multi_cross_attn)
                    
                cam_list.append(cams)
                

                # coarse_cam_list.append(coarse_cam)
                # cross_attn_list.append(cross_attn)
                # multi_cross_attn_list.append(multi_cross_attn)

            

            # sum_cam = np.sum(cam_list, axis=0)
            # sum_cam = torch.from_numpy(sum_cam)
            # sum_cam = sum_cam.unsqueeze(0).to(device)
            sum_cam = cam_tensor.unsqueeze(0)
            
            # ############## Added by Colez.
            # sum_coarse_cam = np.sum(coarse_cam_list, axis=0)
            # sum_coarse_cam = torch.from_numpy(sum_coarse_cam)
            # sum_coarse_cam = sum_coarse_cam.unsqueeze(0).to(device)
            
            # sum_cross_attn = np.sum(cross_attn_list, axis=0)
            # sum_cross_attn = torch.from_numpy(sum_cross_attn)
            # sum_cross_attn = sum_cross_attn.unsqueeze(0).to(device)
            # sum_multi_cross_attn = torch.tensor(multi_cross_attn_list).to(device)
            # sum_multi_cross_attn = torch.sum(sum_multi_cross_attn, dim=0)
            # sum_multi_cross_attn = multi_tensor.unsqueeze(0)
            
            ##############

            output = torch.sigmoid(output)

############## Edited by COlez.
#TODO@: Reformulate: using a score threshold to filter out foreground classes.
        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                print(target[b], type(target[b]))
                if (target[b].sum()) > 0:
                    coarse_cam_dict = {}
                    cross_attn_dict = {}
                    cam_dict = {}
                    multi_cross_attn_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b, cls_ind] > 0:  # Here tarbet[b] : ground true class labels.
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cam = sum_cam[b, cls_ind, :]
                            # coarse_cam = sum_coarse_cam[b, cls_ind, :]
                            # cross_attn = sum_cross_attn[b, cls_ind, :]
                            # multi_cross_attn = sum_multi_cross_attn[b, :, cls_ind, :]


                            cam = (cam - cam.min()) / (
                                    cam.max() - cam.min() + 1e-8)  # normalization.
                            # coarse_cam = (coarse_cam - coarse_cam.min()) / (
                            #         coarse_cam.max() - coarse_cam.min() + 1e-8)
                            # cross_attn = (cross_attn - cross_attn.min()) / (
                            #         cross_attn.max() - cross_attn.min() + 1e-8)
                        
                            # for i in range(multi_cross_attn.shape[0]):
                            #     multi_cross_attn[i] = (multi_cross_attn[i] - multi_cross_attn[i].min()) / (
                            #         multi_cross_attn[i].max() - multi_cross_attn[i].min() + 1e-8)
                            
                            
                            
                            cam = cam.cpu().numpy()
                            # coarse_cam = coarse_cam.cpu().numpy()
                            # cross_attn = cross_attn.cpu().numpy()
                            # multi_cross_attn = multi_cross_attn.cpu().numpy()
                            

                            cam_dict[cls_ind] = cam
                            # coarse_cam_dict[cls_ind] = coarse_cam
                            # cross_attn_dict[cls_ind] = cross_attn
                            # multi_cross_attn_dict[cls_ind] = multi_cross_attn
                            if args.attention_dir is not None and index < 30:
                                pth = args.attention_dir
                                file_name = img_name + '_' + str(cls_ind) + '.png'
                                Path(pth).mkdir(exist_ok=True, parents=True)
                                fname = os.path.join(args.attention_dir, file_name)
                                show_cam_on_image(orig_images[0], cam, fname)
                            # if args.attention_dir is not None and index < 0:
                            #     # specifically for multi_scale cross attention.
                            #     for i in range(multi_cross_attn.shape[0]):
                            #         file_name = img_name + '_' + str(cls_ind) + '_' + 'H' + str(i // k + 1) + '.png'
                            #         layer_num = 'layer' + str(i % k + 1)
                            #         if i < k:
                            #             pth = os.path.join(args.attention_dir, layer_num)
                            #             Path(pth).mkdir(exist_ok=True, parents=True)
                            #         fname = os.path.join(args.attention_dir, layer_num, file_name)
                            #         show_cam_on_image(orig_images[0], multi_cross_attn[i], fname)
                    Path(args.peak_file).mkdir(exist_ok=True, parents=True)
                    all_cnt += do_python_eval(args, cam_dict, img_list, args.t / 100, args.min_distance, \
                        args.border, args.kernel, index, is_multi=False, iscoco=True)
                    
                    torch.cuda.empty_cache()

                    # if args.cam_npy_dir is not None:
                    #     # np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)
                    #     Path(args.cam_npy_dir + '_coarse').mkdir(exist_ok=True, parents=True)
                    #     Path(args.cam_npy_dir + '_cross').mkdir(exist_ok=True, parents=True)
                    #     Path(args.cam_npy_dir + '_multicross').mkdir(exist_ok=True, parents=True)
                        # np.save(os.path.join(args.cam_npy_dir + '_coarse', img_name + '.npy'), coarse_cam_dict)
                        # np.save(os.path.join(args.cam_npy_dir + '_cross', img_name + '.npy'), cross_attn_dict)
                        # np.save(os.path.join(args.cam_npy_dir + '_multicross', img_name + '.npy'), multi_cross_attn_dict)
                        # the saving metric being 
                        # {cls_lable: cam, ...}
    
    avg_peak_num = all_cnt / len(img_list)
    print('average peak point num being:', avg_peak_num)                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)
    

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


def do_python_eval(args, predict_dict, name_list, threshold, min_distance, border, kernel, index, is_multi=False, iscoco=False):
    name = name_list[index - 1].strip()
    if iscoco:
        if 'val' in args.img_ms_list:
            name = 'COCO_val2014_' + str(name)
        else:
            name = 'COCO_train2014_' + str(name)
    predict = None

    # automatic attention map interface

    cams = np.array([predict_dict[key] for key in predict_dict.keys()])
    label_key = np.array([key for key in predict_dict.keys()]).astype(np.uint8)
    # the cam's indices.

    orig_image = np.array(Image.open(os.path.join(args.ori_img_dir, name + ".jpg")))
    if len(orig_image.shape) == 3:
        h, w, c = orig_image.shape
    else :
        h, w = orig_image.shape
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
            
        # use PAM point extraction method
        elif args.point_type == "pam":
            # generating multiple peak points from a CAM file.
            # smooth_cam = smoothing(torch.fromnumpy(cam).unsqueeze(0).unsqueeze(0))
            if is_multi:
                coordinates = []
                for i in range(cam.shape[0]):
                    _, coordinate = peak_extract(torch.tensor(cam[i]).unsqueeze(0).unsqueeze(0), kernel, device=args.device)
                    coordinates.append(coordinate)
            else:
                _, coordinate = peak_extract(torch.tensor(cam).unsqueeze(0).unsqueeze(0), kernel, device=args.device)
                coordinates = coordinate

        else:
            # single peak point generation.
            coordinates = peak_local_max(cam, min_distance=int(min_distance * adapt_weight),
                                            threshold_abs=threshold,
                                            exclude_border=int(border * adapt_weight), num_peaks=25)

        if is_multi:
            cnt = 0
            for i in range(cam.shape[0]):
                for x, y in coordinates[i]:
                    if cam[i][x][y] < threshold:  # the judgement here naming that the threshold means alot to the peak point generation
                        break
                        # args.g = gaussian(args.sigma)
                        # center_map = center_map_gen(center_map, y, x, args.sigma, args.g)
                        # # plotting in the center map and display.
                    else:
                        cnt += 1    
                    if args.peak_file is not None:
                        peak_txt.write("%d %d %d %.3f %d\n" % (y, x, key, cam[i][x][y], i))
        else:
            cnt = 0
            for x, y in coordinates:
                if cam[x][y] < threshold:  # the judgement here naming that the threshold means alot to the peak point generation
                    break
                else:
                    cnt += 1
                args.g = gaussian(args.sigma)
                center_map = center_map_gen(center_map, y, x, args.sigma, args.g)
                    # plotting in the center map and display.
                if args.peak_file is not None:
                    peak_txt.write("%d %d %d %.3f\n" % (y, x, key, cam[x][y]))
            if args.pseudo_cam is not None:
                fname = os.path.join(args.pseudo_cam, name + '_' + str(key) + '.png')
                show_cam_on_image(orig_image, center_map, fname)
    return cnt
