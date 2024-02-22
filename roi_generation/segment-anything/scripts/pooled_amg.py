# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List
import multiprocessing
from multiprocessing import shared_memory


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    default='./checkpoints/sam_vit_h_4b8939.pth',
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda:3", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

parser.add_argument(
    '--n-processes', 
    default=4,
    type=int
)

parser.add_argument(
    '--counting-num', 
    default=0,
    type=int
)

parser.add_argument(
    '--partial-folder',
    type = bool,
    default= False
)

parser.add_argument(
    '--partial-txt',
    type = str,
    default=None
)


amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=32,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=64,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=0.3,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=0.3,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=0.3,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        # for stotage convenience
        # mask = mask_data["segmentation"]
        # filename = f"{i}.png"
        # cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace, input, i) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    device = 0 + i
    device = 'cuda:' + str(device)
    _ = sam.to(device=device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    # if isinstance(input, list):
    #     targets = input
    # else:
    #     targets = [
    #         f for f in os.listdir(input) if not os.path.isdir(os.path.join(args.input, f))
    #         ]
    #     targets = [os.path.join(input, f) for f in targets]

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        if args.partial_folder:
            pth = args.partial_txt
            f = open(pth, encoding='utf-8')
            num_list = []
            for line in f:
                num_list.append(line.strip())
            targets = [args.input + '/' + str(num) + '.jpg' for num in num_list]
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]
            
    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if os.path.exists(save_base):
            continue
        print(f"Processing '{t}'...")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image)

        
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.partial_folder:
        if args.partial_folder:
            pth = args.partial_txt
            f = open(pth, encoding='utf-8')
            num_list = []
            for line in f:
                num_list.append(line.strip())
            targets = [args.input + '/' + str(num) + '.jpg' for num in num_list]
            all_list = targets
        else:
            targets = [
                f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
            ]
            targets = [os.path.join(args.input, f) for f in targets]
            all_list = targets
    else:
        all_list = [os.path.join(args.input, f) for f in os.listdir(args.input)]
    starting_gpu = 0
    
    print(len(all_list))
    partitioned_list = []
    single_pool_num = len(all_list) // args.n_processes
    print(single_pool_num)
    for i in range(args.n_processes):
        if i != args.n_processes - 1:
            partitioned_list.append(all_list[i * single_pool_num : (i+1) * single_pool_num])
        else: partitioned_list.append(all_list[i * single_pool_num :])
    pool = multiprocessing.Pool(processes=args.n_processes)
    # print(len(partitioned_list[0]), len(partitioned_list[3]))
    for i in range(args.n_processes):
        pool.apply_async(func=main, args=(args,partitioned_list[i], i))
    pool.close()
    pool.join()
    print('ALL DONE!')