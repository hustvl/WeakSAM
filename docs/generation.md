
Due to the nuisance generation process, we strongly recommand using our generated proposals for WSOD training. 
All the proposals are uploaded via Google Drive.

|  Dataset | OICR proposal <br /> train/trainval | OICR proposal <br /> test/val | MIST proposal <br /> train/trainval | MIST proposal <br /> test/val |
|:------------:|:----------------:|:--------------:|:---------------:|:---------------:|
|  VOC2007 |    [oicr_trainval](https://drive.google.com/file/d/1gco2QWL6OZJ4hBOrbQPBXaXhSEGdFn8t/view?usp=drive_link)    |   [oicr_test](https://drive.google.com/file/d/1ApvX8GJOhCfwOnZ6hXf71qZEA2gnqxm3/view?usp=drive_link)   |    [mist_trainval](https://drive.google.com/file/d/17TKSr-EQhhO9M3ZUiQUUsMnJwolswWtq/view?usp=drive_link)    |   [mist_test](https://drive.google.com/file/d/1dfMmtGbrs67CnMVr0Z-LXqeFozMgA2AX/view?usp=drive_link)   |
| COCO2014 |    [oicr_train](https://drive.google.com/file/d/1-mSKQFOr8MBWDKAj4tZ5ssTZj_zenjU1/view?usp=drive_link)    |   [oicr_val](https://drive.google.com/file/d/1erGmyxOvBbZ5nVm9G77KXT0NUh5CA_5p/view?usp=drive_link)   |    [mist_train](https://drive.google.com/file/d/1CpCdf_4onAw8MCb5ThMB0E5O1Q07vw7u/view?usp=drive_link)    |   [mist_val](https://drive.google.com/file/d/1YJ2Mgu0GCwjkxRqz1-8SKLQij620pcjx/view?usp=drive_link)   |

### Peak-points generation

For more instructions, please follow the repo below:

[WeakTr: Exploring Plain Vision Transformer for Weakly-supervised Semantic Segmentation](https://github.com/hustvl/WeakTr)

#### Extracting peak-points from CAMs.
```bash
### Noting that the main differences in down below scripts are '--attnmap' , '--peak-file' and '--kernel', which can be adjusted to experiment on more parameter configs.

cd WeakSAM/roi_generation/WeakTr

### Also, for better efficiency, you can run the commands below in parallel with different GPUs.

#-trainval set generation-#
# Later they can be intergrated into another README, especially for peak-point generation.  

# Extracting from Fine CAM 
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/trainval.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc07/VOCdevkit/VOC2007/JPEGImages \
--attnmap fine \
	--device cuda:0 \
	--label-file-path voc07/cls_labels.npy \
--peak-file weaktr_results/VOC07-peak/fine/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 

# Extracting from Coarse CAM
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/trainval.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc07/VOCdevkit/VOC2007/JPEGImages \
--attnmap coarse \ 
	--device cuda:0 \
	--label-file-path voc07/cls_labels.npy \
--peak-file weaktr_results/VOC07-peak/coarse/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90

# Extracting from Cross attn map
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/trainval.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc07/VOCdevkit/VOC2007/JPEGImages \
--attnmap cross \
	--device cuda:0 \
	--label-file-path voc07/cls_labels.npy \
--peak-file weaktr_results/VOC07-peak/coarse/peak-pam-k129-t90 \
--point-type pam \
--kernel 129 \
--t 90

#------------------------------test set generation-------------------------------#

# Here we only take the fine cam as an example, other settings can be adjusted as above.
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/test.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results\
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc07/VOCdevkit/VOC2007/JPEGImages \
--device cuda:1 \
--attnmap fine \
	--attention-dir weaktr_results/VOC07/visible_results_test \
	--label-file-path voc07/cls_labels_test.npy \
--peak-file weaktr_results/VOC07-peak/fine/peak-pam-k17-t90_test \
--point-type pam \
--kernel 17 \
--t 90 \
--gen-test True

```

#### post processing for peak-points
```bash
# Cross attention peak-points from other datasets can also be clustered with the command modified.
python clustering_generation.py --image-set voc07/trainval.txt \
--image-path {your_path}/WeakSAM/data/voc/VOC2007/JPEGImages \
--point-path ./weaktr_results/VOC07-peak/cross/peak-pam-k129-t90 \
--peakfile ./weaktr_results/VOC07-peak/cross/peak-pam-k129-t90-1

```

#### Or use the scripts for convenience
```bash
# The above commands are integrated in this script.
bash ./extraction_scripts/extract_voc07.sh

```

### 3. Prompting SAM using peak-points

#### Spatial Sampling 
```bash

# This command is the original amg with multi-processin pool, which should save the time for generating.
cd ../segment-anything

#--------------------------------------voc trainval set---------------------------------------#
python ./scripts/pooled_amg.py --checkpoint './checkpoints/sam_vit_h_4b8939.pth' \
--model-type 'default' \
--input {your_path}/WeakSAM/data/voc/VOC2007/JPEGImages \
--output ./VOC07/trainvalset \
--partial-folder True \
--partial-txt {your_path}/WeakSAM/data/voc/VOC2007/ImageSets/Main/trainval.txt \
--n-processes 12 \

# Collecting from metadata and finally 

python ./scripts/proposal_generator.py --source-path ./VOC07/trainvalset \
--ann-path {your_path}/WeakSAM/data/voc/VOC2007/ImageSets/Main/trainval.txt \
--saving-path ./peak_proposal/VOC07/SAMbase \
--saving-name trainval-grid32.pkl \
--dataset voc \

# test set can replace the 'trainval' string with 'test' and others the same.


#--------------------------------------coco train set---------------------------------------#
python ./scripts/pooled_amg.py --checkpoint './checkpoints/sam_vit_h_4b8939.pth' \
--model-type 'default' \
--input {your_path}/WeakSAM/data/coco/coco2014/train2014 \
--output ./COCO14/train \
--partial-folder False \
--n-processes 12 \

python ./scripts/proposal_generator.py --source-path './COCO14/train' \
--ann-path {your_path}/WeakSAM/roi_generation/WeakTr/coco/train_id.txt \
--saving-path ./peak_proposal/COCO14/SAMbase \
--saving-name train_COCO14_grid32.pkl \
--dataset coco \
```

#### Prompting with generated peak-points

```bash
# Note that the command below are for GPUs with 24G memory, e.g. RTX3090, for other GPUs, --n-processes should be (--num-gpus * Memory/8)

# trainval set prompting 
# For more layers, adjusting the starting layer and ending layer.

# Prompting using FineCAM points

cd WeakSAM/roi_generation/segment-anything/segment_anything/utils

python pooled_proposal_generation.py --img-set-path {your_path}/WeakSAM/data/voc/VOC2007/ImageSets/Main/trainval.txt \
--peakfile-name coarse/peak-pam-k17-t90/ \
--txt-folder-path {your_path}/WeakSAM/roi_generation/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path {your_path}/WeakSAM/data/voc/VOC2007/JPEGImages \
--proposal-storage-path ./peak_proposal/VOC07/coarse \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

# Prompting using CoarseCAM points
python pooled_proposal_generation.py --img-set-path {your_path}/WeakSAM/data/voc/VOC2007/ImageSets/Main/trainval.txt \
--peakfile-name fine/peak-pam-k17-t90/ \
--txt-folder-path {your_path}/WeakSAM/roi_generation/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path {your_path}/WeakSAM/data/voc/VOC2007/JPEGImages \
--proposal-storage-path ./peak_proposal/VOC07/fine \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

# Prompting using Cross-attn points
python pooled_proposal_generation.py --img-set-path {your_path}/WeakSAM/data/voc/VOC2007/ImageSets/Main/trainval.txt \
--peakfile-name cross/peak-pam-k129-t90/ \
--txt-folder-path {your_path}/WeakSAM/roi_generation/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path {your_path}/WeakSAM/data/voc/VOC2007/JPEGImages \
--proposal-storage-path ./peak_proposal/VOC07/cross \
--proposal-name k129_t90_wopost.pkl \
--device 0 \
--starting-layer 4 \
--ending-layer 12 \
--is-multi True \
--num-gpus 4 \
--n-processes 12 \

# Do NMS without scores on proposals generated with Cross-attn points
python post_suppression.py --target-prop ./peak_proposal/VOC07/cross/k129_t90_wopost.pkl \
--threshold 0.9 \
--save-name k129_t90.pkl \
--save-path ./peak_proposal/VOC07/cross \


# Concat all the generated proposals
python proposal_concatenation.py --pickle-path ./peak_proposal/VOC07/ \
--storage-path ./peak_proposal/VOC07/ \
--file-name voc07trainval.pkl \
```

#### Or using scripts for one-step
```bash
# All integrated in one script.

cd WeakSAM/roi_generation
bash generation_voc07.sh

```

### Adjusting proposal into wetectron format.

```bash
# This file transforms original mmdet format proposals to wetectron format.
python pkl_trans.py --dataset voc \
--image-set trainval \
--ann-path {your_path}/WeakSAM/WSOD2/data/voc/VOC2007/ImageSets/Main \
--tar-proposals {your_path}/WeakSAM/segment-anything/peak_proposal/VOC07/voc07trainval.pkl \
--save-path {your_path}/WeakSAM/MIST/proposal/SAM/VOC07 \
--save-name voc07-trainvalboexes

#--------------------------------Below for COCO dataset conversion------------------------------#
python pkl_trans.py --dataset coco \
--ann-path {your_path}/WeakSAM/WSOD2/data/coco/coco2014/annotations/instances_train2014.json \
--tar-proposals {your_path}/WeakSAM/segment-anything/peak_proposal/COCO14/cocotrain.pkl \
--save-path {your_path}/WeakSAM/MIST/proposal/SAM/COCO \
--save-name coco-trainboxes.pkl

```
