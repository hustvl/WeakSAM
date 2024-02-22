## Baselines
For OICR, WSOD2 and MIST model, please refer to the repos below for more instructions.

[WSOD^2: Learning Bottom-up and Top-down Objectness Distillation for Weakly-supervised Object Detection](https://github.com/researchmm/WSOD2)

[Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection](https://github.com/NVlabs/wetectron)

### Training & Testing on MIST model

```bash
# Note that here we utilize Adam as our optimizer, which is not that sensitive to the learning rate changes.
# Applying SGD and adjusting the learning rate may get a better performance.

cd WeakSAM/baselines/MIST

# Training on VOC07 dataset. 
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
    --config-file "configs/voc/V_16_voc07.yaml" --use-tensorboard \
    OUTPUT_DIR MIST/SAM_voc07/exp_adam

# Testing on VOC07 dataset. 
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 114514 tools/test_net.py \
    --config-file "configs/voc/V_16_voc07.yaml" TEST.IMS_PER_BATCH 8 \
    OUTPUT_DIR MIST/SAM_voc07/exp_adam/inference \
    MODEL.WEIGHT MIST/SAM_voc07/exp_adam/model_final.pth
```

### Training & Testing on OICR model
```bash

cd WeakSAM/baselines/WSOD2

# Training on VOC07 dataset. 
bash tools/dist_train.sh configs/wsod/oicr_vgg16.py 4

# Testing on VOC07 dataset. 
bash tools/dist_test.sh configs/wsod/oicr_vgg16.py {path_to_ckpt} 4 --eval mAP

```

### Adaptive PGT generation.

```bash
cd WeakSAM/retraining/SoS-WSOD
# Generating Pseudo Annotations for VOC07 dataset.
python tools/adaptive_pgt.py --det-path {your_path}/WeakSAM/baselines/MIST/MIST/SAM_voc07/exp_adam/inference/voc_2007_trainval --save-path {your_path}/WeakSAM/data/voc/VOC2007/pseudo_ann --prefix pseudo --dataset voc2007 --t-keep 0.3

# Then should register a new dataset for pseudo training. 
# See 'WeakSAM/retraining/SoS-WSOD/detectron2/detectron2/data/datasets/builtin.py' for registering new datasets and its corresponding configs.
```
## Retraining 
### Retraining with Faster-RCNN model

```bash
# Retraining a plain Faster-RCNN with PGT on VOC07 dataset.
./train_net.py --num-gpus 4 \
--config-file {your_path}/WeakSAM/retraining/SoS-WSOD/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
--resume

# Evaluating Retrain performance on VOC07 dataset.
./train_net.py --num-gpus 4 \
--config-file {your_path}/WeakSAM/retraining/SoS-WSOD/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
--eval-only MODEL.WEIGHTS {path_to_ckpts}

# Retraining on COCO14 dataset.
./train_net.py --num-gpus 4 \
--config-file {your_path}/WeakSAM/retraining/SoS-WSOD/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_COCO14.yaml \
--resume

# Evaluating Retrain performance on COCO2014 dataset.
./train_net.py --num-gpus 4 \
--config-file {your_path}/WeakSAM/retraining/SoS-WSOD/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_COCO14.yaml \
--eval-only MODEL.WEIGHTS {path_to_ckpts}
```

For SoS-WSOD pipeline(STAGE3-SSOD), (e.g. VOC07), please follow the instructions in the below repo for more instructions.
We report the implementation of ours.


[Salvage of Supervision in Weakly Supervised Object Detection](https://github.com/suilin0432/SoS-WSOD)

