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