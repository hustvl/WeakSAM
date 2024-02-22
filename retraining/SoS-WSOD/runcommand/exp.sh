
CUDA_VISIBLE_DEVICES=2,3 python WeakSAM/retraining/SoS-WSOD/detectron2/tools/train_net.py \
--num-gpus 2 \
--resume \
--config WeakSAM/retraining/SoS-WSOD/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
--dist-url tcp://0.0.0.0:21727 \
OUTPUT_DIR WeakSAM/retraining/SoS-WSOD/pseudo_dropvoc07_top2/plain\ 


CUDA_VISIBLE_DEVICES=2,3  python WeakSAM/retraining/SoS-WSOD/detectron2/tools/train_net.py \
--num-gpus 2 \
--resume \
--config WeakSAM/retraining/SoS-WSOD/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
--dist-url tcp://0.0.0.0:21727 \
OUTPUT_DIR WeakSAM/retraining/SoS-WSOD/pseudo_dropvoc07_top1/plain
