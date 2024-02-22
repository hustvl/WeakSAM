CUDA_VISIBLE_DEVICES=2,3
./train_net.py \
--num-gpus 2 \
--resume \
--config /home/junweizhou/WeakSAM/SoS-WSOD/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
--dist-url tcp://0.0.0.0:21727 \
OUTPUT_DIR /home/junweizhou/WeakSAM/SoS-WSOD/pseudo_dropvoc07/drop2k_1_398 MODEL.INSTANCE_DROP.OPERATE_ITER 4000 MODEL.INSTANCE_DROP.THRESH_CLS 1.0 MODEL.INSTANCE_DROP.THRESH_REG 3.98 INPUT.MAX_SIZE_TEST 4000 INPUT.MIN_SIZE_TEST 2000 \

CUDA_VISIBLE_DEVICES=2,3 
./train_net.py \
--num-gpus 2 \
--resume \
--config /home/junweizhou/WeakSAM/SoS-WSOD/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
--dist-url tcp://0.0.0.0:21727 \
OUTPUT_DIR /home/junweizhou/WeakSAM/SoS-WSOD/pseudo_dropvoc07/drop2k_1_396 MODEL.INSTANCE_DROP.OPERATE_ITER 4000 MODEL.INSTANCE_DROP.THRESH_CLS 1.0 MODEL.INSTANCE_DROP.THRESH_REG 3.96 INPUT.MAX_SIZE_TEST 2000 INPUT.MIN_SIZE_TEST 1333 \

