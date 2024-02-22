CUDA_VISIBLE_DEVICES=2,3 python train_net.py \
--num-gpus 2 \
--config configs/code_release/coco_baseline.yaml \
--dist-url tcp://0.0.0.0:21727 \
--resume \
OUTPUT_DIR ./output/coco_baseline \
SOLVER.BASE_LR 0.01 SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 TEST.VAL_LOSS False
