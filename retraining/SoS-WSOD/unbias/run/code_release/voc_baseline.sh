python train_net.py \
--num-gpus 2 \
--resume \
--eval-only \
--config configs/code_release/voc_baseline.yaml \
--dist-url tcp://0.0.0.0:21727 \
OUTPUT_DIR ./output/voc_baseline \
SOLVER.BASE_LR 0.01 SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 TEST.VAL_LOSS False
