python train_net.py \
  --num-gpus 4 \
  --resume \
  --config configs/code_release/voc_ssod.yaml \
  --dist-url tcp://0.0.0.0:21197 \
  MODEL.WEIGHTS ./output/voc_baseline/model_final.pth \
  OUTPUT_DIR output/voc_ssod/ \
  SOLVER.BASE_LR 0.01 SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 DATALOADER.SUP_PERCENT 39.92217 TEST.VAL_LOSS False