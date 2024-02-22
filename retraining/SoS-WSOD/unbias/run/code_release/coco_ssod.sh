python train_net.py \
  --num-gpus 4 \
  --config configs/code_release/coco_ssod.yaml \
  --dist-url tcp://0.0.0.0:21197 \
  MODEL.WEIGHTS ./output/coco_baseline_top2/model_final.pth \
  OUTPUT_DIR output/coco_ssod/ \
  SOLVER.BASE_LR 0.01 SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 SEMISUPNET.UNSUP_LOSS_WEIGHT 2.0 DATALOADER.SUP_PERCENT 36.56814 TEST.VAL_LOSS False