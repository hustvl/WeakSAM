# convert model
python tools/convert2detectron2.py ${MODEL_PATH} ${OUTPUT_PATH} -m [teacher(default) | student]

# tta test
python train_net_test_tta.py \
--num-gpus 8 \
--config configs/code_release/voc07_tta_test.yaml \
--dist-url tcp://0.0.0.0:21197 --eval-only \
MODEL.WEIGHTS ${MODEL_PATH} \
OUTPUT_DIR ${OUTPUT_DIR}