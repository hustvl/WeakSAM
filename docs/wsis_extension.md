### Performing WSIS extension.
The extension part is based on Mask2Former codebase and please refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former) for implementation details. Note that you might need to install detectron2 following the Mask2Former repo and implement WSIS task.
Also, make a soft link to avoid repetitive download of datasets.

The generated WSIS dataset are listed below.
|  Dataset | MIST PGT |
|:------------|:-------------------:|
|  VOC2012    |    [trainaug set](https://pan.baidu.com/s/1s-KNqSDmELhy-AWDAOrt9g?pwd=husn)  |
| COCO2017    |   [train set](https://pan.baidu.com/s/1zYBdLWNw6XzQC8pOGXou3Q?pwd=8am4)   |


```bash
# Performing WSIS on VOC12 dataset.
python train_net_maskrcnn.py --num-gpus 4 \
  --config-file ./Mask2Former/configs/voc12/instance-segmentation/mask_rcnn_R_50_FPN_1x.yaml --dist-url tcp://0.0.0.0:12425 \
OUTPUT_DIR ./Mask2Former/results/mrcnn_voc

python train_net.py --num-gpus 4 \
  --config-file ./Mask2Former/configs/voc12/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --dist-url tcp://0.0.0.0:12425 \
OUTPUT_DIR ./Mask2Former/results/m2f_voc

# Performing WSIS on coco dataset.
python train_net_maskrcnn.py --num-gpus 4 \
  --config-file ./Mask2Former/configs/coco/instance-segmentation/mask_rcnn_R_50_FPN_1x_wsss.yaml \
OUTPUT_DIR ./Mask2Former/results/mrcnn_coco

python train_net.py --num-gpus 4 \
  --config-file ./Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_wsss.yaml --dist-url tcp://0.0.0.0:12425 \
OUTPUT_DIR ./Mask2Former/results/m2f_coco
```
