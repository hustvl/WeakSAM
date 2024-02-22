### Ⅰ Dataset preparation & Structures
The data used in this project are structured as the directory below.

```
WeakSAM
    -data
        -voc
            -VOC2007
            -VOC2012
                -Annotations
                -pseudo_anns
                -...
        -coco
            -coco2014
                -annotations
                -train2014
                -val2014
```


```bash
# Here we take VOC dataset as an example.
cd WeakSAM/data/voc
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
# For aligning of the data structure, we move the main part of VOC dataset from 'data/VOCdevkit' to 'data' and deleted the folder 'VOCdevkit'
mv {path_to_WeakSAM}/WeakSAM/data/VOCdevkit/VOC2007 {path_to_WeakSAM}/WeakSAM/data
mv {path_to_WeakSAM}/WeakSAM/data/VOCdevkit/VOC2012 {path_to_WeakSAM}/WeakSAM/data
rm -r {path_to_WeakSAM}/WeakSAM/data/VOCdevkit
```

### Ⅱ Downloading SAM Ckpts and pretrained ViTs

In this work, we utilize the default SAM model for proposal generation and WeakTr(DeiT-S) for peak-points extraction. If interested, you can replace WeakTr with another WSSS model for more explorations.

Downloading SAM checkpoints with the official url: 
SAM ViT-H (default model) : [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

And WeakTr checkpoints with google drive: 
WeakTr(DeiT-S) for voc dataset : [DeiT-S VOC](https://drive.google.com/file/d/1uUHGLIDFm49Gh41Ddc8_T93-bRWnJzeF/view?usp=share_link)
WeakTr(DeiT-S) for coco dataset : [DeiT-S COCO](https://drive.google.com/file/d/1Q5UdcGCHFitSJXljZmTA4USKPpyZIVDw/view?usp=drive_link)

### Ⅲ Downloading pretrained VGG16 for WSOD2

# download vgg16 imagenet pre-trained weights
cd WeakSAM/baselines/WSOD2
mkdir -p pretrain
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10Vh2qFmGucO-9DZ3eY3HAvcAmtPFcFg2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10Vh2qFmGucO-9DZ3eY3HAvcAmtPFcFg2" -O pretrain/vgg16.pth && rm -rf /tmp/cookies.txt
