## Installation
Since we integrated several repos in this project, we need to build independent conda environments to reproduce our experiments.

Note that all the dependencies are built upon RTX3090 GPUs. The first environment(i.e. **Env1**) is built for [MIST](https://github.com/NVlabs/wetectron), [Segment-anything](https://github.com/facebookresearch/segment-anything) and [WeakTr](https://github.com/hustvl/WeakTr). And the second one(i.e. **Env2**) is for [WSOD2](https://github.com/researchmm/WSOD2) and [SoS-WSOD](https://github.com/suilin0432/SoS-WSOD/tree/main)(detectron2). If training DINO, please refer to [mmdetection](https://github.com/open-mmlab/mmdetection) official repo for instructions and we included its source code in 'WeakSAM/retraining/mmdetection'. 
We will illustrate these environments in the following parts.

## Env1. 
The first environment is for MIST, segment-anything and WeakTr. If you found not compatible, please refer to the original repos for more instructions.
### Requirements:
- Python 3.8
- Pytorch 1.8.1
- Torchvision 0.9.1
- numpy 1.24.3

### Getting Started:
```bash
# Creating current conda environment
conda create -n WeakSAM python==3.8
conda activate WeakSAM

# MIST dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python tensorboardX pycocotools

# installing pytorch using conda, and we here utilizing CUDA 11.1 as an example. If you change the CUDA version, please refer to the original repos for more instructions.
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge

# Cloning whole project
git clone https://github.com/Colezwhy/WeakSAM.git
cd WeakSAM/baselines/MIST

# building apex for mist
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../


python setup.py build develop

# building segment-anything dependency.
cd ../../roi_generation/segment-anything
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx

cd ../WeakTr
pip install -r requirements.txt
```


## Env2.
This environment is for WSOD2 and SoS-WSOD(detectron2).
### Requirements
- Python 3.7
- Pytorch 1.7.0
- Torchvision 0.8.0

### Getting started
```bash
# Creating environment
conda create -n wsod2 python==3.7.0
conda activate wsod2

cd WeakSAM/baselines/WSOD2

# building requirements for mmdet
pip install -r requirements.txt

# installing pytorch with CUDA 11.0 and mmcv-full 1.2.6
pip install mmcv-full==1.2.6+torch1.7.0+cu110 -f https://download.openmmlab.com/mmcv/dist/index.html
# installing mmdet
python setup.py install


cd ../../retraining/SoS-WSOD/detectron2
# installing detectron2 for retraining.
python3 -m pip install -v -e .
```






