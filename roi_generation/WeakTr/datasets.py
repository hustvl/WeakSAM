import os
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image

from pathlib import Path


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list


def load_img_name_list_VOC07(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = ['2007_' + img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

# Here for the evaluation label interface. Adding a case with no image labels.
def load_image_label_list_from_npy(img_name_list, label_file_path):
    label_list = []
    if label_file_path is None:  # For evaluation.
        for id in img_name_list:
            pseudo_empty = np.zeros(20)
            label_list.append(pseudo_empty)
        return label_list
    
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()

    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list
    # return [cls_labels_dict[img_name] for img_name in img_name_list ]


class COCOClsDataset(Dataset):
    def __init__(self, data_root, label_file_path="coco/cls_labels.npy", train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join("coco", f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.data_root = Path(data_root) / "coco" if "coco" not in data_root else Path(data_root)
        self.gt_dir = self.data_root / "voc_format" / "class_labels"
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.data_root, 'images', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_name_list)


class COCOClsDatasetMS(Dataset):
    def __init__(self, img_name_list_path, data_root, scales, label_file_path="coco/label_cls.npy", train=True, transform=None,
                 gen_attn=False, unit=1):
        self.img_name_list = load_img_name_list(img_name_list_path)
        if 'val' in label_file_path:
            self.istrain = False
        else:
            self.istrain = True
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.data_root = Path(data_root) / "coco" if "coco" not in data_root else Path(data_root)
        self.gt_dir = self.data_root / "voc_format" / "class_labels"
        self.transform = transform
        self.train = train
        self.unit = unit
        self.scales = scales
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.istrain:
            img = PIL.Image.open(os.path.join(self.data_root,'COCO_train2014_' +  name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.data_root,'COCO_val2014_' +  name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (
        int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])

            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12Dataset(Dataset):
    def __init__(self, data_root, label_file_path="voc12/cls_labels.npy", train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join("voc12", f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        data_root = Path(data_root) / "voc12" if "voc12" not in data_root else data_root
        self.data_root = Path(data_root) / "VOCdevkit" / "VOC2012"
        self.gt_dir = self.data_root / "SegmentationClass"
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.data_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12DatasetMS(Dataset):
    def __init__(self, img_name_list_path, data_root, scales, label_file_path="voc12/cls_labels.npy", train=True, transform=None, gen_attn=False, unit=1):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        data_root = Path(data_root) / "voc12" if "voc12" not in data_root else data_root
        self.data_root = Path(data_root) / "VOCdevkit" / "VOC2012"
        self.gt_dir = self.data_root / "SegmentationClassAug"
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.data_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (
        int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


class VOC07DatasetMS(Dataset):
    def __init__(self, img_name_list_path, data_root, scales, label_file_path="voc07/cls_labels.npy", train=True, transform=None, gen_attn=False, unit=1):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        data_root = Path(data_root) / "voc07" if "voc07" not in data_root else data_root
        self.data_root = Path(data_root) / "VOCdevkit" / "VOC2007"
        self.gt_dir = self.data_root / "SegmentationClass"
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.data_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (
        int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)


def build_dataset(is_train, data_set, args, gen_attn=False):
    transform = build_transform(is_train, args, gen_attn)
    dataset = None
    nb_classes = None
    if data_set == 'VOC07MS':
        dataset = VOC07DatasetMS(img_name_list_path=args.img_ms_list, data_root=args.data_path,
                                 scales=tuple(args.scales), label_file_path=args.label_file_path, 
                                 train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20

    if data_set == 'VOC12':
        dataset = VOC12Dataset(data_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(img_name_list_path=args.img_ms_list, data_root=args.data_path, label_file_path=args.label_file_path,
                                 scales=tuple(args.scales),
                                 train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif data_set == 'COCO':
        dataset = COCOClsDataset(data_root=args.data_path,
                                 train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 90
    elif data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(img_name_list_path=args.img_ms_list, data_root=args.data_path,
                                   scales=tuple(args.scales), label_file_path=args.label_file_path,
                                   train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 90

    return dataset, nb_classes


def build_transform(is_train, args, if_gen_attention_maps):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not if_gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
