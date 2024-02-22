# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
import math
from wetectron.modeling import registry
from wetectron.modeling.backbone import resnet
from wetectron.modeling.poolers import Pooler
from wetectron.modeling.make_layers import group_norm
from wetectron.modeling.make_layers import make_fc
from wetectron.modeling.dropblock.drop_block import DropBlock2D

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        # self.head.layer4[0].conv1.stride = (1,1)
        # self.head.layer4[0].downsample[0].stride = (1,1)
        self.out_channels = head.out_channels


        # self.pooler = pooler
        # self.out_channels = 4096
        # #self.out_channels = 2048

        # if config.DB.METHOD == 'dropblock':
        #     self.dropblock = DropBlock2D(block_size=3, drop_prob=0.3)
        # self.sim_drop = DropBlock2D(block_size=1, drop_prob=0.3)

        # self.classifier = nn.Sequential(
        #     nn.Linear(7*7*2048, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(2048, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout()
        # )
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        ### modify fc layer - 4096###
        pooled_feats = self.pooler(x, proposals)
        # print(pooled_feats.shape)
        x = self.head(pooled_feats)
        ### modify fc layer - 4096###

        ### pool -> head -> avg 2048 ###
        #pooled_feats = self.pooler(x, proposals)
        #x = self.head(pooled_feats).mean(3).mean(2)
        ### pool -> head -> avg 2048 ###

        ### head -> pool -> avg ###
        #pooled_feats = self.pooler(x, proposals)
        #x = pooled_feats.mean(3).mean(2)
        ### head -> pool -> avg ###
        # pooled feats for object discovery. (Not used in MIST) ?
        return x, pooled_feats

    def forward_pooler(self, x, proposals):
        x = self.pooler(x, proposals)
        return x

    def forward_neck(self, x):
        ### added for ResC5 ###
        if isinstance(x, tuple):
            x = x[1]
        ### added for ResC5 ###
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        ### C4 ###
        #x = self.head(x).mean(3).mean(2)
        ### C4 ###
        return x

    ### add original dropblock ###
    def forward_dropblock(self, pooled_feat, proposals):
        db_pooled_feat = self.dropblock(pooled_feat)
        #x = db_pooled_feat.view(db_pooled_feat.shape[0], -1)
        #x = self.classifier(x)

        ### C4 ###
        #x = self.head(db_pooled_feat).mean(3).mean(2)
        ### C4 ###
        return db_pooled_feat

    def drop_pool(self, pooled_feats):
        db_pooled_feats = self.sim_drop(pooled_feats)
        return db_pooled_feats

    def noise_pool(self, pooled_feats):
        noise_pooled_feats = torch.normal(0, 1**2, size=pooled_feats.shape, device=pooled_feats[0].device) * pooled_feats + pooled_feats
        return noise_pooled_feats

    def flip_pool(self, pooled_feats):
        flip_pooled_feats = torch.flip(pooled_feats, (3,))
        return flip_pooled_feats

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet101Conv5ROIFeatureExtractor")
class ResNet101Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet101Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        '''stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )
        self.pooler = pooler
        self.head = head
        self.head.layer4[0].conv1.stride = (1,1)
        self.head.layer4[0].downsample[0].stride = (1,1)
        self.out_channels = head.out_channels
        '''
        self.pooler = pooler
        self.out_channels = 4096

        if config.DB.METHOD == 'dropblock':
            self.dropblock = DropBlock2D(block_size=3, drop_prob=0.3)
        self.sim_drop = DropBlock2D(block_size=1, drop_prob=0.3)

        self.classifier = nn.Sequential(
            nn.Linear(7*7*2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        ### modify fc layer - 4096###
        pooled_feats = self.pooler(x, proposals)
        x = self.classifier(torch.flatten(pooled_feats, 1))
        ### modify fc layer - 4096###

        ### pool -> head -> avg 2048 ###
        #pooled_feats = self.pooler(x, proposals)
        #x = self.head(pooled_feats).mean(3).mean(2)
        ### pool -> head -> avg 2048 ###

        ### head -> pool -> avg ###
        #pooled_feats = self.pooler(x, proposals)
        #x = pooled_feats.mean(3).mean(2)
        ### head -> pool -> avg ###

        return x, pooled_feats

    def forward_pooler(self, x, proposals):
        x = self.pooler(x, proposals)
        return x

    def forward_neck(self, x):
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        ### C4 ###
        #x = self.head(x).mean(3).mean(2)
        ### C4 ###
        return x

    ### add original dropblock ###
    def forward_dropblock(self, pooled_feat, proposals):
        #pooled_feat = self.pooler(x, proposals)
        db_pooled_feat = self.dropblock(pooled_feat)
        #x = db_pooled_feat.view(db_pooled_feat.shape[0], -1)
        #x = self.classifier(x)

        ### C4 ###
        #x = self.head(db_pooled_feat).mean(3).mean(2)
        ### C4 ###
        return db_pooled_feat

    def drop_pool(self, pooled_feats):
        db_pooled_feats = self.sim_drop(pooled_feats)
        return db_pooled_feats

    def noise_pool(self, pooled_feats):
        noise_pooled_feats = torch.normal(0, 1**2, size=pooled_feats.shape, device=pooled_feats[0].device) * pooled_feats + pooled_feats
        return noise_pooled_feats

    def flip_pool(self, pooled_feats):
        flip_pooled_feats = torch.flip(pooled_feats, (3,))
        return flip_pooled_feats

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet18Conv5ROIFeatureExtractor")
class ResNet18Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet18Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        '''stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )
        self.pooler = pooler
        self.head = head
        self.head.layer4[0].conv1.stride = (1,1)
        self.head.layer4[0].downsample[0].stride = (1,1)
        self.out_channels = head.out_channels
        '''
        self.pooler = pooler
        self.out_channels = 4096

        if config.DB.METHOD == 'dropblock':
            self.dropblock = DropBlock2D(block_size=3, drop_prob=0.3)
        self.spatial_dropblock = DropBlock2D(block_size=1, drop_prob=0.3)

        self.classifier = nn.Sequential(
            nn.Linear(7*7*2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        ### DRN WSOD structure. replace avg_pool -> two fc layer with C5 backbone###
        pooled_feats = self.pooler(x, proposals)
        x = self.classifier(torch.flatten(pooled_feats, 1))
        ### DRN WSOD structure###

        ### pool -> head -> avg 2048 ### -> C4 combination
        #pooled_feats = self.pooler(x, proposals)
        #x = self.head(pooled_feats).mean(3).mean(2)
        ### pool -> head -> avg 2048 ###

        ### head -> pool -> avg ###
        #x = self.head(x[0])
        #pooled_feats = self.pooler([x], proposals)
        #x = pooled_feats.mean(3).mean(2)
        ### head -> pool -> avg ###

        return x, pooled_feats

    def forward_pooler(self, x, proposals):
        x = self.pooler(x, proposals)
        return x

    def forward_neck(self, x):
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    ### add original dropblock ###
    def forward_dropblock(self, pooled_feat):
        #pooled_feat = self.pooler(x, proposals)
        db_pooled_feat = self.dropblock(pooled_feat)
        #x = db_pooled_feat.view(db_pooled_feat.shape[0], -1)
        #x = self.classifier(x)

        ### C4 ###
        #x = self.head(db_pooled_feat).mean(3).mean(2)
        ### C4 ###
        return db_pooled_feat

    def forward_dropblock_pool(self, pooled_feats):
        x = self.spatial_dropblock(pooled_feats)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        ### C4 ###
        #x = self.head(x).mean(3).mean(2)
        ### C4 ###
        return x

    def forward_noise_pool(self, pooled_feats):
        noise_pooled_feats = torch.normal(0, 1**2, size=pooled_feats.shape, device=pooled_feats[0].device) * pooled_feats + pooled_feats
        x = noise_pooled_feats.view(noise_pooled_feats.shape[0], -1)
        x = self.classifier(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
