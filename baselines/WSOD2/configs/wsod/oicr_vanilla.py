_base_ = './base.py'

# model settings
model = dict(
    type='WeakRCNN',
    pretrained=None,
    backbone=dict(
        type='VanillaNet',
        dims=[128*4, 256*4, 512*4, 1024*4, 1024*4],
        #1024*4
        strides=[2,2,2,1]),  
    neck=None,
    roi_head=dict( 
        type='OICRRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIPool', output_size=7),
            out_channels=512,
            featmap_strides=[8]),
        bbox_head=dict(
            type='OICRHead',
            in_channels=4096,  # changed for aligning dimension of VanillaNet
            hidden_channels=4096,
            roi_feat_size=7,
            num_classes=20))
)

load_from = '/home/junweizhou/WeakSAM/WSOD2/pretrain/vanilla_6.pth'
work_dir = '/home/junweizhou/WeakSAM/WSOD2/work_dirs/oicr_vanilla/'
fp16 = dict(loss_scale=512.)
