_base_ = './base.py'
# model settings
model = dict(
    type='WeakRCNN',
    pretrained=None,
    backbone=dict(type='VGG16'),
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
            in_channels=512,
            hidden_channels=4096,
            roi_feat_size=7,
            num_classes=80))
) 
work_dir = 'work_dirs/coco/oicr_vgg16/finproposal'
resume_from = '/home/junweizhou/WeakSAM/WSOD2/work_dirs/coco/oicr_vgg16/finproposal/epoch_12.pth'
evaluation = dict(interval=12, metric='bbox' )