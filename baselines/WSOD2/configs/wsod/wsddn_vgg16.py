_base_ = './base.py'
# model settings
model = dict(
    type='WeakRCNN',
    pretrained=None,
    backbone=dict(type='VGG16'),
    neck=None,
    roi_head=dict(
        type='WSDDNRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIPool', output_size=7),
            out_channels=512,
            featmap_strides=[8]),
        bbox_head=dict(
            type='WSDDNHead',
            in_channels=512,
            hidden_channels=4096,
            roi_feat_size=7,
            num_classes=20))
)
work_dir = 'work_dirs/wsddn_vgg16/'
