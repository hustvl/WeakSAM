_base_ = './base.py'
# model settings
model = dict(
    type='WeakRCNN',
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        xattn=True,
        swiglu=True,
        rope=True,
        pt_hw_seq_len=16,   # 224/14
        intp_freq=True),
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
            num_classes=20))
)
work_dir = '/home/fcy/WSOD2/work_dirs/oicr_vt/'
