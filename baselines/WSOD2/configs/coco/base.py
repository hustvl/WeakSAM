# model training and testing settings
train_cfg = dict(
    rcnn=dict())
test_cfg = dict(
    rcnn=dict(
        score_thr=0.0000,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))

# dataset settings
dataset_type = 'WeakCOCODataset'
data_root = '/home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadWeakAnnotations', num_classes=80),
    dict(type='LoadProposals'),
    dict(type='Resize', img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],
        keep_ratio=True, multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels', 'proposals']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(2000, 800),
        img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2014.json',
        # ann_file=data_root + 'VOC2012/train_aug_id.txt',
        img_prefix=data_root + 'train2014/',
        proposal_file = '/home/junweizhou/WeakSAM/segment-anything/peak_proposal/COCO14/cocotrain.pkl',
        # proposal_file = '/home/junweizhou/WeakSAM/segment-anything/peak_proposal/COCO14/MCGtrain.pkl',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        proposal_file = '/home/junweizhou/WeakSAM/segment-anything/peak_proposal/COCO14/cocoval.pkl',
        # proposal_file = '/home/junweizhou/WeakSAM/segment-anything/peak_proposal/COCO14/MCGval.pkl',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2014.json',
        img_prefix=data_root + 'train2014/',
        proposal_file = '/home/junweizhou/WeakSAM/segment-anything/peak_proposal/COCO14/cocotrain.pkl',
        # proposal_file = '/home/junweizhou/WeakSAM/segment-anything/peak_proposal/COCO14/MCGval.pkl',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(
    type='Adam', 
    lr=1e-5,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        bias_decay_mult=0.,
        bias_lr_mult=2.,
        custom_keys={
            'refine': dict(lr_mult=10),
        })
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

checkpoint_config = dict(interval=4)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrain/vgg16.pth'

workflow = [('train', 1)]

