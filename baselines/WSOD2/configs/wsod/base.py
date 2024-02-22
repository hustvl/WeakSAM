# model training and testing settings
train_cfg = dict(
    rcnn=dict())
test_cfg = dict(
    rcnn=dict(
        score_thr=0.0000,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=100))

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'WeakSAM/data/voc/'
img_norm_cfg = dict(
    mean=[104., 117., 124.], std=[1., 1., 1.], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadWeakAnnotations'),
    dict(type='LoadProposals'),
    dict(type='Resize', img_scale=[(488, 2000), (576, 2000), (688, 2000), (864, 2000), (1200, 2000)], keep_ratio=True, multiscale_mode='value'),
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
        # img_scale=(688, 2000),  # For fast validation .
        img_scale=[(488, 2000), (576, 2000), (688, 2000), (864, 2000), (1200, 2000)],
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
        ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007/',
        proposal_file = 'WeakSAM/roi_generation/segment-anything/peak_proposal/VOC07/voc07trainval.pkl',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        proposal_file = 'WeakSAM/roi_generation/segment-anything/peak_proposal/VOC07/voc07test.pkl',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007/',
        proposal_file = 'WeakSAM/roi_generation/segment-anything/peak_proposal/VOC07/voc07trainval.pkl',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')

# optimizer
optimizer = dict(
    type='Adam', 
    lr=8e-6,
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
    step=[36])
total_epochs = 50

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
load_from = 'WeakSAM/baselines/WSOD2/pretrain'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(2, 4)
