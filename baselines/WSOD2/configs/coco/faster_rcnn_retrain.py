_base_ = [
    '../_base_/models/faster_rcnn_fpn_1x_res50.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',          
            pos_iou_thr=0.7,                
            neg_iou_thr=0.3,                
            min_pos_iou=0.3,                 
            ignore_iof_thr=-1),            
        sampler=dict(
            type='RandomSampler',          
            num=256,                       
            pos_fraction=0.5,               
            neg_pos_ub=-1,                  
            add_gt_as_proposals=False),    
        allowed_border=0,                  
        pos_weight=-1,                      
        smoothl1_beta=1 / 9.0,            
        debug=False),                   
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',            
            pos_iou_thr=0.5,               
            neg_iou_thr=0.5,               
            min_pos_iou=0.5,               
            ignore_iof_thr=-1),             
        sampler=dict(
            type='RandomSampler',          
            num=512,                        
            pos_fraction=0.25,              
            neg_pos_ub=-1,                   
            add_gt_as_proposals=True),        
        pos_weight=-1,                        
        debug=False))                        
test_cfg = dict(
    rpn=dict(                                 
        nms_across_levels=False,            
        nms_pre=2000,                         
        nms_post=2000,                      
        max_num=2000,                         
        nms_thr=0.7,                          
        min_bbox_size=0),                     
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)   
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)            # soft_nms参数
)

dataset_type = 'COCODetDataset'
data_root = 'data/coco/coco2014/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],
        keep_ratio=True, multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/pseudo_top2_train2014.json',
        img_prefix=data_root + 'train2014/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        pipeline=test_pipeline))

log_config = dict(
    interval=100,                          
    hooks=[
        dict(type='TextLoggerHook'),    
        # dict(type='TensorboardLoggerHook')
    ])

total_epochs = 12
evaluation = dict(interval=1, metric='bbox')

work_dir = '/home/junweizhou/WeakSAM/WSOD2/work_dirs/faster_rcnn_r50_fpn_1x' 
load_from = None                                
resume_from = None  