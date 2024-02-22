tta_model = dict(
    type = 'DetTTAModel', 
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=300))  # 100 for DeformDETR

img_scales = [(488, 2000), (576, 2000), (688, 2000), (864, 2000), (1200, 2000)]

tta_pipeline = [
    dict(type='LoadImageFromFile',
        backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], 
        [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.),

        ], 
        [
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        [
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape',
                        'img_shape', 'scale_factor', 'flip',
                        'flip_direction'))
        ]])]