_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/schedules/schedule_2x.py', './_base_/default_runtime.py'
]

# model
num_classes = 4
model = dict(
    type='EnDiffDet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth'
    ),
    diff_cfg=dict(
        type='EnDiff',
        net=dict(type='PM', channels=16, time_channels=16),
        diffuse_ratio=0.6,
        sample_times=15,
        land_loss_weight=1,
        uw_loss_weight=10),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100))
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ResizeLqHqImages', 
        img_scale=(1333, 800),
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='FormatBundle'),
    dict(type='Collect', keys=['img', 'hq_img', 'gt_bboxes', 'gt_labels'], meta_keys=['filename', 'ori_filename',
         'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'hq_img_filename']),
]
test_pipeline = [
    dict(type='LoadLqHqImages'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800)],
        flip=True,
        transforms=[
            dict(type='ResizeLqHqImages', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'hq_img']),
            dict(type='Collect', keys=['img', 'hq_img'], meta_keys=['filename', 'ori_filename', 'ori_shape',
                 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'hq_img_filename']),
        ])
]
dataset_type = 'HqLqCocoDataset'
hq_img_prefix = './data/coco2017/train2017/'
data_root = './data/urpc2020/'
train_ann = './data/urpc2020/annotations/instances_train.json'
test_ann = './data/urpc2020/annotations/instances_test.json'
classes = ['echinus', 'holothurian',  'scallop', 'starfish']
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        ann_file=train_ann,
        classes=classes,
        img_prefix=data_root+'images/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        ann_file=test_ann,
        classes=classes,
        img_prefix=data_root+'images/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        hq_img_prefix=hq_img_prefix,
        ann_file=test_ann,
        classes=classes,
        img_prefix=data_root+'images/',
        pipeline=test_pipeline
    ))
evaluation = dict(interval=1, save_best='auto', classwise=True)

optimizer = dict(
    lr=0.0025,
    paramwise_cfg=dict(
        custom_keys=dict(diffusion=dict(lr_mult=0.1, decay_mult=5.0))))

epoch_iter = 2262
lr_config = dict(
    _delete_=True,
    policy='MulStep',
    step=[0, 6 * epoch_iter, 12 * epoch_iter, 20 * epoch_iter, 23 * epoch_iter],
    lr_mul=[1, 0.1, 1, 0.1, 0.01],
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    warmup_start=0)

runner = dict(max_epochs=24)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='TrainModeControlHook', train_modes=['sample', 'det'], num_epoch=[12, 12])
]
fp16 = dict(loss_scale=512.0)
