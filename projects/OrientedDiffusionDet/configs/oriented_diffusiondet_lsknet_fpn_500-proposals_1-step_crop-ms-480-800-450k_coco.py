_base_ = [
    'mmrotate::_base_/datasets/dota.py',
    'mmrotate::_base_/schedules/schedule_1x.py',
    'mmrotate::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmrotate.models.task_modules',
             'projects.OrientedDiffusionDet.oriented_diffusiondet'], allow_failed_imports=False)

# model settings
angle_version = 'le90'
model = dict(
    type='OrientedDiffusionDet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmrotate/v1.0/lsknet/\
        backbones/lsk_s_backbone-e9d2e551.pth'),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    bbox_head=dict(
        type='DynamicDiffusionDetHead',
        num_classes=15,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        single_head=dict(
            type='SingleDiffusionDetHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            # clockwise=True 为使用顺时针旋转
            roi_layer=dict(type='RoIAlignRotated',
                           output_size=7,
                           sampling_ratio=2,
                           clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # criterion
        criterion=dict(
            type='DiffusionDetCriterion',
            num_classes=15,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='mmdet.FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='MopBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='RIoUCost', iou_mode='iou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss',
                           beta=0.1111111111111111,
                           reduction='sum',
                           loss_weight=5.0),
            loss_riou=dict(type='RotatedIoULoss', reduction='sum',
                           loss_weight=2.0))),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type='nms_rotated', iou_threshold=0.5),
    ))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2))

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=75000,
    val_interval=7500)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=450000,
        by_epoch=False,
        milestones=[350000, 420000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=75000, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)
