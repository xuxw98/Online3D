n_points = 100000

model = dict(
    neck_with_head=dict(
        n_classes=18,
        n_reg_outs=6,
        loss_bbox=dict(with_yaw=False)))

dataset_type = 'ScanNetMVDataset'
data_root = './data/scannet-mv1/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

voxel_size = 0.01


evaluator_mode = 'slice_num_constant'
num_slice = 1
len_slice = 5

model = dict(
    evaluator_mode=evaluator_mode,
    num_slice=num_slice,
    len_slice=len_slice)

model = dict(
    type='SingleStageSparse3DDetector',
    voxel_size=voxel_size,
    backbone=dict(
        type='MEResNet3D',
        in_channels=3,
        depth=34),
    neck_with_head=dict(
        type='Fcaf3DNeckWithHead',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        pts_threshold=100000,
        n_classes=18,
        n_reg_outs=6,
        #loss_bbox=dict(with_yaw=False),
        voxel_size=voxel_size,
        assigner=dict(
            type='Fcaf3DAssigner',
            limit=27,
            topk=18,
            n_scales=4),
        loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0, with_yaw=False)),
    evaluator_mode=evaluator_mode,
    num_slice=num_slice,
    len_slice=len_slice,
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        iou_thr=.5,
        score_thr=.01,
        loss_bbox=dict(with_yaw=False)))

train_pipeline = [
    dict(
        type='LoadAdjacentViewsFromFiles',
        coord_type='DEPTH',
        num_frames=-1,
        shift_height=False,
        use_color=True,
        use_dim=[0, 1, 2, 3, 4, 5],
        ),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='MultiViewFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadAdjacentViewsFromFiles',
        coord_type='DEPTH',
        num_frames=-1,
        shift_height=False,
        use_color=True,
        use_dim=[0, 1, 2, 3, 4, 5],
        ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(
                type='MultiViewFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_mv_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_mv_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        evaluator_mode=evaluator_mode,
        num_slice=num_slice,
        len_slice=len_slice),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_mv_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        evaluator_mode=evaluator_mode,
        num_slice=num_slice,
        len_slice=len_slice))



optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]