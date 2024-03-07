_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_3x.py'
]

# dataset settings
dataset_type = 'ScanNetMVSegDataset'
data_root = './data/scannet-mv/'
class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')

voxel_size = 0.02
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
evaluator_mode = 'slice_num_constant'
num_slice = 1
len_slice = 5

load_from = "work_dirs/minkunet_svFF/latest.pth"


model = dict(type='MinkUnetSemsegFF_Online',
    voxel_size=voxel_size,
    evaluator_mode=evaluator_mode,
    num_slice=num_slice,
    len_slice=len_slice,
    img_backbone=dict(type='Resnet_Unet_Backbone',),
    img_memory=dict(type='MultilevelImgMemory', in_channels=[64, 128, 256, 512], ada_layer=(2,3), semseg=True),
    backbone=dict(type='MinkUNet34C_SemsegFF', in_channels=3, out_channels=20, D=3),
    memory=dict(type='MultilevelMemory', in_channels=[32, 64, 128, 256], vmp_layer=(0,1,2,3)),
    head=dict(
        type='MinkUnetSemHead',
        voxel_size=voxel_size,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=20,
            class_weight=[
                2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
                4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
                5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
                5.3954206, 4.6971426
            ],  # should be modified with dataset
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict())

train_pipeline = [ 
    dict(
        type='LoadAdjacentViewsFromFiles',
        coord_type='DEPTH',
        num_frames=8,
        shift_height=False,
        use_ins_sem=True,
        use_color=True,
        use_amodal_points=False,
        use_box=False,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiImgsAug',
        img_scales=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
                   (1333, 576), (1333, 600)],
        transforms=[
            dict(
                type='Resize',
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)
        ]
        ),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39),
        max_cat_id=40),
    dict(
        type='RandomFlip3DV2',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTransV2',
        rot_range_z=[-3.14, 3.14],
        rot_range_x_y=[-0.1308, 0.1308],
        scale_ratio_range=[.8, 1.2],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='MultiViewFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask', 'img'])
]

test_pipeline = [
    dict(
        type='LoadAdjacentViewsFromFiles',
        coord_type='DEPTH',
        num_frames=-1,
        max_frames=50,
        num_sample=20000,
        shift_height=False,
        use_ins_sem=True,
        use_color=True,
        use_box=False,
        use_dim=[0, 1, 2, 3, 4, 5],
        ),
    dict(
        type='MultiImgsAug',
        img_scales=(1333, 600),
        #img_scales=(1333, 800),
        transforms=[
            dict(
                type='Resize',
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)
        ]
        ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='MultiViewFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadAdjacentViewsFromFiles',
        coord_type='DEPTH',
        num_frames=-1,
        max_frames=50,
        num_sample=20000,
        shift_height=False,
        use_ins_sem=True,
        use_color=True,
        use_box=False,
        use_dim=[0, 1, 2, 3, 4, 5],
        ),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39),
        max_cat_id=40),
    dict(
        type='MultiViewFormatBundle3D',
        with_label=False,
        class_names=class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask', 'img'])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(       
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_mv_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            ignore_index=len(class_names))),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_mv_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names),
        evaluator_mode=evaluator_mode,
        num_slice=num_slice,
        len_slice=len_slice,
        use_voxel_eval=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_mv_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        ignore_index=len(class_names),
        evaluator_mode=evaluator_mode,
        num_slice=num_slice,
        len_slice=len_slice,
        use_voxel_eval=False))


# data settings
evaluation = dict(pipeline=eval_pipeline, interval=5)
checkpoint_config = dict(interval=5)
# runner = dict(max_epochs=200)
