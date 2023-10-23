_base_ = ['seggroup3d.py']
n_points = 100000
semantic_threshold = 0.15
model = dict(
    type='SegGroupFF',
    semantic_min_threshold=0.04,
    semantic_iter_value=0.02, # 0.01,
    img_backbone=dict(type='Resnet_FPN_Backbone',),
    backbone=dict(
        type='MEDualFFResNet',
        in_channels=3,
        out_channels=64),
    neck_with_head=dict(
        type='SegGroup3DHeadDDR',
        semantic_threshold=semantic_threshold,
        n_classes=18,
        n_reg_outs=6,
        cls_kernel=9,
        use_fusion_feat=False,
        loss_bbox=dict(with_yaw=False)),
    roi_head=dict(
    #   type='VoxelROIHeadMulti',
        type='VoxelROIHead',
    #   middle_feature_source=[3,4],
        middle_feature_source=[3],
        grid_size=7,
        voxel_size=0.02,
        coord_key=2,
    #   mlps=[[128,128,128],[3,128,128]],
    #   mlps=[[128,64,64],[128,64,64],[128,64,64],[128,128,128]],
        mlps=[[64,128,128]],
        roi_per_image=128, roi_fg_ratio=0.9, reg_fg_thresh=0.3, roi_conv_kernel=5,
        enlarge_ratio=None,
        use_corner_loss=False, 
        use_grid_offset=False,
        use_simple_pooling=True, 
        use_center_pooling=True,
    #   use_center_pooling=False,
        pooling_pose_only=False),)

find_unused_parameters=True

dataset_type = 'ScanNetSVDataset'
# dataset_type = 'ScanNetSVInstanceSegV2Dataset'
data_root = './data/scannet-sv1/'

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')
train_pipeline = [
    dict(
        type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_mask_3d=True,
         with_seg_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='Resize',
        img_scale=[(1333, 600)],  
        # img_scale=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
        #            (1333, 576), (1333, 600)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39),
        max_cat_id=40),
    # dict(type='IndoorPointSample', num_points=n_points),
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'pts_semantic_mask', 'pts_instance_mask'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
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
            dict(type='IndoorPointSample', num_points=n_points),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_sv_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_sv_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_sv_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

# lr_config = dict(policy='step', warmup=None, step=[12, 16])
# runner = dict(type='EpochBasedRunner', max_epochs=18)
lr_config = dict(policy='step', warmup=None, step=[7, 9])
runner = dict(type='EpochBasedRunner', max_epochs=10)
# lr_config = dict(policy='step', warmup=None, step=[15, 20])
# runner = dict(type='EpochBasedRunner', max_epochs=22)
# lr_config = dict(policy='step', warmup=None, step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
