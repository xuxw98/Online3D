_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_3x.py'
]

# dataset settings
dataset_type = 'SceneNNMVSegDataset' # TODO: may need to provide resampled scene_idx
data_root = './data/scenenn-mv1/'
class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'sink')
all_class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')

voxel_size = 0.02
evaluator_mode = 'slice_num_constant'
num_slice = 1
len_slice = 5

model = dict(type='FusionAwareConv',
    evaluator_mode = evaluator_mode,
    num_slice = num_slice,
    len_slice = len_slice,
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
    dict(type='MultiViewFormatBundle3D', class_names=all_class_names),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask', 'img'])
]

test_pipeline = [
    dict(
        type='LoadAdjacentViewsFromFiles_FSA',
        coord_type='DEPTH',
        num_frames=-1,
        max_frames=50,
        num_sample=20000,
        shift_height=False,
        use_ins_sem=True,
        use_color=True,
        use_box=False,
        use_dim=[0, 1, 2, 3, 4, 5],
        scenenn_rot=True,
        ),
    dict(type='MultiViewFormatBundle3D', class_names=all_class_names, with_label=False),
    dict(type='Collect3D', keys=['points', 'img', 'depth_fsa', 'poses', 'img_fsa'])
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
    dict(type='MultiViewFormatBundle3D', class_names=all_class_names, with_label=False),
    dict(type='Collect3D', keys=['points', 'pts_semantic_mask', 'img'])
]

data = dict(
    # 6 8
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(       
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scenenn_mv_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            ignore_index=len(class_names))),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scenenn_mv_infos_val.pkl',
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
        ann_file=data_root + 'scenenn_mv_infos_val.pkl',
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