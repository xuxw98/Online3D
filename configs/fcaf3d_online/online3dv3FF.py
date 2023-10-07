evaluator_mode = 'slice_num_constant'
num_slice = 1
len_slice = 5

model = dict(
    type='SingleStageSparse3DDetectorFF_OnlineV3',
    voxel_size=.01,
    evaluator_mode=evaluator_mode,
    num_slice=num_slice,
    len_slice=len_slice,
    img_backbone=dict(type='Resnet_FPN_Backbone',),
    img_memory=dict(type='MultilevelImgMemory'),
    backbone=dict(type='MEFFResNet3D', in_channels=3, depth=34),
    memory=dict(type='MultilevelMemory', in_channels=[64, 128, 256, 512]),
    neck_with_head=dict(
        type='Fcaf3DNeckWithHead_OnlineV3',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        voxel_size=.01,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        n_classes=18,
        n_reg_outs=6),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=300, nms_pre_merge=1000, iou_thr=.5, iou_thr_merge=.5, score_thr=.02))
