# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

import torch
from torch import nn
from functools import partial
from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
import numpy as np
import torch.nn as nn
import os
import pdb


@DETECTORS.register_module()
class SingleStageSparse3DDetector_OnlineV3(Base3DDetector):
    r"""Single stage detector based on MinkowskiEngine `GSDN
    <https://arxiv.org/abs/2006.12356>`_.

    Args:
        backbone (dict): Config of the backbone.
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck_with_head,
                 voxel_size,
                 evaluator_mode,
                 num_slice=0,
                 len_slice=0,
                 memory=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(SingleStageSparse3DDetector_OnlineV3, self).__init__(init_cfg)
        assert evaluator_mode in ['slice_len_constant','slice_num_constant']
        self.evaluator_mode=evaluator_mode
        self.num_slice=num_slice
        self.len_slice=len_slice
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        if memory is not None:
            self.memory = build_neck(memory)
        self.voxel_size = voxel_size

    def extract_feat(self, points):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        if hasattr(self, 'memory'):
            x = self.memory(x)
        return x

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
        losses = {}
        bbox_data_list = []
        if hasattr(self, 'memory'):
            self.memory.reset()
        for i in range(img_metas[0]['num_frames']):
            current_feats = self.extract_feat([scene_points[i] for scene_points in points])
            loss, bbox_data_list = self.neck_with_head.forward_train(current_feats, gt_bboxes_3d, gt_labels_3d, bbox_data_list, img_metas)
            for key, value in loss.items():
                if key in losses: 
                    losses[key] += value
                else:
                    losses[key] = value
        return losses

    def simple_test(self, points, img_metas, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Note:
            During test time the batchsize should be 1,
            as each scene contains different number of frames.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        # Benchmark
        timestamps = []
        if self.evaluator_mode == 'slice_len_constant':
            i=1
            while i*self.len_slice<len(points[0]):
                timestamps.append(i*self.len_slice)
                i=i+1
            timestamps.append(len(points[0]))
        else:
            num_slice = min(len(points[0]),self.num_slice)
            for i in range(1,num_slice):
                timestamps.append(i*(len(points[0])//num_slice))
            timestamps.append(len(points[0]))

        # Process
        bbox_results = [[]]
        for i in range(len(timestamps)):
            if i == 0:
                ts_start, ts_end = 0, timestamps[i]
            else:
                ts_start, ts_end = timestamps[i-1], timestamps[i]
            bbox_data_list = []
            if hasattr(self, 'memory'):
                self.memory.reset()
            for j in range(ts_start, ts_end):
                current_feats = self.extract_feat([scene_points[i] for scene_points in points])
                bbox_list, bbox_data_list = self.neck_with_head.forward_test(current_feats, bbox_data_list, img_metas)
                bboxes, scores, labels = bbox_list[0]
                bbox_results[0].append(bbox3d2result(bboxes, scores, labels))

        return bbox_results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
