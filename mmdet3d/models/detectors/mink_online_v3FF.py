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
class MinkOnline3DDetector_V3FF(Base3DDetector):
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
                 img_backbone,
                 backbone,
                 head,
                 voxel_size,
                 evaluator_mode,
                 num_slice=0,
                 len_slice=0,
                 vmp_layer=(0,1,2,3),
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(MinkOnline3DDetector_V3FF, self).__init__(init_cfg)
        assert evaluator_mode in ['slice_len_constant','slice_num_constant']
        self.evaluator_mode=evaluator_mode
        self.num_slice=num_slice
        self.len_slice=len_slice
        self.img_backbone = build_backbone(img_backbone)
        self.backbone = build_backbone(backbone)
        self.has_neck = False
        if neck is not None:
            self.neck = build_neck(neck)
            self.has_neck = True
        self.vmp_layer = list(vmp_layer)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(256, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True))
        self.voxel_size = voxel_size

        self.scale = 2.5
        self.conv_d1 = nn.ModuleList()
        self.conv_d3 = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        for i, C in enumerate([64, 128, 256, 512]):
            if i in self.vmp_layer:
                self.conv_d1.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_d3.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=3,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_convert.append(nn.Sequential(
                    ME.MinkowskiConvolutionTranspose(
                        in_channels=3*C,
                        out_channels=C,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C)))
            else:
                self.conv_d1.append(nn.Identity())
                self.conv_d3.append(nn.Identity())
                self.conv_convert.append(nn.Identity())
        self.relu = ME.MinkowskiReLU()
        
        # self.init_weights()

    def init_weights(self, pretrained=None):
        self.img_backbone.init_weights()
        # self.img_neck.init_weights()
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        #for param in self.img_neck.parameters():
        #    param.requires_grad = False
        self.img_backbone.eval()
        #self.img_neck.eval()
        self.backbone.init_weights()
        if self.has_neck:
            self.neck.init_weights()
        self.head.init_weights()
             

    def extract_feat(self, points):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        B, C = len(points), points[0].shape[-1]
        points = [scene_points.reshape(-1, C) for scene_points in points]

        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        return x
    
    def global_avg_pool_and_cat(self, feat1, feat2, feat3):
        coords1 = feat1.decomposed_coordinates
        feats1 = feat1.decomposed_features
        coords2 = feat2.decomposed_coordinates
        feats2 = feat2.decomposed_features
        coords3 = feat3.decomposed_coordinates
        feats3 = feat3.decomposed_features
        for i in range(len(coords3)):
            # shape 1 N
            global_avg_feats3 = torch.mean(feats3[i], dim=0).unsqueeze(0).repeat(coords3[i].shape[0],1)
            feats1[i] = torch.cat([feats1[i], feats2[i]], dim=1)     
            feats1[i] = torch.cat([feats1[i], global_avg_feats3], dim=1)      
        coords_sp, feats_sp = ME.utils.sparse_collate(coords1, feats1)
        feat_new = ME.SparseTensor(
            coordinates=coords_sp,
            features=feats_sp,
            tensor_stride=feat1.tensor_stride,
            coordinate_manager=feat1.coordinate_manager
        )
        return feat_new
    
    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))

        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        projected_features = self.conv(projected_features)
        return projected_features + x

    def accumulate(self, accumulated_feat, current_feat, index):
        """Accumulate features for a single stage.

        Args:
            accumulated_feat (ME.SparseTensor)
            current_feat (ME.SparseTensor)

        Returns:
            ME.SparseTensor: refined accumulated features
            ME.SparseTensor: current features after accumulation
        """
        if index in self.vmp_layer:
            # VMP
            tensor_stride = current_feat.tensor_stride
            accumulated_feat = ME.TensorField(
                features=torch.cat([current_feat.features, accumulated_feat.features], dim=0),
                coordinates=torch.cat([current_feat.coordinates, accumulated_feat.coordinates], dim=0),
                quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
            ).sparse()
            accumulated_feat = ME.SparseTensor(
                coordinates=accumulated_feat.coordinates,
                features=accumulated_feat.features,
                tensor_stride=tensor_stride,
                coordinate_manager=accumulated_feat.coordinate_manager
            )

            # Select neighbor region for current frame
            accumulated_coords = accumulated_feat.decomposed_coordinates
            current_coords = current_feat.decomposed_coordinates
            accumulated_coords_select_list=[]
            zero_batch_feature_list=[]
            for i in range(len(current_coords)):
                accumulated_coords_batch = accumulated_coords[i]
                current_coords_batch = current_coords[i]
                current_coords_batch_max, _ = torch.max(current_coords_batch,dim=0)
                current_coords_batch_min, _ = torch.min(current_coords_batch,dim=0)
                current_box_size = current_coords_batch_max - current_coords_batch_min
                current_box_add = ((self.scale-1)/2) * current_box_size
                margin_positive = accumulated_coords_batch-current_coords_batch_max-current_box_add
                margin_negative = accumulated_coords_batch-current_coords_batch_min+current_box_add
                in_criterion = torch.mul(margin_positive,margin_negative)
                zero = torch.zeros_like(in_criterion)
                one = torch.ones_like(in_criterion)
                in_criterion = torch.where(in_criterion<=0,one,zero)
                mask = in_criterion[:,0]*in_criterion[:,1]*in_criterion[:,2]
                mask = mask.type(torch.bool)
                mask = mask.reshape(mask.shape[0],1)
                accumulated_coords_batch_select = torch.masked_select(accumulated_coords_batch,mask)
                accumulated_coords_batch_select = accumulated_coords_batch_select.reshape(-1,3)
                zero_batch_feature = torch.zeros_like(accumulated_coords_batch_select)
                accumulated_coords_select_list.append(accumulated_coords_batch_select)
                zero_batch_feature_list.append(zero_batch_feature)
            accumulated_coords_select_coords, _ = ME.utils.sparse_collate(accumulated_coords_select_list, zero_batch_feature_list)
            current_feat_new = ME.SparseTensor(
                coordinates=accumulated_coords_select_coords,
                features=accumulated_feat.features_at_coordinates(accumulated_coords_select_coords.float()),
                tensor_stride=tensor_stride
            )

            branch1 = self.conv_d1[index](current_feat_new)
            branch3 = self.conv_d3[index](current_feat_new)
            branch  = self.global_avg_pool_and_cat(branch1, branch3, current_feat_new)
            branch = self.conv_convert[index](branch)
            current_feat_new = branch + current_feat_new
            current_feat_new = self.relu(current_feat_new)
            current_feat = ME.SparseTensor(
                coordinates=current_feat.coordinates,
                features=current_feat_new.features_at_coordinates(current_feat.coordinates.float()),
                tensor_stride=tensor_stride,
                coordinate_manager=current_feat.coordinate_manager
            )
        return accumulated_feat, current_feat
    
    def process_one_frame(self, accumulated_feats, points, img, img_metas):
        """Extract and accumulate features from current frame.

        Args:
            accumulated_feats (list[ME.SparseTensor]) --> list of different stages
            current_frames (list[Tensor]) --> list of batch

        Returns:
            list[ME.SparseTensor]: refined accumulated features
            list[ME.SparseTensor]: current features after accumulation
        """
        with torch.no_grad():
            img_features = self.img_backbone(img)['p2']   
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x,partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        if accumulated_feats is None:
            accumulated_feats = x
            for i in range(len(x)):
                if i in self.vmp_layer:
                    branch1 = self.conv_d1[i](x[i])
                    branch3 = self.conv_d3[i](x[i])
                    branch  = self.global_avg_pool_and_cat(branch1, branch3, x[i])
                    branch = self.conv_convert[i](branch)
                    x[i] = branch + x[i]
                    x[i] = self.relu(x[i])
            return accumulated_feats, x
        else:
            tuple_feats = [self.accumulate(accumulated_feats[i], x[i], i) for i in range(len(x))]
            return [tuple_feats[i][0] for i in range(len(x))], [tuple_feats[i][1] for i in range(len(x))]

    def forward_train(self, points, img, gt_bboxes_3d, gt_labels_3d, img_metas):
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
        accumulated_feats = None
        depth2img = [img_meta['depth2img'] for img_meta in img_metas]
        for i in range(img_metas[0]['num_frames']):
            for j in range(len(img_metas)):
                img_metas[j]['depth2img'] = depth2img[j][i]
            accumulated_feats, current_feats = self.process_one_frame(accumulated_feats,
                 [scene_points[i] for scene_points in points],torch.stack([scene_img[i] for scene_img in img],dim=0), img_metas)
            if self.has_neck:
                current_feats = self.neck(current_feats)
            loss, bbox_data_list = self.head.forward_train(current_feats, gt_bboxes_3d, gt_labels_3d, bbox_data_list, img_metas)
            for key, value in loss.items():
                if key in losses: 
                    losses[key] += value
                else:
                    losses[key] = value
        return losses

    def simple_test(self, points, img_metas, img, *args, **kwargs):
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
        depth2img = img_metas[0]['depth2img']


        for i in range(len(timestamps)):
            if i == 0:
                ts_start, ts_end = 0, timestamps[i]
            else:
                ts_start, ts_end = timestamps[i-1], timestamps[i]
            bbox_data_list = []
            accumulated_feats = None

            for j in range(ts_start, ts_end):
                img_metas[0]['depth2img'] = depth2img[j]
                accumulated_feats, current_feats = self.process_one_frame(accumulated_feats, 
                        [points[0][j]], torch.stack([img[0][j]],dim=0), img_metas)
                if self.has_neck:
                    current_feats = self.neck(current_feats)
                bbox_list, bbox_data_list = self.head.forward_test(current_feats, bbox_data_list, img_metas)
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
