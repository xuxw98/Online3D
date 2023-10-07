# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
import torch
from torch import nn
from mmdet3d.models.builder import NECKS
from mmcv.runner import BaseModule
from mmcv.cnn.utils.weight_init import constant_init
from mmdet3d.models.fusion_layers.point_fusion import point_project
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import pdb


@NECKS.register_module()
class MultilevelImgMemory(BaseModule):
    def __init__(self, voxel_size, in_channels=[256, 512, 1024, 2048], keys=['res2', 'res3', 'res4', 'res5'], ada_layer=(1,2,3), semseg=False):
        super(MultilevelImgMemory, self).__init__()
        self.voxel_size = voxel_size
        self.ada_layer = list(ada_layer)
        self.reorg_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.orgback_list = nn.ModuleList()
        self.spconv2d_list = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            C = in_channel // 2
            if i in self.ada_layer:
                self.reorg_list.append(nn.Linear(in_channel, C))
                self.conv_list.append(nn.Sequential(
                    nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(C)))
                self.orgback_list.append(nn.Linear(C, in_channel))
                self.spconv2d_list.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C if semseg else C // 2,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=2),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU(),
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=2),
                    ME.MinkowskiBatchNorm(C)))
            else:
                self.reorg_list.append(nn.Identity())
                self.conv_list.append(nn.Identity())
                self.orgback_list.append(nn.Identity())
                self.spconv2d_list.append(nn.Identity())
        self.keys = keys
        self.cached = {k: None for k in keys}
        self.acc_feats_3d = None
    
    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                constant_init(m.weight, 0)
                constant_init(m.bias, 0)
            if isinstance(m, nn.Linear):
                constant_init(m.weight, 0)
                constant_init(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)
            if isinstance(m, ME.MinkowskiConvolution):
                constant_init(m.kernel, 0)
            if isinstance(m, ME.MinkowskiBatchNorm):
                constant_init(m.bn.weight, 1)
                constant_init(m.bn.bias, 0)
    
    def register(self, acc_feats_3d):
        self.acc_feats_3d = acc_feats_3d
    
    def reset(self):
        self.cached = {k: None for k in self.keys}
        self.acc_feats_3d = None
    
    def forward(self, x, img_metas):
        # B C H W
        for i, key in enumerate(self.keys):
            if i in self.ada_layer:
                x_reorg = self.reorg_list[i](x[key].permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

                ## 3D-->2D
                if self.acc_feats_3d is not None:
                    points = self.acc_feats_3d[i].decomposed_coordinates
                    for k in range(len(points)):
                        points[k] = points[k] * self.voxel_size
                    features = self.acc_feats_3d[i].decomposed_features
                    img_coords_feats = []
                    for point, feature, img_meta in zip(points, features, img_metas):
                        coord_type = 'DEPTH'
                        img_scale_factor = (
                            point.new_tensor(img_meta['scale_factor'][:2])
                            if 'scale_factor' in img_meta.keys() else 1)
                        img_crop_offset = (
                            point.new_tensor(img_meta['img_crop_offset'])
                            if 'img_crop_offset' in img_meta.keys() else 0)
                        proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
                        img_coords_feats.append(point_project( # project: consider points behind camera
                            img_meta=img_meta,
                            points=point,
                            features=feature,
                            proj_mat=point.new_tensor(proj_mat),
                            coord_type=coord_type,
                            img_scale_factor=img_scale_factor,
                            img_crop_offset=img_crop_offset))
                    # coords in raw image, should convert to corresponding stage
                    coordinates, features = ME.utils.batch_sparse_collate(
                        [(c / 2**(i+2), f) for c, f in img_coords_feats],
                        device=features[0].device)
                    keep_idx = (coordinates[:,1] >= 0) * (coordinates[:,1] < x[key].shape[-2]) * \
                        (coordinates[:,2] >= 0) * (coordinates[:,2] < x[key].shape[-1])
                    coordinates, features = coordinates[keep_idx], features[keep_idx]
                    if len(coordinates) > 0:
                        acc_imgs = ME.TensorField(coordinates=coordinates, features=features, 
                            quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL).sparse()
                        acc_imgs = self.spconv2d_list[i](acc_imgs)
                        acc_imgs = acc_imgs.dense(shape=x_reorg.shape, min_coordinate=torch.IntTensor([0,0]))[0]
                        x_reorg += acc_imgs

                ## Temporal Shift
                fold = x_reorg.shape[1] // 8
                out = torch.zeros_like(x_reorg)
                out[:, fold:] = x_reorg[:, fold:]
                if self.cached[key] is not None:
                    out[:, :fold] = self.cached[key]
                self.cached[key] = x_reorg[:, :fold]

                ## 2D Aggregation
                # TODO: remove this relu for both RGB and D
                # TODO: bottleneck for D
                x[key] = self.orgback_list[i](self.conv_list[i](out).permute(0,2,3,1).contiguous()). \
                    permute(0,3,1,2).contiguous() + x[key]
        return x



# This single-stage ImgMemory should be inserted after decoder.
@NECKS.register_module()
class ImgMemory(BaseModule):
    def __init__(self, voxel_size, in_channels=256):
        super(ImgMemory, self).__init__()
        self.voxel_size = voxel_size
        C = in_channels
        # C = in_channels // 2
        self.reorg = nn.Linear(in_channels, C)
        self.conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C)
        )
        self.orgback = nn.Linear(C, in_channels)
        self.spconv2d = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=C*8,
                # in_channels=C // 2,
                out_channels=C,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=2),
            ME.MinkowskiBatchNorm(C),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=C,
                out_channels=C,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=False,
                dimension=2),
            ME.MinkowskiBatchNorm(C))
        self.cached = None
        self.acc_feat_3d = None
    
    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                constant_init(m.weight, 0)
                constant_init(m.bias, 0)
            if isinstance(m, nn.Linear):
                constant_init(m.weight, 0)
                constant_init(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)
            if isinstance(m, ME.MinkowskiConvolution):
                constant_init(m.kernel, 0)
            if isinstance(m, ME.MinkowskiBatchNorm):
                constant_init(m.bn.weight, 1)
                constant_init(m.bn.bias, 0)
    
    def register(self, acc_feats_3d):
        # TODO: which level of 3D feature is the best?
        if acc_feats_3d is not None:
            self.acc_feat_3d = acc_feats_3d[-1]
        else:
            self.acc_feat_3d = None

    def reset(self):
        self.cached = None
        self.acc_feat_3d = None
    
    def forward(self, x, img_metas):
        x_reorg = self.reorg(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

        ## 3D-->2D
        if self.acc_feat_3d is not None:
            points = self.acc_feat_3d.decomposed_coordinates
            for k in range(len(points)):
                points[k] = points[k] * self.voxel_size
            features = self.acc_feat_3d.decomposed_features
            img_coords_feats = []
            for point, feature, img_meta in zip(points, features, img_metas):
                coord_type = 'DEPTH'
                img_scale_factor = (
                    point.new_tensor(img_meta['scale_factor'][:2])
                    if 'scale_factor' in img_meta.keys() else 1)
                img_crop_offset = (
                    point.new_tensor(img_meta['img_crop_offset'])
                    if 'img_crop_offset' in img_meta.keys() else 0)
                proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
                img_coords_feats.append(point_project( # project: consider points behind camera
                    img_meta=img_meta,
                    points=point,
                    features=feature,
                    proj_mat=point.new_tensor(proj_mat),
                    coord_type=coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset))
            # coords in raw image, should convert to corresponding stage. c / 2 for ResUnet
            coordinates, features = ME.utils.batch_sparse_collate(
                [(c / 2, f) for c, f in img_coords_feats],
                device=features[0].device)
            keep_idx = (coordinates[:,1] >= 0) * (coordinates[:,1] < x.shape[-2]) * \
                (coordinates[:,2] >= 0) * (coordinates[:,2] < x.shape[-1])
            coordinates, features = coordinates[keep_idx], features[keep_idx]
            if len(coordinates) > 0:
                acc_imgs = ME.TensorField(coordinates=coordinates, features=features, 
                    quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL).sparse()
                acc_imgs = self.spconv2d(acc_imgs)
                acc_imgs = acc_imgs.dense(shape=x_reorg.shape, min_coordinate=torch.IntTensor([0,0]))[0]
                x_reorg += acc_imgs

        ## Temporal Shift
        fold = x_reorg.shape[1] // 8
        out = torch.zeros_like(x_reorg)
        out[:, fold:] = x_reorg[:, fold:]
        if self.cached is not None:
            out[:, :fold] = self.cached
        self.cached = x_reorg[:, :fold]

        ## 2D Aggregation
        # TODO: remove this relu for both RGB and D
        # TODO: bottleneck for D
        x = self.orgback(self.conv(out).permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous() + x
        return x