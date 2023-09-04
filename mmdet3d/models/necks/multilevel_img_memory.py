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
    def __init__(self, in_channels=[256, 512, 1024, 2048], keys=['res2', 'res3', 'res4', 'res5'], ada_layer=(3,)):
        super(MultilevelImgMemory, self).__init__()
        self.ada_layer = list(ada_layer)
        self.conv_list = nn.ModuleList()
        for i, C in enumerate(in_channels):
            if i in self.ada_layer:
                self.conv_list.append(nn.Sequential(
                    # pw
                    nn.Conv2d(C, C // 4, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(C // 4),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(C // 4, C // 4, 3, 1, 1, groups=C // 4, bias=False),
                    nn.BatchNorm2d(C // 4),
                    nn.ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(C // 4, C, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(C)))
            else:
                self.conv_list.append(nn.Identity())
        self.keys = keys
        self.cached = {k: None for k in keys}
    
    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                constant_init(m.weight, 0)

            if isinstance(m, nn.BatchNorm2d):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)
    
    def reset(self):
        self.cached = {k: None for k in self.keys}
    
    def forward(self, x):
        # B C H W
        for i, key in enumerate(self.keys):
            if i in self.ada_layer:
                fold = x[key].shape[1] // 8
                out = torch.zeros_like(x[key])
                out[:, fold:] = x[key][:, fold:]
                if self.cached[key] is not None:
                    out[:, :fold] = self.cached[key]
                self.cached[key] = x[key][:, :fold]
                # TODO: remove this relu for both RGB and D
                x[key] = F.relu(self.conv_list[i](out) + x[key])
        return x


@NECKS.register_module()
class ImgMemory(BaseModule):
    def __init__(self, in_channels=256, with_depth=True):
        super(ImgMemory, self).__init__()
        C = in_channels // 2
        self.reorg = nn.Linear(in_channels, C)
        self.conv = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C)
        )
        self.orgback = nn.Linear(C, in_channels)
        self.with_depth = with_depth
        self.cached = None
        self.pre_points, self.num_points, self.pre_features = None, None, None
    
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
    
    def register(self, inputs, mode):
        assert mode in ['point', 'feature']
        if mode == 'point':
            self.pre_points = inputs
            self.num_points = inputs[0].shape[0]
            # if self.pre_points is None:
            #     self.pre_points = inputs
            #     self.num_points = inputs[0].shape[0]
            # else:
            #     self.pre_points = [torch.cat([pp, inputs[i]], dim=0)[-self.max_points:]
            #                         for i, pp in enumerate(self.pre_points)]
            #     self.num_points = min(self.max_points, self.num_points + inputs[0].shape[0])
        else:
            self.pre_features = inputs
            # if self.pre_features is None:
            #     self.pre_features = inputs
            # else:
            #     self.pre_features = [torch.cat([pf, inputs[i]], dim=0)[-self.max_points:]
            #                         for i, pf in enumerate(self.pre_features)]

    def reset(self):
        self.cached = None
        self.pre_points, self.num_points, self.pre_features = None, None, None
    
    def forward(self, x, img_metas):
        ## Project and Pixel-Max-Pooling
        acc_imgs = None
        if self.pre_points is not None and self.with_depth:
            acc_points = self.pre_points
            acc_features = self.pre_features
            img_coords_feats = []
            for point, feature, img_meta in zip(acc_points, acc_features, img_metas):
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
            # coords in raw image, should convert to 'p2'
            coordinates, features = ME.utils.batch_sparse_collate(
                [(c / 4, f) for c, f in img_coords_feats],
                device=acc_features[0].device)
            keep_idx = (coordinates[:,1] >= 0) * (coordinates[:,1] < x.shape[-2]) * \
                (coordinates[:,2] >= 0) * (coordinates[:,2] < x.shape[-1])
            coordinates, features = coordinates[keep_idx], features[keep_idx]
            if len(coordinates) > 0:
                acc_imgs = ME.TensorField(coordinates=coordinates, features=features, 
                    quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL).sparse()
                acc_imgs = acc_imgs.dense(shape=x.shape, min_coordinate=torch.IntTensor([0,0]))[0]

        ## Temporal Shift and 2D Aggregation
        x_ = x.clone()
        if acc_imgs is not None:
            pmp_index = (acc_imgs != 0)
            x_[pmp_index] = torch.maximum(x_[pmp_index], acc_imgs[pmp_index]) # PMP
        x_reorg = self.reorg(x_.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
        fold = x_reorg.shape[1] // 8
        out = torch.zeros_like(x_reorg)
        out[:, fold:] = x_reorg[:, fold:]
        if self.cached is not None:
            out[:, :fold] = self.cached
        self.cached = x_reorg[:, :fold]
        # TODO: remove this relu for both RGB and D
        # TODO: bottleneck for D
        # TODO: pre_feature should be raw 'p2' or 'p2' after PMP
        x = self.orgback(self.conv(out).permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous() + x
        return x, x_