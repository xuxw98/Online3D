# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

import torch
from torch import nn
from mmdet3d.models.builder import NECKS
from mmcv.runner import BaseModule
import numpy as np
import torch.nn as nn
import os
import pdb


@NECKS.register_module()
class MultilevelMemory(BaseModule):
    def __init__(self, in_channels=[64, 128, 256, 512], scale=2.5, vmp_layer=(0,1,2,3)):
        super(MultilevelMemory, self).__init__()
        self.scale = scale
        self.vmp_layer = list(vmp_layer)
        self.conv_k5d1 = nn.ModuleList()
        self.conv_k3d5 = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        for i, C in enumerate(in_channels):
            if i in self.vmp_layer:
                self.conv_k5d1.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=5,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_k3d5.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=5,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_convert.append(nn.Sequential(
                    ME.MinkowskiConvolutionTranspose(
                        in_channels=2*C,
                        out_channels=C,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C)))
            else:
                self.conv_k5d1.append(nn.Identity())
                self.conv_k3d5.append(nn.Identity())
                self.conv_convert.append(nn.Identity())
        self.relu = ME.MinkowskiReLU()
        self.accumulated_feats = None
    
    def reset(self):
        self.accumulated_feats = None
    
    def two_cat(self, feat1, feat2):
        coords1 = feat1.decomposed_coordinates
        feats1 = feat1.decomposed_features
        coords2 = feat2.decomposed_coordinates
        feats2 = feat2.decomposed_features
        for i in range(len(coords1)):
            # shape 1 N
            feats1[i] = torch.cat([feats1[i], feats2[i]], dim=1)       
        coords_sp, feats_sp = ME.utils.sparse_collate(coords1, feats1)
        feat_new = ME.SparseTensor(
            coordinates=coords_sp,
            features=feats_sp,
            tensor_stride=feat1.tensor_stride,
            coordinate_manager=feat1.coordinate_manager
        )
        return feat_new
    
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

            branch1 = self.conv_k5d1[index](current_feat_new)
            branch2 = self.conv_k3d5[index](current_feat_new)
            branch  = self.two_cat(branch1, branch2)
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
    
    def forward(self, x):
        if self.accumulated_feats is None:
            accumulated_feats = x
            for i in range(len(x)):
                if i in self.vmp_layer:
                    branch1 = self.conv_k5d1[i](x[i])
                    branch2 = self.conv_k3d5[i](x[i])
                    branch  = self.two_cat(branch1, branch2)
                    branch = self.conv_convert[i](branch)
                    x[i] = branch + x[i]
                    x[i] = self.relu(x[i])
            self.accumulated_feats = accumulated_feats
            return x
        else:
            tuple_feats = [self.accumulate(self.accumulated_feats[i], x[i], i) for i in range(len(x))]
            self.accumulated_feats = [tuple_feats[i][0] for i in range(len(x))]
            return [tuple_feats[i][1] for i in range(len(x))]