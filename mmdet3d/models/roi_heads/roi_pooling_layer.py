import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmdet3d.ops.pointnet2.pointnet2_stack import pointnet2_utils
# from mmdet3d.ops.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from mmcv.ops.group_points import QueryAndGroup

from mmdet3d.models.detectors.SegGroupFF import SAVE, PRINT

class PoolingLayer(nn.Module):
    def __init__(self, radius=0.3, nsample=16, mlps=[128,128,128], pool_method='max_pool'):
        super(PoolingLayer, self).__init__()
        self.point_sampler = QueryAndGroup(radius=radius, nsample=nsample, use_xyz=True)
        self.mlps = mlps
        self.pool_method = pool_method
        self.mlp_in = nn.Sequential(nn.Conv1d(mlps[0], mlps[1], 1, bias=False), nn.BatchNorm1d(mlps[1]))
        self.mlp_pos = nn.Sequential(nn.Conv2d(3, mlps[1], 1, bias=False), nn.BatchNorm2d(mlps[1]))
        self.mlp_out = nn.Sequential(nn.Conv1d(mlps[1], mlps[2], 1, bias=False), nn.BatchNorm1d(mlps[2]))

        self.relu = nn.ReLU()

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch_size, features, xyz, xyz_bs_cnt, grid_points, grid_bs_cnt):
        """
        Args:
            sp_tensor: minkowski tensor
            grid_points: npoint, 4 (b,x,y,z)
            grid_bs_cnt: batch_size
        """
         # N,C
        features_in = features.permute(1,0).unsqueeze(0) # 1,C,N
        features_in = self.mlp_in(features_in)
        features_in = features_in.permute(0,2,1).contiguous().squeeze(0) # N, C

        # xyz = coords[:, 1:4].contiguous()
        # 
        assert xyz.is_contiguous()
        grid_features, _ = self.point_sampler(xyz, xyz_bs_cnt, grid_points, grid_bs_cnt, features_in) # npoint, 3+C, nsample
        grid_xyz = grid_features[:, :3, :] # npoint, 3, nsample
        grid_feat = grid_features[:, 3:, :]
        if SAVE:
            np.savez('debug/pooling_layer.npz', xyz=xyz.detach().cpu().numpy(), xyz_bs_cnt=xyz_bs_cnt.detach().cpu().numpy(),\
                    grid_points=grid_points.detach().cpu().numpy(), grid_bs_cnt=grid_bs_cnt.detach().cpu().numpy(), \
                        grid_xyz=grid_xyz.detach().cpu().numpy())

        grid_xyz = grid_xyz.permute(1,0,2).unsqueeze(0) # 1, 3, npoint, nsample
        grid_feat = grid_feat.permute(1,0,2).unsqueeze(0) # 1, C, npoint, nsample

        position_featuress = self.mlp_pos(grid_xyz) # 1, C, npoint, nsample
        new_features = position_featuress + grid_feat # 1, C, npoint, nsample
        new_features = self.relu(new_features)

        if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
        elif self.pool_method == 'avg_pool':
            new_features = F.avg_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
        else:
            raise NotImplementedError
            
        new_features = self.mlp_out(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)

        return new_features

