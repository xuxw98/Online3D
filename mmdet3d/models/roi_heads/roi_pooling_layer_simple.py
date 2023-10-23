from enum import unique
from matplotlib.pyplot import box
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME
from mmdet3d.models.detectors.SegGroupFF import SAVE, PRINT

class SimplePoolingLayer(nn.Module):
    def __init__(self, channels=[128,128,128], grid_kernel_size = 5, grid_num = 7, voxel_size=0.04, coord_key=2,
                    point_cloud_range=[-5.12*3, -5.12*3, -5.12*3, 5.12*3, 5.12*3, 5.12*3],
                    corner_offset_emb=False, pooling=False):
        super(SimplePoolingLayer, self).__init__()
        # build conv
        self.voxel_size = voxel_size
        self.coord_key = coord_key
        grid_size = [int((point_cloud_range[3] - point_cloud_range[0])/voxel_size), 
                     int((point_cloud_range[4] - point_cloud_range[1])/voxel_size), 
                     int((point_cloud_range[5] - point_cloud_range[2])/voxel_size)] # simple set as -5.12*3, 5.12*3 / 0.08
        self.grid_size = grid_size
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.grid_num = grid_num
        self.corner_offset_emb = corner_offset_emb
        self.pooling = pooling
        # self.grid_conv = ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=grid_kernel_size, dimension=3, expand_coordinates=True)
        self.grid_conv = ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=grid_kernel_size, dimension=3)
        self.grid_bn = ME.MinkowskiBatchNorm(channels[1])
        self.grid_relu = ME.MinkowskiELU()
        if self.pooling:
            self.pooling_conv = ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=grid_num, dimension=3)
            self.pooling_bn = ME.MinkowskiBatchNorm(channels[1])

        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.grid_conv.kernel, std=.01)
        if self.pooling:
            nn.init.normal_(self.pooling_conv.kernel, std=.01)

    def forward(self, sp_tensor, grid_points, grid_corners=None, box_centers=None, batch_size=None):
        """
        Args:
            sp_tensor: minkowski tensor
            grid_points: bxnum_roisx216, 4 (b,x,y,z)
            grid_corners (optional): bxnum_roisx216, 8, 3
            box_centers: bxnum_rois, 4 (b,x,y,z)
        """
        grid_coords = grid_points.long()
        # print("=== grid_points shape ===", grid_points.shape)
        # print("=== grid_points ===", grid_points[:100])
        grid_coords[:, 1:4] = torch.floor(grid_points[:, 1:4] / self.voxel_size) # get coords (grid conv center)
        # assert grid_coords[:, 1:4].max() <= self.grid_size[0] / 2  and grid_coords[:, 1:4].min() >= -self.grid_size[0] / 2, \
        #     "points max {}, points min {}, voxel max {}, voxel min {}".format(grid_points[:, 1:4].max(), grid_points[:, 1:4].min(), grid_coords[:, 1:4].max(), grid_coords[:, 1:4].min())
        grid_coords[:, 1:4] = torch.clamp(grid_coords[:, 1:4], min=-self.grid_size[0] / 2 + 1, max=self.grid_size[0] / 2 - 1) # -192 ~ 192
        if SAVE:
            np.save("debug/sp_tensor.npy", sp_tensor.C.detach().cpu().numpy())
            np.save("debug/grid_points.npy", grid_points.detach().cpu().numpy())
            np.save("debug/grid_coords.npy", grid_coords.detach().cpu().numpy())
        grid_coords_positive = grid_coords[:, 1:4] + self.grid_size[0] // 2 # -192 ~ 192 -> 0 ~ 384
        merge_coords = grid_coords[:, 0] * self.scale_xyz + \
                        grid_coords_positive[:, 0] * self.scale_yz + \
                        grid_coords_positive[:, 1] * self.scale_z + \
                        grid_coords_positive[:, 2] # (N,)
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        unq_grid_coords = torch.stack((torch.div(unq_coords, self.scale_xyz),
                                    torch.div((unq_coords % self.scale_xyz), self.scale_yz),
                                    torch.div((unq_coords % self.scale_yz), self.scale_z),
                                    unq_coords % self.scale_z), dim=1) # num_voxels, 4(b, x, y, z) positive
        unq_grid_coords[:, 1:4] -= self.grid_size[0] // 2
        unq_grid_coords[:, 1:4] *= self.coord_key
        if SAVE:
            np.save("debug/unq_grid_coords.npy", unq_grid_coords.detach().cpu().numpy())
        # conv at given coords
        # grid_sp_tensor = self.grid_relu(self.grid_bn(self.grid_conv(sp_tensor, grid_coords))) # bxnum_roisx216, C/4
        if PRINT:
            print("=====roi get key==========", sp_tensor.coordinate_map_key.get_key())
        unq_grid_sp_tensor = self.grid_relu(self.grid_bn(self.grid_conv(sp_tensor, unq_grid_coords.int()))) # N_unq, C/4
        unq_features = unq_grid_sp_tensor.F
        unq_coords = unq_grid_sp_tensor.C
        new_features = unq_features[unq_inv]
        # new_coords = unq_coords[unq_inv]
        # print(new_features.shape)
        # assert (new_coords == grid_coords).all(), "some bug exists !!"
        # print("new coords:", new_coords[:50])
        # print("grid coords:", grid_coords[:50])

        if self.pooling:
            # fake grid
            fake_grid_coords = torch.ones(self.grid_num, self.grid_num, self.grid_num, device=unq_grid_coords.device)
            fake_grid_coords = torch.nonzero(fake_grid_coords) - self.grid_num // 2 # grid_num**3, 3
            fake_grid_coords = fake_grid_coords.unsqueeze(0).repeat(grid_coords.shape[0] // fake_grid_coords.shape[0], 1, 1) # bxnum_rois, grid_num**3, 3
            # fake center
            fake_centers = fake_grid_coords.new_zeros(fake_grid_coords.shape[0], 3) # bxnum_rois, 3
            fake_batch_idx = torch.arange(fake_grid_coords.shape[0]).to(fake_grid_coords.device) # bxnum_rois
            fake_center_idx = fake_batch_idx.reshape([-1, 1])
            fake_center_coords = torch.cat([fake_center_idx, fake_centers], dim=-1).int() # bxnum_rois, 4
            
            fake_grid_idx = fake_batch_idx.reshape([-1, 1, 1]).repeat(1, fake_grid_coords.shape[1], 1) # bxnum_rois, grid_num**3, 1
            fake_grid_coords = torch.cat([fake_grid_idx, fake_grid_coords], dim=-1).reshape([-1, 4]).int()

            grid_sp_tensor = ME.SparseTensor(coordinates=fake_grid_coords, features=new_features)
            pooled_sp_tensor = self.pooling_conv(grid_sp_tensor, fake_center_coords) # bxnum_rois, C/4
            pooled_sp_tensor = self.pooling_bn(pooled_sp_tensor) # bxnum_rois, C
            return pooled_sp_tensor.F
        else:
            return new_features


