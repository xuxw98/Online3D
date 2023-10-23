import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from mmcv.runner import BaseModule, _load_checkpoint_with_prefix, load_state_dict
import numpy as np
import torch.nn.functional as F

class FusionAwareFuseConv(nn.Module):
    def __init__(self, num_class):
        super(FusionAwareFuseConv, self).__init__()
        self.cmid = 32
        self.feature_dim = 128

        self.mlp_conv1 = torch.nn.Conv2d(3, 8, (1, 1))
        self.mlp_conv2 = torch.nn.Conv2d(8, 16, (1, 1))
        self.mlp_conv3 = torch.nn.Conv2d(16, 16, (1, 1))
        self.mlp_conv4 = torch.nn.Conv2d(16, 32, (1, 1))
        
        self.fc_1_256 = torch.nn.Conv1d(self.cmid * self.feature_dim, 1024, 1)
        self.fc_2 = torch.nn.Conv1d(1024, 256, 1)
        self.fc_3 = torch.nn.Conv1d(256, 128, 1)
        self.fc_output_20 = torch.nn.Conv1d(128, num_class, 1)



        self.bn_conv1 = nn.BatchNorm2d(8)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.bn_conv3 = nn.BatchNorm2d(16)
        self.bn_conv4 = nn.BatchNorm2d(32)


        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3_bn = nn.BatchNorm1d(128)

    def init_weights(self, pretrained=None):
        ckpt_path = './mmdet3d/models/fs_conv/model3d.pth'
        ckpt = torch.load(ckpt_path)
        load_state_dict(self, ckpt, strict=False)
        # for param in self.parameters():
        #     param.requires_grad = False
        # self.eval()


    def forward(self, feature2d,points, pre_result):
        self.batch_size = points.shape[0]
        self.node_size=points.shape[3]
        

        points = F.relu(self.bn_conv1(self.mlp_conv1(points)))
        points = F.relu(self.bn_conv2(self.mlp_conv2(points)))
        points = F.relu(self.bn_conv3(self.mlp_conv3(points)))
        points = F.relu(self.bn_conv4(self.mlp_conv4(points)))

        feature2d = feature2d.permute(0, 3, 1, 2)
        points = points.permute(0, 3, 2, 1)
        combine = torch.matmul(feature2d, points)

        combine = combine.permute(0, 2, 3, 1)
        combine = combine.view(self.batch_size, -1, self.node_size)

        combine = F.relu(self.fc1_bn(self.fc_1_256(combine)))
        combine = F.relu(self.fc2_bn(self.fc_2(combine)))
        combine = F.relu(self.fc3_bn(self.fc_3(combine)))
        combine=torch.max(combine,pre_result)

        combine_result = self.fc_output_20(combine)
        result = F.log_softmax(combine_result, dim=1)
        sorted_result,indices=torch.sort(result,dim=1,descending=True)

        uncertainty=sorted_result[:,0,:]/sorted_result[:,1,:]

        return result,combine,uncertainty




def create_FusionAwareFuseConv(num_class):
    return FusionAwareFuseConv(num_class=num_class)
