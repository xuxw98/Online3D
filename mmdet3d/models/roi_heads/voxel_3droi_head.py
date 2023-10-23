from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet3d.core.bbox import bbox3d2result
from mmdet.models import HEADS
from ..builder import build_head
from .base_3droi_head import Base3DRoIHead
from .proposal_target_layer import ProposalTargetLayer
from .roi_pooling_layer import PoolingLayer
from .roi_pooling_layer_simple import SimplePoolingLayer
from mmdet3d.core.bbox.coders.box_coder_utils import ResidualCoder
from mmdet3d.models.losses.weighted_smooth_l1_loss import WeightedSmoothL1Loss, get_corner_loss_lidar
# from mmdet3d.ops.pcdet_nms import pcdet_nms_gpu, pcdet_nms_normal_gpu
from mmcv.ops import nms3d, nms3d_normal


from mmdet3d.models.detectors.SegGroupFF import SAVE, PRINT

@HEADS.register_module()
class VoxelROIHead(nn.Module):
    def __init__(self, middle_feature_source, num_class=18, code_size=6, \
                 grid_size=6, voxel_size=0.02, mlps=[[128,128,128]], coord_key=2,\
                 pool_radius=[0.4], nsample=[16], enlarge_ratio=None,
                 shared_fc=[256,256], cls_fc=[256,256], reg_fc=[256,256], \
                 dp_ratio=0.3, test_score_thr=0.01, test_iou_thr=0.5, \
                 roi_per_image=128, roi_fg_ratio=0.5, reg_fg_thresh=0.3, \
                 cls_loss_type='BinaryCrossEntropy', reg_loss_type='smooth-l1', roi_conv_kernel=5,
                 use_corner_loss=False, use_grid_offset=False, use_simple_pooling=False, use_center_pooling=False, pooling_pose_only=False,
                 loss_weight={'rcnn_cls_weight': 1.0, 'rcnn_reg_weight': 1.0, 'rcnn_corner_weight':1.0, 'rcnn_corner_weight':1.0,
                                'code_weight':[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}):
        super(VoxelROIHead, self).__init__()
        self.middle_feature_source = middle_feature_source # [3] the last feature
        self.scale_list = list(torch.tensor([64, 32, 16, 8])[middle_feature_source])
        self.num_class = num_class
        self.code_size = code_size
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.enlarge_ratio = enlarge_ratio
        self.mlps = mlps
        self.shared_fc = shared_fc
        self.test_score_thr = test_score_thr
        self.test_iou_thr = test_iou_thr
        self.cls_fc = cls_fc
        self.reg_fc = reg_fc
        self.cls_loss_type = cls_loss_type
        self.reg_loss_type = reg_loss_type

        # features
        self.use_corner_loss = use_corner_loss
        self.use_grid_offset = use_grid_offset
        self.use_simple_pooling = use_simple_pooling
        self.use_center_pooling = use_center_pooling
        self.pooling_pose_only = pooling_pose_only

        self.loss_weight = loss_weight
        self.proposal_target_layer = ProposalTargetLayer(roi_per_image=roi_per_image, 
                                                         fg_ratio=roi_fg_ratio, 
                                                         reg_fg_thresh=reg_fg_thresh,)
        self.box_coder = ResidualCoder(code_size=code_size)
        self.reg_loss_func = WeightedSmoothL1Loss(code_weights=loss_weight['code_weight'])

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for i in range(len(self.mlps)):# different feature source
            mlp = self.mlps[i] # [128, 128]
            if not self.use_simple_pooling:
                pool_layer = PoolingLayer(
                    nsample=nsample[i],
                    radius=pool_radius[i],
                    mlps=mlp,
                    corner_offset_emb=self.use_grid_offset,
                    pose_only=self.pooling_pose_only,
                    pool_method='max_pool',
                )
            else:
                pool_layer = SimplePoolingLayer(channels=mlp, grid_kernel_size=roi_conv_kernel, grid_num=grid_size, \
                                                voxel_size=voxel_size*coord_key, coord_key=coord_key, pooling=self.use_center_pooling)
            self.roi_grid_pool_layers.append(pool_layer)
            # cout += sum(x[-1] for x in mlps) # TODO: check this
        
        if not self.use_center_pooling:
            c_out = sum([x[-1] for x in self.mlps])
            GRID_SIZE = self.grid_size
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

            shared_fc_list = []
            for k in range(0, self.shared_fc.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.shared_fc[k], bias=False),
                    nn.BatchNorm1d(self.shared_fc[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.shared_fc[k]

                if k != self.shared_fc.__len__() - 1 and dp_ratio > 0:
                    shared_fc_list.append(nn.Dropout(dp_ratio))
            self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        else:
            pre_channel = sum([x[-1] for x in self.mlps])

        # cls_fc_list = []
        # for k in range(0, self.cls_fc.__len__()):
        #     cls_fc_list.extend([
        #         nn.Linear(pre_channel, self.cls_fc[k], bias=False),
        #         nn.BatchNorm1d(self.cls_fc[k]),
        #         nn.ReLU()
        #     ])
        #     pre_channel = self.cls_fc[k]

        #     if k != self.cls_fc.__len__() - 1 and dp_ratio > 0:
        #         cls_fc_list.append(nn.Dropout(dp_ratio))
        # self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        # self.cls_pred_layer = nn.Linear(pre_channel, 1, bias=True) # only predict scores

        reg_fc_list = []
        for k in range(0, self.reg_fc.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.reg_fc[k], bias=False),
                nn.BatchNorm1d(self.reg_fc[k]),
                nn.ReLU()
            ])
            pre_channel = self.reg_fc[k]

            if k != self.reg_fc.__len__() - 1 and dp_ratio > 0:
                reg_fc_list.append(nn.Dropout(dp_ratio))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.code_size, bias=True)

        self.init_weights()
    
    def init_weights(self): # TODO: finish it 
        init_func = nn.init.xavier_normal_
        # for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
        layers_list = [self.shared_fc_layer, self.reg_fc_layers] if not self.use_center_pooling else [self.reg_fc_layers]
        for module_list in layers_list:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        # nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        # nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
    
    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1]) # (BxN, 7)
        # print("---debug rois----", rois[:100])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points, local_roi_grid_offset_corners = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        # no need to rotate cause with yaw is False
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points, local_roi_grid_offset_corners
    
    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        # get corner
        template = rois.new_tensor((
                    [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                    [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                    )) / 2
        proposal_corners = local_roi_size.unsqueeze(1).repeat(1,8,1) * template.unsqueeze(0) # BxN, 8, 3
        # proposal_corners += rois.view(batch_size_rcnn, -1)[:, None, :3]
        # caculate offset
        roi_grid_offset_corners = proposal_corners.unsqueeze(1) - roi_grid_points.unsqueeze(2) # BxN, 6x6x6, 8, 3
            
        return roi_grid_points, roi_grid_offset_corners
    
    def roi_grid_pool(self, input_dict):
        """
        Args:
            input_dict:
                rois: b, num_max_rois, 7
                batch_size: b
                middle_feature_list: List[mink_tensor]
        """
        rois = input_dict['rois']
        batch_size = input_dict['batch_size']
        middle_feature_list = [input_dict['middle_feature_list'][i] for i in self.middle_feature_source]
        if not isinstance(middle_feature_list, list):
            middle_feature_list = [middle_feature_list]
        
        roi_grid_xyz, _ , local_roi_grid_offset = self.get_global_grid_points_of_roi(
            rois, grid_size=self.grid_size
        )  # (BxN, 6x6x6, 3) _ (BxN, 6x6x6, 8, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        if SAVE:
            # print(rois[:100])
            np.save("debug/rois.npy", rois.detach().cpu().numpy())
            np.save("debug/roi_grid_xyz.npy", roi_grid_xyz.detach().cpu().numpy())

        # compute the voxel coordinates of grid points
        # roi_grid_coords: (B, Nx6x6x6, 3)
        # roi_grid_coords = roi_grid_coords // self.voxel_size

        batch_idx = rois.new_zeros(batch_size, roi_grid_xyz.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_xyz.shape[1])
        # TODO: use voxel-rcnn style to gather grid feature
        
        pooled_features_list = []
        for k, cur_sp_tensors in enumerate(middle_feature_list):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = self.scale_list[k]
            if not self.use_simple_pooling:
                # NOTE: reorder the batch idx to avoid bug
                points = cur_sp_tensors.decomposed_coordinates
                xyz_bs_cnt = cur_sp_tensors.C.new_zeros(batch_size)
                for i in range(len(points)):
                    points[i] = points[i].float()
                    points[i] = points[i] * self.voxel_size
                    xyz_bs_cnt[i] = len(points[i])
                xyz = torch.cat(points, dim=0)            
                # features = cur_sp_tensors.F
                if not self.pooling_pose_only:
                    feature_list = cur_sp_tensors.decomposed_features
                    features = torch.cat(feature_list, dim=0)
                else:
                    features = None
                if self.use_grid_offset:
                    new_features = pool_layer(batch_size, features, xyz, xyz_bs_cnt, roi_grid_xyz.reshape([-1,3]), roi_grid_batch_cnt, grid_corners=local_roi_grid_offset.reshape([-1, 8, 3]))
                else:
                    new_features = pool_layer(batch_size, features, xyz, xyz_bs_cnt, roi_grid_xyz.reshape([-1,3]), roi_grid_batch_cnt)
            else:
                #use simple pooling
                batch_grid_points = torch.cat([batch_idx, roi_grid_xyz], dim=-1) # B, Nx6x6x6, 4
                batch_grid_points = batch_grid_points.reshape([-1, 4])
                # print("------batch grid points----", batch_grid_points[:100])
                # if self.use_center_pooling:
                #     center_batch_idx = rois.new_zeros(batch_size, rois.shape[1], 1) # B, N, 1
                #     for bs_idx in range(batch_size):
                #         center_batch_idx[bs_idx, :, 0] = bs_idx
                #     batch_box_centers = torch.cat([center_batch_idx, rois[:, :, :3]], dim=-1) # B, N, 4
                #     batch_box_centers = batch_box_centers.reshape([-1, 4])
                # else:
                #     batch_box_centers = None

                # new_features = pool_layer(cur_sp_tensors, grid_points=batch_grid_points, box_centers=batch_box_centers) # BxNx216 or BxN , C
                new_features = pool_layer(cur_sp_tensors, grid_points=batch_grid_points) # BxNx216 or BxN , C
                
            # BxNx6x6x6, C
            if not self.use_center_pooling:
                new_features = new_features.reshape([-1, self.grid_size**3, new_features.shape[-1]])
            # BxN, 6x6x6, C or BxN, C
            pooled_features_list.append(new_features)

        ms_pooled_feature = torch.cat(pooled_features_list, dim=-1)
        return ms_pooled_feature

    def forward_train(self, input_dict):
        pred_boxes_3d = input_dict['pred_bbox_list']

        # preprocess rois, padding to same number
        rois, roi_scores, roi_labels, batch_size = self.reoder_rois_for_refining(pred_boxes_3d) # b, num_max_rois, 6
        if self.enlarge_ratio is not None:
            rois[..., 3:6] *= self.enlarge_ratio
        input_dict['rois'] = rois
        input_dict['roi_scores'] = roi_scores
        input_dict['roi_labels'] = roi_labels
        input_dict['batch_size'] = batch_size

        # assign targets
        targets_dict = self.assign_targets(input_dict)
        input_dict.update(targets_dict)
        # input_dict['rois'] = targets_dict['rois']
        # input_dict['roi_labels'] = targets_dict['roi_labels']
        # input_dict['rcnn_cls_labels'] = targets_dict['rcnn_cls_labels']

        # roi pooling
        pooled_features = self.roi_grid_pool(input_dict)  # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  # (BN, 6x6x6xC)
        if not self.use_center_pooling:
            shared_features = self.shared_fc_layer(pooled_features)  # (BN, C)
        else:
            shared_features = pooled_features
        # rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))  # (BN, 1)
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) # (BN, 6)

        # input_dict['rcnn_cls'] = rcnn_cls
        input_dict['rcnn_reg'] = rcnn_reg

        return input_dict       

    def assign_targets(self, input_dict):
        with torch.no_grad():
            targets_dict = self.proposal_target_layer(input_dict)
        rois = targets_dict['rois'] # b, num_max_rois, 7
        gt_of_rois = targets_dict['gt_of_rois'] # b, num_max_rois, 7
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        gt_label_of_rois = targets_dict['gt_label_of_rois'] # b, num_max_rois
        
        # canonical transformation, only change center to center-offset
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry # 0 - 0 = 0
        targets_dict['gt_of_rois'] = gt_of_rois

        return targets_dict
    
    def reoder_rois_for_refining(self, pred_boxes_3d):
        """
        Args:
            pred_boxes_3d: List[(box, score, label), (), ...]
        """
        batch_size = len(pred_boxes_3d)
        num_max_rois = max([len(preds[0]) for preds in pred_boxes_3d])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_boxes_3d[0][0].tensor

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_boxes_3d[bs_idx][0])
            # remember to change origin from 0.5,0.5,0 -> 0.5,0.5,0.5
            rois[bs_idx, :num_boxes, :] = torch.cat((pred_boxes_3d[bs_idx][0].gravity_center, pred_boxes_3d[bs_idx][0].tensor[:, 3:]), dim=1)
            roi_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][1]
            roi_labels[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][2]
        return rois, roi_scores, roi_labels, batch_size

    def simple_test(self, input_dict):
        pred_boxes_3d = input_dict['pred_bbox_list']

        # preprocess rois, padding to same number
        rois, roi_scores, roi_labels, batch_size = self.reoder_rois_for_refining(pred_boxes_3d) # b, num_max_rois, 6
        input_dict['rois'] = rois
        input_dict['roi_scores'] = roi_scores
        input_dict['roi_labels'] = roi_labels
        input_dict['batch_size'] = batch_size        

        # roi pooling
        pooled_features = self.roi_grid_pool(input_dict)  # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  # (BN, 6x6x6xC)
        if not self.use_center_pooling:
            shared_features = self.shared_fc_layer(pooled_features)  # (BN, C)
        else:
            shared_features = pooled_features
        # rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))  # (BN, num_classes)
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) #(BN, 6)

        # input_dict['rcnn_cls'] = rcnn_cls
        input_dict['rcnn_reg'] = rcnn_reg

        # batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
        #         batch_size=input_dict['batch_size'], rois=input_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
        #     )
        # input_dict['batch_cls_preds'] = batch_cls_preds # B,N
        # input_dict['batch_box_preds'] = batch_box_preds # B,N,6
        # input_dict['cls_preds_normalized'] = False

        return input_dict
    
    def get_boxes(self, input_dict, img_meta):
        batch_size = input_dict['batch_size']
        # rcnn_cls = input_dict['rcnn_cls']
        rcnn_cls = None
        rcnn_reg = input_dict['rcnn_reg']
        roi_labels = input_dict['roi_labels']
        roi_scores = input_dict['roi_scores']

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_size, rois=input_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
        input_dict['cls_preds_normalized'] = False
        if not input_dict['cls_preds_normalized'] and batch_cls_preds is not None:
            batch_cls_preds = torch.sigmoid(batch_cls_preds)
        input_dict['batch_cls_preds'] = batch_cls_preds # B,N
        input_dict['batch_box_preds'] = batch_box_preds # B,N,6

        results = []
        for bs_id in range(batch_size):
            # nms
            boxes = batch_box_preds[bs_id]
            # scores = batch_cls_preds[bs_id].squeeze(-1)
            scores = roi_scores[bs_id]
            labels = roi_labels[bs_id]

            result = self._nms(boxes, scores, labels, img_meta[bs_id])
            results.append(result)
        
        return results
    
    def _nms(self, bboxes, scores, labels, img_meta):
        n_classes = self.num_class
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = (labels == i) & (scores > self.test_score_thr) & (bboxes.sum() != 0)
            if not ids.any():
                continue
            class_scores = scores[ids]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                # nms_function = pcdet_nms_gpu
                nms_function = nms3d
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                # nms_function = pcdet_nms_normal_gpu
                nms_function = nms3d_normal

            # nms_ids, _ = nms_function(class_bboxes, class_scores, self.test_iou_thr)
            nms_ids = nms_function(class_bboxes, class_scores, self.test_iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes, box_dim=box_dim, with_yaw=with_yaw, origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels
    
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        # batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1]) # B, N, 1
        batch_cls_preds = None
        batch_box_preds = box_preds.view(batch_size, -1, code_size) # B,N,6

        # decode box
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()[..., :code_size]
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
    
    def loss(self, input_dict):
        rcnn_loss_dict = {}
        # rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(input_dict)
        if not self.use_corner_loss:
            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(input_dict)
            rcnn_loss_dict['rcnn_loss_reg'] = rcnn_loss_reg
        else:
            rcnn_loss_reg, rcnn_loss_corner, reg_tb_dict = self.get_box_reg_layer_loss(input_dict)
            rcnn_loss_dict['rcnn_loss_reg'] = rcnn_loss_reg
            rcnn_loss_dict['rcnn_loss_corner'] = rcnn_loss_corner
        # rcnn_loss_dict['rcnn_loss_cls'] = rcnn_loss_cls
        return rcnn_loss_dict
    
    def get_box_cls_layer_loss(self, forward_ret_dict):
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if self.cls_loss_type == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif self.cls_loss_type == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * self.loss_weight['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict
    
    def get_box_reg_layer_loss(self, forward_ret_dict):
        code_size = self.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois'][..., 0:code_size]
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if self.reg_loss_type == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0

            # encode box
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 6]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * self.loss_weight['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
            loss_corner = torch.tensor(0., device=fg_mask.device)
            if self.use_corner_loss and fg_sum > 0:
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()

                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d[:, 0:3] += roi_xyz
                loss_corner = get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:code_size],
                    gt_of_rois_src[fg_mask][:, 0:code_size]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * self.loss_weight['rcnn_corner_weight']
                tb_dict['rcnn_corner_weight'] = loss_corner.item()
        else:
            raise NotImplementedError
        
        if not self.use_corner_loss:
            return rcnn_loss_reg, tb_dict
        else:
            return rcnn_loss_reg, loss_corner, tb_dict

    def forward(self, input_dict):
        if self.training:
            return self.forward_train(input_dict)
        else:
            return self.simple_test(input_dict)