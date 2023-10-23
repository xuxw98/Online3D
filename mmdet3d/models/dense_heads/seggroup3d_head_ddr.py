import torch
from torch import nn
import numpy as np
import MinkowskiEngine as ME
from mmdet.core import BaseAssigner, reduce_mean, build_assigner
# from mmdet.models.builder import HEADS, build_loss
from mmdet3d.models.builder import HEADS, build_loss, MMDET_DETECTORS
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmcv.cnn import Scale, bias_init_with_prob

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
# from mmdet3d.ops.pcdet_nms import pcdet_nms_gpu, pcdet_nms_normal_gpu
from mmcv.ops import nms3d, nms3d_normal
from mmcv.ops.knn import knn
# from mmdet3d.ops.knn import knn

# from mmdet3d.models.voxel_encoders import DynamicVFE
# from mmdet3d.models.voxel_encoders.dynamic_vfe_v2 import DynamicVFEv2


@HEADS.register_module()
class SegGroup3DHeadDDR(nn.Module):
    def __init__(self,
                 n_classes,
                 in_channels,#  [64, 128, 256, 512],
                 out_channels,# 128
                 n_reg_outs,
                 voxel_size,# 0.01
                 pts_threshold,
                 semantic_threshold,
                 expand_ratio,
                 assigner,
                 yaw_parametrization='fcaf3d',
                 point_cloud_range=(-5.12*3-1e-6, -5.12*3-1e-6, -5.12*3-1e-6, 5.12*3+1e-6, 5.12*3+1e-6, 5.12*3+1e-6), # TODO: find the best point cloud range for scannet, now simply use a large one
                 use_fusion_feat=False,
                 cls_kernel=9,
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 #loss_bbox=dict(type='AxisAlignedIoULoss'),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 #loss_bbox=dict(type='RotatedIoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_sem=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_offset=dict(
                     type='SmoothL1Loss', beta=0.04, reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(SegGroup3DHeadDDR, self).__init__()
        self.voxel_size = voxel_size
        self.yaw_parametrization = yaw_parametrization
        self.use_fusion_feat = use_fusion_feat
        self.cls_kernel = cls_kernel
        self.assigner = build_assigner(assigner)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.loss_sem = build_loss(loss_sem)
        self.loss_offset = build_loss(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pts_threshold = pts_threshold
        self.semantic_threshold = semantic_threshold
        # self.thres_topk = thres_topk
        self.n_classes = n_classes
        self.voxel_size_list = [[0.2309, 0.2435, 0.2777],
                                [0.5631, 0.5528, 0.3579],
                                [0.1840, 0.1845, 0.2155],
                                [0.4187, 0.4536, 0.2503],
                                [0.2938, 0.3203, 0.1899],
                                [0.1595, 0.1787, 0.5250],
                                [0.2887, 0.2174, 0.3445],
                                [0.2497, 0.3147, 0.5063],
                                [0.0634, 0.1262, 0.1612],
                                [0.4332, 0.5691, 0.0810],
                                [0.3088, 0.4212, 0.2627],
                                [0.4130, 0.1966, 0.5044],
                                [0.1995, 0.2133, 0.3897],
                                [0.1260, 0.1137, 0.5254],
                                [0.1781, 0.1774, 0.2218],
                                [0.1526, 0.1520, 0.0904],
                                [0.3453, 0.3164, 0.1491],
                                [0.1426, 0.1477, 0.1741]]
        lower_size = 0.04
        self.voxel_size_list = np.clip(np.array(self.voxel_size_list) / 2., lower_size, 1.0).tolist()
        self.expand = expand_ratio

        self.point_cloud_range = point_cloud_range
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)

            
    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_block_with_kernels(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.ModuleList([
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3),
            nn.Sequential(
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU(),
                ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU())])

    @staticmethod
    def _make_up_block_with_parameters(in_channels, out_channels, kernel_size, stride):
        return nn.ModuleList([
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=3),
            nn.Sequential(
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiELU(),
                # ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
                # ME.MinkowskiBatchNorm(out_channels),
                # ME.MinkowskiELU()
                )])

    @staticmethod
    def _make_offset_block(in_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(in_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(in_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(in_channels, 3, kernel_size=1, dimension=3),
        )

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        # for i in range(len(in_channels)):
        #     if i > 0:
        #         self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))

        #     self.__setattr__(f'out_block_{i}', self._make_block(in_channels[i], out_channels))
        self.__setattr__(f'offset_block', self._make_offset_block(out_channels))
        self.__setattr__(f'feature_offset', self._make_block(out_channels, out_channels))

        # head layers
        self.semantic_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.centerness_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, dimension=3)
        self.reg_conv = ME.MinkowskiConvolution(out_channels, n_reg_outs, kernel_size=1, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(n_classes)])
        self.cls_individual_out = nn.ModuleList([self._make_block_with_kernels(out_channels, out_channels, self.cls_kernel) for _ in range(n_classes)])
        self.cls_individual_up = nn.ModuleList([self._make_up_block_with_parameters(out_channels,
                                                        out_channels, self.expand, self.expand) for _ in range(n_classes)])
        self.cls_individual_fuse = nn.ModuleList([self._make_block_with_kernels(out_channels*2, out_channels, 1) for _ in range(n_classes)])
        self.cls_individual_expand_out = nn.ModuleList([self._make_block_with_kernels(out_channels, out_channels, 5) for _ in range(n_classes)])

    def init_weights(self):
        nn.init.normal_(self.centerness_conv.kernel, std=.01)
        nn.init.normal_(self.reg_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))
        nn.init.normal_(self.semantic_conv.kernel, std=.01)
        nn.init.constant_(self.semantic_conv.bias, bias_init_with_prob(.01))
        for cls_id in range(self.n_classes):
            nn.init.normal_(self.cls_individual_out[cls_id][0].kernel, std=.01)

    def forward(self, x, return_middle_feature=False):
        outs = []
        semantic_input, inputs = x[-2], x[-1]
        out = semantic_input
        # inputs = x
        # x = inputs[-1]

        decode_out = [inputs[3], inputs[2], inputs[1], semantic_input]
        # for i in range(len(inputs) - 1, -1, -1):
        #     if i < len(inputs) - 1:
        #         x = self.__getattr__(f'up_block_{i + 1}')[0](x, inputs[i].coordinate_map_key)
        #         x = self.__getattr__(f'up_block_{i + 1}')[1](x)
        #         x = inputs[i] + x

        #     out_ = self.__getattr__(f'out_block_{i}')(x)
        #     decode_out.append(out_)
        #     if i == 0:
        #         curr_coordinates = out.C.float()
        #         semantic_features = out.F.clone()
        #         for level in range(3):
        #             interpolate_feats = decode_out[level].features_at_coordinates(curr_coordinates).clone()
        #             semantic_features += interpolate_feats
        #         semantic_input = ME.SparseTensor(features=semantic_features,
        #                                          # coordinates=out.C,
        #                                          coordinate_map_key=out.coordinate_map_key,
        #                                          coordinate_manager=out.coordinate_manager)
        if self.use_fusion_feat:
            decode_out[-1] = semantic_input
        semantic_scores = self.semantic_conv(semantic_input)

        pad_id = semantic_scores.C.new_tensor([permutation[0] for permutation in semantic_scores.decomposition_permutations]).long()
        # compute points range
        scene_coord = out.C[:, 1:].clone()
        max_bound = (scene_coord.max(0)[0] + out.coordinate_map_key.get_key()[0][0]) * self.voxel_size
        min_bound = (scene_coord.min(0)[0] - out.coordinate_map_key.get_key()[0][0]) * self.voxel_size

        voxel_offsets = self.__getattr__(f'offset_block')(out)
        offset_features = self.__getattr__(f'feature_offset')(out).F

        voted_coordinates = out.C[:, 1:].clone() * self.voxel_size + voxel_offsets.F.clone().detach()
        voted_coordinates[:, 0] = torch.clamp(voted_coordinates[:, 0], max=max_bound[0], min=min_bound[0])
        voted_coordinates[:, 1] = torch.clamp(voted_coordinates[:, 1], max=max_bound[1], min=min_bound[1])
        voted_coordinates[:, 2] = torch.clamp(voted_coordinates[:, 2], max=max_bound[2], min=min_bound[2])

        for cls_id in range(self.n_classes):
            with torch.no_grad():
                cls_semantic_scores = semantic_scores.F[:, cls_id].sigmoid()
                cls_selected_id = torch.nonzero(cls_semantic_scores > self.semantic_threshold).squeeze(1)
                cls_selected_id = torch.cat([cls_selected_id, pad_id])

            coordinates = out.C.float().clone()[cls_selected_id]

            coordinates[:, 1:4] = voted_coordinates[cls_selected_id]  # N,4 (b,x,y,z)
            ori_coordinates = out.C.float().clone()[cls_selected_id]
            ori_coordinates[:, 1:4] *= self.voxel_size
            fuse_coordinates = torch.cat([coordinates, ori_coordinates], dim=0)
            fuse_features = torch.cat([offset_features[cls_selected_id], out.F[cls_selected_id]], dim=0)
            # fuse_coordinates = ori_coordinates
            # fuse_features = cls_selected_voxels_map.F
            voxel_size = torch.tensor(self.voxel_size_list[cls_id], device=fuse_features.device)
            voxel_coord = fuse_coordinates.clone().int()
            voxel_coord[:, 1:] = (fuse_coordinates[:, 1:] / voxel_size).floor()
            cls_individual_map = ME.SparseTensor(coordinates=voxel_coord, features=fuse_features,
                                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            cls_individual_map = self.cls_individual_out[cls_id](cls_individual_map)

            # expand feature map
            cls_voxel_coord = fuse_coordinates.clone().int()
            expand = self.expand
            cls_voxel_coord[:, 1:] = (fuse_coordinates[:, 1:] / (voxel_size * expand)).floor()
            cls_individual_map_expand = ME.SparseTensor(coordinates=cls_voxel_coord, features=fuse_features,
                                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            expand_coord = cls_individual_map_expand.C
            expand_coord[:, 1:] *= expand
            cls_individual_map_expand = ME.SparseTensor(coordinates=expand_coord, features=cls_individual_map_expand.F,
                                                        tensor_stride=expand,
                                                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

            cls_individual_map_expand = self.cls_individual_expand_out[cls_id](cls_individual_map_expand)
            cls_individual_map_up = self.cls_individual_up[cls_id][0](cls_individual_map_expand, cls_individual_map.C)
            cls_individual_map_up = self.cls_individual_up[cls_id][1](cls_individual_map_up)
            cls_individual_map_out = ME.SparseTensor(coordinates=cls_individual_map.C,
                                                    features=torch.cat([cls_individual_map_up.F, cls_individual_map.F], dim=-1))
            cls_individual_map_out = self.cls_individual_fuse[cls_id](cls_individual_map_out)

            prediction = self.forward_single(cls_individual_map_out, self.scales[cls_id], self.voxel_size_list[cls_id])
            scores = prediction[-1]
            outs.append(list(prediction[:-1]))

        # return zip(*outs[::-1]), semantic_scores
        if not return_middle_feature:
            return zip(*outs), semantic_scores, voxel_offsets
        else:
            return zip(*outs), semantic_scores, voxel_offsets, decode_out # [scale: 0.32, 0.16, 0.08, 0.04]

    def _prune(self, x, scores):
        if self.pts_threshold < 0:
            return x

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros((len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             points,
             semantic_scores,
             voxel_offset,
             gt_bboxes,
             gt_labels,
             scene_points,
             img_metas,
             pts_semantic_mask,
             pts_instance_mask):

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for _ in range(len(centernesses[0]))]
            pts_instance_mask = pts_semantic_mask
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas) == len(gt_bboxes) == len(gt_labels) \
               == len(pts_instance_mask) == len(pts_semantic_mask) == len(scene_points)

        semantic_scores_list = []
        semantic_points_list = []
        for permutation in semantic_scores.decomposition_permutations:
            semantic_scores_list.append(semantic_scores.F[permutation])
            semantic_points_list.append(semantic_scores.C[permutation, 1:] * self.voxel_size)

        voxel_offset_list = []
        voxel_points_list = []
        for permutation in voxel_offset.decomposition_permutations:
            voxel_offset_list.append(voxel_offset.F[permutation])
            voxel_points_list.append(voxel_offset.C[permutation, 1:] * self.voxel_size)

        loss_centerness, loss_bbox, loss_cls, loss_sem, loss_vote = [], [], [], [], []
        for i in range(len(img_metas)):
            img_loss_centerness, img_loss_bbox, img_loss_cls, img_loss_sem, img_loss_vote = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                voxel_offset_preds=voxel_offset_list[i],
                original_points=voxel_points_list[i],
                semantic_scores=semantic_scores_list[i],
                semantic_points=semantic_points_list[i],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                scene_points=scene_points[i],
                pts_semantic_mask=pts_semantic_mask[i],
                pts_instance_mask=pts_instance_mask[i],
            )
            loss_centerness.append(img_loss_centerness)
            loss_bbox.append(img_loss_bbox)
            loss_cls.append(img_loss_cls)
            loss_sem.append(img_loss_sem)
            loss_vote.append(img_loss_vote)

        return dict(
            loss_centerness=torch.mean(torch.stack(loss_centerness)),
            loss_bbox=torch.mean(torch.stack(loss_bbox)),
            loss_cls=torch.mean(torch.stack(loss_cls)),
            loss_sem=torch.mean(torch.stack(loss_sem)),
            loss_vote=torch.mean(torch.stack(loss_vote))
        )

    # per image
    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     points,
                     voxel_offset_preds,
                     original_points,
                     semantic_scores,
                     semantic_points,
                     img_meta,
                     gt_bboxes,
                     gt_labels,
                     scene_points,
                     pts_semantic_mask,
                     pts_instance_mask):
        with torch.no_grad():
            semantic_labels = self.assigner.assign_semantic(semantic_points, gt_bboxes, gt_labels, self.n_classes)
            centerness_targets, bbox_targets, labels = self.assigner.assign(points, gt_bboxes, gt_labels)
            # compute offset targets
            if pts_semantic_mask is not None and pts_instance_mask is not None:
                # ScanNet V2 with mask annotations
                # compute original all points offsets and masks
                allp_offset_targets = torch.zeros_like(scene_points[:, :3])
                allp_offset_masks = scene_points.new_zeros(len(scene_points))
                instance_center = scene_points.new_zeros((pts_instance_mask.max()+1, 3))
                instance_match_gt_id = -scene_points.new_ones((pts_instance_mask.max()+1)).long()
                for i in torch.unique(pts_instance_mask):
                    indices = torch.nonzero(
                        pts_instance_mask == i, as_tuple=False).squeeze(-1)
                    if pts_semantic_mask[indices[0]] < self.n_classes:
                        selected_points = scene_points[indices, :3]
                        center = 0.5 * (
                                selected_points.min(0)[0] + selected_points.max(0)[0])
                        allp_offset_targets[indices, :] = center - selected_points
                        allp_offset_masks[indices] = 1

                        match_gt_id = torch.argmin(torch.cdist(center.view(1, 1, 3),
                                                               gt_bboxes.gravity_center.unsqueeze(0).to(center.device)).view(-1))
                        instance_match_gt_id[i] = match_gt_id
                        instance_center[i] = gt_bboxes.gravity_center[match_gt_id].to(center.device)
                    else:
                        instance_center[i] = torch.ones_like(instance_center[i]) * (-10000.)
                        instance_match_gt_id[i] = -1
                # compute points offsets of each scale seed points
                offset_targets = []
                offset_masks = []
                knn_number = 1
                idx = knn(knn_number, scene_points[None, :, :3].contiguous(), original_points[None, ::])[0].long()
                instance_idx = pts_instance_mask[idx.view(-1)].view(idx.shape[0], idx.shape[1])

                # condition1: all the points must belong to one instance
                valid_mask = (instance_idx == instance_idx[0]).all(0)

                max_instance_num = pts_instance_mask.max()+1
                arange_tensor = torch.arange(max_instance_num).unsqueeze(1).unsqueeze(2).to(instance_idx.device)
                arange_tensor = arange_tensor.repeat(1, instance_idx.shape[0], instance_idx.shape[1]) # instance_num, k, points
                instance_idx = instance_idx[None, ::].repeat(max_instance_num, 1, 1)

                max_instance_idx = torch.argmax((instance_idx == arange_tensor).sum(1), dim=0)
                offset_t = instance_center[max_instance_idx] - original_points
                offset_m = torch.where(offset_t < -100., torch.zeros_like(offset_t), torch.ones_like(offset_t)).all(1)
                offset_t = torch.where(offset_t < -100., torch.zeros_like(offset_t), offset_t)
                offset_m *= valid_mask

                offset_targets.append(offset_t)
                offset_masks.append(offset_m)

            else:
                # TODO: need to imple for sun rgbd
                raise NotImplementedError

        centerness = torch.cat(centernesses)
        bbox_preds = torch.cat(bbox_preds)
        cls_scores = torch.cat(cls_scores)
        points = torch.cat(points)

        offset_targets = torch.cat(offset_targets)
        offset_masks = torch.cat(offset_masks)

        # vote loss
        offset_weights_expand = (offset_masks.float() / torch.ones_like(offset_masks).float().sum() + 1e-6).unsqueeze(1).repeat(1, 3)
        loss_offset = self.loss_offset(voxel_offset_preds, offset_targets, weight=offset_weights_expand)

        # semantic loss
        sem_n_pos = torch.tensor(len(torch.nonzero(semantic_labels >= 0).squeeze(1)), dtype=torch.float, device=centerness.device)
        sem_n_pos = max(reduce_mean(sem_n_pos), 1.)
        loss_sem = self.loss_sem(semantic_scores, semantic_labels, avg_factor=sem_n_pos)

        # skip background
        # centerness loss and bbox loss
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centerness.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_centerness = centerness[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )
            loss_bbox = self.loss_bbox(
                self._bbox_pred_to_bbox(pos_points, pos_bbox_preds),
                pos_bbox_targets,
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            )
        else:
            loss_centerness = pos_centerness.sum()
            loss_bbox = pos_bbox_preds.sum()
        return loss_centerness, loss_bbox, loss_cls, loss_sem, loss_offset


    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   points,
                   img_metas,
                   rescale=False):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas)
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i]
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta):
        mlvl_bboxes, mlvl_scores = [], []
        for centerness, bbox_pred, cls_score, point in zip(
            centernesses, bbox_preds, cls_scores, points
        ):
            scores = cls_score.sigmoid() * centerness.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        return bboxes, scores, labels

    # per scale
    def forward_single(self, x, scale, voxel_size):
        centerness = self.centerness_conv(x).features
        scores = self.cls_conv(x)
        cls_score = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        reg_final = self.reg_conv(x).features
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        voxel_size = torch.tensor(voxel_size, device=cls_score.device)
        for i in range(len(points)):
            points[i] = points[i] * voxel_size
            assert len(points[i]) > 0, "forward empty"

        return centernesses, bbox_preds, cls_scores, points, prune_scores

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        if bbox_pred.shape[1] == 6:
            return base_bbox

        if self.yaw_parametrization == 'naive':
            # ..., alpha
            return torch.cat((
                base_bbox,
                bbox_pred[:, 6:7]
            ), -1)
        elif self.yaw_parametrization == 'sin-cos':
            # ..., sin(a), cos(a)
            norm = torch.pow(torch.pow(bbox_pred[:, 6:7], 2) + torch.pow(bbox_pred[:, 7:8], 2), 0.5)
            sin = bbox_pred[:, 6:7] / norm
            cos = bbox_pred[:, 7:8] / norm
            return torch.cat((
                base_bbox,
                torch.atan2(sin, cos)
            ), -1)
        else:  # self.yaw_parametrization == 'fcaf3d'
            # ..., sin(2a)ln(q), cos(2a)ln(q)
            scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
            q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
            alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
            return torch.stack((
                x_center,
                y_center,
                z_center,
                scale / (1 + q),
                scale / (1 + q) * q,
                bbox_pred[:, 5] + bbox_pred[:, 4],
                alpha
            ), dim=-1)

    def _nms(self, bboxes, scores, img_meta):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                # nms_function = pcdet_nms_gpu
                nms_function = nms3d
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                # nms_function = pcdet_nms_normal_gpu
                nms_function = nms3d_normal

            # nms_ids, _ = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_ids = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
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


def compute_centerness(bbox_targets):
    x_dims = bbox_targets[..., [0, 1]]
    y_dims = bbox_targets[..., [2, 3]]
    z_dims = bbox_targets[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
                         y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
                         z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    return torch.sqrt(centerness_targets)


@BBOX_ASSIGNERS.register_module()
class SegGroup3DAssigner(BaseAssigner):
    def __init__(self, limit, topk, n_scales):
        self.limit = limit
        self.topk = topk
        self.n_scales = n_scales

    def assign(self, points_list, gt_bboxes_ori, gt_labels_ori):

        centerness_targets_all = []
        gt_bboxes_all = []
        labels_all = []
        class_num = len(points_list)
        for cls_id in range(class_num):
            float_max = 1e8
            points = points_list[cls_id]
            # points = torch.cat(points, dim=0)

            # below is based on FCOSHead._get_target_single
            n_points = len(points)
            assert n_points > 0, "empty points in class {}".format(cls_id)
            select_inds = torch.nonzero((gt_labels_ori == cls_id)).squeeze(1)
            if len(select_inds) == 0:
                labels = torch.zeros((len(points)), dtype=torch.long).to(points.device).fill_(-1)
                gt_bbox_targets = torch.zeros((len(points), 7), dtype=torch.float).to(points.device)
                centerness_targets = torch.zeros((len(points)), dtype=torch.float).to(points.device)
            else:
                n_boxes = len(select_inds)
                volumes = gt_bboxes_ori.volume.to(points.device)[select_inds]
                volumes = volumes.expand(n_points, n_boxes).contiguous()
                gt_bboxes = torch.cat((gt_bboxes_ori.gravity_center[select_inds].clone(), gt_bboxes_ori.tensor[select_inds, 3:].clone()), dim=1)
                gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 7)
                gt_labels = gt_labels_ori[select_inds].clone()
                expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
                shift = torch.stack((
                    expanded_points[..., 0] - gt_bboxes[..., 0],
                    expanded_points[..., 1] - gt_bboxes[..., 1],
                    expanded_points[..., 2] - gt_bboxes[..., 2]
                ), dim=-1).permute(1, 0, 2)
                # print("====debug=== gt {} vs shift {} =".format(gt_bboxes.shape, shift.shape))
                shift = rotation_3d_in_axis(shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
                centers = gt_bboxes[..., :3] + shift
                dx_min = centers[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
                dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - centers[..., 0]
                dy_min = centers[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
                dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - centers[..., 1]
                dz_min = centers[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
                dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - centers[..., 2]
                bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)

                # condition1: inside a gt bbox
                inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

                # condition3: limit topk locations per box by centerness
                centerness = compute_centerness(bbox_targets)
                centerness = torch.where(inside_gt_bbox_mask, centerness, torch.ones_like(centerness) * -1)
                top_centerness = torch.topk(centerness, min(self.topk + 1, len(centerness)), dim=0).values[-1]
                inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

                # if there are still more than one objects for a location,
                # we choose the one with minimal area
                volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
                volumes = torch.where(inside_top_centerness_mask, volumes, torch.ones_like(volumes) * float_max)
                min_area, min_area_inds = volumes.min(dim=1)

                labels = gt_labels[min_area_inds]
                labels = torch.where(min_area == float_max, -1, labels)
                bbox_targets = bbox_targets[range(n_points), min_area_inds]
                centerness_targets = compute_centerness(bbox_targets)
                gt_bbox_targets = gt_bboxes[range(n_points), min_area_inds].clone()

            centerness_targets_all.append(centerness_targets)
            gt_bboxes_all.append(gt_bbox_targets)
            labels_all.append(labels)
        centerness_targets_all = torch.cat(centerness_targets_all)
        gt_bboxes_all = torch.cat(gt_bboxes_all)
        labels_all = torch.cat(labels_all)
        return centerness_targets_all, gt_bboxes_all, labels_all


    def assign_semantic(self, points, gt_bboxes, gt_labels, n_classes):
        float_max = 1e8

        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device)
        volumes = volumes.expand(n_points, n_boxes).contiguous()
        gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 7)
        expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        shift = torch.stack((
            expanded_points[..., 0] - gt_bboxes[..., 0],
            expanded_points[..., 1] - gt_bboxes[..., 1],
            expanded_points[..., 2] - gt_bboxes[..., 2]
        ), dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(shift, -gt_bboxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = gt_bboxes[..., :3] + shift
        dx_min = centers[..., 0] - gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2
        dx_max = gt_bboxes[..., 0] + gt_bboxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2
        dy_max = gt_bboxes[..., 1] + gt_bboxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2
        dz_max = gt_bboxes[..., 2] + gt_bboxes[..., 5] / 2 - centers[..., 2]
        bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max, gt_bboxes[..., 6]), dim=-1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets[..., :6].min(-1)[0] > 0  # skip angle

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        volumes = torch.where(inside_gt_bbox_mask, volumes, torch.ones_like(volumes) * float_max)
        min_area, min_area_inds = volumes.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels = torch.where(min_area == float_max, -1, labels)

        return labels

