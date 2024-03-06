try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn import Scale, bias_init_with_prob
from mmdet.core.bbox.builder import build_assigner
from mmdet3d.models.builder import HEADS, build_backbone, build_loss
from mmcv.ops import nms3d, nms3d_normal

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet3d.models.builder import (ROI_EXTRACTORS, build_roi_extractor)
from mmdet3d.models.dense_heads.ngfc_head import get_face_distances
import pdb

@HEADS.register_module()
class TD3DInstanceHead_Online(BaseModule):
    def __init__(self,
        n_classes,
        in_channels,
        n_levels,
        unet,
        n_reg_outs,
        voxel_size,
        padding,
        first_assigner,
        second_assigner,
        roi_extractor,
        reg_loss=dict(type='SmoothL1Loss'),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode="diou"),
        cls_loss=dict(type='FocalLoss'),
        inst_loss=build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True)),
        train_cfg=None,
        test_cfg=None):
        super(TD3DInstanceHead_Online, self).__init__()
        self.voxel_size = voxel_size
        self.unet = build_backbone(unet)
        self.first_assigner = build_assigner(first_assigner)
        self.second_assigner = build_assigner(second_assigner)
        self.roi_extractor = build_roi_extractor(roi_extractor)
        self.reg_loss = build_loss(reg_loss)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.inst_loss = inst_loss
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.padding = padding
        self.n_classes = n_classes
        self._init_layers(n_classes, in_channels, n_levels, n_reg_outs)

    def _init_layers(self, n_classes, in_channels, n_levels, n_reg_outs):
        self.reg_conv = ME.MinkowskiConvolution(
            in_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        
    def init_weights(self):
        # pdb.set_trace()
        nn.init.normal_(self.reg_conv.kernel, std=0.01)
        nn.init.normal_(self.cls_conv.kernel, std=0.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(0.01))

    # per level
    def _forward_first_single(self, x):
        reg_pred = torch.exp(self.reg_conv(x).features)
        cls_pred = self.cls_conv(x).features

        reg_preds, cls_preds, locations = [], [], []
        for permutation in x.decomposition_permutations:
            reg_preds.append(reg_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            locations.append(x.coordinates[permutation][:, 1:] * self.voxel_size)
        return reg_preds, cls_preds, locations

    def _forward_first(self, x):
        reg_preds, cls_preds, locations = [], [], []
        for i in range(len(x)):
            reg_pred, cls_pred, point = self._forward_first_single(x[i])
            reg_preds.append(reg_pred)
            cls_preds.append(cls_pred)
            locations.append(point)
        return reg_preds, cls_preds, locations

    def _forward_second(self, x, targets, bbox_list):
        rois = [b[0] for b in bbox_list]
        scores = [b[1] for b in bbox_list]
        labels = [b[2] for b in bbox_list]
        levels = [torch.zeros(len(b[0])) for b in bbox_list]
        feats_with_targets = ME.SparseTensor(torch.cat((x.features, targets), axis=1), x.coordinates)
        tensors, ids, rois, scores, labels = self.roi_extractor.extract(
            [feats_with_targets], levels, rois, scores, labels)
        if tensors[0].features.shape[0] == 0:
            return (targets.new_zeros((0, 1)),
                    targets.new_zeros((0, 1)),
                    targets.new_zeros(0),
                    targets.new_zeros(0),
                    [targets.new_zeros((0, 7)) for i in range(len(bbox_list))],
                    [targets.new_zeros(0) for i in range(len(bbox_list))],
                    [targets.new_zeros(0) for i in range(len(bbox_list))])
           

        feats = ME.SparseTensor(tensors[0].features[:, :-2], tensors[0].coordinates)
        targets = tensors[0].features[:, -2:]

        preds = self.unet(feats).features
        
        return preds, targets, feats.coordinates[:, 0].long(), ids[0], rois[0], scores[0], labels[0]


    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
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

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
                bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    # per scene
    def _loss_first_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):

        assigned_ids = self.first_assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        avg_factor = max(pos_mask.sum(), 1)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=avg_factor)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            pos_bbox_targets = torch.cat((
                pos_bbox_targets[:, :3],
                pos_bbox_targets[:, 3:6] + self.padding,
                pos_bbox_targets[:, 6:]), dim=1)
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = pos_bbox_preds.sum()
        return bbox_loss, cls_loss

    def _loss_first(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses = [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss = self._loss_first_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(bbox_loss=torch.mean(torch.stack(bbox_losses)),
                    cls_loss=torch.mean(torch.stack(cls_losses)))

    def _loss_second(self, cls_preds, targets, v2r, r2scene, rois, gt_idxs,
                    gt_bboxes, gt_labels, img_metas):
        v2scene = r2scene[v2r]
        inst_losses = []
        for i in range(len(img_metas)):
            inst_loss = self._loss_second_single(
                cls_preds=cls_preds[v2scene == i],
                targets=targets[v2scene == i],
                v2r=v2r[v2scene == i],
                rois=rois[i],
                gt_idxs=gt_idxs[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                img_meta=img_metas[i])
            inst_losses.append(inst_loss)
        return dict(inst_loss=torch.mean(torch.stack(inst_losses)))

    def _loss_second_single(self, cls_preds, targets, v2r, rois, gt_idxs, gt_bboxes, gt_labels, img_meta):
        if len(rois) == 0 or cls_preds.shape[0] == 0:
            return cls_preds.sum().float()
        v2r = v2r - v2r.min()
        assert len(torch.unique(v2r)) == len(rois)
        assert torch.all(torch.unique(v2r) == torch.arange(0, v2r.max() + 1).to(v2r.device))
        assert torch.max(gt_idxs) < len(gt_bboxes)

        v2bbox = gt_idxs[v2r.long()]
        assert torch.unique(v2bbox)[0] != -1
        inst_targets = targets[:, 0]
        seg_targets = targets[:, 1]

        seg_preds = cls_preds[:, :-1]
        inst_preds = cls_preds[:, -1]

        # check point num
        labels = v2bbox == inst_targets

        seg_targets[seg_targets == -1] = self.n_classes
        seg_loss = self.cls_loss(seg_preds, seg_targets.long())

        inst_loss = self.inst_loss(inst_preds, labels)
        return inst_loss + seg_loss
 
    def forward_train(self,
        x,
        targets,
        points,
        gt_bboxes,
        gt_labels,
        pts_semantic_mask,
        pts_instance_mask,
        bbox_data_list,
        img_metas):
        # first stage
        bbox_preds, cls_preds, locations = self._forward_first(x[1:])
        losses = self._loss_first(bbox_preds, cls_preds, locations, 
                            gt_bboxes, gt_labels, img_metas)
        
        # second stage
        bbox_list = self._get_bboxes_train(bbox_preds, cls_preds, locations, gt_bboxes, gt_labels, img_metas)
        assigned_bbox_list = []
        for i in range(len(bbox_list)):
            assigned_ids = self.second_assigner.assign(bbox_list[i][0], gt_bboxes[i])
            gt_idxs = bbox_list[i][2]
            gt_idxs[gt_idxs != assigned_ids] = -1

            boxes = bbox_list[i][0][gt_idxs != -1]
            scores = bbox_list[i][1][gt_idxs != -1]
            boxes_data = bbox_list[i][3][gt_idxs != -1]
            scores_data = bbox_list[i][4][gt_idxs != -1]
            gt_idxs = gt_idxs[gt_idxs != -1]

            if len(boxes) != 0:
                gt_idxs_one_hot = torch.nn.functional.one_hot(gt_idxs)
                mask, idxs = torch.topk(gt_idxs_one_hot, min(self.train_cfg.num_rois, len(boxes)), 0)
                sampled_boxes = img_metas[i]['box_type_3d'](boxes.tensor[idxs].view(-1, 7), with_yaw=gt_bboxes[i].with_yaw)
                sampled_scores = scores[idxs].view(-1)
                sampled_gt_idxs = gt_idxs[idxs].view(-1)
                sampled_boxes_data = boxes_data[idxs].view(-1,6)
                sampled_scores_data = scores_data[idxs].view(-1,scores_data.shape[1])
                mask = mask.view(-1).bool()
                assigned_bbox_list.append((sampled_boxes[mask],
                                           sampled_scores[mask],
                                           sampled_gt_idxs[mask],
                                           sampled_boxes_data[mask],
                                           sampled_scores_data[mask]))
            else:
                assigned_bbox_list.append((boxes,
                                           scores,
                                           gt_idxs,
                                           boxes_data,
                                           scores_data))

        if len(bbox_data_list) == 0:
            for i in range(len(img_metas)):
                bbox_data_list.append([])
        bbox_all_list, _ = self._merge_and_update(bbox_data_list, [(assigned_bbox_list[i][3], assigned_bbox_list[i][4], assigned_bbox_list[i][2]) for i in range(len(img_metas))], self.test_cfg, img_metas, mode='train')
        
        _, bbox_now_list = self._merge_and_update([[]], [(assigned_bbox_list[i][3], assigned_bbox_list[i][4], assigned_bbox_list[i][2]) for i in range(len(img_metas))], self.test_cfg, img_metas, mode='train')
        for i in range(len(img_metas)):
            if len(bbox_data_list[i]) == self.train_cfg.acc_tot:
                for j in range(len(bbox_data_list[i])-1):
                    bbox_data_list[i][j] = bbox_data_list[i][j+1]
                bbox_data_list[i][self.train_cfg.acc_tot-1] = bbox_now_list[i]
            else:
                bbox_data_list[i].append(bbox_now_list[i])                        


        # remember gt_ids problem
        cls_preds, targets, v2r, r2scene, rois, scores, gt_idxs = self._forward_second(x[0], targets, bbox_all_list)

        losses.update(self._loss_second(cls_preds, targets, v2r, r2scene, rois, gt_idxs,
                                    gt_bboxes, gt_labels, img_metas))
        

        return losses, bbox_data_list

    # per scene
    def _get_instances_single(self, cls_preds, idxs, v2r, scores, labels, inverse_mapping):
        if scores.shape[0] == 0:
            return (inverse_mapping.new_zeros((1, len(inverse_mapping)), dtype=torch.bool),
                    inverse_mapping.new_tensor([0], dtype=torch.long),
                    inverse_mapping.new_tensor([0], dtype=torch.float32))
        v2r = v2r - v2r.min()
        assert len(torch.unique(v2r)) == scores.shape[0]
        assert torch.all(torch.unique(v2r) == torch.arange(0, v2r.max() + 1).to(v2r.device))

        cls_preds = cls_preds.sigmoid()
        binary_cls_preds = cls_preds > self.test_cfg.binary_score_thr
        v2r_one_hot = torch.nn.functional.one_hot(v2r).bool()
        n_rois = v2r_one_hot.shape[1]
        # todo: why convert from float to long here? can it be long or even int32 before this function?
        idxs_expand = idxs.unsqueeze(-1).expand(idxs.shape[0], n_rois).long()
        # todo: can we not convert to ofloat here?
        binary_cls_preds_expand = binary_cls_preds.unsqueeze(-1).expand(binary_cls_preds.shape[0], n_rois)
        cls_preds[cls_preds <= self.test_cfg.binary_score_thr] = 0
        cls_preds_expand = cls_preds.unsqueeze(-1).expand(cls_preds.shape[0], n_rois)
        idxs_expand[~v2r_one_hot] = inverse_mapping.max() + 1

        # toso: idxs is float. can these tensors be constructed with .new_zeros(..., dtype=bool) ?
        voxels_masks = idxs.new_zeros(inverse_mapping.max() + 2, n_rois, dtype=bool)
        voxels_preds = idxs.new_zeros(inverse_mapping.max() + 2, n_rois)
        voxels_preds = voxels_preds.scatter_(0, idxs_expand, cls_preds_expand)[:-1, :]
        # todo: is it ok that binary_cls_preds_expand is float?
        voxels_masks = voxels_masks.scatter_(0, idxs_expand, binary_cls_preds_expand)[:-1, :]
        scores = scores * voxels_preds.sum(axis=0) / voxels_masks.sum(axis=0)
        points_masks = voxels_masks[inverse_mapping].T.bool()
        return points_masks, labels, scores

    def _get_bboxes_single_train(self, bbox_preds, cls_preds, locations, gt_bboxes, gt_labels, img_meta):
        assigned_ids = self.first_assigner.assign(locations, gt_bboxes, gt_labels, img_meta)
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        locations = torch.cat(locations)

        pos_mask = assigned_ids >= 0
        scores = scores[pos_mask]
        bbox_preds = bbox_preds[pos_mask]
        locations = locations[pos_mask]
        assigned_ids = assigned_ids[pos_mask]

        max_scores, _ = scores.max(dim=1)
        boxes = self._bbox_pred_to_bbox(locations, bbox_preds)
        boxes_data = torch.cat((
            boxes[:, :3],
            boxes[:, 3:6] - self.padding,
            boxes.new_zeros(boxes.shape[0], 1)), dim=1)
        boxes = img_meta['box_type_3d'](boxes_data,
                                        with_yaw=False,
                                        origin=(.5, .5, .5))
        # what about padding
        return boxes, max_scores, assigned_ids, boxes_data[:,:6], scores

    def _get_instances(self, cls_preds, idxs, v2r, r2scene, scores, labels, inverse_mapping, img_metas):
        v2scene = r2scene[v2r]
        results = []
        for i in range(len(img_metas)):
            result = self._get_instances_single(
                cls_preds=cls_preds[v2scene == i],
                idxs=idxs[v2scene == i],
                v2r=v2r[v2scene == i],
                scores=scores[i],
                labels=labels[i],
                inverse_mapping=inverse_mapping)
            results.append(result)
        return results

    def _get_bboxes_train(self, bbox_preds, cls_preds, locations, gt_bboxes, gt_labels, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single_train(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                locations=[x[i] for x in locations],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def _get_bboxes_single_test(self, bbox_preds, cls_preds, locations, cfg, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        locations = torch.cat(locations)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > cfg.nms_pre > 0:
            _, ids = max_scores.topk(cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            locations = locations[ids]

        boxes = self._bbox_pred_to_bbox(locations, bbox_preds)
        boxes = torch.cat((
            boxes[:, :3],
            boxes[:, 3:6] - self.padding,
            boxes[:, 6:]), dim=1)
        boxes, scores, labels, boxes_data, scores_data = self._nms(boxes, scores, cfg, img_meta, get_data=True)
        return boxes, scores, labels, boxes_data, scores_data

    def _get_bboxes_test(self, bbox_preds, cls_preds, locations, cfg, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single_test(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                locations=[x[i] for x in locations],
                cfg=cfg,
                img_meta=img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, func_points, bbox_data_list, is_last_frame, img_metas):
        #first stage
        bbox_preds, cls_preds, locations = self._forward_first(x[1:])
        bbox_list = self._get_bboxes_test(bbox_preds, cls_preds, locations, self.test_cfg, img_metas)
        bbox_list, bbox_data_list = self._merge_and_update(bbox_data_list, [(bbox_list[i][3], bbox_list[i][4], bbox_list[i][2].float()) for i in range(len(img_metas))], self.test_cfg, img_metas, mode='test')
        
        if not is_last_frame:
            return bbox_data_list

        #second stage
        all_field = func_points[0](func_points[1], ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        all_sp = all_field.sparse()
        all_sp = ME.SparseTensor(
            features=x[0].features_at_coordinates(all_sp.coordinates.float()),
            coordinate_manager=all_sp.coordinate_manager,
            coordinate_map_key=all_sp.coordinate_map_key
        )
        inverse_mapping = all_field.inverse_mapping(all_sp.coordinate_map_key).long()
        src_idxs = torch.arange(0, all_sp.features.shape[0]).to(inverse_mapping.device)
        src_idxs = src_idxs.unsqueeze(1).expand(src_idxs.shape[0], 2)
        
        cls_preds, idxs, v2r, r2scene, rois, scores, labels = self._forward_second(all_sp, src_idxs, bbox_list)
        instances = self._get_instances(cls_preds[:, -1], idxs[:, 0], v2r, r2scene, scores, labels, inverse_mapping, img_metas)
        # instances_index = instances[0][0].new_ones((instances[0][0].shape[0])) * index
        # # merge and updates
        # instance_list, instance_data_list = self._merge_and_update(instance_data_list, [[bboxes_data[0], scores_data[0], instances[0][0], instances[0][1], instances[0][2], instances_index]], self.test_cfg, img_metas, mode='test')
        
        # # ic
        return instances, bbox_data_list


    def _nms(self, bboxes, scores, cfg, img_meta, get_data=False):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        nms_bboxes_data, nms_scores_data = [], []
        if get_data:
            keep_ids = torch.zeros(bboxes.shape[0]).bool()
        for i in range(n_classes):
            ids = scores[:, i] > cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   cfg.iou_thr)
            if get_data:
                keep_ids[ids.nonzero()[nms_ids].squeeze(-1)] = True

            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))
            nms_bboxes_data.append(class_bboxes[nms_ids,:6])
            nms_scores_data.append(scores[ids][nms_ids])

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
            nms_bboxes_data = torch.cat(nms_bboxes_data, dim=0)
            nms_scores_data = torch.cat(nms_scores_data, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))
            nms_bboxes_data = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores_data = bboxes.new_zeros((0, scores.shape[1]))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))


        # shape is different
        # latter adapt to former

        if get_data:
            return nms_bboxes, nms_scores, nms_labels, nms_bboxes_data, nms_scores_data
        return nms_bboxes, nms_scores, nms_labels
    


    def _nms_merge(self, bboxes, scores, gt_idxs, cfg, img_meta, mode='train'):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        max_scores, _ = scores.max(dim=1)

        if len(scores) > cfg.nms_pre_merge > 0:
            _, ids = max_scores.topk(cfg.nms_pre_merge)
            bboxes = bboxes[ids]
            scores = scores[ids]
            gt_idxs = gt_idxs[ids]

        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        nms_gt_idxs = []
        
        keep_ids = torch.zeros(bboxes.shape[0]).bool()
        for i in range(n_classes):
            ids = scores[:, i] > cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            class_gt_idxs = gt_idxs[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   cfg.iou_thr)
            
            keep_ids[ids.nonzero()[nms_ids].squeeze(-1)] = True

            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))
            nms_gt_idxs.append(class_gt_idxs[nms_ids])

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
            nms_gt_idxs = torch.cat(nms_gt_idxs, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))
            nms_gt_idxs = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))
        # attention!!! different from train and test 
        if mode == 'train':
            return (nms_bboxes, nms_scores, nms_gt_idxs), \
                   (bboxes[keep_ids], scores[keep_ids], gt_idxs[keep_ids])
        else:
            return (nms_bboxes, nms_scores, nms_labels), \
                   (bboxes[keep_ids], scores[keep_ids], gt_idxs[keep_ids])

    def _merge_and_update(self, bbox_pre, bbox_now, cfg, input_metas, mode):
        """Add current boxes to bbox history list.

        Args:
            bbox_pre (list[tuple[Tensor]]): Previous boxes.
            bbox_now (list[tuple[Tensor]]): Current boxes.
        
        Content: bboxes, scores, levels, points, proposals

        Returns:
            list[tuple[Tensor]]: Merged bbox list.
        """
            
        if mode == "train":
            if len(bbox_pre[0]) != 0:
                bbox_pre_new = []
                try:
                    for i in range(len(bbox_pre)):
                        bbox_pre_acc = torch.cat([bbox_pre[i][j][0] for j in range(len(bbox_pre[i]))], dim=0)
                        score_pre_acc = torch.cat([bbox_pre[i][j][1] for j in range(len(bbox_pre[i]))], dim=0)
                        gt_idx_pre_acc = torch.cat([bbox_pre[i][j][2] for j in range(len(bbox_pre[i]))], dim=0)
                        bbox_pre_new.append((bbox_pre_acc,score_pre_acc,gt_idx_pre_acc))
                except:
                    pdb.set_trace()
                bbox_now = [(torch.cat([bboxes, bboxes2], dim=0), torch.cat([scores, scores2], dim=0), torch.cat([gt_idxs, gt_idxs2], dim=0))
                    for (bboxes, scores, gt_idxs), 
                        (bboxes2, scores2, gt_idxs2) in zip(bbox_pre_new, bbox_now)]
            bbox_all_list = [self._nms_merge(bbox_now[i][0], bbox_now[i][1], bbox_now[i][2], cfg, input_metas[i], mode='train') for i in range(len(bbox_now))]
        else:
            bbox_all_list = self._merge_bbox(bbox_pre, bbox_now, cfg, input_metas)
        return [bbox_all_list[i][0] for i in range(len(bbox_all_list))], [bbox_all_list[i][1] for i in range(len(bbox_all_list))]


    def _merge_bbox(self, bboxes_pre, bboxes_now, cfg, input_metas):
        if len(bboxes_pre) == 0:
            bbox_all_list = [self._nms_merge(bboxes_now[i][0], bboxes_now[i][1], bboxes_now[i][2], cfg, input_metas[i], mode='test')
                for i in range(len(bboxes_now))]
        else:
            bbox_all_list = []
            for i in range(len(input_metas)):
                # if boxes_now
                if bboxes_pre[i][1].shape[0] == 0:
                    bbox_all_list.append(self._nms_merge(bboxes_now[i][0], bboxes_now[i][1], bboxes_now[i][2], cfg, input_metas[i], mode='test'))
                else:
                    bbox_all_list.append(self._merge_bbox_single(bboxes_pre[i], bboxes_now[i], cfg, input_metas[i]))
        return bbox_all_list
     
    def _merge_bbox_single(self, bbox_pre, bbox_now, cfg, input_meta):
        def keep_max(scores):
            idx = scores.max(1).indices
            scores_ = scores.clone()
            scores_[F.one_hot(idx, 18) == 0] = 0
            return scores_

        delta = self.test_cfg.delta; sigma = -delta
        def change_now(score_now, score_pre):
            score_now_ = score_now.clone()
            assert score_pre[score_now<self.test_cfg.score_thr].sum() == 0
            zero_cond = (score_now - score_pre < sigma) * (score_pre != 0)
            normal_cond = (sigma < score_now - score_pre) * (score_now - score_pre < delta) * (score_pre != 0)
            score_now_[normal_cond] += abs(sigma) # 0.1-->67.26/46.34; 0.2-->67.22/46.23; 0-->67.70/46.18; 0.05-->67.44/46.36
            # 0.03(2)-->67.65/46.63; 0.02(2)-->67.56/46.31; 0.025(2)-->67.68/46.69
            return score_now_

        ## handle conflict
        category_mat_pre = (keep_max(bbox_pre[1]) > 0).unsqueeze(0).repeat(bbox_now[1].shape[0], 1, 1)
        category_mat_now = (keep_max(bbox_now[1]) > 0).unsqueeze(1).repeat(1, bbox_pre[1].shape[0], 1)
        non_cat_conflict = (category_mat_pre * category_mat_now).float()
        bbox_pre_ = input_meta['box_type_3d'](bbox_pre[0], box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        bbox_now_ = input_meta['box_type_3d'](bbox_now[0], box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        ious = bbox_now_.overlaps(bbox_now_, bbox_pre_) # num_now x num_pre
        ious = ious.unsqueeze(-1) * non_cat_conflict
        score_mat_pre = bbox_pre[1].unsqueeze(0).repeat(bbox_now[1].shape[0], 1, 1)
        score_mat_now = bbox_now[1].unsqueeze(1).repeat(1, bbox_pre[1].shape[0], 1)
        score_mat_pre[ious < 0.5] = 0; score_mat_now[ious < 0.5] = 0
        ious[ious < 0.5] = 0; pre_index = ious.sum([0,2]) == 0; now_index = ious.sum([1,2]) == 0
        
        bboxes_conflict = torch.cat([bbox_pre[0][~pre_index], bbox_now[0][~now_index]], dim=0)
        scores_conflict = torch.cat([bbox_pre[1][~pre_index], 
                                    change_now(bbox_now[1][~now_index], score_mat_pre.max(1).values[~now_index])], dim=0)
        gt_idxs_conflict = torch.cat([bbox_pre[2][~pre_index], bbox_now[2][~now_index]], dim=0)

        ## nms_merge
        bbox_all = (torch.cat([bbox_pre[0][pre_index], bbox_now[0][now_index], bboxes_conflict], dim=0), 
                    torch.cat([bbox_pre[1][pre_index], bbox_now[1][now_index], scores_conflict], dim=0), 
                    torch.cat([bbox_pre[2][pre_index], bbox_now[2][now_index], gt_idxs_conflict], dim=0),)
        return self._nms_merge(bbox_all[0], bbox_all[1], bbox_all[2], cfg, input_meta, mode='test')

