import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
from mmdet3d.models import DETECTORS
from mmdet3d.models import build_backbone, build_head
from mmdet3d.core import bbox3d2result
from .base import Base3DDetector

import pdb

SAVE=False
PRINT=False

@DETECTORS.register_module()
class SegGroupFF(Base3DDetector):
    def __init__(self,
                 img_backbone,
                 backbone,
                 neck_with_head,
                 voxel_size,
                 semantic_min_threshold,
                 semantic_iter_value,
                 pretrained=False,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SegGroupFF, self).__init__()
        self.img_backbone = build_backbone(img_backbone)
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        # build roi head
        self.use_roi_head = False
        if roi_head is not None:
            self.roi_head = build_head(roi_head)
            self.use_roi_head = True
        self.voxel_size = voxel_size
        self.semantic_min_threshold = semantic_min_threshold
        self.semantic_iter_value = semantic_iter_value
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.test_count = 0
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(256, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True))
        self.init_weights()

    def init_weights(self, pretrained=None, pretrain_path=None):
        self.img_backbone.init_weights()
        self.backbone.init_weights()
        self.neck_with_head.init_weights()
    
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
    
    def extract_feat(self, points, img, img_metas, return_middle_feature=False):
        """Extract features from points."""
        with torch.no_grad():
            img_features = self.img_backbone(img)['p2'] 
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:] / 255.) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x,partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        return self.neck_with_head(x, return_middle_feature=return_middle_feature)

    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      img_metas=None):
        if self.use_roi_head:
            roi_input_dict = {}
            x, semantic_scores, voxel_offset, middile_features = self.extract_feat(points, torch.stack(img, dim=0), img_metas, return_middle_feature=True)
            centernesses, bbox_preds, cls_scores, voxel_points = list(x)
            losses = self.neck_with_head.loss(centernesses, bbox_preds, cls_scores, voxel_points, semantic_scores, voxel_offset, \
                                            gt_bboxes_3d, gt_labels_3d, points, img_metas,
                                            pts_semantic_mask, pts_instance_mask)
            ### roi head
            roi_input_dict['middle_feature_list'] = middile_features # [mink_tensor, ...]  
            # get box
            bbox_list = self.neck_with_head.get_bboxes(centernesses, bbox_preds, cls_scores, voxel_points, img_metas, rescale=False)
            roi_input_dict['pred_bbox_list'] = bbox_list # [[box, score, label], [], ...]
            roi_input_dict['gt_bboxes_3d'] = gt_bboxes_3d
            roi_input_dict['gt_labels_3d'] = gt_labels_3d
            roi_out_dict = self.roi_head(roi_input_dict)
            # also pred boxes during training for debug
            # refine_bbox_list = self.roi_head.get_boxes(roi_out_dict, img_metas)
            losses.update(self.roi_head.loss(roi_out_dict))
            # if len(bbox_list[0][0]) > 10:
            #     exit()
        else:
            x, semantic_scores, voxel_offset = self.extract_feat(points, img_metas)
            losses = self.neck_with_head.loss(*x, semantic_scores, voxel_offset, gt_bboxes_3d, gt_labels_3d, points, img_metas,
                                            pts_semantic_mask, pts_instance_mask)
        return losses

    def simple_test(self, points, img_metas, img, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        if self.use_roi_head:
            roi_input_dict = {}
            x, semantic_scores, voxel_offset, middile_features = self.extract_feat(points, torch.stack(img, dim=0), img_metas, return_middle_feature=True)
            centernesses, bbox_preds, cls_scores, voxel_points = list(x)
            bbox_list = self.neck_with_head.get_bboxes(centernesses, bbox_preds, cls_scores, voxel_points, img_metas, rescale=rescale)
            # roi head
            roi_input_dict['middle_feature_list'] = middile_features # [mink_tensor, ...]
            roi_input_dict['pred_bbox_list'] = bbox_list # [[box, score, label], [], ...]
            roi_out_dict = self.roi_head(roi_input_dict)
            bbox_list = self.roi_head.get_boxes(roi_out_dict, img_metas)
        else:
            x, semantic_scores, voxel_offset = self.extract_feat(points, img_metas)
            bbox_list = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # only batch == 1
        bbox_results[0]['semantic_scores'] = semantic_scores.F.sigmoid().cpu()
        bbox_results[0]['semantic_preds'] = self.get_semantic_pred(semantic_scores.F.sigmoid().cpu())
        bbox_results[0]['semantic_coords'] = semantic_scores.C[:, 1:].cpu() * self.voxel_size
        self.test_count += 1
        # TODO: temp for scannet
        if self.test_count % 156 == 0:
            print("***************** semantic threshold ", self.neck_with_head.semantic_threshold, "*****************")
            self.neck_with_head.semantic_threshold -= self.semantic_iter_value
            self.neck_with_head.semantic_threshold = max(self.neck_with_head.semantic_threshold, self.semantic_min_threshold)
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass

    def get_semantic_pred(self, semantic_scores):
        semantic_pred = torch.zeros_like(semantic_scores).long().fill_(self.neck_with_head.n_classes)
        for cls_id in range(self.neck_with_head.n_classes):
            cls_selected_id = torch.nonzero(semantic_scores[:, cls_id] > self.neck_with_head.semantic_threshold).squeeze(1)
            # topk_indices = torch.topk(semantic_scores, self.neck_with_head.thres_topk, -1)[1]
            # topk_selected_masks = (topk_indices == cls_id).sum(-1).bool()
            # cls_thres_selected_masks = (semantic_scores[:, cls_id] > self.neck_with_head.semantic_threshold)
            # cls_selected_id = torch.nonzero((topk_selected_masks & cls_thres_selected_masks)).squeeze(1)
            semantic_pred[cls_selected_id, cls_id] = cls_id
        return semantic_pred