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
class MinkUnetSemHead(BaseModule):
    def __init__(self,
        voxel_size,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            ignore_index=20,
            loss_weight=1.0),
        train_cfg=None,
        test_cfg=None):
        super(MinkUnetSemHead, self).__init__()
        self.voxel_size = voxel_size
        self.loss_decode = build_loss(loss_decode)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _loss(self, preds, targets, img_metas):
        sem_losses = []
        for i in range(len(img_metas)):
            sem_loss = self._loss_single(
                preds=preds[i],
                targets=targets[i],
                img_meta=img_metas[i])
            sem_losses.append(sem_loss)
        return dict(sem_loss=torch.mean(torch.stack(sem_losses)))

    def _loss_single(self, preds, targets, img_meta):
        seg_targets = targets[:, 0]
        seg_loss = self.loss_decode(preds, seg_targets.long())
        return seg_loss
 
    def forward_train(self, x, targets, points, pts_semantic_mask, img_metas):
        preds_list = []
        targets_list = []
        preds = x.features
        for permutation in x.decomposition_permutations:
            preds_list.append(preds[permutation])
            targets_list.append(targets[permutation])
        losses = self._loss(preds_list, targets_list, img_metas)
        return losses

    def forward_test(self, x, points, img_metas):
        inverse_mapping = points.inverse_mapping(x.coordinate_map_key).long()
        point_semseg = x.features[inverse_mapping]
        return point_semseg.argmax(dim=-1)
