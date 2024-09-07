import MinkowskiEngine as ME

from mmdet.models import DETECTORS
from mmdet3d.models import build_backbone, build_head
from mmdet3d.core import bbox3d2result
from .base import Base3DDetector
import numpy as np
import pdb
import torch


@DETECTORS.register_module()
class SingleStageSparse3DDetector(Base3DDetector):
    def __init__(self,
                backbone,
                neck_with_head,
                voxel_size,
                pretrained=False,
                evaluator_mode=None,
                num_slice=None,
                len_slice=None,
                train_cfg=None,
                test_cfg=None):
        super(SingleStageSparse3DDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.evaluator_mode=evaluator_mode
        self.num_slice=num_slice
        self.len_slice=len_slice
        self.init_weights()

    def init_weights(self, pretrained=None):
        self.backbone.init_weights()
        self.neck_with_head.init_weights()

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:] / 255.) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x)
        x = self.neck_with_head(x)
        return x

    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas):
        x = self.extract_feat(points, img_metas)
        losses = self.neck_with_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
               """Test function without augmentaiton."""
        x = self.extract_feat([points[0]], img_metas)
        bbox_list = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass
