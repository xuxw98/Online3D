import MinkowskiEngine as ME

from mmdet.models import DETECTORS
from mmdet3d.models import build_backbone, build_head
from mmdet3d.core import bbox3d2result
from .base import Base3DDetector
from functools import partial
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
import torch.nn as nn
import torch


@DETECTORS.register_module()
class SingleStageSparse3DDetectorFF(Base3DDetector):
    def __init__(self,
                 img_backbone,
                 backbone,
                 neck_with_head,
                 voxel_size,
                 pretrained=False,
                 train_cfg=None,
                 test_cfg=None):
        super(SingleStageSparse3DDetectorFF, self).__init__()
        self.img_backbone = build_backbone(img_backbone)
        self.backbone = build_backbone(backbone)
        neck_with_head.update(train_cfg=train_cfg)
        neck_with_head.update(test_cfg=test_cfg)
        self.neck_with_head = build_head(neck_with_head)
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(256, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True))
        self.init_weights()

    def init_weights(self, pretrained=None):
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


    def extract_feat(self, points, img, img_metas):
        """Extract features from points."""
        with torch.no_grad():
            img_features = self.img_backbone(img)['p2'] 
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:] / 255.) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x,partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        x = self.neck_with_head(x)
        return x

    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img,
                      img_metas):
        x = self.extract_feat(points, torch.stack(img, dim=0), img_metas)
        losses = self.neck_with_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, img, imgs=None, rescale=False):
        """Test function without augmentaiton."""
    
                # Benchmark
        timestamps = []
        if self.evaluator_mode == 'slice_len_constant':
            i=1
            while i*self.len_slice<len(points[0]):
                timestamps.append(i*self.len_slice)
                i=i+1
            timestamps.append(len(points[0]))
        else:
            num_slice = min(len(points[0]),self.num_slice)
            for i in range(1,num_slice):
                timestamps.append(i*(len(points[0])//num_slice))
            timestamps.append(len(points[0]))

        # Process
        bbox_results = [[]]
        depth2img = img_metas[0]['depth2img']

        for i in range(len(timestamps)):
            if i == 0:
                ts_start, ts_end = 0, timestamps[i]
            else:
                ts_start, ts_end = timestamps[i-1], timestamps[i]
            bbox_data_list = []

            x = self.extract_feat([points[0][ts_start:ts_end]], torch.stack([img[0][ts_start:ts_end]], dim=0), img_metas)
            bbox_list = self.neck_with_head.get_bboxes(*x, img_metas, rescale=rescale)
            bboxes, scores, labels = bbox_list[0]
            ret_res = bbox3d2result(bboxes, scores, labels)
            for j in range(ts_start, ts_end):
                bbox_results[0].append(ret_res)

        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass