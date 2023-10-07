try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from .base import Base3DDetector
from functools import partial
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
import torch.nn as nn
import torch
import pdb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


@DETECTORS.register_module()
class MinkUnetSemsegFF(Base3DDetector):
    def __init__(self,
                 img_backbone,
                 backbone,
                 head,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(MinkUnetSemsegFF, self).__init__(init_cfg)
        self.img_backbone = build_backbone(img_backbone)
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        # 128 for Res50Unet 32 for Res18UNet
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(32, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))      
        self.init_weights()

    def init_weights(self, pretrained=None):
        self.img_backbone.init_weights()
        self.backbone.init_weights()
        self.head.init_weights()

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
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        with torch.no_grad():
            img_features = self.img_backbone(img)

        x = self.backbone(points,partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        return x   
    
    def collate(self, points, quantization_mode):
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            dtype=points[0].dtype,
            device=points[0].device)
        return ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=quantization_mode,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device,
        )

    def forward_train(self, points, pts_semantic_mask, img, img_metas):
        """Forward of training.

        Returns:
            dict: Loss values.
        """
        # points = [torch.cat([p, torch.unsqueeze(m, 1)], dim=1) for p, m in zip(points, pts_instance_mask)]
        points = [torch.cat([p, torch.unsqueeze(sem, 1)], dim=1) for p, sem in zip(points, pts_semantic_mask)]
        field = self.collate(points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        x = field.sparse()
        
        targets = x.features[:, 3:].round().long()
        x = ME.SparseTensor(
            x.features[:, :3],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.extract_feat(x, torch.stack(img, dim=0), img_metas)
        
        losses = self.head.forward_train(x, targets, field, pts_semantic_mask, img_metas)
        return losses

    def simple_test(self, points, img_metas, img, *args, **kwargs):
        """Test without augmentations.
        """

        field = self.collate(points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        x = self.extract_feat(field.sparse(), torch.stack(img, dim=0), img_metas)
        
        preds = self.head.forward_test(x, field, img_metas)
        results = [dict(semantic_mask=preds.cpu())]
        return results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError


@DETECTORS.register_module()
class MinkUnetSemseg(Base3DDetector):
    def __init__(self,
                 backbone,
                 head,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(MinkUnetSemseg, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.voxel_size = voxel_size

    def init_weights(self, pretrained=None):
        # self.backbone.init_weights()
        self.head.init_weights()

    def extract_feat(self, points, img_metas):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        x = self.backbone(points)
        return x   
    
    def collate(self, points, quantization_mode):
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            dtype=points[0].dtype,
            device=points[0].device)
        return ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=quantization_mode,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device,
        )

    def forward_train(self, points, pts_semantic_mask, img_metas):
        """Forward of training.

        Returns:
            dict: Loss values.
        """
        
        points = [point.reshape(-1,6) for point in points]
        # points = [torch.cat([p, torch.unsqueeze(m, 1)], dim=1) for p, m in zip(points, pts_instance_mask)]
        points = [torch.cat([p, torch.unsqueeze(sem, 1)], dim=1) for p, sem in zip(points, pts_semantic_mask)]
        field = self.collate(points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        x = field.sparse()
        
        targets = x.features[:, 3:].round().long()
        x = ME.SparseTensor(
            x.features[:, :3],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.extract_feat(x, img_metas)
        
        losses = self.head.forward_train(x, targets, field, pts_semantic_mask, img_metas)
        return losses

    def simple_test(self, points, img_metas, *args, **kwargs):
        """Test without augmentations.
        """

        points = [point.reshape(-1,6) for point in points]
        field = self.collate(points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        x = self.extract_feat(field.sparse(), img_metas)
        
        preds = self.head.forward_test(x, field, img_metas)
        results = [dict(semantic_mask=preds.cpu())]
        return results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
