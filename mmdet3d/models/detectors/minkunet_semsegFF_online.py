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
class MinkUnetSemsegFF_Online(Base3DDetector):
    def __init__(self,
                 img_backbone,
                 backbone,
                 head,
                 voxel_size,                 
                 evaluator_mode,
                 num_slice=0,
                 len_slice=0,
                 memory=None,
                 img_memory=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(MinkUnetSemsegFF_Online, self).__init__(init_cfg)
        self.img_backbone = build_backbone(img_backbone)
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        self.evaluator_mode=evaluator_mode
        self.num_slice=num_slice
        self.len_slice=len_slice
        # 128 for Res50Unet 32 for Res18UNet
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(32, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))
        if memory is not None:
            self.memory = build_neck(memory)
        if img_memory is not None:
            img_memory['voxel_size'] = voxel_size
            self.img_memory = build_neck(img_memory)
        self.init_weights()

    def init_weights(self, pretrained=None):
        self.img_backbone.init_weights()
        self.backbone.init_weights()
        self.head.init_weights()
        if hasattr(self, 'memory'):
            self.memory.init_weights()
        if hasattr(self, 'img_memory'):
            self.img_memory.init_weights()

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
        if hasattr(self, 'img_memory'):
            self.img_memory.register(self.memory.accumulated_feats)
            img_features = self.img_backbone(img, memory=partial(self.img_memory, img_metas=img_metas))
        else:
            with torch.no_grad():
                img_features = self.img_backbone(img)
        x = self.backbone(points,partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape),
            memory=self.memory if hasattr(self,'memory') else None)
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

        losses = {}
        depth2img = [img_meta['depth2img'] for img_meta in img_metas]
        if hasattr(self, 'img_memory'):
            self.img_memory.reset()
        if hasattr(self, 'memory'):
            self.memory.reset()
        for i in range(img_metas[0]['num_frames']):
            for j in range(len(img_metas)):
                img_metas[j]['depth2img'] = depth2img[j][i]
            current_points = [scene_points[i] for scene_points in points]
            current_pts_semantic_mask = [scene_pts_semantic_mask.reshape(img_metas[0]['num_frames'],-1)[i] for scene_pts_semantic_mask in pts_semantic_mask]
            current_img = [scene_img[i] for scene_img in img]
            cur_points = [torch.cat([p, torch.unsqueeze(sem, 1)], dim=1) for p, sem in zip(current_points, current_pts_semantic_mask)]

            field = self.collate(cur_points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
            x = field.sparse()
        
            targets = x.features[:, 3:].round().long()
            x = ME.SparseTensor(
                x.features[:, :3],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
            
            x = self.extract_feat(x, torch.stack(current_img, dim=0), img_metas)
        
            loss = self.head.forward_train(x, targets, field, current_pts_semantic_mask, img_metas)

            for key, value in loss.items():
                if key in losses: 
                    losses[key] += value
                else:
                    losses[key] = value 
        return losses

    def simple_test(self, points, img_metas, img, *args, **kwargs):
        """Test without augmentations.
        """
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
        semseg_results = []
        depth2img = img_metas[0]['depth2img']

        for i in range(len(timestamps)):
            if i == 0:
                ts_start, ts_end = 0, timestamps[i]
            else:
                ts_start, ts_end = timestamps[i-1], timestamps[i]
            sem_result = []
            if hasattr(self, 'img_memory'):
                self.img_memory.reset()
            if hasattr(self, 'memory'):
                self.memory.reset()
            for j in range(ts_start, ts_end):
                img_metas[0]['depth2img'] = depth2img[j]
                field = self.collate([points[0][j]], ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
                x = self.extract_feat(field.sparse(), torch.stack([img[0][j]], dim=0), img_metas)
                
                preds = self.head.forward_test(x, field, img_metas)
                sem_result.append(preds)
                if j == ts_end-1:
                    sem_preds = torch.cat(sem_result, dim=0) 
                    semseg_results.append(sem_preds.cpu())


        results = [dict(semantic_mask=torch.cat(semseg_results,dim=0))]    
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