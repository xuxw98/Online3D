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
from mmdet3d.core.bbox import BaseInstance3DBoxes
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


@DETECTORS.register_module()
class TD3DInstanceSegmentorFF_Online(Base3DDetector):
    r"""Two-stage instance segmentor based on MinkowskiEngine.
    The first stage is bbox detector. The second stage is two-class pointwise segmentor (foreground/background).

    Args:
        backbone (dict): Config of the backbone.
        neck (dict): Config of the neck.
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """
    def __init__(self,
                 img_backbone,
                 backbone,
                 neck,
                 head,
                 voxel_size,
                 evaluator_mode,
                 num_slice=0,
                 len_slice=0,
                 img_memory=None,
                 memory=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(TD3DInstanceSegmentorFF_Online, self).__init__(init_cfg)
        assert evaluator_mode in ['slice_len_constant','slice_num_constant']
        self.evaluator_mode=evaluator_mode
        self.num_slice=num_slice
        self.len_slice=len_slice
        self.img_backbone = build_backbone(img_backbone)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        if memory is not None:
            self.memory = build_neck(memory)
        if img_memory is not None:
            img_memory['voxel_size'] = voxel_size
            self.img_memory = build_neck(img_memory)
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(256, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True))
        self.voxel_size = voxel_size

    def init_weights(self, pretrained=None):
        self.img_backbone.init_weights()
        self.backbone.init_weights()
        if hasattr(self, 'memory'):
            self.memory.init_weights()
        if hasattr(self, 'img_memory'):
            self.img_memory.init_weights()
        self.neck.init_weights()
        self.head.init_weights()
    
    def extract_feat(self, points, img, img_metas):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        with torch.no_grad():
            img_features = self.img_backbone(img)['p2']
        if hasattr(self, 'img_memory'):
            img_features = self.img_memory(img_features, img_metas)

        x = self.backbone(points, partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape, 
            img_memory=self.img_memory if hasattr(self, 'img_memory') else None))
        if hasattr(self, 'memory'):
            x = self.memory(x)
        x = self.neck(x)
        return x

    def _f(self, x, img_features, img_metas, img_shape, img_memory):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
            if img_memory is not None:
                points[i] = torch.cat([points[i], img_memory.pre_points[i]], dim=0)
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
            
        if img_memory is not None:
            img_memory.register([pf[-img_memory.num_points:] for pf in projected_features], 'feature')
            projected_features = [pf[:-img_memory.num_points] for pf in projected_features]

        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        projected_features = self.conv(projected_features)
        return projected_features + x
    
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
            device=points[0].device)
    
    def bool_select(self, data, select):
        if isinstance(data, BaseInstance3DBoxes):
            assert data.tensor.shape[0] == select.shape[0]
            data_new = []
            for i in range(len(select)):
                if select[i] == 1:
                    data_new.append(data.tensor[i])
            data.tensor = torch.cat(data_new, dim=0)
            return data         
        else:
            assert data.shape[0] == select.shape[0]
            data_new = []
            for i in range(len(select)):
                if select[i] == 1:
                    data_new.append(data[i])
            return torch.cat(data_new, dim=0)

    def forward_train(self, points, modal_box, modal_label, amodal_box_mask,
                      pts_semantic_mask, pts_instance_mask, gt_bboxes_3d, gt_labels_3d, 
                     img, img_metas):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes_3d (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (list[torch.Tensor]): Per point semantic labels
                of each sample.
            pts_instance_mask (list[torch.Tensor]): Per point instance labels
                of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Loss values.
        """

        # Process ground-truth
        modal_box_for_each_frame = []
        amodal_box_for_each_frame = []
        label_for_each_frame = []
        for i in range(img_metas[0]['num_frames']):
            modal_boxes, modal_labels, amodal_boxes = [], [], []
            for j in range(len(img_metas)):
                modal_boxes.append(img_metas[j]['box_type_3d'](modal_box[j][i],
                     box_dim=modal_box[j][i].shape[-1], with_yaw=False, origin=(.5, .5, .5)))
                modal_labels.append(modal_label[j][i])
                if (amodal_box_mask[j][i] == False).all():
                    amodal_boxes.append(img_metas[j]['box_type_3d'](torch.zeros(1,7), with_yaw=False, origin=(.5, .5, .5)))
                else:
                    amodal_boxes.append(gt_bboxes_3d[j][amodal_box_mask[j][i]])
                    assert (modal_label[j][i] == gt_labels_3d[j][amodal_box_mask[j][i]]).all()
            modal_box_for_each_frame.append(modal_boxes)
            amodal_box_for_each_frame.append(amodal_boxes)
            label_for_each_frame.append(modal_labels)

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
            self.img_memory.register([p[:,:3] for p in current_points], 'point')
            current_pts_semantic_mask = [scene_pts_semantic_mask[i] for scene_pts_semantic_mask in pts_semantic_mask]
            current_pts_instance_mask = [scene_pts_instance_mask[i] for scene_pts_instance_mask in pts_instance_mask]
            current_img = [scene_img[i] for scene_img in img]
            cur_points = [torch.cat([p, torch.unsqueeze(inst, 1), torch.unsqueeze(sem, 1)], dim=1) for p, inst, sem in zip(current_points, current_pts_instance_mask, current_pts_semantic_mask)]
            cur_field = self.collate(cur_points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
            x = cur_field.sparse()
            cur_targets = x.features[:, 3:].round().long()
            x = ME.SparseTensor(
                x.features[:, :3],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager)
            
            current_feats = self.extract_feat(x, torch.stack(current_img, dim=0), img_metas)
            loss = self.head.forward_train(current_feats, cur_targets, cur_field, 
                                            modal_box_for_each_frame[i], label_for_each_frame[i],
                                            current_pts_semantic_mask, current_pts_instance_mask, img_metas)
            for key, value in loss.items():
                if key in losses: 
                    losses[key] += value
                else:
                    losses[key] = value
        return losses

    def simple_test(self, points, img_metas, img, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d instances.
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
        depth2img = img_metas[0]['depth2img']
        instances_results = [[]]

        for i in range(len(timestamps)):
            if i == 0:
                ts_start, ts_end = 0, timestamps[i]
            else:
                ts_start, ts_end = timestamps[i-1], timestamps[i]
            if hasattr(self, 'img_memory'):
                self.img_memory.reset()
            if hasattr(self, 'memory'):
                self.memory.reset()
            for j in range(ts_start, ts_end):
                img_metas[0]['depth2img'] = depth2img[j]
                current_points = [scene_points[j] for scene_points in points]
                current_img = [scene_img[j] for scene_img in img]

                cur_field = self.collate(current_points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
                x = cur_field.sparse()
                current_feats = self.extract_feat(x, torch.stack(current_img, dim=0), img_metas)
                instances = self.head.forward_test(current_feats, cur_field, img_metas)
                results = []
                for mask, label, score in instances:
                    results.append(dict(
                        instance_mask=mask.cpu(),
                        instance_label=label.cpu(),
                        instance_score=score.cpu()))
                instances_results[0].append(results)
                
        return instances_results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
