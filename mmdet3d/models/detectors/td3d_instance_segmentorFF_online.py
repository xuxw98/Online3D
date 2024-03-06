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
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox import BaseInstance3DBoxes
import numpy as np
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
                 memory_insseg=None,
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
        if memory_insseg is not None:
            self.memory_insseg = build_neck(memory_insseg)
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
        if hasattr(self, 'memory_insseg'):
            self.memory_insseg.init_weights()
        self.neck.init_weights()
        self.head.init_weights()


    
    def extract_feat(self, points, img, img_metas, targets=None, mode='train', ts=0):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        if hasattr(self, 'img_memory'):
            self.img_memory.register(self.memory.accumulated_feats)
            img_features = self.img_backbone(img, memory=partial(self.img_memory, img_metas=img_metas))['p2']
        else:
            with torch.no_grad():
                img_features = self.img_backbone(img)['p2']

        x = self.backbone(points, partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        
        if hasattr(self, 'memory'):
            x = self.memory(x)
        x = self.neck(x)


        if hasattr(self, 'memory_insseg'):
            if mode=='test':
                zero = ME.SparseTensor(x[0].features.new_zeros((x[0].shape[0],2)),coordinate_map_key=x[0].coordinate_map_key,coordinate_manager=x[0].coordinate_manager)
                x.append(zero)
                x = self.memory_insseg(x, mode='test', ts=ts)
                targets = x[-1]
                x = x[:-1]
            else:
                x.append(targets)
                x = self.memory_insseg(x, mode='train', ts=ts)
                targets = x[-1]
                x = x[:-1]


        if mode == 'train':
            return x, targets
        else:
            return x

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
        amodal_box_for_all_frame = []
        label_for_each_frame = []
        label_for_all_frame = []

        for i in range(img_metas[0]['num_frames']):
            modal_boxes, modal_labels, amodal_boxes = [], [], []
            all_labels, all_amodal_boxes = [], []
            for j in range(len(img_metas)):
                # I'm not sure
                # need amodal_box             
                modal_boxes.append(img_metas[j]['box_type_3d'](modal_box[j][i],
                    box_dim=modal_box[j][i].shape[-1], with_yaw=False, origin=(.5, .5, .5)))
                modal_labels.append(modal_label[j][i])
                if (amodal_box_mask[j][i] == False).all():
                    amodal_boxes.append(img_metas[j]['box_type_3d'](torch.zeros(1,7), with_yaw=False, origin=(.5, .5, .5)))
                else:
                    amodal_boxes.append(gt_bboxes_3d[j][amodal_box_mask[j][i]])
                    assert (modal_label[j][i] == gt_labels_3d[j][amodal_box_mask[j][i]]).all()

                all_amodal_box_mask = (amodal_box_mask[j][:i+1].sum(dim=0)>=1)
                all_amodal_boxes.append(gt_bboxes_3d[j][all_amodal_box_mask])
                all_labels.append(gt_labels_3d[j][all_amodal_box_mask])
                

            modal_box_for_each_frame.append(modal_boxes)
            amodal_box_for_each_frame.append(amodal_boxes)
            label_for_each_frame.append(modal_labels)
            amodal_box_for_all_frame.append(all_amodal_boxes)
            label_for_all_frame.append(all_labels)

        amodal_box_for_all_frame_new = [gt_bboxes_3d[i] for i in range(len(img_metas))]
        label_for_all_frame_new = [gt_labels_3d[i] for i in range(len(img_metas))]

        losses = {}
        bbox_data_list = []
        depth2img = [img_meta['depth2img'] for img_meta in img_metas]
        if hasattr(self, 'img_memory'):
            self.img_memory.reset()
        if hasattr(self, 'memory'):
            self.memory.reset()
        if hasattr(self, 'memory_insseg'):
            self.memory_insseg.reset()
        for i in range(img_metas[0]['num_frames']):
            for j in range(len(img_metas)):
                img_metas[j]['depth2img'] = depth2img[j][i]
            current_points = [scene_points[i] for scene_points in points]
            current_pts_semantic_mask = [scene_pts_semantic_mask[i] for scene_pts_semantic_mask in pts_semantic_mask]
            current_pts_instance_mask = [scene_pts_instance_mask[i] for scene_pts_instance_mask in pts_instance_mask]
            current_img = [scene_img[i] for scene_img in img]
            cur_points = [torch.cat([p, torch.unsqueeze(inst, 1), torch.unsqueeze(sem, 1)], dim=1) for p, inst, sem in zip(current_points, current_pts_instance_mask, current_pts_semantic_mask)]
            cur_field = self.collate(cur_points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
            xf = cur_field.sparse()
            x = ME.SparseTensor(
                xf.features[:, :3],
                coordinate_map_key=xf.coordinate_map_key,
                coordinate_manager=xf.coordinate_manager)
            cur_targets = ME.SparseTensor(
                xf.features[:, 3:],
                coordinate_map_key=xf.coordinate_map_key,
                coordinate_manager=xf.coordinate_manager)
            # current_feats = self.extract_feat(x, torch.stack(current_img, dim=0), img_metas)
            current_feats, acc_targets = self.extract_feat(x, torch.stack(current_img, dim=0), img_metas, targets=cur_targets, mode='train', ts=i)
            loss, bbox_data_list = self.head.forward_train(current_feats, acc_targets.features_at_coordinates(current_feats[0].coordinates.float()).long(), cur_field, 
                                            amodal_box_for_all_frame_new, label_for_all_frame_new,
                                            current_pts_semantic_mask, current_pts_instance_mask, bbox_data_list, img_metas)
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
            bbox_data_list = []
            if hasattr(self, 'img_memory'):
                self.img_memory.reset()
            if hasattr(self, 'memory'):
                self.memory.reset()
            if hasattr(self, 'memory_insseg'):
                self.memory_insseg.reset()
            for j in range(ts_start, ts_end):
                img_metas[0]['depth2img'] = depth2img[j]
                current_points = [scene_points[j] for scene_points in points]
                current_img = [scene_img[j] for scene_img in img]
                x = self.collate(current_points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE).sparse()
                current_feats = self.extract_feat(x, torch.stack(current_img, dim=0), img_metas, mode='test', ts=j)
                all_points = [scene_points[ts_start: j+1].view(-1, 6) for scene_points in points]
                
                if j == ts_end - 1:
                    instances, bbox_data_list = self.head.forward_test(current_feats, (self.collate, all_points), bbox_data_list, True, img_metas)
                    results = []
                    for mask, label, score in instances:
                        results.append(dict(
                            instance_mask=mask.cpu(),
                            instance_label=label.cpu(),
                            instance_score=score.cpu()))
                    instances_results[0].append(results)
                else:
                    bbox_data_list = self.head.forward_test(current_feats, None, bbox_data_list, False, img_metas)

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
