try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from .base import Base3DDetector
import torch
import numpy as np
import pdb

@DETECTORS.register_module()
class TD3DInstanceSegmentor(Base3DDetector):
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
                 backbone,
                 neck,
                 head,
                 voxel_size,
                 evaluator_mode,
                 num_slice=0,
                 len_slice=0,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(TD3DInstanceSegmentor, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        # self.init_weights()
        self.evaluator_mode=evaluator_mode
        self.num_slice=num_slice
        self.len_slice=len_slice


    def view_model_param(self):
        total_param = 0
        print("MODEL DETAILS:\n")
        #print(model)
        for param in self.parameters():
            # print(param.data.size())
            total_param += np.prod(list(param.data.size()))
        print('MODEL/Total parameters:', total_param)
        
        # 假设每个参数是一个 32 位浮点数（4 字节）
        bytes_per_param = 4
        
        # 计算总字节数
        total_bytes = total_param * bytes_per_param
        
        # 转换为兆字节（MB）和千字节（KB）
        total_megabytes = total_bytes / (1024 * 1024)
        total_kilobytes = total_bytes / 1024

        print("Total parameters in MB:", total_megabytes)
        print("Total parameters in KB:", total_kilobytes)
        return total_param


    def extract_feat(self, points):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        x = self.backbone(points)
        x = self.neck(x)
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

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d,
                      pts_semantic_mask, pts_instance_mask, img_metas):
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

        # points = [torch.cat([p, torch.unsqueeze(m, 1)], dim=1) for p, m in zip(points, pts_instance_mask)]
        points = [torch.cat([p, torch.unsqueeze(inst, 1), torch.unsqueeze(sem, 1)], dim=1) for p, inst, sem in zip(points, pts_instance_mask, pts_semantic_mask)]
        field = self.collate(points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        x = field.sparse()
        targets = x.features[:, 3:].round().long()
        x = ME.SparseTensor(
            x.features[:, :3],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.extract_feat(x)
        
        losses = self.head.forward_train(x, targets, field, gt_bboxes_3d, gt_labels_3d,
                                         pts_semantic_mask, pts_instance_mask, img_metas)
        return losses

    def simple_test(self, points, img_metas, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d instances.
        """
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
                
            points_new = [points[0][ts_start:ts_end,:,:].reshape(-1,points[0].shape[-1])]
            # if points_new[0].shape[0] > 100000: 
            #     sample = torch.randint(size=(100000,), high=points_new[0].shape[0], low=0)
            #     points_new[0] = points_new[0][sample]
            field = self.collate(points_new, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            x = self.extract_feat(field.sparse())
            
            instances = self.head.forward_test(x, field, img_metas)
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