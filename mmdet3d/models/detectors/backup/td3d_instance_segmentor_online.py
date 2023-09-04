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
class TD3DInstanceSegmentor_Online(Base3DDetector):
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
                 vmp_layer=(0,1,2,3),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 num_slice=0,
                 len_slice=0):
        super(TD3DInstanceSegmentor_Online, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        assert evaluator_mode in ['slice_len_constant','slice_num_constant']
        self.evaluator_mode = evaluator_mode
        self.num_slice = num_slice
        self.len_slice = len_slice


        self.scale = 2.5
        self.conv1 = nn.ModuleList()
        self.conv12 = nn.ModuleList()
        # conv_convert conv_d3 conv_d1
        self.conv2 = nn.ModuleList()
        for i, C in enumerate([64]):
            #if i in self.vmp_layer:
            self.conv1.append(nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=5,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=3),
                ME.MinkowskiBatchNorm(C),
                ME.MinkowskiReLU()))
            self.conv12.append(nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=3,
                    stride=1,
                    dilation=5,
                    bias=False,
                    dimension=3),
                ME.MinkowskiBatchNorm(C),
                ME.MinkowskiReLU()))
            self.conv2.append(nn.Sequential(
                ME.MinkowskiConvolutionTranspose(
                    in_channels=2*C,
                    out_channels=C,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    bias=False,
                    dimension=3),
                ME.MinkowskiBatchNorm(C)))
        # else:
        #     self.conv1.append(nn.Identity())
        #     self.conv12.append(nn.Identity())
        #     self.conv2.append(nn.Identity())
        self.relu = ME.MinkowskiReLU()
        

        self.init_weights()

    def init_weights(self, pretrained=None):
        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def extract_feat(self, points, img, img_metas):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        x = self.backbone(points)
        #x = self.backbone(points)
        x = self.neck(x)
        return x   
    
    def two_cat(self, feat1, feat2):
        coords1 = feat1.decomposed_coordinates
        feats1 = feat1.decomposed_features
        coords2 = feat2.decomposed_coordinates
        feats2 = feat2.decomposed_features
        for i in range(len(coords1)):
            # shape 1 N
            feats1[i] = torch.cat([feats1[i], feats2[i]], dim=1)       
        coords_sp, feats_sp = ME.utils.sparse_collate(coords1, feats1)
        feat_new = ME.SparseTensor(
            coordinates=coords_sp,
            features=feats_sp,
            tensor_stride=feat1.tensor_stride,
            coordinate_manager=feat1.coordinate_manager
        )
        return feat_new
    

    def accumulate_conv(self, accumulated_feat, current_feat, index):
        """Accumulate features for a single stage.

        Args:
            accumulated_feat (ME.SparseTensor)
            current_feat (ME.SparseTensor)

        Returns:
            ME.SparseTensor: refined accumulated features
            ME.SparseTensor: current features after accumulation
        """
        #if index in self.vmp_layer:
            # VMP
        tensor_stride = current_feat.tensor_stride
        accumulated_feat = ME.TensorField(
            features=torch.cat([current_feat.features, accumulated_feat.features], dim=0),
            coordinates=torch.cat([current_feat.coordinates, accumulated_feat.coordinates], dim=0),
            quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
        ).sparse()
        accumulated_feat = ME.SparseTensor(
            coordinates=accumulated_feat.coordinates,
            features=accumulated_feat.features,
            tensor_stride=tensor_stride,
            coordinate_manager=accumulated_feat.coordinate_manager
        )

        # Select neighbor region for current frame
        accumulated_coords = accumulated_feat.decomposed_coordinates
        current_coords = current_feat.decomposed_coordinates
        accumulated_coords_select_list=[]
        zero_batch_feature_list=[]
        for i in range(len(current_coords)):
            accumulated_coords_batch = accumulated_coords[i]
            current_coords_batch = current_coords[i]
            current_coords_batch_max, _ = torch.max(current_coords_batch,dim=0)
            current_coords_batch_min, _ = torch.min(current_coords_batch,dim=0)
            current_box_size = current_coords_batch_max - current_coords_batch_min
            current_box_add = ((self.scale-1)/2) * current_box_size
            margin_positive = accumulated_coords_batch-current_coords_batch_max-current_box_add
            margin_negative = accumulated_coords_batch-current_coords_batch_min+current_box_add
            in_criterion = torch.mul(margin_positive,margin_negative)
            zero = torch.zeros_like(in_criterion)
            one = torch.ones_like(in_criterion)
            in_criterion = torch.where(in_criterion<=0,one,zero)
            mask = in_criterion[:,0]*in_criterion[:,1]*in_criterion[:,2]
            mask = mask.type(torch.bool)
            mask = mask.reshape(mask.shape[0],1)
            accumulated_coords_batch_select = torch.masked_select(accumulated_coords_batch,mask)
            accumulated_coords_batch_select = accumulated_coords_batch_select.reshape(-1,3)
            zero_batch_feature = torch.zeros_like(accumulated_coords_batch_select)
            accumulated_coords_select_list.append(accumulated_coords_batch_select)
            zero_batch_feature_list.append(zero_batch_feature)
        accumulated_coords_select_coords, _ = ME.utils.sparse_collate(accumulated_coords_select_list, zero_batch_feature_list)
        current_feat_new = ME.SparseTensor(
            coordinates=accumulated_coords_select_coords,
            features=accumulated_feat.features_at_coordinates(accumulated_coords_select_coords.float()),
            tensor_stride=tensor_stride
        )

        branch1 = self.conv1[index](current_feat_new)
        branch2 = self.conv12[index](current_feat_new)
        branch  = self.two_cat(branch1, branch2)
        branch = self.conv2[index](branch)
        current_feat_new = branch + current_feat_new
        current_feat_new = self.relu(current_feat_new)
        current_feat = ME.SparseTensor(
            coordinates=current_feat.coordinates,
            features=current_feat_new.features_at_coordinates(current_feat.coordinates.float()),
            tensor_stride=tensor_stride,
            coordinate_manager=current_feat.coordinate_manager
        )
        return accumulated_feat, current_feat

    def accumulate_vmp(self, accumulated_feat, current_feat, index):
        """Accumulate features for a single stage.

        Args:
            accumulated_feat (ME.SparseTensor)
            current_feat (ME.SparseTensor)

        Returns:
            ME.SparseTensor: refined accumulated features
            ME.SparseTensor: current features after accumulation
        """
        if index in self.vmp_layer:
            tensor_stride = current_feat.tensor_stride
            accumulated_feat = ME.TensorField(
                features=torch.cat([current_feat.features, accumulated_feat.features], dim=0),
                coordinates=torch.cat([current_feat.coordinates, accumulated_feat.coordinates], dim=0),
                quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
            ).sparse()
            accumulated_feat = ME.SparseTensor(
                coordinates=accumulated_feat.coordinates,
                features=accumulated_feat.features,
                tensor_stride=tensor_stride,
                coordinate_manager=accumulated_feat.coordinate_manager
            )

            current_feat = ME.SparseTensor(
                features=accumulated_feat.features_at_coordinates(current_feat.coordinates.float()),
                tensor_stride=tensor_stride,
                coordinate_map_key=current_feat.coordinate_map_key,
                coordinate_manager=current_feat.coordinate_manager
            )
        return accumulated_feat, current_feat

    def process_one_frame(self, accumulated_feats, points, img_metas):
        """Extract and accumulate features from current frame.

        Args:
            accumulated_feats (list[ME.SparseTensor]) --> list of different stages
            current_frames (list[Tensor]) --> list of batch

        Returns:
            list[ME.SparseTensor]: refined accumulated features
            list[ME.SparseTensor]: current features after accumulation
        """  
        x = self.backbone(points)

        if accumulated_feats is None:
            accumulated_feats = x
    
            branch1 = self.conv1[0](x[0])
            branch2 = self.conv12[0](x[0])
            branch  = self.two_cat(branch1, branch2)
            branch = self.conv2[0](branch)
            x[0] = branch + x[0]
            x[0] = self.relu(x[0])
            return accumulated_feats, x
        else:
            tuple_feats_0 = [self.accumulate_conv(accumulated_feats[0], x[0], 0)]
            tuple_feats_1 = [self.accumulate_vmp(accumulated_feats[1][i], x[1][i], i) for i in range(len(x[1]))]
            return [tuple_feats_0[0][0], [tuple_feats_1[i][0] for i in range(len(x[1]))]], [tuple_feats_0[0][1],[tuple_feats_1[i][1] for i in range(len(x[1]))]]
        

    
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

    def forward_train(self, points, modal_box, modal_label, amodal_box, amodal_label,
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
        losses = {}
        accumulated_feats = None
        depth2img = [img_meta['depth2img'] for img_meta in img_metas]
        #if self.use_roi_head:
        for i in range(img_metas[0]['num_frames']):
            for j in range(len(img_metas)):
                img_metas[j]['depth2img'] = depth2img[j][i]
            current_points = [scene_points[i] for scene_points in points]
            current_pts_semantic_mask = [scene_pts_semantic_mask[i] for scene_pts_semantic_mask in pts_semantic_mask]
            current_pts_instance_mask = [scene_pts_instance_mask[i] for scene_pts_instance_mask in pts_instance_mask]
            current_gt_bboxes_3d = [scene_modal_box[i] for scene_modal_box in modal_box]
            current_gt_labels_3d = [scene_modal_label[i] for scene_modal_label in modal_label]
            # points = [torch.cat([p, torch.unsqueeze(m, 1)], dim=1) for p, m in zip(points, pts_instance_mask)]
            cur_points = [torch.cat([p, torch.unsqueeze(inst, 1), torch.unsqueeze(sem, 1)], dim=1) for p, inst, sem in zip(current_points, current_pts_instance_mask, current_pts_semantic_mask)]
            cur_field = self.collate(cur_points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
            x = cur_field.sparse()
            cur_targets = x.features[:, 3:].round().long()
            x = ME.SparseTensor(
                x.features[:, :3],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
            accumulated_feats, current_feats = self.process_one_frame(accumulated_feats,
                    x,img_metas)
            current_feats = self.neck(current_feats)
        
            loss = self.head.forward_train(current_feats, cur_targets, cur_field, 
                                           current_gt_bboxes_3d, current_gt_labels_3d,
                                           current_pts_semantic_mask, current_pts_instance_mask, img_metas)
            for key, value in loss.items():
                if key in losses: 
                    losses[key] += value
                else:
                    losses[key] = value
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
        # special condition  only 4 frames but require 5 slices so just 4slices
        # as short as I can 

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

        #bbox_data_list = []
        depth2img = img_metas[0]['depth2img']
        instances_results = [[]]
        for i in range(len(timestamps)):
            if i==0:
                ts_start,ts_end=0,timestamps[i]
            else:
                ts_start,ts_end=timestamps[i-1],timestamps[i]
            accumulated_feats = None
            for j in range(ts_start,ts_end):
                img_metas[0]['depth2img'] = depth2img[j]
                current_points = [scene_points[j] for scene_points in points]

                cur_field = self.collate(current_points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
                x = cur_field.sparse()
                accumulated_feats, current_feats = self.process_one_frame(accumulated_feats,
                    x,img_metas)
                current_feats = self.neck(current_feats)
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
