# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp

import numpy as np
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass
from mmdet3d.core import (
    instance_seg_eval, instance_seg_eval_v2, show_result, show_seg_result)
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose
import torch
import pdb

@DATASETS.register_module()
class SceneNNMVSegDataset(Custom3DSegDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
            'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
            'curtain', 'refrigerator', 'sink')
    ALL_CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')

    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 34,)    
    ALL_VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)


    ALL_CLASS_IDS = tuple(range(41))

    PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 evaluator_mode='slice_len_constant',
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_depth=True),
                 test_mode=False,
                 palette=None,
                 num_slice=0,
                 len_slice=0,
                 ignore_index=None,
                 voxel_size=.02,
                 use_voxel_eval=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            palette=palette,
            modality=modality,
            ignore_index=ignore_index,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']
        assert evaluator_mode in ['slice_len_constant','slice_num_constant']
        self.evaluator_mode = evaluator_mode
        self.num_slice = num_slice
        self.len_slice = len_slice

        self.cat_ids = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }

        self.voxel_size = voxel_size
        self.use_voxel_eval = use_voxel_eval

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['scene_idx']
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth'] and self.modality['use_camera']:
            pts_info = []
            for pts_path in info['pts_paths']:
                pts_info.append(
                    dict(filename=osp.join(self.data_root, pts_path)))
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            semantic_info = []
            for semantic_path in info['semantic_paths']:
                semantic_info.append(
                    dict(filename=osp.join(self.data_root, semantic_path)))
            
            # depth2img = []
            # intrinsic = np.array([[288.9353025,0,159.5,0],[0,288.9353025,119.5,0],[0,0,1,0],[0,0,0,1]])
            # for pose in info['poses']:
            #     depth2img.append(
            #         intrinsic @ np.linalg.inv(pose))
            poses = info['poses']

            input_dict['pts_info'] = pts_info
            input_dict['img_info'] = img_info
            #input_dict['depth2img'] = depth2img
            input_dict['poses'] = poses
            input_dict['semantic_info'] = semantic_info

        if not self.test_mode:
            #annos = self.get_ann_info(index)
            input_dict['ann_info'] = {}
        return input_dict


    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        # input_dict['ann_info'] = self.get_ann_info(index)
        # input_dict['ann_info'] = dict(
        #     axis_align_matrix=self._get_axis_align_matrix(
        #         self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in semantic segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Defaults to False.
            out_dir (str, optional): Path to save the visualization results.
                Defaults to None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import multiview_seg_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        load_pipeline = self._build_default_pipeline()
        pred_sem_masks = [result['semantic_mask'] for result in results]

        points, gt_sem_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['points', 'pts_semantic_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        points = [point.reshape(-1,point.shape[-1])[:,:3] for point in points]

        # np.save('/home/ubuntu/xxw/Online3D/Online3D/work_dirs/vis_scenenn/point_0.npy',points[0])
        # np.save('/home/ubuntu/xxw/Online3D/Online3D/work_dirs/vis_scenenn/gt_sem_0.npy',gt_sem_masks[0])
        # np.save('/home/ubuntu/xxw/Online3D/Online3D/work_dirs/vis_scenenn/pred_sem_0.npy',pred_sem_masks[0])
            
        if self.use_voxel_eval:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            gt_sem_masks_new = []
            pred_sem_masks_new = []
            
            for point, gt_sem_mask, pred_sem_mask in zip(points, gt_sem_masks, pred_sem_masks):
                sparse_tensor_coordinates = (torch.cat((torch.zeros(point.shape[0], 1), (point / self.voxel_size).floor().int()), dim=1)).contiguous().to(device)
                gt_sparse_feature = gt_sem_mask.unsqueeze(1).to(device)
                pred_sparse_feature = pred_sem_mask.unsqueeze(1).float().to(device)

                gt_sparse = ME.SparseTensor(
                    features=gt_sparse_feature,
                    coordinates=sparse_tensor_coordinates,
                    quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
                )
                pred_sparse = ME.SparseTensor(
                    features=pred_sparse_feature,
                    coordinates=sparse_tensor_coordinates,
                    quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
                )
                print('vNum:%d'%(pred_sparse.coordinates.shape[0]),end="  ")
                gt_sem_masks_new.append(gt_sparse.features.cpu())
                pred_sem_masks_new.append(pred_sparse.features_at_coordinates(gt_sparse.coordinates.float()).cpu())



            gt_sem_masks = gt_sem_masks_new
            pred_sem_masks = pred_sem_masks_new

        for gt_sem_mask in gt_sem_masks:
            gt_sem_mask[gt_sem_mask==15] = 16
            gt_sem_mask[gt_sem_mask==17] = 15
            gt_sem_mask[gt_sem_mask>=16] = 16
        for pred_sem_mask in pred_sem_masks:
            pred_sem_mask[pred_sem_mask==15] = 16
            pred_sem_mask[pred_sem_mask==17] = 15
            pred_sem_mask[pred_sem_mask>=16] = 16
        ret_dict = multiview_seg_eval(
            gt_sem_masks,
            pred_sem_masks,
            self.label2cat,
            self.ignore_index,
            logger=logger,
            evaluator_mode = self.evaluator_mode,
            num_slice = self.num_slice,
            len_slice = self.len_slice)

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
    


    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        return np.eye(4).astype(np.float32)

    # To be modified.
    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadAdjacentViewsFromFiles',
                coord_type='DEPTH',
                num_frames=-1,
                max_frames=50,
                num_sample=20000,
                shift_height=False,
                use_ins_sem=True,
                use_color=True,
                use_box=False,
                use_dim=[0, 1, 2, 3, 4, 5],
                scenenn_rot=True
                ),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.ALL_VALID_CLASS_IDS,
                max_cat_id=np.max(self.ALL_CLASS_IDS)),
            dict(
                type='MultiViewFormatBundle3D',
                with_label=False,
                class_names=self.ALL_CLASSES, 
                dataset_type='scenenn'),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ]
        return Compose(pipeline)

    # To be modified.

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        pass

