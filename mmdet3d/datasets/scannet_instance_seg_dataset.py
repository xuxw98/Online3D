# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp
import mmcv
import numpy as np

from mmdet3d.core import (
    instance_seg_eval, instance_seg_eval_v2, multiview_instance_seg_eval_v2,show_result, show_seg_result, show_online_seg_result)
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose
from .scannet_dataset import ScanNetDataset, ScanNetSVDataset, ScanNetMVDataset
import pdb
import torch


@DATASETS.register_module()
@SEG_DATASETS.register_module()
class ScanNetInstanceSegDataset(Custom3DSegDataset):
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    VALID_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
                       36, 39)

    ALL_CLASS_IDS = tuple(range(41))

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:
                - pts_semantic_mask_path (str): Path of semantic masks.
                - pts_instance_mask_path (str): Path of instance masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset. Palette is simply ignored for
        instance segmentation.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
                Defaults to None.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Defaults to None.
        """
        if classes is not None:
            return classes, None
        return self.CLASSES, None

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.VALID_CLASS_IDS,
                max_cat_id=40),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
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
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        load_pipeline = self._get_pipeline(pipeline)
        pred_instance_masks = [result['instance_mask'] for result in results]
        pred_instance_labels = [result['instance_label'] for result in results]
        pred_instance_scores = [result['instance_score'] for result in results]
        gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.VALID_CLASS_IDS,
            class_labels=self.CLASSES,
            options=options,
            logger=logger)

        if show:
            raise NotImplementedError('show is not implemented for now')

        return ret_dict
    
@DATASETS.register_module()
class ScanNetInstanceSegV2Dataset(ScanNetDataset):
    VALID_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
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
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        load_pipeline = self._build_default_pipeline()
        pred_instance_masks = [result['instance_mask'] for result in results]
        pred_instance_labels = [result['instance_label'] for result in results]
        pred_instance_scores = [result['instance_score'] for result in results]
        gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        ret_dict = instance_seg_eval_v2(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.VALID_CLASS_IDS,
            class_labels=self.CLASSES,
            options=options,
            logger=logger)

        if show:
            self.show(results, out_dir)

        return ret_dict

    def show(self, results, out_dir, show=True, pipeline=None):
        assert out_dir is not None, 'Expect out_dir, got none.'
        load_pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_instance_mask, gt_sem_mask = self._extract_data(
                i, load_pipeline, ['points', 'pts_instance_mask', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            gt_inst_mask_final = np.zeros_like(gt_instance_mask)
            for cls_idx in self.VALID_CLASS_IDS:
                mask = gt_sem_mask == cls_idx
                gt_inst_mask_final += mask.numpy()
            gt_instance_mask[gt_inst_mask_final == 0] = -1

            pred_instance_masks = result['instance_mask']
            pred_instance_scores = result['instance_score']

            pred_instance_masks_sort = pred_instance_masks[pred_instance_scores.argsort()]
            pred_instance_masks_label = pred_instance_masks_sort[0].long() - 1
            for i in range(1, pred_instance_masks_sort.shape[0]):
                pred_instance_masks_label[pred_instance_masks_sort[i].bool()] = i

            palette = np.random.random((max(max(pred_instance_masks_label) + 2, max(gt_instance_mask) + 2), 3)) * 255
            palette[-1] = 255

            show_seg_result(points, gt_instance_mask,
                            pred_instance_masks_label, out_dir, file_name,
                            palette)


@DATASETS.register_module()
class ScanNetSVInstanceSegV2Dataset(ScanNetSVDataset):
    VALID_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
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
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        load_pipeline = self._build_default_pipeline()
        pred_instance_masks = [result['instance_mask'] for result in results]
        pred_instance_labels = [result['instance_label'] for result in results]
        pred_instance_scores = [result['instance_score'] for result in results]
        gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        ret_dict = instance_seg_eval_v2(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.VALID_CLASS_IDS,
            class_labels=self.CLASSES,
            options=options,
            logger=logger)

        if show:
            self.show(results, out_dir)

        return ret_dict

    def show(self, results, out_dir, show=True, pipeline=None):
        assert out_dir is not None, 'Expect out_dir, got none.'
        load_pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_instance_mask, gt_sem_mask = self._extract_data(
                i, load_pipeline, ['points', 'pts_instance_mask', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            gt_inst_mask_final = np.zeros_like(gt_instance_mask)
            for cls_idx in self.VALID_CLASS_IDS:
                mask = gt_sem_mask == cls_idx
                gt_inst_mask_final += mask.numpy()
            gt_instance_mask[gt_inst_mask_final == 0] = -1

            pred_instance_masks = result['instance_mask']
            pred_instance_scores = result['instance_score']

            pred_instance_masks_sort = pred_instance_masks[pred_instance_scores.argsort()]
            pred_instance_masks_label = pred_instance_masks_sort[0].long() - 1
            for i in range(1, pred_instance_masks_sort.shape[0]):
                pred_instance_masks_label[pred_instance_masks_sort[i].bool()] = i

            palette = np.random.random((max(max(pred_instance_masks_label) + 2, max(gt_instance_mask) + 2), 3)) * 255
            palette[-1] = 255

            show_seg_result(points, gt_instance_mask,
                            pred_instance_masks_label, out_dir, file_name,
                            palette)


@DATASETS.register_module()
class ScanNetMVInstanceSegV2Dataset(ScanNetMVDataset):
    VALID_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39)

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
                use_dim=[0, 1, 2, 3, 4, 5],
                ),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=False),
            dict(
                type='MultiViewFormatBundle3D',
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 scene_name=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
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

        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)

        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []
        #pred_instance_indexes = []
        for i in range(len(results)):
            pred_instance_masks.append([result[0]['instance_mask'] for result in results[i]])
            pred_instance_labels.append([result[0]['instance_label'] for result in results[i]])
            pred_instance_scores.append([result[0]['instance_score'] for result in results[i]])
            #pred_instance_indexes.append([result[0]['instance_index'] for result in results[i]])


        load_pipeline = self._build_default_pipeline()
        points, gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['points', 'pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        
        use_voxel_eval = False

        if show:
            self.show(results, out_dir, scene_name)
            
        ret_dict = multiview_instance_seg_eval_v2(
            points,
            [gt_semantic_masks[i].reshape(points[i].shape[0], points[i].shape[1]).int() for i in range(len(points))],
            [gt_instance_masks[i].reshape(points[i].shape[0], points[i].shape[1]).int() for i in range(len(points))],
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            #pred_instance_indexes,
            valid_class_ids=self.VALID_CLASS_IDS,
            class_labels=self.CLASSES,
            options=options,
            logger=logger,
            evaluator_mode=self.evaluator_mode,
            num_slice=self.num_slice,
            len_slice=self.len_slice,
            voxel_size=0.02,
            use_voxel_eval=use_voxel_eval)


        return ret_dict

    def show(self, results, out_dir, scene_name=None, show=True, pipeline=None):
        f = open('./data/scannet-mv/meta_data/scannetv2_val.txt', 'r')
        scene_names = f.readlines()
        for idx in range(len(scene_names)):
            if scene_name == scene_names[idx][:-1]:
                break
        assert idx < len(results), 'No expected scene in results'
        assert out_dir is not None, 'Expect out_dir, got none.'
        load_pipeline = self._build_default_pipeline()
        data_info = self.data_infos[idx]
        scene_name = data_info['point_cloud']['lidar_idx']
        out_dir = os.path.join(out_dir,scene_name)
        mmcv.mkdir_or_exist(out_dir)

        points, gt_semantic_masks, gt_instance_masks = self._extract_data(
                idx, load_pipeline, ['points', 'pts_semantic_mask', 'pts_instance_mask'], load_annos=True)
        pred_instance_mask = results[idx][0][0]['instance_mask']
        pred_instance_score = results[idx][0][0]['instance_score']
        score_thr = 0.2
        pred_instance_mask = pred_instance_mask.type(torch.float32)
        pred_instance_mask[pred_instance_score < score_thr] = 0
        pred_instance_mask[pred_instance_score > score_thr] =  pred_instance_score[pred_instance_score > 0.2].unsqueeze(1).repeat(1,pred_instance_mask.shape[1]) * pred_instance_mask[pred_instance_score > 0.2]
        pred_instance_label = pred_instance_mask.argmax(dim=0)
        pred_instance_label[pred_instance_mask.sum(dim=0)==0] = pred_instance_score.shape[0]
        
        points_show = None
        gt_instance_masks_show = None
        pred_instance_masks_show = None
        
        palette = np.random.randint(0, 256, size=(100, 3))

        for i in range(len(points)):
            points_show = points[i].numpy() if i==0 else np.concatenate([points_show, points[i].numpy()], axis = 0)
            gt_instance_masks_show = gt_instance_masks[i*20000:(i+1)*20000].numpy().astype(np.int32) if i ==0 else np.concatenate([gt_instance_masks_show, gt_instance_masks[i*20000:(i+1)*20000].numpy()], axis=0)
            pred_instance_masks_show = pred_instance_label[i*20000:(i+1)*20000].numpy().astype(np.int32) if i ==0 else np.concatenate([pred_instance_masks_show, pred_instance_label[i*20000:(i+1)*20000].numpy()], axis=0)
 
            show_seg_result(points_show, gt_instance_masks_show, pred_instance_masks_show, out_dir, 'frame%d'%i, palette, ignore_index=None, show=True, snapshot=True)


