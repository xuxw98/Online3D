# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import os
from os import path as osp
import numpy as np
import mmcv
import warnings
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox import DepthInstance3DBoxes
from concurrent import futures as futures

class ScanNetMVDataConverter(object):
    def __init__(self, root_path, split='train', interval=40):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'scannetv2_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = mmcv.list_from_file(split_file)
        self.test_mode = (split == 'test')
        self.interval = interval

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_train_detection_data',
                            f'{idx}_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, '3D', 'scans', idx,
                               f'{idx}.txt')
        mmcv.check_file_exist(matrix_file)
        # Load scene axis alignment matrix
        lines = open(matrix_file).readlines()
        # test set data doesn't have align_matrix
        axis_align_matrix = np.eye(4)
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [
                    float(x)
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')
                ]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        return axis_align_matrix

    def get_points_images_masks_poses(self, idx):
        point_paths = []
        image_paths = []
        box_masks = []
        poses = []
        path = osp.join(self.root_dir, '2D', idx, 'point')
        files = os.listdir(path); files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
        for file in files:
            frame_id = int(file.split('.')[0])
            if file.endswith('.npy') and (frame_id % self.interval == 0):
                point_paths.append(osp.join('2D', idx, 'point', file))
                image_paths.append(osp.join('2D', idx, 'color', file.replace('npy', 'jpg')))
                box_masks.append(np.load(osp.join(path.replace('point', 'box_mask'), file)))
                pose = np.asarray(
                    [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                    (x.split(" ") for x in open(osp.join(path.replace('point', 'pose'), file.replace('npy', 'txt'))).read().splitlines())]
                )
                poses.append(pose)
        return point_paths, image_paths, box_masks, poses
    
    def align_poses(self, axis_align_matrix, poses):
        aligned_poses = []
        for pose in poses:
            aligned_poses.append(np.dot(axis_align_matrix, pose))
        return aligned_poses

    @staticmethod
    def _get_axis_align_matrix(info):
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def process_single_scene(self,sample_idx,cfg,has_label=True):
        # data process stage 1
        info = dict()
        pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info

        pts_paths, img_paths, box_masks, poses = self.get_points_images_masks_poses(sample_idx)
        axis_align_matrix = self.get_axis_align_matrix(sample_idx)
        poses = self.align_poses(axis_align_matrix, poses)
        # TODO: check if any path is invalid
        info['poses'] = poses
        info['img_paths'] = img_paths
        info['pts_paths'] = pts_paths
        # info['box_masks'] = box_masks

        if has_label:
            annotations = {}
            # box is of shape [k, 6 + class]
            aligned_box_label = self.get_aligned_box_label(sample_idx)
            annotations['gt_num'] = aligned_box_label.shape[0]
            if annotations['gt_num'] != 0:
                aligned_box = aligned_box_label[:, :-1]  # k, 6
                classes = aligned_box_label[:, -1]  # k
                annotations['name'] = np.array([
                    self.label2cat[self.cat_ids2class[classes[i]]]
                    for i in range(annotations['gt_num'])
                ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                annotations['location'] = aligned_box[:, :3]
                annotations['dimensions'] = aligned_box[:, 3:6]
                annotations['gt_boxes_upright_depth'] = aligned_box
                annotations['index'] = np.arange(
                    annotations['gt_num'], dtype=np.int32)
                annotations['class'] = np.array([
                    self.cat_ids2class[classes[i]]
                    for i in range(annotations['gt_num'])
                ])
            annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
            annotations['box_masks'] = box_masks
            info['annos'] = annotations

        # data process stage 2
        sample_idx = info['point_cloud']['lidar_idx']
        input_dict = dict(sample_idx=sample_idx)

        pts_info = []
        for pts_path in info['pts_paths']:
            pts_info.append(
                dict(filename=osp.join(self.root_dir, pts_path)))
        img_info = []
        for img_path in info['img_paths']:
            img_info.append(
                dict(filename=osp.join(self.root_dir, img_path)))
        axis_align_matrix = self._get_axis_align_matrix(info)
        poses = info['poses']

        input_dict['pts_info'] = pts_info
        input_dict['img_info'] = img_info
        input_dict['poses'] = poses

        input_dict['img_fields']=[]
        input_dict['bbox3d_fields']=[]
        input_dict['pts_mask_fields']=[]
        input_dict['pts_seg_fields']=[]
        input_dict['bbox_fields']=[]
        input_dict['mask_fields']=[]
        input_dict['seg_fields']=[]
        box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
        input_dict['box_type_3d'] = box_type_3d
        input_dict['box_mode_3d'] = box_mode_3d


        annos = self.get_ann_info(info=info,cfg=cfg)
        input_dict['ann_info'] = annos
        if ~(annos['gt_labels_3d'] != -1).any():
            return None

        return input_dict


    def get_ann_info(self, info,cfg):
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)
        
        box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)

        axis_align_matrix = self._get_axis_align_matrix(info)
        box_masks = info['annos']['box_masks']

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            box_masks=box_masks,
            axis_align_matrix=axis_align_matrix)
        return anns_results