# Copyright (c) OpenMMLab. All rights reserved.
import os
from concurrent import futures as futures
from os import path as osp

import mmcv
import numpy as np
import math
import pdb


class SceneNNMVData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train', interval=20):
        self.root_dir = root_path
        self.split_dir = osp.join(root_path)
        self.sample_id_list =  ['015', '005', '030', '054', '322', '263', '243', '080', '089', '093', '096', '011']
        self.test_mode = (split == 'test')
        self.interval = interval

    def __len__(self):
        return len(self.sample_id_list)

    def get_points_images_semantic_poses(self, idx):
        point_paths = []
        image_paths = []
        semantic_paths = []
        poses = []
        path = osp.join(self.root_dir, idx, 'point')
        files = os.listdir(path); files.sort(key=lambda x: int(x.split('/')[-1][:-5]))
        for file in files:
            frame_id = int(file.split('.')[0])
            if file.endswith('.npy'):
                point_paths.append(osp.join(idx,'point', file))
                image_paths.append(osp.join(idx,'image', 'image'+file.replace('npy','png')))
                semantic_paths.append(osp.join(idx, 'label', file))
                poses.append(np.load(osp.join(self.root_dir, idx, 'pose', file)))
        return point_paths, image_paths, semantic_paths, poses
    
    def align_poses(self, axis_align_matrix, poses):
        aligned_poses = []
        for pose in poses:
            aligned_poses.append(np.dot(axis_align_matrix, pose))
        return aligned_poses

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'scene_idx': sample_idx}
            info['point_cloud'] = pc_info

            pts_paths, img_paths, semantic_paths, poses = self.get_points_images_semantic_poses(sample_idx)
            axis_align_matrix = np.eye(4)
            info['poses'] = poses
            info['img_paths'] = img_paths
            info['pts_paths'] = pts_paths
            info['semantic_paths'] = semantic_paths

            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        # list organization
        return list(infos)
    

