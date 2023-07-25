# Copyright (c) OpenMMLab. All rights reserved.
import os
from concurrent import futures as futures
from os import path as osp

import mmcv
import numpy as np
import math


class ScanNetData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
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

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_instance_data',
                            f'{idx}_aligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_instance_data',
                            f'{idx}_unaligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'scannet_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_images(self, idx):
        paths = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.jpg'):
                paths.append(osp.join('posed_images', idx, file))
        return paths

    def get_extrinsics(self, idx):
        extrinsics = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.txt') and not file == 'intrinsic.txt':
                extrinsics.append(np.loadtxt(osp.join(path, file)))
        return extrinsics

    def get_intrinsics(self, idx):
        matrix_file = osp.join(self.root_dir, 'posed_images', idx,
                               'intrinsic.txt')
        mmcv.check_file_exist(matrix_file)
        return np.loadtxt(matrix_file)

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
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'scannet_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            points.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            # update with RGB image paths if exist
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):
                info['intrinsics'] = self.get_intrinsics(sample_idx)
                all_extrinsics = self.get_extrinsics(sample_idx)
                all_img_paths = self.get_images(sample_idx)
                # some poses in ScanNet are invalid
                extrinsics, img_paths = [], []
                for extrinsic, img_path in zip(all_extrinsics, all_img_paths):
                    if np.all(np.isfinite(extrinsic)):
                        img_paths.append(img_path)
                        extrinsics.append(extrinsic)
                info['extrinsics'] = extrinsics
                info['img_paths'] = img_paths

            if not self.test_mode:
                pts_instance_mask_path = osp.join(
                    self.root_dir, 'scannet_instance_data',
                    f'{sample_idx}_ins_label.npy')
                pts_semantic_mask_path = osp.join(
                    self.root_dir, 'scannet_instance_data',
                    f'{sample_idx}_sem_label.npy')

                pts_instance_mask = np.load(pts_instance_mask_path).astype(
                    np.int64)
                pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                    np.int64)

                mmcv.mkdir_or_exist(osp.join(self.root_dir, 'instance_mask'))
                mmcv.mkdir_or_exist(osp.join(self.root_dir, 'semantic_mask'))

                pts_instance_mask.tofile(
                    osp.join(self.root_dir, 'instance_mask',
                             f'{sample_idx}.bin'))
                pts_semantic_mask.tofile(
                    osp.join(self.root_dir, 'semantic_mask',
                             f'{sample_idx}.bin'))

                info['pts_instance_mask_path'] = osp.join(
                    'instance_mask', f'{sample_idx}.bin')
                info['pts_semantic_mask_path'] = osp.join(
                    'semantic_mask', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 6
                    unaligned_box = unaligned_box_label[:, :-1]
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
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


class ScanNetSegData(object):
    """ScanNet dataset used to generate infos for semantic segmentation task.

    Args:
        data_root (str): Root path of the raw data.
        ann_file (str): The generated scannet infos.
        split (str, optional): Set split type of the data. Default: 'train'.
        num_points (int, optional): Number of points in each data input.
            Default: 8192.
        label_weight_func (function, optional): Function to compute the
            label weight. Default: None.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split='train',
                 num_points=8192,
                 label_weight_func=None):
        self.data_root = data_root
        self.data_infos = mmcv.load(ann_file)
        self.split = split
        assert split in ['train', 'val', 'test']
        self.num_points = num_points

        self.all_ids = np.arange(41)  # all possible ids
        self.cat_ids = np.array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36,
            39
        ])  # used for seg task
        self.ignore_index = len(self.cat_ids)

        self.cat_id2class = np.ones((self.all_ids.shape[0],), dtype=np.int) * \
            self.ignore_index
        for i, cat_id in enumerate(self.cat_ids):
            self.cat_id2class[cat_id] = i

        # label weighting function is taken from
        # https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py#L24
        self.label_weight_func = (lambda x: 1.0 / np.log(1.2 + x)) if \
            label_weight_func is None else label_weight_func

    def get_seg_infos(self):
        if self.split == 'test':
            return
        scene_idxs, label_weight = self.get_scene_idxs_and_label_weight()
        save_folder = osp.join(self.data_root, 'seg_info')
        mmcv.mkdir_or_exist(save_folder)
        np.save(
            osp.join(save_folder, f'{self.split}_resampled_scene_idxs.npy'),
            scene_idxs)
        np.save(
            osp.join(save_folder, f'{self.split}_label_weight.npy'),
            label_weight)
        print(f'{self.split} resampled scene index and label weight saved')

    def _convert_to_label(self, mask):
        """Convert class_id in loaded segmentation mask to label."""
        if isinstance(mask, str):
            if mask.endswith('npy'):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.int64)
        label = self.cat_id2class[mask]
        return label

    def get_scene_idxs_and_label_weight(self):
        """Compute scene_idxs for data sampling and label weight for loss
        calculation.

        We sample more times for scenes with more points. Label_weight is
        inversely proportional to number of class points.
        """
        num_classes = len(self.cat_ids)
        num_point_all = []
        label_weight = np.zeros((num_classes + 1, ))  # ignore_index
        for data_info in self.data_infos:
            label = self._convert_to_label(
                osp.join(self.data_root, data_info['pts_semantic_mask_path']))
            num_point_all.append(label.shape[0])
            class_count, _ = np.histogram(label, range(num_classes + 2))
            label_weight += class_count

        # repeat scene_idx for num_scene_point // num_sample_point times
        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points))
        scene_idxs = []
        for idx in range(len(self.data_infos)):
            scene_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        scene_idxs = np.array(scene_idxs).astype(np.int32)

        # calculate label weight, adopted from PointNet++
        label_weight = label_weight[:-1].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = self.label_weight_func(label_weight).astype(np.float32)

        return scene_idxs, label_weight


class ScanNetMVData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train', interval=20):
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
        matrix_file = osp.join(self.root_dir, 'scans', idx,
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
    
    # Points and images are paths.
    # Poses are unaligned.
    # The points are located in aligned world coordinates. Depth x Aligned_Pose --> Points
    # Downsampled to 5000 pts per frame.
    # Due to some NAN in poses, we skip some frames when generating the points.
    def get_points_images_masks_poses(self, idx):
        point_paths = []
        image_paths = []
        boxes = []
        amodal_boxes = []
        poses = []
        path = osp.join(self.root_dir, 'point', idx)
        files = os.listdir(path); files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
        for file in files:
            frame_id = int(file.split('.')[0])
            if file.endswith('.npy') and (frame_id % self.interval == 0):
                point_paths.append(osp.join('point', idx, file))
                image_paths.append(osp.join('2D', idx, 'color', file.replace('npy', 'jpg')))
                boxes.append(np.load(osp.join(path.replace('point', 'box'), file)))
                amodal_boxes.append(np.load(osp.join(path.replace('point', 'amodal_box'), file)))
                pose = np.asarray(
                    [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                    (x.split(" ") for x in open(osp.join('2D', idx, 'pose', file.replace('npy', 'txt'))).read().splitlines())]
                )
                poses.append(pose)
        return point_paths, image_paths, boxes, amodal_boxes, poses
    
    def get_points_images_instance_semantic_masks_poses(self, idx):
        point_paths = []
        image_paths = []
        instance_paths = []
        semantic_paths = []
        boxes = []
        amodal_boxes = []
        poses = []
        path = osp.join(self.root_dir, 'point', idx)
        files = os.listdir(path); files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
        for file in files:
            frame_id = int(file.split('.')[0])
            if file.endswith('.npy') and (frame_id % self.interval == 0):
                point_paths.append(osp.join('point', idx, file))
                image_paths.append(osp.join('2D', idx, 'color', file.replace('npy', 'jpg')))
                instance_paths.append(osp.join('instance_mask', idx, file))
                semantic_paths.append(osp.join('semantic_mask', idx, file))
                boxes.append(np.load(osp.join(path.replace('point', 'box'), file)))
                amodal_boxes.append(np.load(osp.join(path.replace('point', 'amodal_box'), file)))
                pose = np.asarray(
                    [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                    (x.split(" ") for x in open(osp.join('2D', idx, 'pose', file.replace('npy', 'txt'))).read().splitlines())]
                )
                poses.append(pose)
        return point_paths, image_paths, instance_paths, semantic_paths, boxes, amodal_boxes, poses
    
    def align_poses(self, axis_align_matrix, poses):
        aligned_poses = []
        for pose in poses:
            aligned_poses.append(np.dot(axis_align_matrix, pose))
        return aligned_poses

    # Use a unified intrinsics, same for all scenes
    # def get_intrinsics(self):
    #     return np.array([[288.9353025,0,159.5,0],[0,288.9353025,119.5,0],[0,0,1,0],[0,0,0,1]])

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
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # pts_paths, img_paths, box_masks, poses = self.get_points_images_masks_poses(sample_idx)
            pts_paths, img_paths, instance_paths, semantic_paths, boxes, amodal_boxes, poses = self.get_points_images_instance_semantic_masks_poses(sample_idx)
            axis_align_matrix = self.get_axis_align_matrix(sample_idx)
            poses = self.align_poses(axis_align_matrix, poses)
            # TODO: check if any path is invalid
            info['poses'] = poses
            info['img_paths'] = img_paths
            info['pts_paths'] = pts_paths
            info['instance_paths'] = instance_paths
            info['semantic_paths'] = semantic_paths
            # info['box_masks'] = box_masks


            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = np.concatenate(boxes, axis=0)
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
                annotations['box'] = boxes
                annotations['amodal_box'] = amodal_boxes
                per_frame_class = []
                for k in range(len(boxes)):
                    per_frame_class.append(np.array([
                    self.cat_ids2class[boxes[i,6]]
                    for i in range(boxes[k].shape[0])
                ]))
                annotations['per_frame_class'] = per_frame_class
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        # list organization
        return list(infos)
    


def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


class ScanNetSVData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train', save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path,'scannet_sv_18cls_%s'%split)
        self.data_dir = osp.join(root_path, 'scannet_frames_25k')
        self.classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
        ]
        self.cat_ids = np.arange(18)
        #self.cat_ids = np.array(
        #    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        # assert split in ['train', 'val', 'test']
        # split_file = osp.join('/data1/SCANNET',
        #                       f'scannetv2_{split}.txt')
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'Test_GT_Maker',
                              f'{split}.txt')
        # split_file = osp.join("/data1/SCANNET",
        #                       f'scannetv2_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list =  sorted(
                list(
                    set([os.path.basename(x)[0:19] for x in os.listdir(self.split_dir) if ("intrinsic" not in x) and ("pc" in x)])
                )
            )
        self.test_mode = (split == 'test')
        self.button =False

    def __len__(self):
        return len(self.sample_id_list)


    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.split_dir, f'{idx}_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_images(self, idx):
        paths = []
        path = osp.join('scannet_sv_18cls_%s'%self.split, f'{idx}.jpg')
        return path
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = img_norm(img)

        # image = np.zeros([968,1296,3], dtype=np.float32)
        # img_width = img.shape[1]
        # img_height = img.shape[0]
        # image_size = np.array([img_width, img_height])
        # image[:img_height, :img_width, :3] = img
        # return image, image_size

    def make_intrinsic(self, fx, fy, mx, my):
        intrinsic = np.eye(4)
        intrinsic[0][0] = fx
        intrinsic[1][1] = fy
        intrinsic[0][2] = mx
        intrinsic[1][2] = my
        return intrinsic

    def adjust_intrinsic(self, intrinsic, intrinsic_image_dim, image_dim):
        if intrinsic_image_dim == image_dim:
            return intrinsic
        resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
        intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
        intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
        # account for cropping here
        intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
        intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
        return intrinsic

    def get_intrinsics(self, idx):
        #unify_dim = (640, 480)
        #matrix = self.adjust_intrinsic(self.make_intrinsic(577.870605,577.870605,319.5,239.5), [640,480], unify_dim)
        path = osp.join(self.split_dir, f'{idx[:-7]}_image_intrinsic.txt')
        matrix = load_matrix_from_txt(path)
        return matrix

    def get_pose(self, idx):
        path = osp.join(self.split_dir, f'{idx}_pose.txt') 
        pose = load_matrix_from_txt(path)
        return  pose
    

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
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            info['intrinsics'] = self.get_intrinsics(sample_idx)  

            info['pts_path'] = osp.join(self.split_dir, f'{sample_idx}_pc.npy')
            
            # update with RGB image paths if exist

            image_path = self.get_images(sample_idx)

            info['img_path'] = image_path
            info['pose'] = self.get_pose(sample_idx)

            info['pts_instance_mask_path'] = osp.join(
                self.split_dir,
                f'{sample_idx}_ins_label.npy')
            info['pts_semantic_mask_path'] = osp.join(
                self.split_dir, 
                f'{sample_idx}_sem_label.npy')

            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 7
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
                axis_align_matrix =  np.eye(4)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


