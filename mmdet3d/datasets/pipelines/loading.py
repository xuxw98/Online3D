# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import Compose, LoadAnnotations, LoadImageFromFile
from ..builder import PIPELINES
import pdb
from skimage import io
import os


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class PointSegClassMappingV2(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        self.cat_id2class = -np.ones(
            self.max_cat_id + 1, dtype=np.int)
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
                - pts_instance_mask (np.ndarray): Mapped instance masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']
        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        pts_instance_mask = results['pts_instance_mask']
        instance_ids = np.unique(pts_instance_mask)

        mapping = np.zeros(
            pts_instance_mask.max() + 1, dtype=np.int)
        for i, instance_id in enumerate(instance_ids):
            mapping[instance_id] = i
        converted_pts_instance_mask = mapping[pts_instance_mask] - 1

        results['pts_semantic_mask'] = converted_pts_sem_mask
        results['pts_instance_mask'] = converted_pts_instance_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class MultiViewsPointSegClassMappingV2(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        self.cat_id2class = -np.ones(
            self.max_cat_id + 1, dtype=np.int)
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
                - pts_instance_mask (np.ndarray): Mapped instance masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = np.clip(results['pts_semantic_mask'],0,40)
        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        # mask = converted_pts_sem_mask >= 0
        pts_instance_mask = results['pts_instance_mask']
        instance_ids = np.unique(pts_instance_mask)

        mapping = np.zeros(
             pts_instance_mask.max() + 1, dtype=np.int)
        for i, instance_id in enumerate(instance_ids):
            mapping[instance_id] = i
        converted_pts_instance_mask = mapping[pts_instance_mask] - 1

        results['pts_semantic_mask'] = converted_pts_sem_mask
        results['pts_instance_mask'] = converted_pts_instance_mask
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        mmcv.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)
      
        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        # pdb.set_trace()
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@PIPELINES.register_module()
class LoadAdjacentViewsFromFiles(object):
    def __init__(self,
                 coord_type,
                 num_frames=8,
                 max_frames=-1,
                 use_dim=[0, 1, 2],
                 num_sample=5000,
                 use_ins_sem=False,
                 use_amodal_points=False,
                 interval=2,
                 shift_height=False,
                 use_color=False,
                 use_box=True,
                 file_client_args=dict(backend='disk'),
                 sum_num_sample=-1,
                 scenenn_rot=False):
        self.shift_height = shift_height
        self.use_color = use_color
        self.use_box = use_box
        self.use_ins_sem = use_ins_sem
        self.use_amodal_points = use_amodal_points
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.num_frames = num_frames
        self.num_sample = num_sample
        self.interval = interval
        self.use_dim = use_dim
        self.max_frames = max_frames
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.sum_num_sample = sum_num_sample
        self.loader = Compose([dict(type = 'LoadImageFromFile')])
        self.scenenn_rot = scenenn_rot
        self.rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ])
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3,:3] = self.rotation_matrix

    # Only for detection
    def _load_points(self, pts_filenames):
        # points: num_frames, 5000, 3+C
        points = [np.load(info['filename']) for info in pts_filenames]
        points = [point[np.random.choice(point.shape[0], self.num_sample, replace=False)] for point in points]
        points = np.concatenate(points, axis=0)
        return points
    
    def _load_amodal_points_ins_sem(self, pts_filenames, instance_filenames, semantic_filenames):
        # points: num_frames, 5000, 3+C
        points = [np.load(info['filename']) for info in pts_filenames]
        instance = [np.load(info['filename']).astype(np.int64) for info in instance_filenames]
        semantic = [np.load(info['filename']).astype(np.int64) for info in semantic_filenames]

        points_new = []
        instance_new = []
        semantic_new = []
        for i in range(len(points)):
            choice = np.random.choice(points[i].shape[0], self.num_sample, replace=False)
            points_new.append(points[i][choice])
            instance_new.append(instance[i][choice])
            semantic_new.append(semantic[i][choice])

        points = np.concatenate(points_new, axis=0)
        instance = np.concatenate(instance_new, axis=0)
        semantic = np.concatenate(semantic_new, axis=0)

        instance_unique = np.unique(instance)
        points_all = []
        instance_all = []
        semantic_all = []
        for i, instance_id in enumerate(instance_unique):
            if instance_id == 0:
                continue
            mask = (instance == instance_id)
            points_all.append(points[mask]) 
            instance_all.append(instance[mask])
            semantic_all.append(semantic[mask])

        points = np.concatenate(points_all, axis=0)
        instance = np.concatenate(instance_all, axis=0)
        semantic = np.concatenate(semantic_all, axis=0)

        return points, instance, semantic      

    def _load_points_sem_ins(self, pts_filenames, semantic_filenames, instance_filenames=None):
        # points: num_frames, 5000, 3+C
        if self.scenenn_rot:
            points = []
            for info in pts_filenames:
                point = np.load(info['filename'])
                point[:,:3] = np.dot(self.rotation_matrix, point[:,:3].T).T
                points.append(point)
        else:
            points = [np.load(info['filename']) for info in pts_filenames]
        if instance_filenames != None:
            instance = [np.load(info['filename']).astype(np.int64) for info in instance_filenames]
        semantic = [np.load(info['filename']).astype(np.int64) for info in semantic_filenames]

        points_new = []
        instance_new = []
        semantic_new = []
        for i in range(len(points)):
            if self.num_sample < points[i].shape[0]:
                choice = np.random.choice(points[i].shape[0], self.num_sample, replace=False)
            else:
                choice = np.arange(points[i].shape[0])
            points_new.append(points[i][choice])
            if instance_filenames != None:
                instance_new.append(instance[i][choice])
            semantic_new.append(semantic[i][choice])

        points = np.concatenate(points_new, axis=0)
        if instance_filenames != None:
            instance = np.concatenate(instance_new, axis=0)
        semantic = np.concatenate(semantic_new, axis=0)
        
        if instance_filenames != None:
            return points, semantic, instance
        else:
            return points, semantic

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filenames = results['pts_info']
        img_filenames = results['img_info']
        if self.use_ins_sem:
            if 'instance_info' in results:
                instance_filenames = results['instance_info']
            semantic_filenames = results['semantic_info']
        poses = results['poses']
        if self.use_box:
            modal_boxes = results['ann_info']['modal_boxes']
            modal_labels = results['ann_info']['modal_labels']
            amodal_box_masks = results['ann_info']['amodal_box_masks']

        if self.use_amodal_points:
            amodal_points, amodal_instance, amodal_semantic = self._load_amodal_points_ins_sem(pts_filenames, instance_filenames, semantic_filenames)
            results['num_amodal_points']  = amodal_points.shape[0]

        if self.num_frames > 0:
            begin_idx = np.random.randint(0, len(pts_filenames))
            keep_view_idx = np.arange(begin_idx, begin_idx + self.num_frames * self.interval, self.interval)
            keep_view_idx %= len(pts_filenames)
            pts_filenames = [pts_filenames[idx] for idx in keep_view_idx]
            img_filenames = [img_filenames[idx] for idx in keep_view_idx]
            if self.use_ins_sem:
                instance_filenames = [instance_filenames[idx] for idx in keep_view_idx]
                semantic_filenames = [semantic_filenames[idx] for idx in keep_view_idx]
            poses = [poses[idx] for idx in keep_view_idx]
            if self.use_box:
                modal_boxes = [modal_boxes[idx] for idx in keep_view_idx]
                modal_labels = [modal_labels[idx] for idx in keep_view_idx]
                amodal_box_masks = [amodal_box_masks[idx] for idx in keep_view_idx]
            
        if self.max_frames > 0 and len(pts_filenames) > self.max_frames:
            # choose_seq = np.floor(np.arange(0,len(pts_filenames),len(pts_filenames)/self.max_frames)).astype(np.int_)
            choose_seq = np.floor(np.linspace(0, len(pts_filenames) - 1, num=self.max_frames)).astype(np.int_)
            pts_filenames = [pts_filenames[idx] for idx in choose_seq]
            img_filenames = [img_filenames[idx] for idx in choose_seq]
            if self.use_ins_sem:
                if 'instance_info' in results:
                    instance_filenames = [instance_filenames[idx] for idx in choose_seq]
                semantic_filenames = [semantic_filenames[idx] for idx in choose_seq]
            poses = [poses[idx] for idx in choose_seq]
            if self.use_box:
                modal_boxes = [modal_boxes[idx] for idx in choose_seq]
                modal_labels = [modal_labels[idx] for idx in choose_seq]
                amodal_box_masks = [amodal_box_masks[idx] for idx in choose_seq]

        if self.use_box:
            results['modal_box'] = modal_boxes
            results['modal_label'] = modal_labels
            results['amodal_box_mask'] = amodal_box_masks
        if self.scenenn_rot:
            results['poses'] = [(self.transform_matrix @ pose) for pose in poses]
        else:
            results['poses'] = poses

        if self.use_ins_sem:
            if 'instance_info' in results:
                points, semantic, instance = self._load_points_sem_ins(pts_filenames, semantic_filenames, instance_filenames)
            else:
                points, semantic = self._load_points_sem_ins(pts_filenames, semantic_filenames)
        else:
            points = self._load_points(pts_filenames)

        results['num_frames'] = self.num_frames
        results['num_sample'] = self.num_sample

        if self.use_amodal_points:
            points = np.concatenate([amodal_points, points], axis=0)
            if self.use_ins_sem:
                if 'instance_info' in results:
                    instance = np.concatenate([amodal_instance, instance], axis=0)
                semantic = np.concatenate([amodal_semantic, semantic], axis=0)

        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        if self.sum_num_sample > 0 and points.shape[0] > self.sum_num_sample:
            choice = np.random.choice(points.shape[0], self.sum_num_sample, replace=False)
            points = points[choice]
            if self.use_ins_sem:
                if 'instance_info' in results:
                    instance = instance[choice]
                semantic = semantic[choice]

        results['points'] = points
        if self.use_ins_sem:
            if 'instance_info' in results:
                results['pts_instance_mask'] = instance
            results['pts_semantic_mask'] = semantic

        imgs = []
        for i in range(len(pts_filenames)):
            _results = dict()
            _results['img_prefix'] = None
            _results['img_info'] = img_filenames[i]
            _results = self.loader(_results)
            # list
            imgs.append(_results['img'])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['imgs'] = imgs
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadAdjacentViewsFromFiles_FSA(object):
    def __init__(self,
                 coord_type,
                 num_frames=8,
                 max_frames=-1,
                 use_dim=[0, 1, 2],
                 num_sample=5000,
                 use_ins_sem=False,
                 use_amodal_points=False,
                 interval=2,
                 shift_height=False,
                 use_color=False,
                 use_box=True,
                 file_client_args=dict(backend='disk'),
                 sum_num_sample=-1,
                 scenenn_rot=False):
        self.shift_height = shift_height
        self.use_color = use_color
        self.use_box = use_box
        self.use_ins_sem = use_ins_sem
        self.use_amodal_points = use_amodal_points
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.num_frames = num_frames
        self.num_sample = num_sample
        self.interval = interval
        self.use_dim = use_dim
        self.max_frames = max_frames
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.sum_num_sample = sum_num_sample
        self.loader = Compose([dict(type = 'LoadImageFromFile')])
        self.scenenn_rot = scenenn_rot
        self.rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ])
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3,:3] = self.rotation_matrix

    def _load_points_sem_ins(self, pts_filenames, semantic_filenames, instance_filenames=None):
        # points: num_frames, 5000, 3+C
        if self.scenenn_rot:
            points = []
            for info in pts_filenames:
                point = np.load(info['filename'])
                point[:,:3] = np.dot(self.rotation_matrix, point[:,:3].T).T
                points.append(point)
        else:
            points = [np.load(info['filename']) for info in pts_filenames]
        if instance_filenames != None:
            instance = [np.load(info['filename']).astype(np.int64) for info in instance_filenames]
        semantic = [np.load(info['filename']).astype(np.int64) for info in semantic_filenames]

        points_new = []
        instance_new = []
        semantic_new = []
        for i in range(len(points)):
            if self.num_sample < points[i].shape[0]:
                choice = np.random.choice(points[i].shape[0], self.num_sample, replace=False)
            else:
                choice = np.arange(points[i].shape[0])
            points_new.append(points[i][choice])
            if instance_filenames != None:
                instance_new.append(instance[i][choice])
            semantic_new.append(semantic[i][choice])

        points = np.concatenate(points_new, axis=0)
        if instance_filenames != None:
            instance = np.concatenate(instance_new, axis=0)
        semantic = np.concatenate(semantic_new, axis=0)
        
        if instance_filenames != None:
            return points, semantic, instance
        else:
            return points, semantic

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filenames = results['pts_info']
        img_filenames = results['img_info']
        if self.use_ins_sem:
            if 'instance_info' in results:
                instance_filenames = results['instance_info']
            semantic_filenames = results['semantic_info']
        poses = results['poses']
        
        if self.max_frames > 0 and len(pts_filenames) > self.max_frames:
            # choose_seq = np.floor(np.arange(0,len(pts_filenames),len(pts_filenames)/self.max_frames)).astype(np.int_)
            choose_seq = np.floor(np.linspace(0, len(pts_filenames) - 1, num=self.max_frames)).astype(np.int_)
            pts_filenames = [pts_filenames[idx] for idx in choose_seq]
            img_filenames = [img_filenames[idx] for idx in choose_seq]
            if self.use_ins_sem:
                if 'instance_info' in results:
                    instance_filenames = [instance_filenames[idx] for idx in choose_seq]
                semantic_filenames = [semantic_filenames[idx] for idx in choose_seq]
            poses = [poses[idx] for idx in choose_seq]
           
        if self.scenenn_rot:
            results['poses'] = [(self.transform_matrix @ pose) for pose in poses]
        else:
            results['poses'] = poses

        depth_fsa_list = []
        img_fsa_list = []
        for info in img_filenames:
            img_filename = info['filename']
            if self.scenenn_rot:
                spilt = img_filename.split("/",6)
                depth_filename = os.path.join(spilt[0],spilt[1],spilt[2],spilt[3],'depth','depth'+spilt[-1][5:])
            else:
                spilt = img_filename.split("/",6)
                depth_filename = os.path.join(spilt[0],spilt[1],spilt[2], spilt[3], spilt[4],'depth',spilt[-1][:-3]+'png')            
            img_fsa = io.imread(img_filename).astype(np.int16)
            depth_fsa = io.imread(depth_filename).astype(np.int16)/1000

            depth_fsa_list.append(depth_fsa)
            img_fsa_list.append(img_fsa)

        results['depth_fsa'] = depth_fsa_list
        results['img_fsa'] = img_fsa_list


        if 'instance_info' in results:
            points, semantic, instance = self._load_points_sem_ins(pts_filenames, semantic_filenames, instance_filenames)
        else:
            points, semantic = self._load_points_sem_ins(pts_filenames, semantic_filenames)


        results['num_frames'] = self.num_frames
        results['num_sample'] = self.num_sample

        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        results['points'] = points
        if self.use_ins_sem:
            if 'instance_info' in results:
                results['pts_instance_mask'] = instance
            results['pts_semantic_mask'] = semantic

        imgs = []
        for i in range(len(pts_filenames)):
            _results = dict()
            _results['img_prefix'] = None
            _results['img_info'] = img_filenames[i]
            _results = self.loader(_results)
            # list
            imgs.append(_results['img'])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['imgs'] = imgs
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


# Need to be updated
@PIPELINES.register_module()
class LoadAdjacentPointsFromFiles(object):
    def __init__(self,
                 coord_type,
                 num_frames=8,
                 load_dim=7,
                 use_dim=[0, 1, 2],
                 num_sample=5000,
                 interval=2,
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.num_frames = num_frames
        self.num_sample = num_sample
        self.interval = interval
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filenames):
        # points: num_frames, 5000, 3+C
        points = [np.load(info['filename']) for info in pts_filenames]
        points = [point[np.random.choice(point.shape[0],self.num_sample,replace=False)] for point in points]
        points = np.stack(points, axis=0)
        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filenames = results['pts_info']

        box_masks = results['ann_info']['box_masks']
        if self.num_frames > 0:
            begin_idx = np.random.randint(0, len(pts_filenames))
            keep_view_idx = np.arange(begin_idx, begin_idx + self.num_frames * self.interval, self.interval)
            keep_view_idx %= len(pts_filenames)
            pts_filenames = [pts_filenames[idx] for idx in keep_view_idx]
            img_filenames = [img_filenames[idx] for idx in keep_view_idx]
            poses = [poses[idx] for idx in keep_view_idx]
            box_masks = [box_masks[idx] for idx in keep_view_idx]
        results['box_masks'] = box_masks
        results['poses'] = poses


        points = self._load_points(pts_filenames)
        results['num_frames'] = self.num_frames

        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        mmcv.check_file_exist(pts_instance_mask_path)
        pts_instance_mask = np.load(pts_instance_mask_path)
        pts_instance_mask = pts_instance_mask.astype('int')


        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        pts_semantic_mask = np.load(pts_semantic_mask_path)
        pts_semantic_mask = pts_semantic_mask.astype('int')

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
