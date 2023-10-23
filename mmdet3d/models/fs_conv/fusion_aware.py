try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
import torch.nn as nn
import torch
import pdb
from mmdet3d.models.detectors.base import Base3DDetector
from .model2d import FuseNet_feature
from .model3d import create_FusionAwareFuseConv
from .GLtree.interval_tree import RedBlackTree, Node, BLACK, RED, NIL
from .GLtree.octree import point3D
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from .utils.ply_utils import write_ply,create_color_palette,label_mapper
import torchvision.transforms as transforms
from mmcv.ops import PointsSampler as Points_Sampler
import random
import h5py
from scipy.spatial import cKDTree
import math


SCANNET_TYPES = {'scannet': (40, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])} 



@DETECTORS.register_module()
class FusionAwareConv(Base3DDetector):
    def __init__(self,
                 evaluator_mode,
                 num_slice = 1,
                 len_slice = 5,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FusionAwareConv, self).__init__(None)   
        self.point_size = 512
        self.min_octree_threshold = 0.04
        self.max_octree_threshold = 0.15
        self.interval_size = 0.035
        self.transform_image = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(SCANNET_TYPES['scannet'][1], SCANNET_TYPES['scannet'][2])])

        self.num_classes = 20
        self.near_node_num = 8
        self.max_node = 8
        self.model2d=FuseNet_feature(self.num_classes)
        self.model3d = create_FusionAwareFuseConv(self.num_classes)
        self.valid_cat_ids=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

        self.evaluator_mode = evaluator_mode
        self.num_slice = num_slice
        self.len_slice = len_slice
        self.init_weights()


    def init_weights(self, pretrained=None):
        self.model2d.init_weights()
        self.model3d.init_weights()

    def extract_feat(self, points, img, img_metas):
        pass
    
    def make_intrinsic(self, fx, fy, mx, my):
        intrinsic = np.eye(4)
        intrinsic[0][0] = fx
        intrinsic[1][1] = fy
        intrinsic[0][2] = mx
        intrinsic[1][2] = my
        return intrinsic

    def nearest_neighbor_interpolation(self,tree, target_coords, source_labels):
        _, nearest_neighbors = tree.query(target_coords[:,:3])
        interpolated_labels = source_labels[nearest_neighbors]
        return interpolated_labels


    def forward_train(self, points, pts_semantic_mask, img, img_metas):
        """Forward of training.

        Returns:
            dict: Loss values.
        """
        pass

    def simple_test(self, points, img_metas, img, depth_fsa, poses, img_fsa, *args, **kwargs):
        """Test without augmentations.
        """
        timestamps = []
        if self.evaluator_mode == 'slice_len_constant':
            i=1
            while i*self.len_slice<len(points):
                timestamps.append(i*self.len_slice)
                i=i+1
            timestamps.append(len(points))
        else:
            num_slice = min(len(points),self.num_slice)
            for i in range(1,num_slice):
                timestamps.append(i*(len(points)//num_slice))
            timestamps.append(len(points))

        # Process
        semseg_results = []

        for w in range(len(timestamps)):
            if w == 0:
                ts_start, ts_end = 0, timestamps[w]
            else:
                ts_start, ts_end = timestamps[w-1], timestamps[w]

            points_slice = points[ts_start:ts_end,:,:]
            depth_fsa_slice = depth_fsa[ts_start:ts_end]
            poses_slice = poses[ts_start:ts_end]
            img_fsa_slice = img_fsa[ts_start:ts_end]

            fx = 577.870605
            fy = 577.870605
            cx = 319.5
            cy = 239.5
            points_slice = points_slice[:,:,:3]
            
            # finsh img
            img_fsa_slice = torch.cat(img_fsa_slice,dim=0).permute(0,3,1,2)
            torch_resize = transforms.Resize([240,320])
            color_image_array = torch_resize(img_fsa_slice).permute(0,2,3,1)

            # finish depth
            depth_list = []
            points_list = []
            masks_list = []
            for i in range(points_slice.shape[0]):
                
                depth = depth_fsa_slice[i].repeat(1,3,1,1)
                depth = torch_resize(depth).permute(2,3,1,0)[:,:,0,0].unsqueeze(0)

                point_count = 0
                point_cloud=np.zeros((320*240,3))
                point_mask=np.zeros((320*240,2))
                for k in range(10, depth.shape[1] - 10, 2):
                    for j in range(10, depth.shape[2] - 10, 2):
                        d = depth[0, k, j]
                        if d == 0:
                            continue
                        z = float(d)
                        x = (j * 2 - cx) * z / fx
                        y = (k * 2 - cy) * z / fy
                        world_point = np.dot(poses_slice[i][0].cpu(), [x, y, z, 1])
                        point_cloud[point_count, :] = world_point[:3]
                        point_mask[point_count, :] = np.array([int(k/4),int(j/4)])
                        point_count = point_count + 1
                if point_count>=self.point_size:
                    sample = np.random.choice(point_count, self.point_size ,replace=False)
                else:
                    sample = np.random.choice(point_count, self.point_size)
                point_cloud=point_cloud[sample,:]
                point_mask=point_mask[sample,:]
                points_list.append(torch.from_numpy(point_cloud).unsqueeze(0))
                masks_list.append(torch.from_numpy(point_mask).unsqueeze(0))
                depth = (depth - depth.min())/(depth.max() - depth.min())
                depth_list.append(depth)


            depth_map_array = torch.cat(depth_list, dim=0)
            points_array = torch.cat(points_list, dim=0)
            mask_array = torch.cat(masks_list,dim=0)


            color_image_array = color_image_array.cpu().numpy()
            depth_map_array = depth_map_array.cpu().numpy()
            points_array = points_array.cpu().numpy()
            mask_array = mask_array.cpu().numpy()


            x_rb_tree = RedBlackTree(self.interval_size)
            y_rb_tree = RedBlackTree(self.interval_size)
            z_rb_tree = RedBlackTree(self.interval_size)

            frame_index=0

            for i in range(0,color_image_array.shape[0]):
                color_image=color_image_array[i,:,:,:].astype(np.uint8)
                depth_image=depth_map_array[i,:,:]
                points_single=points_array[i,:,:]
                points_mask=mask_array[i,:,:]
                color_image_cuda = self.transform_image(color_image).cuda()
                depth_image=transforms.ToTensor()(depth_image).type(torch.FloatTensor).cuda()
                input_color = torch.unsqueeze(color_image_cuda, 0)
                depth_image = torch.unsqueeze(depth_image, 0)
                imageft=self.model2d(input_color,depth_image).detach().cpu().numpy()
                x_tree_node_list=[]
                y_tree_node_list=[]
                z_tree_node_list=[]
                per_image_node_set=set()

                for p in range(self.point_size):
                
                    x_temp_node = x_rb_tree.add(points_single[p,0])
                    y_temp_node = y_rb_tree.add(points_single[p,1])
                    z_temp_node = z_rb_tree.add(points_single[p,2])
                    x_tree_node_list.append(x_temp_node)
                    y_tree_node_list.append(y_temp_node)
                    z_tree_node_list.append(z_temp_node)

                for p in range(self.point_size):

                    x_set_union = x_tree_node_list[p].set_list
                    y_set_union = y_tree_node_list[p].set_list
                    z_set_union = z_tree_node_list[p].set_list
                    set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0]
                    temp_branch = [None, None, None, None, None, None, None, None]
                    temp_branch_distance = np.full((8),self.max_octree_threshold)
                    is_find_nearest = False
                    branch_record = set()
                    list_intersection=list(set_intersection)
                    random.shuffle(list_intersection)

                    for point_iter in list_intersection:
                        distance = np.sum(np.absolute(point_iter.point_coor - points_single[p,:]))
                        if distance < self.min_octree_threshold:
                            is_find_nearest = True
                            if frame_index!=point_iter.frame_id:
                                point_iter.feature_fuse = np.maximum(imageft[0, :, int(points_mask[p, 0]),
                                            int(points_mask[p, 1])].copy() , point_iter.feature_fuse)
                                point_iter.frame_id=frame_index
                            per_image_node_set.add(point_iter)
                            break
                        x = int(point_iter.point_coor[0] >= points_single[p, 0])
                        y = int(point_iter.point_coor[1] >= points_single[p, 1])
                        z = int(point_iter.point_coor[2] >= points_single[p, 2])
                        branch_num= x * 4 + y * 2 + z
                        if distance < point_iter.branch_distance[7-branch_num]:
                            branch_record.add((point_iter, 7 - branch_num, distance))
                            if distance < temp_branch_distance[branch_num]:
                                temp_branch[branch_num] = point_iter
                                temp_branch_distance[branch_num] = distance

                    if not is_find_nearest:
                        new_3dpoint = point3D(points_single[p, :].T, imageft[0, :, int(points_mask[p, 0]),
                                                            int(points_mask[p, 1])].copy(),self.max_octree_threshold)
                        for point_branch in branch_record:
                            point_branch[0].branch_array[point_branch[1]] = new_3dpoint
                            point_branch[0].branch_distance[point_branch[1]] = point_branch[2]

                        new_3dpoint.branch_array = temp_branch
                        new_3dpoint.branch_distance = temp_branch_distance
                        per_image_node_set.add(new_3dpoint)

                        for x_set in x_set_union:
                            x_set.add(new_3dpoint)
                        for y_set in y_set_union:
                            y_set.add(new_3dpoint)
                        for z_set in z_set_union:
                            z_set.add(new_3dpoint)

                node_lengths=len(per_image_node_set)
                input_feature = np.zeros([1, 128, self.near_node_num, node_lengths])
                input_coor = np.zeros([1, 3,self.near_node_num, node_lengths])
                result_feature = np.zeros([1, 128, node_lengths])
                points_single = np.zeros([node_lengths, 3])
                points_color = np.zeros([node_lengths,3])
                
                set_count=0
                for set_point in per_image_node_set:
                    neighbor_2dfeature, neighbor_coor,_ =set_point.findNearPoint(self.near_node_num,self.max_node)
                    input_feature[0, :, :, set_count] = neighbor_2dfeature
                    input_coor[0, :, :, set_count] = neighbor_coor
                    result_feature[0,:,set_count]=set_point.result_feature
                    points_single[set_count,:]=set_point.point_coor
                    set_count+=1

                input_feature=torch.from_numpy(input_feature).cuda()
                input_coor=torch.from_numpy(input_coor).cuda()
                result_feature=torch.from_numpy(result_feature).cuda()
                output,combine_result,uncertainty = self.model3d(input_feature.float(), input_coor.float(),result_feature.float())
                result_array = combine_result.detach().cpu().numpy()
                uncertainty_array= uncertainty.detach().cpu().numpy()
                point_pred_label=label_mapper[torch.argmax(output, 1).long().squeeze().cpu().numpy()]
                set_count=0
                for set_point in per_image_node_set:
                    if uncertainty_array[0][set_count]<set_point.uncertainty:
                        set_point.result_feature= result_array[0, :, set_count]
                        set_point.uncertainty=uncertainty_array[0][set_count]
                    set_point.pred_result=point_pred_label[set_count]
                    set_count+=1

                frame_index+=1
                point_result=x_rb_tree.all_points_from_tree(return_label=True)

            point_result=x_rb_tree.all_points_from_tree(return_label=True)

            this_tree = cKDTree(point_result[:,:3])
            this_labels = self.nearest_neighbor_interpolation(this_tree, points_slice.reshape(-1,3).cpu(), point_result[:,3])
            for i in range(len(self.valid_cat_ids)):
                this_labels[this_labels==self.valid_cat_ids[i]] = i
            semseg_results.append(torch.tensor(this_labels))
            del x_rb_tree
            del y_rb_tree
            del z_rb_tree
        
        results = [dict(semantic_mask=torch.cat(semseg_results,dim=0))]    
        return results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError