from tqdm import tqdm
import os, struct
import numpy as np
import zlib
import imageio.v2 as imageio
import cv2
import math
import sys
import mmcv
from os import path as osp
import open3d as o3d
from sklearn.decomposition import PCA
import trimesh
from sklearn.cluster import DBSCAN
import pdb
import torch
from load_scannet_data import export

path_2d = './2D/'

valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
            36, 39)
max_cat_id=40


def compute_bbox_aabb(in_pc):
    x_min = in_pc[:,0].min()
    x_max = in_pc[:,0].max()
    y_min = in_pc[:,1].min()
    y_max = in_pc[:,1].max()
    z_min = in_pc[:,2].min()
    z_max = in_pc[:,2].max()
    cx = (x_min+x_max)/2
    cy = (y_min+y_max)/2
    cz = (z_min+z_max)/2
    dx = x_max-x_min
    dy = y_max-y_min
    dz = z_max-z_min
    bbox_aabb = np.expand_dims(np.array([cx, cy, cz, dx/2, dy/2, dz/2, 0]), axis=0)
    return bbox_aabb

def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()
    
    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0,0)
    step_index.append(labels.shape[0])
    return step_index,index

def calculate_box_label(pts_semantic_mask_expand):
    pts_semantic_mask_expand = pts_semantic_mask_expand.transpose(1,0)
    label_match_num = [torch.sum(pts_semantic_mask_expand == i, dim=1).unsqueeze(1) for i in range(torch.max(pts_semantic_mask_expand)+1)]
    label_match_res = torch.cat(label_match_num, dim=1)
    label_match_res = label_match_res.max(axis=1)[1]
    return label_match_res

def select_scene(bboxes, instances, max_box_labels):
    assert bboxes.shape[0] == instances.shape[0]
    assert max_box_labels.shape[0] == instances.shape[0]
    scene_bboxes = []
    scene_instances = []
    scene_labels = []
    for i in range(max_box_labels.shape[0]):
        if max_box_labels[i] != torch.max(max_box_labels):
            scene_bboxes.append(bboxes[i,:].unsqueeze(0))
            scene_instances.append(instances[i].unsqueeze(0))
            scene_labels.append(max_box_labels[i].unsqueeze(0))
    scene_bboxes = np.concatenate(scene_bboxes, axis=0)
    scene_instances = np.concatenate(scene_instances, axis=0)
    scene_labels = np.concatenate(scene_labels, axis=0)
    return scene_bboxes, scene_instances, scene_labels

def get_3d_bbox(xyzrgb):
    label = xyzrgb[:,-1].copy()
    instance = xyzrgb[:,-2].copy()
    xyz = xyzrgb[:,:3].copy()
    
    
    # build cat_id to class index mapping
    neg_cls = len(valid_cat_ids)
    cat_id2class = np.ones(
        max_cat_id + 1, dtype=np.int8) * neg_cls
    # 0~40 18
    for cls_idx, cat_id in enumerate(valid_cat_ids):
        cat_id2class[cat_id] = cls_idx
    label = np.where(label.astype(np.int8)<=40, label.astype(np.int8), 40)
    label = np.where(label.astype(np.int8)>=0, label.astype(np.int8), 40)
    label = cat_id2class[label]
    
    pts_instance_mask = instance
    instance_unique = np.array(list(set(pts_instance_mask)))
    
    instance_convert = {instance_unique[i]: i for i in range(instance_unique.shape[0])}
    for j in range(pts_instance_mask.shape[0]):
        pts_instance_mask[j] = instance_convert[pts_instance_mask[j]]
    pts_instance_mask = torch.tensor(pts_instance_mask).to(torch.int64)


    if torch.sum(pts_instance_mask == -1) != 0:
        # throw things you don't want
        pts_instance_mask[pts_instance_mask == -1] = torch.max(pts_instance_mask) + 1
        pts_instance_mask_one_hot = torch.nn.functional.one_hot(pts_instance_mask)[
            :, :-1
        ]
    else:
        pts_instance_mask_one_hot = torch.nn.functional.one_hot(pts_instance_mask)

    points = torch.tensor(xyz)
    points_for_max = points.unsqueeze(1).expand(points.shape[0], pts_instance_mask_one_hot.shape[1], points.shape[1]).clone()
    points_for_min = points.unsqueeze(1).expand(points.shape[0], pts_instance_mask_one_hot.shape[1], points.shape[1]).clone()
    points_for_max[~pts_instance_mask_one_hot.bool()] = float('-inf')
    points_for_min[~pts_instance_mask_one_hot.bool()] = float('inf')
    bboxes_max = points_for_max.max(axis=0)[0]
    bboxes_min = points_for_min.min(axis=0)[0]
    bboxes_sizes = bboxes_max - bboxes_min
    bboxes_centers = (bboxes_max + bboxes_min) / 2
    bboxes = torch.hstack((bboxes_centers, bboxes_sizes, torch.zeros_like(bboxes_sizes[:, :1])))
    instances = torch.arange(pts_instance_mask_one_hot.shape[1])
    # input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"].__class__(bboxes, with_yaw=False, origin=(.5, .5, .5))

    pts_semantic_mask = torch.tensor(label)
    pts_semantic_mask_expand = pts_semantic_mask.unsqueeze(1).expand(pts_semantic_mask.shape[0], pts_instance_mask_one_hot.shape[1]).clone()
    pts_semantic_mask_expand[~pts_instance_mask_one_hot.bool()] = -1
    
    scene_bboxes = bboxes
    scene_instances = torch.tensor(instance_unique)
    scene_labels = np.zeros(scene_instances.shape)


    if scene_bboxes.shape[0] > 0:
        bbox_3d = np.concatenate([scene_bboxes, scene_labels.reshape(-1,1)], axis=1)
    else:
        bbox_3d=np.empty([0,8])
        
    return bbox_3d, scene_instances




def get_3d_bbox_new(xyzrgb):
    label = xyzrgb[:,-1].copy()
    instance = xyzrgb[:,-2].copy()
    xyz = xyzrgb[:,:3].copy()
    
    
    # build cat_id to class index mapping
    neg_cls = len(valid_cat_ids)
    cat_id2class = np.ones(
        max_cat_id + 1, dtype=np.int8) * neg_cls
    # 0~40 18
    for cls_idx, cat_id in enumerate(valid_cat_ids):
        cat_id2class[cat_id] = cls_idx
    label = np.where(label.astype(np.int8)<=40, label.astype(np.int8), 40)
    label = np.where(label.astype(np.int8)>=0, label.astype(np.int8), 40)
    label = cat_id2class[label]
    
    pts_instance_mask = instance
    instance_unique = np.array(list(set(pts_instance_mask)))
    
    instance_convert = {instance_unique[i]: i for i in range(instance_unique.shape[0])}
    for j in range(pts_instance_mask.shape[0]):
        pts_instance_mask[j] = instance_convert[pts_instance_mask[j]]
    pts_instance_mask = torch.tensor(pts_instance_mask).to(torch.int64)

    instances = torch.tensor(instance_unique)
    bboxes = []
    instance_valid = torch.zeros(instances.shape[0]).bool()
    for ins in range(instances.shape[0]):
        mask_ = pts_instance_mask == ins
        if len(mask_) == 1:
            if mask_:
                cur_tmp_xyz = xyz
            else:
                cur_tmp_xyz = np.empty([0,3])
        elif len(mask_) == 0:
            cur_tmp_xyz = np.empty([0,3])
        else:
            cur_tmp_xyz = xyz[mask_]
        
        # 100
        # if cur_tmp_xyz.shape[0] > 50:
            # save_path =  os.path.join('/home/ubuntu/xxw/SmallDet/mmdetection3d/dataset/OVD_sv_real_gt/OVD_sv_real_gt_train', "%s_pc_before_%s.obj"%(name,ins))
            # _write_obj(cur_tmp_xyz,  save_path)
            ###剔除离群点
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cur_tmp_xyz)
        res = pcd.remove_statistical_outlier(20, 2.0)   #统计法
        # res = pcd.remove_radius_outlier(nb_points=10, radius=0.05)#半径方法剔除
        cur_tmp_xyz = np.asarray(res[0].points)

            # ###DBSCAN聚类
            # db = DBSCAN(eps=0.1, min_samples=100).fit(cur_tmp_xyz)
            # # ic(np.unique(db.labels_))
            # clusters = []
            # max_mask = np.unique(db.labels_)
            # for i, cluster in enumerate(np.unique(db.labels_)):
            #     if cluster < 0:
            #         max_mask[i] = 0
            #         clusters.append(np.array([0]))
            #         continue
                
            #     cluster_ind = np.where(db.labels_ == cluster)[0]
            #     max_mask[i] = cluster_ind.shape[0]
            #     clusters.append(cluster_ind)
            #     # if cluster_ind.shape[0] / cur_tmp_xyz.shape[0] < 0.1 or cluster_ind.shape[0] <= 100:
            #     #     continue
                
            # # pdb.set_trace()
            # if clusters[np.argmax(max_mask)].shape[0] > 100 and clusters[np.argmax(max_mask)].shape[0] / cur_tmp_xyz.shape[0] > 0.1:
            #     cur_tmp_xyz = cur_tmp_xyz[clusters[np.argmax(max_mask)],:]
            # # cur_tmp_xyz = cur_tmp_xyz[cluster_ind,:]
            # # pdb.set_trace()
            # # if name == 'scene0000_00_000900' and ins == 4:
            # #     save_path =  os.path.join('/home/ubuntu/xxw/SmallDet/mmdetection3d/dataset/OVD_sv_real_gt/OVD_sv_real_gt_train', "%s_pc_after_%s.obj"%(name,ins))
            # #     _write_obj(cur_tmp_xyz,  save_path)
            # #     ic('okkkk!!!!!!')
            # # ic(cur_tmp_xyz.shape, ins)
              
            # # 150
        if cur_tmp_xyz.shape[0] > 150:   #200的效果目前2好    ##先过离群点过滤，再150更好
            cur_bbox_3d = np.zeros(7)
            cur_bbox_3d[:7] = compute_bbox_aabb(cur_tmp_xyz)[0,:7]
            instance_valid[ins] = True
            bboxes.append(torch.Tensor(cur_bbox_3d))
            
        # else:
        #     continue

    if len(bboxes) != 0:
        bboxes = torch.stack(bboxes, dim=0)
        # instances = torch.arange(bboxes.shape[0])
        #pdb.set_trace()
        instances = instances[instance_valid]
        scene_bboxes = bboxes
        scene_instances = torch.tensor(instances)
        scene_labels = np.zeros(scene_instances.shape)
    else:
        scene_bboxes=np.empty([0,8])
        scene_instances=torch.tensor(np.empty(0))

    if scene_bboxes.shape[0] > 0:
        bbox_3d = np.concatenate([scene_bboxes, scene_labels.reshape(-1,1)], axis=1)
    else:
        bbox_3d=np.empty([0,8])
        
    return bbox_3d, scene_instances

# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def depth_image_to_point_cloud(rgb, depth, K, pose, ins, sem):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float)
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    I = np.ravel(ins[:, :])[valid]
    S = np.ravel(sem[:, :])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B, valid.nonzero()[0], I, S)))
    return points


        
def match_box(modal_data, scene_data):
    # model bboxes Mx8
    # scene bboxes Mx8
    [modal_bboxes, modal_instances] = modal_data 
    [scene_bboxes, scene_instances] = scene_data

    modal_bboxes_new = []
    modal_labels_new = []
    amodal_box_mask = []
    amodal_bboxes_new = []
    amodal_labels_new = []
    for i in range(scene_instances.shape[0]):
        match_res = False
        for j in range(modal_instances.shape[0]):
            if modal_instances[j]==scene_instances[i]:
                modal_bboxes[j,-1] = scene_bboxes[i,-1]
                modal_bboxes_new.append(modal_bboxes[j,:].reshape(1,-1))
                amodal_bboxes_new.append(scene_bboxes[i,:].reshape(1,-1))
                match_res = True
        amodal_box_mask.append(match_res)
    
    if len(modal_bboxes_new) == 0:
        modal_bboxes_new = np.empty([0,8])
        amodal_bboxes_new = np.empty([0,8])
        amodal_box_mask = np.zeros((scene_bboxes.shape[0]))
        return modal_bboxes_new, amodal_bboxes_new, amodal_box_mask
    
    modal_bboxes_new = np.concatenate(modal_bboxes_new, axis=0)
    amodal_bboxes_new = np.concatenate(amodal_bboxes_new, axis=0)
    amodal_box_mask = np.array(amodal_box_mask)

    return modal_bboxes_new, amodal_bboxes_new, amodal_box_mask


def export_one_scan(scan_name):    
    mesh_file = os.path.join('3D/scans', scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join('3D/scans', scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join('3D/scans', scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join('3D/scans', scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
    mesh_vertices, semantic_labels, instance_labels, bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, 'meta_data/scannetv2-labels.combined.tsv', None)

    num_instances = len(np.unique(instance_labels))
    instance_labels = np.arange(1,bboxes.shape[0]+1)

    print('Num of instances: ', num_instances)
    OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
    bbox_mask = np.in1d(bboxes[:,-1], OBJ_CLASS_IDS)
    bboxes = bboxes[bbox_mask,:]
    bboxes = np.concatenate([bboxes[:,:6],np.zeros(bboxes.shape[0]).reshape(-1,1),bboxes[:,6].reshape(-1,1)], axis=1)
    # 0~66  0 valid 
    instance_labels = instance_labels[bbox_mask]
    print('Num of care instances: ', bboxes.shape[0])

    return bboxes, instance_labels

def select_points_in_bbox(xyzrgb, bboxes, bbox_instance_labels):
    semantic = xyzrgb[:,-1].copy()
    instance = xyzrgb[:,-2].copy()
    xyz = xyzrgb[:,:3].copy()

    semantic_new = []
    instance_new = []
    xyz_new = []

    for i in range(bboxes.shape[0]):
        instance_target = bbox_instance_labels[i]
        x_max = bboxes[i,0] + bboxes[i,3]/2
        x_min = bboxes[i,0] - bboxes[i,3]/2
        y_max = bboxes[i,1] + bboxes[i,4]/2
        y_min = bboxes[i,1] - bboxes[i,4]/2
        z_max = bboxes[i,2] + bboxes[i,5]/2
        z_min = bboxes[i,2] - bboxes[i,5]/2
        max_range = np.array([x_max, y_max, z_max])
        min_range = np.array([x_min, y_min, z_min])
        mask_single = np.in1d(instance, instance_target)
        xyz_single = xyz[mask_single,:]
        semantic_single = semantic[mask_single]
        instance_single = instance[mask_single]
        margin_positive = xyz_single-min_range
        margin_negative = xyz_single-max_range
        in_criterion = margin_positive * margin_negative
        zero = np.zeros(in_criterion.shape)
        one = np.ones(in_criterion.shape)
        in_criterion = np.where(in_criterion<=0,one,zero)
        mask = in_criterion[:,0]*in_criterion[:,1]*in_criterion[:,2]
        mask = mask.astype(np.bool_)
        xyz_single = xyz_single[mask,:]
        semantic_single = semantic_single[mask]
        instance_single = instance_single[mask]
        if xyz_single.shape[0] == 0:
            continue
        else:
            xyz_new.append(xyz_single)
            semantic_new.append(semantic_single)
            instance_new.append(instance_single) 

    if len(xyz_new) == 0:
        return np.empty([0,5])
    xyz_new = np.concatenate(xyz_new, axis=0)
    semantic_new = np.concatenate(semantic_new, axis=0)
    instance_new = np.concatenate(instance_new, axis=0)

    xyz_all = np.concatenate([xyz_new,instance_new.reshape(-1,1), semantic_new.reshape(-1,1)], axis=1)
    return xyz_all


if __name__ == '__main__':
    f = open('meta_data/scannet_train.txt', 'r')
    unify_dim = (640, 480)
    unify_intrinsic = adjust_intrinsic(make_intrinsic(577.870605,577.870605,319.5,239.5), [640,480], unify_dim)
    scene_names = f.readlines()
    try:
        os.makedirs('instance_mask')
    except:
        pass
    try:
        os.makedirs('semantic_mask')
    except:
        pass
    try:
        os.makedirs('point')
    except:
        pass
    try:
        os.makedirs('modal_box')
    except:
        pass
    try:
        os.makedirs('amodal_box_mask')
    except:
        pass
    try:
        os.makedirs('scene_amodal_box')
    except:
        pass
    for scene_name in scene_names:
        scene_name = scene_name[:-1]
        print(scene_name)
        num_frames = len(os.listdir('2D/%s/color/' % scene_name))
        # Load scene axis alignment matrix
        lines = open('3D/scans/%s/%s.txt' % (scene_name, scene_name)).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        sum_views = 0
        try:
            os.makedirs('instance_mask/%s' % scene_name)
        except:
            pass
        try:
            os.makedirs('semantic_mask/%s' % scene_name)
        except:
            pass
        try:
            os.makedirs('point/%s' % scene_name)
        except:
            pass
        try:
            os.makedirs('modal_box/%s' % scene_name)
        except:
            pass
        try:
            os.makedirs('amodal_box_mask/%s' % scene_name)
        except:
            pass

        
        
        scene_bboxes, bbox_instance_labels = export_one_scan(scene_name)
        np.save('scene_amodal_box/%s.npy' % (scene_name), scene_bboxes)

        for i in range(num_frames):
            frame_id = i * 20
            f = os.path.join(path_2d, scene_name, 'color', str(frame_id)+'.jpg')
            img = imageio.imread(f)
            depth = imageio.imread(f.replace('color', 'depth').replace('jpg', 'png')) / 1000.0  # convert to meter
            label = imageio.imread(f.replace('color','label').replace('jpg','png'))
            posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                (x.split(" ") for x in open(posePath).read().splitlines())]
            )
            ins = imageio.imread(f.replace('color','instance').replace('jpg','png'))
            sem = imageio.imread(f.replace('color','label').replace('jpg','png'))
            pc = depth_image_to_point_cloud(img, depth, unify_intrinsic[:3,:3], pose, ins, sem)
            # to skip bad point cloud
            if np.isnan(pc).any():
                continue

            try:
                pc = pc[np.random.choice(pc.shape[0], 20000, replace=False)]
            except:
                try:
                    pc = pc[np.random.choice(pc.shape[0], 20000, replace=True)]
                except:
                    continue

            # There has been axis-aligned
            sum_views += 1
            pts = np.ones((pc.shape[0], 4))
            pts[:,0:3] = pc[:,0:3]
            pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
            pc[:,0:3] = pts[:,0:3]
            pose_ = np.dot(axis_align_matrix, pose)

            # u must merge two files to one file
            # 7 -2 ins 
            # 8 -1 label
            xyzrgb = np.concatenate([pc[:,0:6],pc[:,7:9]], axis=1) 
            xyz_for_bbox = select_points_in_bbox(xyzrgb, scene_bboxes, bbox_instance_labels)
            # xyzrgb = remove_far_points(xyzrgb)
            if xyz_for_bbox.shape[0] != 0:
                modal_bboxes, modal_instances = get_3d_bbox(xyz_for_bbox)
                # modal_bboxes, modal_instances = get_3d_bbox_new(xyz_for_bbox)
                #pdb.set_trace()
                modal_bboxes, amodal_bboxes, amodal_box_mask = match_box([modal_bboxes, modal_instances], [scene_bboxes, bbox_instance_labels])
            else:
                modal_bboxes_new = np.empty([0,8])
                amodal_bboxes_new = np.empty([0,8])
                amodal_box_mask = np.zeros((scene_bboxes.shape[0]))

            print(i*20)
            print(modal_bboxes.shape)
            np.save('point/%s/%s.npy' % (scene_name, frame_id), pc[:,:7])
            np.save('instance_mask/%s/%s.npy' % (scene_name, frame_id), pc[:,7])
            np.save('semantic_mask/%s/%s.npy' % (scene_name, frame_id), pc[:,8])
                
            if modal_bboxes.shape[0] > 0 and scene_bboxes.shape[0] > 0:
                np.save('modal_box/%s/%s.npy' % (scene_name, frame_id), modal_bboxes)
                np.save('amodal_box_mask/%s/%s.npy' % (scene_name, frame_id), amodal_box_mask)
            elif scene_bboxes.shape[0] == 0:
                np.save('modal_box/%s/%s.npy' % (scene_name, frame_id), np.zeros((1,8)))
                np.save('amodal_box_mask/%s/%s.npy' % (scene_name, frame_id), np.ones((1)))
            else:
                np.save('modal_box/%s/%s.npy' % (scene_name, frame_id), np.empty([0,8]))
                np.save('amodal_box_mask/%s/%s.npy' % (scene_name, frame_id), np.zeros((scene_bboxes.shape[0])))


            
            
