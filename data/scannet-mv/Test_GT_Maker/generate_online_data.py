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
from sklearn.cluster import DBSCAN
from box_util import box3d_iou
import pdb

path_2d = './2D/'


type2class_semseg = {
    "cabinet": 0,
    "bed": 1,
    "chair": 2,
    "sofa": 3,
    "table": 4,
    "door": 5,
    "window": 6,
    "bookshelf": 7,
    "picture": 8,
    "counter": 9,
    "desk": 10,
    "curtain": 11,
    "refridgerator": 12,
    "shower curtain": 13,
    "toilet": 14,
    "sink":15,
    "bathtub": 16,
    "garbage bin": 17
}
class2type_semseg = {
    type2class_semseg[t]: t for t in type2class_semseg
}

nyu40ids_semseg = np.array(
    [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
)
nyu40id2class_semseg = {
    nyu40id: i for i, nyu40id in enumerate(list(nyu40ids_semseg))
}


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

def get_3d_bbox(xyzrgb):
    instance = xyzrgb[:,-1].copy()
    label = xyzrgb[:,-2].copy()
    xyz = xyzrgb[:,:3].copy()

    step_idx, index = get_split_point(label)
    xyz = xyz[index,:]
    label = label[index]
    
    instance = instance[index]
    #print(step_idx)
    #print(index)

    cur_start = 0
    bbox_3d = []


    for ind, cur_end in enumerate(step_idx):
        if  cur_end - cur_start < 100:
            #print("not engouh", cur_start," ", cur_end)
            cur_start = cur_end
            #print("insufficient points 0")
            continue

        cur_xyz = xyz[cur_start:cur_end, :]
        cur_label = label[cur_start:cur_end]
        
        assert np.unique(cur_label).shape[0] == 1

        cur_cls = np.unique(cur_label)[0]

        if cur_cls not in list(nyu40ids_semseg):
            #print("outiler class 0")
            cur_start = cur_end
            continue

        step_interval = max((1, int(cur_xyz.shape[0]/30000)))
        cur_xyz = cur_xyz[0:cur_xyz.shape[0]:step_interval, :]
        db = DBSCAN(eps=0.1, min_samples=100).fit(cur_xyz)

        for cluster in np.unique(db.labels_):
            if cluster < 0:
                continue

            cluster_ind = np.where(db.labels_ == cluster)[0]
            if cluster_ind.shape[0] / cur_xyz.shape[0] < 0.1 or cluster_ind.shape[0] <= 100:
                continue

            cur_tmp_xyz = cur_xyz[cluster_ind,:]
            cur_bbox_3d = np.zeros(8)

            # cur_bbox_3d[:7] = compute_bbox(cur_tmp_xyz)[0,:7]
            cur_bbox_3d[:7] = compute_bbox_aabb(cur_tmp_xyz)[0,:7]
            cur_bbox_3d[7] = nyu40id2class_semseg[cur_cls]
            bbox_3d.append(cur_bbox_3d)
        
        cur_start = cur_end

    if len(bbox_3d) > 0:
        bbox_3d = np.stack(bbox_3d, axis=0)
    else:
        bbox_3d=np.empty([0,8])
        
    return bbox_3d

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


def my_compute_box_3d(center, size, heading_angle):
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    R = rotz(-1 * heading_angle)
    l, w, h = size
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)

def batch_compute_box_3d(bbox):
    rst_bbox = []

    for cur_bbox in bbox:
        cur_bbox_corner = my_compute_box_3d(cur_bbox[:3], cur_bbox[3:6], cur_bbox[6])
        rst_bbox.append(cur_bbox_corner)

    rst_bbox = np.stack(rst_bbox, axis=0)
    return rst_bbox

def select_3d_bbox(local_bbox, global_bbox):
    local_bbox_corner = batch_compute_box_3d(local_bbox)
    global_bbox_corner = batch_compute_box_3d(global_bbox)

    global_bbox_cnt = global_bbox_corner.shape[0]
    local_bbox_cnt = local_bbox_corner.shape[0]

    valid_local_bbox = []
    corresponding_global_bbox = []
    for ind_local in range(local_bbox_cnt):
        cur_local_corner = local_bbox_corner[ind_local, ...]
        cur_local_cls = local_bbox[ind_local, -1]

        for ind_global in range(global_bbox_cnt):
            cur_global_corner = global_bbox_corner[ind_global, ...]
            cur_global_cls = global_bbox[ind_global, -1]

            if not cur_global_cls == cur_local_cls:
                continue


            cur_iou, iou_2d = box3d_iou(cur_local_corner, cur_global_corner)
            if cur_iou > 0.25:
                corresponding_global_bbox.append(ind_global)
                valid_local_bbox.append(ind_local)
            
    rst_local_bbox = local_bbox[valid_local_bbox, :]
    rst_global_bbox = global_bbox[corresponding_global_bbox, :]

    return rst_local_bbox, rst_global_bbox

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
            os.makedirs('box/%s' % scene_name)
        except:
            pass
        try:
            os.makedirs('amodal_box/%s' % scene_name)
        except:
            pass

        xyz_global = []
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

            # There has been axis-aligned
            pts = np.ones((pc.shape[0], 4))
            pts[:,0:3] = pc[:,0:3]
            pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
            pc[:,0:3] = pts[:,0:3]
            pose_ = np.dot(axis_align_matrix, pose)

            # u must merge two files to one file
            xyzrgb = np.concatenate([pc[:,0:6],pc[:,7:9]], axis=1)  
            xyz_global.append(xyzrgb)      
        xyz_global = np.concatenate(xyz_global, axis=0)
        bbox_3d_global = get_3d_bbox(xyz_global)

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
            xyzrgb = np.concatenate([pc[:,0:6],pc[:,7:9]], axis=1)        
            bbox_3d = get_3d_bbox(xyzrgb)
                
            if bbox_3d.shape[0] > 0 and bbox_3d_global.shape[0] > 0:
                bbox_3d, amodal_bbox_3d = select_3d_bbox(bbox_3d, bbox_3d_global)
                print(bbox_3d.shape)
            else:
                continue
                
            if bbox_3d.shape[0] > 0 and bbox_3d_global.shape[0] > 0:
                np.save('point/%s/%s.npy' % (scene_name, frame_id), pc[:,:7])
                np.save('instance_mask/%s/%s.npy' % (scene_name, frame_id), pc[:,7])
                np.save('semantic_mask/%s/%s.npy' % (scene_name, frame_id), pc[:,8])
                np.save('box/%s/%s.npy' % (scene_name, frame_id), bbox_3d)
                np.save('amodal_box/%s/%s.npy' % (scene_name, frame_id), amodal_bbox_3d)
