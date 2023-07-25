import cv2
import shutil
import numpy as np
from scipy import stats
import os
from plyfile import PlyData,PlyElement
from scipy import stats
import open3d as o3d
from sklearn.decomposition import PCA
import trimesh
from sklearn.cluster import DBSCAN
from box_util import box3d_iou
import multiprocessing
from icecream import ic
import pdb
import imageio
import skimage.transform as sktf 
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

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def heading2rotmat(heading_angle):
    rotmat = np.zeros((3,3))
    rotmat[2,2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
    return rotmat

def write_ply(save_path,points,text=True):
    points = [tuple(x) for x in points.tolist()]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def convert_from_uvd(u, v, d, intr, pose):
    # u is width index, v is height index
    depth_scale = 1000
    z = d/depth_scale

    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)
    
    uv = np.concatenate([u,v,padding], axis=0)
    xyz = (np.linalg.inv(intr[:3,:3]) @ uv) * np.expand_dims(z,axis=0)
    xyz_local = xyz.copy()
    
    xyz = np.concatenate([xyz,padding], axis=0)



    xyz = pose @ xyz
    xyz[:3,:] /= xyz[3,:] 

    #np.savetxt("xyz.txt", xyz.T, fmt="%.3f")
    return xyz[:3, :].T, xyz_local.T

def get_color_label(xyz, intrinsic_image, rgb, ins, label):
    height, width = ins.shape
    intrinsic_image = intrinsic_image[:3,:3]

    xyz_uniform = xyz/xyz[:,2:3]
    xyz_uniform = xyz_uniform.T

    uv = intrinsic_image @ xyz_uniform

    uv /= uv[2:3, :]
    uv = np.around(uv).astype(int)
    uv = uv.T

    uv[:, 0] = np.clip(uv[:, 0], 0, width-1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height-1)

    uv_ind = uv[:, 1]*width + uv[:, 0]
    
    pc_label = np.take(label.reshape(-1), uv_ind)
    pc_ins = np.take(ins.reshape(-1), uv_ind)
    
    pc_rgb = np.take_along_axis(rgb.reshape([-1,3]), np.expand_dims(uv_ind, axis=1), axis=0)
    
    return pc_rgb, np.expand_dims(pc_label, axis=1), np.expand_dims(pc_ins, axis=1)

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

def get_2d_bbox(img, ins, label, sample_idx):
    height, width = ins.shape
    w_ind = np.arange(width)
    h_ind = np.arange(height)
    ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
    ww_ind = ww_ind.reshape(-1)
    hh_ind = hh_ind.reshape(-1)

    ins = ins.reshape([-1,1])
    label = label.reshape([-1,1])

    step_idx, index = get_split_point(ins[:,0])

    ww_ind = ww_ind[index]
    hh_ind = hh_ind[index]
    label = label[index]

    cur_start = 0

    bbox_2d = []


    for ind, cur_end in enumerate(step_idx):
        if  cur_end - cur_start < 100:
            #print("not engouh", cur_start," ", cur_end)
            cur_start = cur_end
            continue

        cur_ww_ind = ww_ind[cur_start:cur_end]
        cur_hh_ind = hh_ind[cur_start:cur_end]
        cur_label = label[cur_start:cur_end]

        assert np.unique(cur_label).shape[0] == 1

        cur_cls = np.unique(cur_label)[0]

        if cur_cls not in list(nyu40ids_semseg):
            cur_start = cur_end
            continue

        top_left=(np.min(cur_ww_ind), np.min(cur_hh_ind))
        down_right=(np.max(cur_ww_ind), np.max(cur_hh_ind))

        cur_bbox_2d = np.array([top_left[0], top_left[1], down_right[0], down_right[1], nyu40id2class_semseg[cur_cls]])

        bbox_2d.append(cur_bbox_2d)

        '''
        img_ = img.copy()

        cv2.rectangle(img_, top_left, down_right, (255,0,0))
        image = cv2.putText(img_, class2type_semseg[nyu40id2class_semseg[cur_cls]], (top_left[0]+20, top_left[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
        
        #if class2type[cur_cls] not in ["wall", "floor"]:
        cv2.imwrite(os.path.join(TARGET_DIR, "%s_%05d.png"%(sample_idx, ind)), img_)
        '''

        cur_start = cur_end

    if len(bbox_2d) > 0:
        bbox_2d = np.stack(bbox_2d, axis=0)
    else:
        bbox_2d=np.empty([0,5])

    return bbox_2d

def compute_bbox(in_pc):
    pca = PCA(2)
    pca.fit(in_pc[:,:2])
    yaw_vec = pca.components_[0,:]
    yaw = np.arctan2(yaw_vec[1],yaw_vec[0])
    
    in_pc_tmp = in_pc.copy()
    in_pc_tmp = heading2rotmat(-yaw) @ in_pc_tmp[:,:3].T
    x_min = in_pc_tmp[0,:].min()
    x_max = in_pc_tmp[0,:].max()
    y_min = in_pc_tmp[1,:].min()
    y_max = in_pc_tmp[1,:].max()
    z_min = in_pc_tmp[2,:].min()
    z_max = in_pc_tmp[2,:].max()

    dx = x_max-x_min
    dy = y_max-y_min
    dz = z_max-z_min

    bbox = heading2rotmat(yaw) @ np.array([[x_min,y_min,z_min],[x_max,y_max,z_max]]).T
    bbox = bbox.T
    x_min,y_min,z_min = bbox[0]
    x_max,y_max,z_max = bbox[1]

    cx = (x_min+x_max)/2
    cy = (y_min+y_max)/2
    cz = (z_min+z_max)/2

    rst_bbox = np.expand_dims(np.array([cx, cy, cz, dx/2, dy/2, dz/2, -1*yaw]), axis=0)
    #print(rst_bbox.shape)
    return rst_bbox


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
    
def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3,3] = 1.0            
        trns[0:3,0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))        
    
    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file    
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')
    
    return

def merge_cur_scan(path_dict, rgb_map_list, depth_map_list, instance_map_list, label_map_list, poses, intrinsic_depth, intrinsic_image):
    DATA_PATH = path_dict["DATA_PATH"]
    TARGET_DIR = path_dict["TARGET_DIR"]
    RGB_PATH = path_dict["RGB_PATH"]
    DEPTH_PATH = path_dict["DEPTH_PATH"]
    INSTANCE_PATH = path_dict["INSTANCE_PATH"]
    LABEL_PATH = path_dict["LABEL_PATH"]
    POSE_PATH = path_dict["POSE_PATH"]
    scan_path = path_dict["scan_path"]

    xyz_global = []
    for rgb_map_name, depth_map_name, instance_map_name, label_map_name, pose in zip(rgb_map_list,depth_map_list,instance_map_list,label_map_list,poses):
        
        instance_map = cv2.imread(os.path.join(scan_path,INSTANCE_PATH,instance_map_name),-1)
        label_map = cv2.imread(os.path.join(scan_path,LABEL_PATH,label_map_name),-1)
        depth_map = cv2.imread(os.path.join(scan_path,DEPTH_PATH,label_map_name),-1)
        color_map = cv2.imread(os.path.join(scan_path,RGB_PATH,rgb_map_name))
        color_map = cv2.cvtColor(color_map,cv2.COLOR_BGR2RGB)
        file_name = rgb_map_name.split(".")[0]

        # convert depth map to point cloud
        height, width = depth_map.shape    
        w_ind = np.arange(width)
        h_ind = np.arange(height)

        ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
        ww_ind = ww_ind.reshape(-1)
        hh_ind = hh_ind.reshape(-1)
        depth_map = depth_map.reshape(-1)

        valid = np.where(depth_map > 0.1)[0]
        ww_ind = ww_ind[valid]
        hh_ind = hh_ind[valid]
        depth_map = depth_map[valid]

        xyz, xyz_local = convert_from_uvd(ww_ind, hh_ind, depth_map, intrinsic_depth, pose)
        rgb, label, ins = get_color_label(xyz_local, intrinsic_image, color_map, instance_map, label_map)
        xyzrgb = np.concatenate([xyz, rgb, label, ins], axis=1)
        step_interval = max((1, int(xyzrgb.shape[0]/10000)))
        xyzrgb = xyzrgb[0:xyzrgb.shape[0]:step_interval,:]
        xyz_global.append(xyzrgb)
        
    xyz_global = np.concatenate(xyz_global, axis=0)
    
    #print(xyz_global.shape)
    #np.savetxt("xyz_global.txt", xyz_global, fmt="%.3f")

    return xyz_global

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
                valid_local_bbox.append(ind_local)
            
    rst_local_bbox = local_bbox[valid_local_bbox, :]


    '''
    print(local_bbox.shape)
    print(global_bbox.shape)
    print(local_bbox_corner.shape)
    print(global_bbox_corner.shape)
    local_bbox_corner = local_bbox_corner.reshape([-1,3])
    global_bbox_corner = global_bbox_corner.reshape([-1,3])
    np.savetxt("global_bbox_corner.txt", global_bbox_corner, fmt="%.3f")
    np.savetxt("local_bbox_corner.txt", local_bbox_corner, fmt="%.3f")
    '''

    return rst_local_bbox
    
def process_cur_scan(cur_scan):
    scan_name_index = cur_scan["scan_name_index"]
    scan_name = cur_scan["scan_name"]
    path_dict = cur_scan["path_dict"]
    scan_num = cur_scan["scan_num"]

    DATA_PATH = path_dict["DATA_PATH"]
    TARGET_DIR = path_dict["TARGET_DIR"]
    AXIS_ALIGN_MATRIX_PATH = path_dict["AXIS_ALIGN_MATRIX_PATH"]
    RGB_PATH = path_dict["RGB_PATH"]
    DEPTH_PATH = path_dict["DEPTH_PATH"]
    INSTANCE_PATH = path_dict["INSTANCE_PATH"]
    LABEL_PATH = path_dict["LABEL_PATH"]
    POSE_PATH = path_dict["POSE_PATH"]

    scan_name = scan_name.strip("\n")
    scan_path = os.path.join(DATA_PATH,scan_name)
    path_dict["scan_path"] = scan_path


    global_bbox_name = os.path.join(TARGET_DIR, "%s_global_bbox.ply"%(scan_name))
    global_pc_name = os.path.join(TARGET_DIR, "%s_global_pc.txt"%(scan_name))
    axis_align_matrix_path = os.path.join(AXIS_ALIGN_MATRIX_PATH, "%s"%(scan_name),"%s.txt"%(scan_name))
    lines = open(axis_align_matrix_path).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    #jump the processed scan
    if os.path.exists(global_pc_name) and os.path.exists(global_bbox_name):
        return

    print("%d/%d"%(scan_name_index+1, scan_num))
    intrinsic_depth = load_matrix_from_txt(os.path.join(scan_path, 'intrinsics_depth.txt'))
    intrinsic_image = load_matrix_from_txt(os.path.join(scan_path, 'intrinsics_color.txt'))

    intrinsic_image_filename = os.path.join(TARGET_DIR, "%s_image_intrinsic.txt"%(scan_name))
    intrinsic_depth_filename = os.path.join(TARGET_DIR, "%s_depth_intrinsic.txt"%(scan_name))
    if not os.path.exists(intrinsic_image_filename):
        shutil.copy(os.path.join(scan_path, 'intrinsics_color.txt'), intrinsic_image_filename)        
    if not os.path.exists(intrinsic_depth_filename):
        shutil.copy(os.path.join(scan_path, 'intrinsics_depth.txt'), intrinsic_depth_filename)

    intrinsic_depth_inv = np.linalg.inv(intrinsic_depth)
    
    
    POSE_txt_list = sorted(os.listdir(os.path.join(scan_path,POSE_PATH)))
    rgb_map_list = sorted(os.listdir(os.path.join(scan_path,RGB_PATH)))
    depth_map_list = sorted(os.listdir(os.path.join(scan_path,DEPTH_PATH)))
    instance_map_list = sorted(os.listdir(os.path.join(scan_path,INSTANCE_PATH)))
    label_map_list = sorted(os.listdir(os.path.join(scan_path,LABEL_PATH)))

    poses = [np.dot(axis_align_matrix ,load_matrix_from_txt(os.path.join(scan_path,POSE_PATH, i))) for i in POSE_txt_list]

    xyzrgb_global = merge_cur_scan(path_dict, rgb_map_list, depth_map_list, instance_map_list, label_map_list, poses, intrinsic_depth, intrinsic_image)

    bbox_3d_global = get_3d_bbox(xyzrgb_global)

    xyzrgb_global = random_sampling(xyzrgb_global, 100000)
    global_pc_name = os.path.join(TARGET_DIR, "%s_global_pc.txt"%(scan_name))
    # np.savetxt(global_pc_name, xyzrgb_global, fmt="%.3f")
    
    #print(xyzrgb_global.shape)
    #print(bbox_3d_global.shape)

    bbox_3d_global_ = bbox_3d_global.copy()
    bbox_3d_global_[:,3:6] *= 2
    bbox_3d_global_[:,6] *= -1

    global_bbox_name = os.path.join(TARGET_DIR, "%s_global_bbox.ply"%(scan_name))
    # if bbox_3d_global_.shape[0] > 0:
    #     write_oriented_bbox(bbox_3d_global_[:,:7], global_bbox_name)
    # else:
    #     return

    for rgb_map_name,\
        depth_map_name,\
        instance_map_name,\
        label_map_name,\
        pose in zip(rgb_map_list,depth_map_list,instance_map_list,label_map_list,poses):

        instance_map = cv2.imread(os.path.join(scan_path,INSTANCE_PATH,instance_map_name),-1)
        label_map = cv2.imread(os.path.join(scan_path,LABEL_PATH,label_map_name),-1)
        depth_map = cv2.imread(os.path.join(scan_path,DEPTH_PATH,label_map_name),-1)
        color_map = cv2.imread(os.path.join(scan_path,RGB_PATH,rgb_map_name))
        color_map = cv2.cvtColor(color_map,cv2.COLOR_BGR2RGB)
        file_name = rgb_map_name.split(".")[0]

        # convert depth map to point cloud
        height, width = depth_map.shape    
        w_ind = np.arange(width)
        h_ind = np.arange(height)

        ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
        ww_ind = ww_ind.reshape(-1)
        hh_ind = hh_ind.reshape(-1)
        depth_map = depth_map.reshape(-1)

        valid = np.where(depth_map > 0.1)[0]
        ww_ind = ww_ind[valid]
        hh_ind = hh_ind[valid]
        depth_map = depth_map[valid]

        xyz, xyz_local = convert_from_uvd(ww_ind, hh_ind, depth_map, intrinsic_depth, pose)

        # xyz_offset = pose[:3,3]
        xyz_offset = np.mean(xyz, axis=0)
        # z_offset = np.mean(np.sort(xyz[:,-1])[100:1000])
        # xyz_offset[-1] = z_offset
        
        rgb, label, ins = get_color_label(xyz_local, intrinsic_image, color_map, instance_map, label_map)
        xyzrgb = np.concatenate([xyz, rgb, label, ins], axis=1)
        #np.savetxt("xyzrgb_xiaobao_0000.txt", xyzrgb, fmt="%.3f")
        #exit()

        bbox_2d = get_2d_bbox(color_map, instance_map, label_map, rgb_map_name.split(".")[0])
        bbox_3d = get_3d_bbox(xyzrgb)
        #print(bbox_3d.shape)

        if bbox_3d.shape[0] > 0 and bbox_2d.shape[0] > 0 and bbox_3d_global.shape[0] > 0:
            bbox_3d = select_3d_bbox(bbox_3d, bbox_3d_global)
            bbox_3d[:,:3] -= xyz_offset
            xyzrgb[:,:3] -= xyz_offset
            print(bbox_3d.shape)
        else:
            continue
            
        if bbox_3d.shape[0] > 0 and bbox_2d.shape[0] > 0 and bbox_3d_global.shape[0] > 0:
            # 1
            rgb_map_name_no = rgb_map_name.split(".")[0]
            
            bbox_2d_name = "%s_%s_2d_bbox"%(scan_name,rgb_map_name_no)
            np.save(os.path.join(TARGET_DIR,bbox_2d_name),bbox_2d)

            bbox_3d_name = "%s_%s_bbox"%(scan_name,rgb_map_name_no)
            np.save(os.path.join(TARGET_DIR,bbox_3d_name),bbox_3d)
            bbox_3d[:,3:6] *= 2
            bbox_3d[:,6] *= -1

            #write_oriented_bbox(bbox_3d[:,:7], os.path.join(TARGET_DIR, "%s.ply"%(bbox_3d_name)))
            #write_oriented_bbox(bbox_3d[:,:7], "local_bbox_%s.ply"%(bbox_3d_name))
            #exit()

            save_rgb_name = "%s_%s"%(scan_name,rgb_map_name)
            
            image = np.array(imageio.v2.imread(os.path.join(scan_path,RGB_PATH,rgb_map_name)))
            image = sktf.resize(image, [968, 1296], order=0, preserve_range=True)
            imageio.v2.imwrite(os.path.join(TARGET_DIR,save_rgb_name), image)
            # 2
            
            xyzrgb = random_sampling(xyzrgb, 50000)
            
            pc_name = "%s_%s_pc"%(scan_name,rgb_map_name_no)
            np.save(os.path.join(TARGET_DIR,pc_name),xyzrgb[:,:6])
            
            pc_name = "%s_%s_sem_label"%(scan_name,rgb_map_name_no)
            np.save(os.path.join(TARGET_DIR,pc_name),xyzrgb[:,-2])

            ins_mapping = { 
                ins:i for i, ins in enumerate(np.unique(xyzrgb[:,-1]).tolist())
            }
            ins_label = xyzrgb[:,-1]
            for i in range(ins_label.shape[0]):
                ins_label[i] = ins_mapping[ins_label[i]] 
            pc_name = "%s_%s_ins_label"%(scan_name,rgb_map_name_no)
            np.save(os.path.join(TARGET_DIR,pc_name),ins_label)

            # pc_name = "%s_%s_pc.txt"%(scan_name,rgb_map_name_no)
            # np.savetxt(os.path.join(TARGET_DIR,pc_name),xyzrgb[:,:6],fmt="%.3f")

            pose_name = "%s_%s_pose.txt"%(scan_name,rgb_map_name_no)
            
            # pose[:3,3] -= pose[:3,3]
            pose[:3,3] -= xyz_offset
            np.savetxt(os.path.join(TARGET_DIR,pose_name),pose,fmt="%.5f")

            # pc_name = "%s_%s_pc_50k.txt"%(scan_name,rgb_map_name_no)
            # np.savetxt(os.path.join(TARGET_DIR,pc_name),xyzrgb[:,:6],fmt="%.3f")
            # exit()
            
            '''
            for box in bbox_3d:
                label_no = box[7]
                if class2type_semseg[label_no] not in num_dict.keys():
                    num_dict[class2type_semseg[label_no]] = 1
                else:
                    num_dict[class2type_semseg[label_no]] += 1
            else:
            continue
            '''

def make_split(path_dict, split="train"):
    DATA_PATH = path_dict["DATA_PATH"]
    TARGET_DIR_PREFIX = path_dict["TARGET_DIR_PREFIX"]
    AXIS_ALIGN_MATRIX_PATH = path_dict["AXIS_ALIGN_MATRIX_PATH"]
    RGB_PATH = path_dict["RGB_PATH"]
    DEPTH_PATH = path_dict["DEPTH_PATH"]
    INSTANCE_PATH = path_dict["INSTANCE_PATH"]
    LABEL_PATH = path_dict["LABEL_PATH"]
    POSE_PATH = path_dict["POSE_PATH"]

    num_dict = {}
    TARGET_DIR = "%s_%s"%(TARGET_DIR_PREFIX,split)
    path_dict["TARGET_DIR"] = TARGET_DIR
    os.makedirs(TARGET_DIR,exist_ok=True)
    f = open("Test_GT_Maker/%s.txt"%(split))
    scan_name_list = sorted(f.readlines())

    multi_process_parameter = []
    #pdb.set_trace()
    for scan_name_index,scan_name in enumerate(scan_name_list):
        cur_parameter = {}
        cur_parameter["scan_name_index"] = scan_name_index
        cur_parameter["scan_name"] = scan_name
        cur_parameter["path_dict"] = path_dict
        cur_parameter["scan_num"] = len(scan_name_list)
        multi_process_parameter.append(cur_parameter)

        process_cur_scan(cur_parameter)

    # pool = multiprocessing.Pool(10)
    # pool.map(process_cur_scan, multi_process_parameter)

    # pool.close()
    # pool.join()


    # for cur_scan in multi_process_parameter:
    #    process_cur_scan(cur_scan)



def main():
    DATA_PATH = "../scannet_frames_25k" # Replace it with the path to scannet_frames_25k
    TARGET_DIR_PREFIX = "../scannet_sv_18cls" # Replace it with the path to output path
    AXIS_ALIGN_MATRIX_PATH = "../scans" # Replace it with the path to axis_align_matrix path
    RGB_PATH = "./color"
    DEPTH_PATH = "./depth"
    INSTANCE_PATH = "./instance"
    LABEL_PATH = "./label"
    POSE_PATH = "./pose"

    path_dict = {"DATA_PATH": DATA_PATH,
                "TARGET_DIR_PREFIX": TARGET_DIR_PREFIX,
                "AXIS_ALIGN_MATRIX_PATH": AXIS_ALIGN_MATRIX_PATH,
                "RGB_PATH": RGB_PATH,
                "DEPTH_PATH": DEPTH_PATH,
                "INSTANCE_PATH": INSTANCE_PATH,
                "LABEL_PATH": LABEL_PATH,
                "POSE_PATH": POSE_PATH,        
                }

    splits = ["train", "val"]
    #splits = ["val"]
    


    for cur_split in splits:
        make_split(path_dict, cur_split)


if __name__ == "__main__":
    main()