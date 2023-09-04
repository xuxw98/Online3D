import cv2
import shutil
import numpy as np
from scipy import stats
import os
from plyfile import PlyData,PlyElement
from scipy import stats
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import trimesh
import multiprocessing
import pdb
import imageio
import skimage.transform as sktf 
import torch
from load_scannet_data import export

valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34,
            36, 39)
max_cat_id=40

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
    uv_ind = np.minimum(uv_ind, np.ones_like(uv_ind)*(label.reshape(-1).shape[0]-1))
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
    if len(scene_bboxes) != 0 :
        scene_bboxes = np.concatenate(scene_bboxes, axis=0)
        scene_instances = np.concatenate(scene_instances, axis=0)
        scene_labels = np.concatenate(scene_labels, axis=0)
    else:
        scene_bboxes = np.empty([0,7])
        scene_instances = np.empty([1])
        scene_labels = np.empty([1])

    return scene_bboxes, scene_instances, scene_labels
   
def get_3d_bbox(xyzrgb):
    if xyzrgb is None:
        return np.empty([0,8]), 0
    if xyzrgb.shape[0] == 0:
        return np.empty([0,8]), 0
    instance = xyzrgb[:,-1].copy()
    label = xyzrgb[:,-2].copy()
    xyz = xyzrgb[:,:3].copy()


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
        # pts_instance_mask_one_hot = torch.nn.functional.one_hot(pts_instance_mask)
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

    scene_bboxes = bboxes
    scene_instances = torch.tensor(instance_unique)
    scene_labels = np.zeros(scene_instances.shape)

    # OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
    # mask = np.in1d(scene_labels, OBJ_CLASS_IDS)
    # scene_bboxes = scene_bboxes[mask,:]
    # scene_labels = scene_labels[mask]
    # scene_instances = scene_instances[mask]


    if scene_bboxes.shape[0] > 0:
        bbox_3d = np.concatenate([scene_bboxes, scene_labels.reshape(-1,1)], axis=1)
    else:
        bbox_3d=np.empty([0,8])
        
    return bbox_3d, scene_instances


        
def match_box(modal_data, scene_data):
    # model bboxes Mx8
    # scene bboxes Mx8
    [modal_bboxes, modal_instances] = modal_data 
    [scene_bboxes, scene_instances] = scene_data
    if modal_bboxes.shape[0] == 0:
        return np.empty([0,8])

    modal_bboxes_new = []
    for i in range(scene_instances.shape[0]):
        for j in range(modal_instances.shape[0]):
            if modal_instances[j]==scene_instances[i]:
                modal_bboxes[j,-1] = scene_bboxes[i,-1]
                modal_bboxes_new.append(modal_bboxes[j,:].reshape(1,-1))
    
    if len(modal_bboxes_new) == 0:
        modal_bboxes_new = np.empty([0,8])
        return modal_bboxes_new
    
    modal_bboxes_new = np.concatenate(modal_bboxes_new, axis=0)
    return modal_bboxes_new





def get_3d_bbox_new(xyzrgb):
    instance = xyzrgb[:,-1].copy()
    label = xyzrgb[:,-2].copy()
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
        # if cur_tmp_xyz.shape[0] > 100:
        #     # save_path =  os.path.join('/home/ubuntu/xxw/SmallDet/mmdetection3d/dataset/OVD_sv_real_gt/OVD_sv_real_gt_train', "%s_pc_before_%s.obj"%(name,ins))
        #     # _write_obj(cur_tmp_xyz,  save_path)
        ###剔除离群点
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cur_tmp_xyz)
        res = pcd.remove_statistical_outlier(20, 0.5)   #统计法
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
                
            # if clusters[np.argmax(max_mask)].shape[0] > 100 and clusters[np.argmax(max_mask)].shape[0] / cur_tmp_xyz.shape[0] > 0.1:
            #     cur_tmp_xyz = cur_tmp_xyz[clusters[np.argmax(max_mask)],:]
            # # cur_tmp_xyz = cur_tmp_xyz[cluster_ind,:]
            # # if name == 'scene0000_00_000900' and ins == 4:
            # #     save_path =  os.path.join('/home/ubuntu/xxw/SmallDet/mmdetection3d/dataset/OVD_sv_real_gt/OVD_sv_real_gt_train', "%s_pc_after_%s.obj"%(name,ins))
            # #     _write_obj(cur_tmp_xyz,  save_path)
            # #     ic('okkkk!!!!!!')
            # # ic(cur_tmp_xyz.shape, ins)
              
            
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
        instances = instances[instance_valid]
        scene_bboxes = bboxes
        scene_instances = torch.tensor(instances)
        scene_labels = (instances//1000).numpy()
        OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        mask = np.in1d(scene_labels, OBJ_CLASS_IDS)
        scene_bboxes = scene_bboxes[mask,:]
        scene_labels = scene_labels[mask]
        scene_instances = scene_instances[mask]
    else:
        scene_bboxes=np.empty([0,8])
        scene_instances=torch.tensor(np.empty(0))

    if scene_bboxes.shape[0] > 0:
        bbox_3d = np.concatenate([scene_bboxes, scene_labels.reshape(-1,1)], axis=1)
    else:
        bbox_3d=np.empty([0,8])
        
    return bbox_3d, scene_instances

def export_one_scan(scan_name):    
    mesh_file = os.path.join('scans', scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join('scans', scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join('scans', scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join('scans', scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
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
    instance = xyzrgb[:,-1].copy()
    instance_new_xyzrgb = np.zeros_like(instance)
    semantic = xyzrgb[:,-2].copy()
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

        cnt = -1
        for j in range(instance.shape[0]):
            if mask_single[j]:
                cnt = cnt+1
                if mask[cnt]:
                    instance_new_xyzrgb[j] = instance_target
                    semantic[j] = bboxes[i,-1]

    xyzrgb_new = np.concatenate([xyzrgb[:,:6].copy(), semantic.reshape(-1,1), instance_new_xyzrgb.reshape(-1,1)], axis=1)
    if len(xyz_new) == 0:
        return np.empty([0,5]), xyzrgb_new
    xyz_new = np.concatenate(xyz_new, axis=0)
    semantic_new = np.concatenate(semantic_new, axis=0)
    instance_new = np.concatenate(instance_new, axis=0)

    xyz_all = np.concatenate([xyz_new,semantic_new.reshape(-1,1), instance_new.reshape(-1,1)], axis=1)
    return xyz_all, xyzrgb_new



def process_cur_scan(cur_scan):
    scan_name_index = cur_scan["scan_name_index"]
    scan_name = cur_scan["scan_name"]
    path_dict = cur_scan["path_dict"]
    scan_num = cur_scan["scan_num"]
    print(scan_name)

    DATA_PATH = path_dict["DATA_PATH"]
    INS_DATA_PATH = path_dict["INS_DATA_PATH"]
    TARGET_DIR = path_dict["TARGET_DIR"]
    AXIS_ALIGN_MATRIX_PATH = path_dict["AXIS_ALIGN_MATRIX_PATH"]
    RGB_PATH = path_dict["RGB_PATH"]
    DEPTH_PATH = path_dict["DEPTH_PATH"]
    LABEL_PATH = path_dict["LABEL_PATH"]
    POSE_PATH = path_dict["POSE_PATH"]

    scan_name = scan_name.strip("\n")
    scan_path = os.path.join(DATA_PATH,scan_name)
    ins_data_path = os.path.join(INS_DATA_PATH,scan_name)
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
    instance_map_list = []
    for depth_map_name in depth_map_list:
        frame_num = int(depth_map_name[-10:-4])
        instance_map_list.append(os.path.join(INS_DATA_PATH,scan_name,'instance',"%s.png"%(frame_num)))
    #instance_map_list = sorted(os.listdir(os.path.join(scan_path,INS_DATA_PATH)))
    label_map_list = sorted(os.listdir(os.path.join(scan_path,LABEL_PATH)))

    poses = [np.dot(axis_align_matrix ,load_matrix_from_txt(os.path.join(scan_path,POSE_PATH, i))) for i in POSE_txt_list]

    scene_bboxes, bbox_instance_labels = export_one_scan(scan_name)

    for rgb_map_name,\
        depth_map_name,\
        instance_map_name,\
        label_map_name,\
        pose in zip(rgb_map_list,depth_map_list,instance_map_list,label_map_list,poses):

        instance_map = cv2.imread(instance_map_name,-1)
        instance_map = cv2.resize(instance_map, (1296,968), interpolation=cv2.INTER_NEAREST)
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
        xyzrgb = random_sampling(xyzrgb, 50000)
        xyz_for_bbox, xyzrgb = select_points_in_bbox(xyzrgb, scene_bboxes, bbox_instance_labels)
        
        modal_bboxes, modal_instances = get_3d_bbox(xyz_for_bbox)
        modal_bboxes = match_box([modal_bboxes, modal_instances], [scene_bboxes, bbox_instance_labels])

        # modal_bboxes, modal_instances = get_3d_bbox_new(xyz_for_bbox)
        #print(bbox_3d.shape)
        #pdb.set_trace()

        if modal_bboxes.shape[0] > 0:       
            # remember to keep it   
            modal_bboxes[:,:3] -= xyz_offset
            xyzrgb[:,:3] -= xyz_offset
            print(modal_bboxes.shape)
        else:
            continue
            
        if modal_bboxes.shape[0] > 0:
            # 1
            rgb_map_name_no = rgb_map_name.split(".")[0]
            

            bbox_3d_name = "%s_%s_bbox"%(scan_name,rgb_map_name_no)
            np.save(os.path.join(TARGET_DIR,bbox_3d_name),modal_bboxes)
            #modal_bboxes[:,3:6] *= 2
            #modal_bboxes[:,6] *= -1


            save_rgb_name = "%s_%s"%(scan_name,rgb_map_name)
            
            image = np.array(imageio.v2.imread(os.path.join(scan_path,RGB_PATH,rgb_map_name)))
            image = sktf.resize(image, [968, 1296], order=0, preserve_range=True)
            imageio.v2.imwrite(os.path.join(TARGET_DIR,save_rgb_name), image)
            # 2
            
            
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
    INS_DATA_PATH = path_dict["INS_DATA_PATH"]
    TARGET_DIR_PREFIX = path_dict["TARGET_DIR_PREFIX"]
    AXIS_ALIGN_MATRIX_PATH = path_dict["AXIS_ALIGN_MATRIX_PATH"]
    RGB_PATH = path_dict["RGB_PATH"]
    DEPTH_PATH = path_dict["DEPTH_PATH"]
    LABEL_PATH = path_dict["LABEL_PATH"]
    POSE_PATH = path_dict["POSE_PATH"]

    num_dict = {}
    TARGET_DIR = "%s_%s"%(TARGET_DIR_PREFIX,split)
    path_dict["TARGET_DIR"] = TARGET_DIR
    os.makedirs(TARGET_DIR,exist_ok=True)
    f = open("meta_data/scannetv2_%s.txt"%(split))
    scan_name_list = sorted(f.readlines())

    multi_process_parameter = []
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
    DATA_PATH = "./scannet_frames_25k" # Replace it with the path to scannet_frames_25k
    TARGET_DIR_PREFIX = "./scannet_sv_18cls" # Replace it with the path to output path
    INS_DATA_PATH = "./2D" # Replace it with the path to 2D
    AXIS_ALIGN_MATRIX_PATH = "./scans" # Replace it with the path to axis_align_matrix path
    RGB_PATH = "./color"
    DEPTH_PATH = "./depth"
    LABEL_PATH = "./label"
    POSE_PATH = "./pose"

    path_dict = {"DATA_PATH": DATA_PATH,
                "TARGET_DIR_PREFIX": TARGET_DIR_PREFIX,
                "INS_DATA_PATH": INS_DATA_PATH,
                "AXIS_ALIGN_MATRIX_PATH": AXIS_ALIGN_MATRIX_PATH,
                "RGB_PATH": RGB_PATH,
                "DEPTH_PATH": DEPTH_PATH,
                "LABEL_PATH": LABEL_PATH,
                "POSE_PATH": POSE_PATH,        
                }

    splits = ["train", "val"]
    #splits = ["val"]
    


    for cur_split in splits:
        make_split(path_dict, cur_split)


if __name__ == "__main__":
    main()