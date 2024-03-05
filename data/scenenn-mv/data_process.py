from os.path import isfile, join, isdir
from os import listdir
from math import fabs
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import det
import math
import random
import cv2
import pickle
import sys
from quality_check import *
import pdb
import scannet_utils
import numpy as np
from scipy.spatial import cKDTree
import os

valid_sequence = ['015', '005', '030', '054', '322', '263', '243', '080', '089', '093', '096', '011']
train_sequence = ['005', '700', '207', '073', '337', '240', '237', '205', '263', '276', '014', '089', '021', '613', '260', '279', '528', '234', '096', '286', '041', '521', '217', '066', '036', '011', '065', '322', '607', '209', '255', '069', '265', '272', '092', '032', '025', '610', '054', '047', '225', '202', '076', '057', '527', '060', '273', '252', '080', '201', '231', '311', '270', '016', '251', '109', '078', '213', '227', '622', '030', '082', '294', '611', '522', '074', '087', '086', '061', '052', '623', '621', '084', '043', '062', '243', '246', '524', '098', '249', '038', '308', '609', '206', '223']
image_folder = './data/scenenn-mv'
trajectory_folder = './data/scenenn-mv/trajectory/'
fx = 544.47329
fy = 544.47329
cx = 320
cy = 240
width = 640
height= 480

def get_pose(info_file, pose_index):
    index = pose_index - 1
    pose = np.zeros((4,4),float)
    for row_i in range(4):
        line = info_file[index*5+row_i+1].split()
        for col_i in range(4):
            pose[row_i,col_i] = float(line[col_i])
    return pose

def angle_diff(transform1, transform2):
	relative_trans = np.dot(inv(transform1), transform2)
	return math.acos(relative_trans[2,2] - 1e-4) / math.pi * 180.0

def trans_diff(transform1, transform2):
	relative_trans = np.dot(inv(transform1), transform2)
	return norm(relative_trans[0:3,3])

def depth_image_to_point_cloud(rgb, depth, K, pose):
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

    points = np.transpose(np.vstack((position[0:3, :], R, G, B, valid.nonzero()[0])))
    return points


def nearest_neighbor_interpolation(tree, target_coords, source_labels):
    _, nearest_neighbors = tree.query(target_coords[:,:3])
    interpolated_labels = source_labels[nearest_neighbors]
    return interpolated_labels


if sys.argv[1] == 'train':
	generate_sqeuence = train_sequence
	save_file_name = 'SceneNN_train.pkl'
	print('generate train sequence')
else:
	generate_sqeuence = valid_sequence
	save_file_name = 'SceneNN_validate.pkl'
	print('generate vaild sequence')


checker = QualityCheck(fx,fy,cx,cy,width,height,8)

image_mean = np.zeros(3)
image_std = np.zeros(3)
image_add = 0.0
pair_list = []

label_map_file = './data/scenenn-mv/scannetv2-labels.combined.tsv'
label_map = scannet_utils.read_label_mapping(label_map_file,
    label_from='raw_category', label_to='nyu40id')    

intrinsic = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1]])
    
for sequence_index, this_sequence in enumerate(generate_sqeuence):
    
    this_mesh_path = image_folder+this_sequence+'/%03d'%(int(this_sequence)) + '.ply'
    this_xml_path = image_folder+this_sequence+'/%03d'%(int(this_sequence)) + '.xml'
    try:
        os.makedirs(image_folder+this_sequence+'/pose')
    except:
        pass
    try:
        os.makedirs(image_folder+this_sequence+'/point')
    except:
        pass
    try:
        os.makedirs(image_folder+this_sequence+'/label')
    except:
        pass
    this_mesh, this_label_txt = scannet_utils.read_mesh_vertices_rgb(this_mesh_path, this_xml_path)
    # read the trajectory
    if this_mesh.shape[0] > 1000000:
        choice = np.random.choice(this_mesh.shape[0], 1000000, replace=False)
        this_mesh = this_mesh[choice]
        this_label_txt = this_label_txt[choice]
    this_label = []
    for i in range(this_label_txt.shape[0]):
        if this_label_txt[i] == 'unknown':
            this_label.append(0)
        elif this_label_txt[i] == 'otherprop':
            this_label.append(40)
        elif this_label_txt[i] == 'fridge':
            this_label.append(24)
        else:
            try:
                this_label.append(label_map[this_label_txt[i]])
            except:
                this_label.append(0)
    this_label = np.array(this_label)
    this_tree = cKDTree(this_mesh[:,:3])


    
    with open(trajectory_folder + '%s_trajectory.log'%this_sequence) as file:
        trajectory_info = file.readlines()
    total_index = len(trajectory_info) / 5
    print('sequence %s has %d poses.'%(this_sequence, total_index))

    down_sample_sequence = range(1, int(total_index), 40)

    this_pc_list = []
    this_label_list = []
    for this_index, this_image_index in enumerate(down_sample_sequence):
        print('process: sequence %f %%, inner sequence %f %%'%(
            (sequence_index+1)/float(len(generate_sqeuence))*100.0,
            (this_index+1)/float(len(down_sample_sequence))*100.0 ))

        this_pose = get_pose(trajectory_info, this_image_index)
        if math.fabs(det(this_pose) - 1.0) > 0.1:
            pdb.set_trace()
            continue
        this_image_path = image_folder+this_sequence+'/image/image'+'%05d'%(this_image_index) + '.png'
        this_depth_path = image_folder+this_sequence+'/depth/depth'+'%05d'%(this_image_index) + '.png'
        this_image = np.asanyarray(Image.open(this_image_path), dtype = float)
        this_depth = np.asanyarray(Image.open(this_depth_path), dtype = float)
        this_depth = this_depth/1000.0

        this_pc = depth_image_to_point_cloud(this_image, this_depth, intrinsic, this_pose)
        this_pc = this_pc[np.random.choice(this_pc.shape[0], 20000, replace=False)]
        
        this_single_labels = nearest_neighbor_interpolation(this_tree, this_pc, this_label)

        np.save(image_folder+this_sequence+'/pose/%05d.npy'%(this_image_index),this_pose)
        np.save(image_folder+this_sequence+'/point/%05d.npy'%(this_image_index),this_pc)
        np.save(image_folder+this_sequence+'/label/%05d.npy'%(this_image_index),this_single_labels)



        this_pair = {}
        this_pair['image'] = image_folder+this_sequence+'/image/image'+'%05d'%(this_image_index) + '.png'
        this_pair['depth'] = image_folder+this_sequence+'/depth/depth'+'%05d'%(this_image_index) + '.png'
        this_pair['pose'] = this_pose
        pair_list.append(this_pair)
    
print('we have total %d pairs'%(len(pair_list)))
dataset = {}
dataset['data_files'] = pair_list
dataset['intrinsic'] = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1]])
dataset['width'] = width
dataset['depth_scale'] = 1000.0
dataset['height'] = height
with open(save_file_name, 'wb') as output:
    pickle.dump(dataset, output, -1)




