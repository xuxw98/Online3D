from tqdm import tqdm
import os, struct
import numpy as np
import zlib
import imageio
import cv2
import math
import sys

path_2d = './2D/'

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

if __name__ == '__main__':
    f = open('meta_data/scannet_train.txt', 'r')
    unify_dim = (640, 480)
    unify_intrinsic = adjust_intrinsic(make_intrinsic(577.870605,577.870605,319.5,239.5), [640,480], unify_dim)
    scene_names = f.readlines()
    
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
        boxes = np.load('scannet_train_detection_data/'+scene_name+'_bbox.npy')
        try:
            os.makedirs('2D/%s/point' % scene_name)
        except:
            pass
        try:
            os.makedirs('2D/%s/box_mask' % scene_name)
        except:
            pass
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
            pc = depth_image_to_point_cloud(img, depth, unify_intrinsic[:3,:3], pose)
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

            # Box Mask
            # box_center = boxes[:,:3].copy()
            # box_center -= pose_[:3,3]
            # box_center = np.dot(pose_[:3,:3].transpose(), box_center.transpose()).transpose() # Nx3
            # uv = np.dot(box_center, unify_intrinsic[:3, :3].transpose())
            # uv[:,0] /= uv[:,2]
            # uv[:,1] /= uv[:,2]
            # mask = (uv[:,0]>=0) * (uv[:,0]<640) * (uv[:,1]>=0) * (uv[:,1]<480) * (box_center[:,2]>0)
            alpha=0.3
            beta=0.1
            mask=np.zeros((boxes.shape[0]))
            #print(mask)
            for j in range(boxes.shape[0]):
                #print('boxes')
                #print(boxes[j])
                l,w,h=boxes[j,3].copy(),boxes[j,4].copy(),boxes[j,5].copy()
                corners=np.array([[0,0,0],
                [-l/2,w/2,h/2],
                [l/2,w/2,h/2],
                [l/2,-w/2,h/2],
                [-l/2,-w/2,h/2],
                [-l/2,w/2,-h/2],
                [l/2,w/2,-h/2],
                [l/2,-w/2,-h/2],
                [-l/2,-w/2,-h/2]]) 
                #1v7 2v8 3v5 4v6
                center=boxes[j,:3].copy()
                corners=corners+center
                #print('corners')
                #print(corners)
                corners -= pose_[:3,3]
                corners = np.dot(pose_[:3,:3].transpose(), corners.transpose()).transpose() # Nx3
                uv = np.dot(corners, unify_intrinsic[:3, :3].transpose())
                uv[:,0] /= uv[:,2]
                uv[:,1] /= uv[:,2]

                pair1=False
                pair2=False
                pair3=False
                pair4=False
                pair5=False
                
                check=(uv[:,0]>=0) * (uv[:,0]<640) * (uv[:,1]>=0) * (uv[:,1]<480) * (corners[:,2]>0)
                # 1v7
                pivotu=uv[1,0]+alpha*(uv[7,0]-uv[1,0])
                pivotv=uv[1,1]+alpha*(uv[7,1]-uv[1,1])
                pivotz=corners[1,2]+alpha*(corners[7,2]-corners[1,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair1=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[7,0]+alpha*(uv[1,0]-uv[7,0])
                pivotv=uv[7,1]+alpha*(uv[1,1]-uv[7,1])
                pivotz=corners[7,2]+alpha*(corners[1,2]-corners[7,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair1=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[1,0]+beta*(uv[7,0]-uv[1,0])
                pivotv=uv[1,1]+beta*(uv[7,1]-uv[1,1])
                pivotz=corners[1,2]+beta*(corners[7,2]-corners[1,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[7,0]+beta*(uv[1,0]-uv[7,0])
                pivotv=uv[7,1]+beta*(uv[1,1]-uv[7,1])
                pivotz=corners[7,2]+beta*(corners[1,2]-corners[7,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                # 2v8
                pivotu=uv[2,0]+alpha*(uv[8,0]-uv[2,0])
                pivotv=uv[2,1]+alpha*(uv[8,1]-uv[2,1])
                pivotz=corners[2,2]+alpha*(corners[8,2]-corners[2,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair2=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[8,0]+alpha*(uv[2,0]-uv[8,0])
                pivotv=uv[8,1]+alpha*(uv[2,1]-uv[8,1])
                pivotz=corners[8,2]+alpha*(corners[2,2]-corners[8,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair2=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[2,0]+beta*(uv[8,0]-uv[2,0])
                pivotv=uv[2,1]+beta*(uv[8,1]-uv[2,1])
                pivotz=corners[2,2]+beta*(corners[8,2]-corners[2,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[8,0]+beta*(uv[2,0]-uv[8,0])
                pivotv=uv[8,1]+beta*(uv[2,1]-uv[8,1])
                pivotz=corners[8,2]+beta*(corners[2,2]-corners[8,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                #3v5
                pivotu=uv[3,0]+alpha*(uv[5,0]-uv[3,0])
                pivotv=uv[3,1]+alpha*(uv[5,1]-uv[3,1])
                pivotz=corners[3,2]+alpha*(corners[5,2]-corners[3,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair3=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[5,0]+alpha*(uv[3,0]-uv[5,0])
                pivotv=uv[5,1]+alpha*(uv[3,1]-uv[5,1])
                pivotz=corners[5,2]+alpha*(corners[3,2]-corners[5,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair3=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[3,0]+beta*(uv[5,0]-uv[3,0])
                pivotv=uv[3,1]+beta*(uv[5,1]-uv[3,1])
                pivotz=corners[3,2]+beta*(corners[5,2]-corners[3,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[5,0]+beta*(uv[3,0]-uv[5,0])
                pivotv=uv[5,1]+beta*(uv[3,1]-uv[5,1])
                pivotz=corners[5,2]+beta*(corners[3,2]-corners[5,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                #4v6
                pivotu=uv[4,0]+alpha*(uv[6,0]-uv[4,0])
                pivotv=uv[4,1]+alpha*(uv[6,1]-uv[4,1])
                pivotz=corners[4,2]+alpha*(corners[6,2]-corners[4,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair4=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[6,0]+alpha*(uv[4,0]-uv[6,0])
                pivotv=uv[6,1]+alpha*(uv[4,1]-uv[6,1])
                pivotz=corners[6,2]+alpha*(corners[4,2]-corners[6,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    pair4=True
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[4,0]+beta*(uv[6,0]-uv[4,0])
                pivotv=uv[4,1]+beta*(uv[6,1]-uv[4,1])
                pivotz=corners[4,2]+beta*(corners[6,2]-corners[4,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True
                pivotu=uv[6,0]+beta*(uv[4,0]-uv[6,0])
                pivotv=uv[6,1]+beta*(uv[4,1]-uv[6,1])
                pivotz=corners[6,2]+beta*(corners[4,2]-corners[6,2])
                if (pivotu>=0) * (pivotu<640) * (pivotv>=0) * (pivotv<480) * (pivotz>0):
                    if label[int(pivotv)][int(pivotu)] == boxes[j,6]:
                        pair5=True

                if check[0]:
                    pair1=True
                    pair2=True
                    pair3=True
                    pair4=True
                    if label[int(uv[0,1])][int(uv[0,0])] == boxes[j,6]:
                        pair5=True
                
                if (pair1 or pair2 or pair3 or pair4) and pair5:
                    mask[j]=1

            np.save('2D/%s/point/%s.npy' % (scene_name, frame_id), pc)
            np.save('2D/%s/box_mask/%s.npy' % (scene_name, frame_id), mask)
