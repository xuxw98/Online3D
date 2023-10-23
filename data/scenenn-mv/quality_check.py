import multiprocessing
import numpy as np
from PIL import Image
from numpy.linalg import inv


class QualityCheck(object):
    """docstring for QualityCheck"""
    def __init__(self, fx, fy, cx, cy, width, height, downsample_step = 1):
        super(QualityCheck, self).__init__()
        self.width = width
        self.height = height
        self.project = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.back_project = np.linalg.inv(self.project)
        self.downsample_step = downsample_step
 
        self.image_dir = np.empty([3, self.height, self.width])
        for row_i in range(self.height):
            for col_i in range(self.width):
                self.image_dir[:, row_i, col_i] = np.dot(self.back_project, np.array([col_i, row_i, 1.0]))
 
    def check(self, left_image, right_image, depth_image, left2right):
 
        space_points = self.image_dir * np.expand_dims(depth_image,0)
        roatation = left2right[0:3,0:3]
        translation = left2right[0:3,3]
        space_points_tranformed = np.dot(roatation, np.reshape(space_points, [3,-1])) + np.expand_dims(translation,1)
        project_map = np.dot(self.project, space_points_tranformed)
        project_map = project_map / project_map[2,:]
        project_map = np.reshape(project_map, [3, self.height, self.width])
 
        rebuild_image = np.zeros([3, self.height, self.width])
        diff_image = np.zeros([self.height, self.width])
        hit_num = 0
        total_sample = 0
        for row_i in range(0, self.height, self.downsample_step):
            for col_i in range(0, self.width, self.downsample_step):
                if depth_image[row_i][col_i] == 0.0:
                    continue
                total_sample = total_sample + 1
                u = int(project_map[0, row_i, col_i])
                v = int(project_map[1, row_i, col_i])
                if u >= 0 and u < self.width and v >= 0 and v < self.height:
                    rebuild_image[:, v, u] = left_image[:, row_i, col_i]
                    diff_image[v, u] = np.abs(rebuild_image[:, v, u] - right_image[:, v, u]).sum()
                    hit_num = hit_num + 1
        return float(hit_num)/(total_sample), diff_image.sum()/(float(total_sample)), rebuild_image, diff_image