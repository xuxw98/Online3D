import  numpy as np
from queue import Queue
import time
class point3D:
    def __init__(self,point_coor,feature_2d,max_octree_threshold):
        self.point_coor=point_coor
        self.feature_fuse=feature_2d
        self.branch_array=[None, None, None, None, None, None, None, None]
        self.branch_distance=np.full((8),max_octree_threshold)
        self.result_feature=np.zeros((128))
        self.pred_result=-1
        self.frame_id=0
        self.scan_times=0
        self.uncertainty=1


   
    def findNearPoint(self,near_node_num,max_node):

        neighbor_2dfeature=np.zeros((max_node+1,128))
        neighbor_node=np.zeros((max_node+1,3))
        count = 0
        neighbor_node[count, :] = 0
        neighbor_2dfeature[count] = self.feature_fuse
        find_queue = Queue()
        count += 1

        for i,node in enumerate(self.branch_array):
            if node is not None:
                neighbor_node[count, :] =  node.point_coor - self.point_coor
                neighbor_2dfeature[count] = node.feature_fuse
                if node.branch_array[i] is not None:
                    find_queue.put((i,node.branch_array[i])) 
                count += 1

        while not find_queue.empty() and count<max_node+1:
            index_node=find_queue.get()
            node=index_node[1]
            index=index_node[0]
            neighbor_node[count, :] =  node.point_coor - self.point_coor
            neighbor_2dfeature[count] = node.feature_fuse
            if node.branch_array[index] is not None:
                find_queue.put((index,node.branch_array[index]))
            count+=1
        
        if count>=near_node_num:
            sample=np.random.choice(count,near_node_num,replace=False)
        else:
            sample=np.random.choice(count,near_node_num,replace=True)
        sample[0] = 0
        return neighbor_2dfeature[sample,:].T,neighbor_node[sample,:].T,count