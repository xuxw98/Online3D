# Copyright (c) OpenMMLab. All rights reserved.
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python demo/online_demo.py
from argparse import ArgumentParser

import os
from os import path as osp
from copy import deepcopy
import numpy as np
import torch
import mmcv
import imageio
from mmcv.parallel import collate, scatter
from mmdet3d.apis import init_model
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import DepthInstance3DBoxes
from concurrent import futures as futures
from demo.online_visualizer import Visualizer
from demo.online_data_converter import ScanNetMVDataConverter

def inference_detector(model, scene_idx):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)

    scannetmv_converter = ScanNetMVDataConverter(root_path="./data/scannet",split='test')
    data = scannetmv_converter.process_single_scene(scene_idx,cfg)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result



def save_result_gif(result,scene_idx,num_frame,score_thr):
    num_frame=min(num_frame,len(result[0]))
    all_points = None
    all_bboxs = None
    png_for_gif=[]
    root_path = './data/scannet/2D/%s'%scene_idx
    vis = Visualizer()
    for i in range(num_frame):
        if i == 0:
            all_points = np.load(os.path.join(root_path,'point','%s.npy'%(i*40)))
            all_bboxs = result[0][i*40]['boxes_3d'].tensor.cpu().numpy()
            scores = result[0][i*40]['scores_3d'].cpu().numpy()
            #accumulated_bboxs  = np.load(os.path.join(root_path,'bboxs','%s.npy'%(i*40)))
            #scores = np.load(os.path.join(root_path,'scores','%s.npy'%(i*40)))
            select_idx = np.where(scores>score_thr)
            all_bboxs = all_bboxs[select_idx[0],:]
            #imgs = np.load(os.path.join(root_path,'imgs','%s.npy'%i*40))
        else:
            current_points = np.load(os.path.join(root_path,'point','%s.npy'%(i*40)))
            current_bboxs = result[0][i*40]['boxes_3d'].tensor.cpu().numpy()
            scores = result[0][i*40]['scores_3d'].cpu().numpy()
            select_idx = np.where(scores>score_thr)
            current_bboxs = current_bboxs[select_idx[0],:]
            all_points = np.concatenate((all_points,current_points),axis=0)
            all_bboxs = np.concatenate((all_bboxs,current_bboxs),axis=0)
            #imgs = np.load(os.path.join(root_path,'imgs','%s.npy'%i*40))
    
        vis.update_points(points=all_points)
        vis.update_bboxs(bbox3d=all_bboxs)
        if os.path.exists('./demo/data/online/%s' % scene_idx):
            pass
        else:
            os.makedirs('./demo/data/online/%s' % scene_idx)
            
        save_path = './demo/data/online/%s/%s.jpg'%(scene_idx,(i*40))
        vis.capture_img(save_path=save_path)
        png_for_gif.append(imageio.imread(save_path))

    imageio.mimsave('./demo/data/online/%s/online.gif'%scene_idx, png_for_gif, fps=5)	# fps值越大，生成的gif图播放就越快

def main():
    parser = ArgumentParser()
    parser.add_argument('--scene-idx', type=str, default="scene0568_00",help='single scene index')
    parser.add_argument('--num-frame', type=int,default=300,help='number of frame to merge')
    parser.add_argument('--score-thr', type=float,default=0.1,help='bbox score threshold')
    parser.add_argument('--config', type=str, default="configs/online3d/online3d_8x2_scannet-3d-18class.py" ,help='Config file')
    parser.add_argument('--checkpoint', type=str, default="work_dirs/online3d_v0/latest.pth", help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # model init
    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)
    # test a single scene
    result = inference_detector(model=model, scene_idx=args.scene_idx)
    # save result gift
    save_result_gif(result=result,scene_idx=args.scene_idx,num_frame=args.num_frame,score_thr=args.score_thr)


if __name__ == '__main__':
    main()



