# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

from .scannet_utils.evaluate_semantic_instance_v2 import scannet_eval_v2
import torch
import pdb
import time

def aggregate_predictions(masks, labels, scores, valid_class_ids):
    """Maps predictions to ScanNet evaluator format.

    Args:
        masks (list[torch.Tensor]): Per scene predicted instance masks.
        labels (list[torch.Tensor]): Per scene predicted instance labels.
        scores (list[torch.Tensor]): Per scene predicted instance scores.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[dict]: Per scene aggregated predictions.
    """
    infos = []
    for id, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        mask = mask.clone().numpy()
        label = label.clone().numpy()
        score = score.clone().numpy()
        info = dict()
        for i in range(mask.shape[0]):
            # match pred_instance['filename'] from assign_instances_for_scan
            file_name = f'{id}_{i}'
            info[file_name] = dict()
            info[file_name]['mask'] = mask[i]
            info[file_name]['label_id'] = valid_class_ids[label[i]]
            info[file_name]['conf'] = score[i]
        infos.append(info)
    return infos


def rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids):
    """Maps gt instance and semantic masks to instance masks for ScanNet
    evaluator.

    Args:
        gt_semantic_masks (list[torch.Tensor]): Per scene gt semantic masks.
        gt_instance_masks (list[torch.Tensor]): Per scene gt instance masks.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[np.array]: Per scene instance masks.
    """
    renamed_instance_masks = []
    for semantic_mask, instance_mask in zip(gt_semantic_masks,
                                            gt_instance_masks):
        semantic_mask = semantic_mask.clone().numpy()
        instance_mask = instance_mask.clone().numpy()
        unique = np.unique(instance_mask)
        assert len(unique) < 1000
        for i in unique:
            if i==0:
               semantic_mask[instance_mask == 0] = -1
            semantic_instance = semantic_mask[instance_mask == i]
            semantic_unique = np.unique(semantic_instance)
            assert len(semantic_unique) == 1
            if semantic_unique[0] in valid_class_ids:
                instance_mask[
                    instance_mask ==
                    i] = 1000 * semantic_unique[0] + i
        renamed_instance_masks.append(instance_mask)
    return renamed_instance_masks


def instance_seg_eval_v2(gt_semantic_masks,
                         gt_instance_masks,
                         pred_instance_masks,
                         pred_instance_labels,
                         pred_instance_scores,
                         valid_class_ids,
                         class_labels,
                         options=None,
                         logger=None):
    """Instance Segmentation Evaluation.

    Evaluate the result of the instance segmentation.

    Args:
        gt_semantic_masks (list[torch.Tensor]): Ground truth semantic masks.
        gt_instance_masks (list[torch.Tensor]): Ground truth instance masks.
        pred_instance_masks (list[torch.Tensor]): Predicted instance masks.
        pred_instance_labels (list[torch.Tensor]): Predicted instance labels.
        pred_instance_scores (list[torch.Tensor]): Predicted instance labels.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Names of valid categories.
        options (dict, optional): Additional options. Keys may contain:
            `overlaps`, `min_region_sizes`, `distance_threshes`,
            `distance_confs`. Default: None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(valid_class_ids) == len(class_labels)
    id_to_label = {
        valid_class_ids[i]: class_labels[i]
        for i in range(len(valid_class_ids))
    }
    preds = aggregate_predictions(
        masks=pred_instance_masks,
        labels=pred_instance_labels,
        scores=pred_instance_scores,
        valid_class_ids=valid_class_ids)
    gts = rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids)
    metrics = scannet_eval_v2(
        preds=preds,
        gts=gts,
        options=options,
        valid_class_ids=valid_class_ids,
        class_labels=class_labels,
        id_to_label=id_to_label)
    header = ['classes', 'AP_0.25', 'AP_0.50', 'AP', 'Prec_0.50', 'Rec_0.50']
    rows = []
    for label, data in metrics['classes'].items():
        aps = [data['ap25%'], data['ap50%'], data['ap'], data['prec50%'], data['rec50%']]
        rows.append([label] + [f'{ap:.4f}' for ap in aps])
    aps = metrics['all_ap_25%'], metrics['all_ap_50%'], metrics['all_ap'], metrics['all_prec_50%'], metrics['all_rec_50%']
    footer = ['Overall'] + [f'{ap:.4f}' for ap in aps]
    table = AsciiTable([header] + rows + [footer])
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    return metrics



def multiview_instance_seg_eval_v2(
                         points,
                         gt_semantic_masks,
                         gt_instance_masks,
                         pred_instance_masks,
                         pred_instance_labels,
                         pred_instance_scores,
                         valid_class_ids,
                         class_labels,
                         options=None,
                         logger=None,
                         evaluator_mode='slice_len_constant',
                         num_slice=0,
                         len_slice=0,
                         voxel_size=.02,
                         use_voxel_eval=True):
    """Instance Segmentation Evaluation.

    Evaluate the result of the instance segmentation.

    Args:
        gt_semantic_masks (list[torch.Tensor]): Ground truth semantic masks.
        gt_instance_masks (list[torch.Tensor]): Ground truth instance masks.
        pred_instance_masks (list[torch.Tensor]): Predicted instance masks.
        pred_instance_labels (list[torch.Tensor]): Predicted instance labels.
        pred_instance_scores (list[torch.Tensor]): Predicted instance labels.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Names of valid categories.
        options (dict, optional): Additional options. Keys may contain:
            `overlaps`, `min_region_sizes`, `distance_threshes`,
            `distance_confs`. Default: None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(valid_class_ids) == len(class_labels)
    id_to_label = {
        valid_class_ids[i]: class_labels[i]
        for i in range(len(valid_class_ids))
    }
    preds = []
    gts = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for scene_idx in range(len(gt_semantic_masks)):
        print('scene: %d'%scene_idx,end="  ")
        t0=time.time()
        timestamps = []
        # as short as I can 
        if evaluator_mode == 'slice_len_constant':
            i=1
            while i*len_slice<len(gt_semantic_masks[scene_idx]):
                timestamps.append(i*len_slice)
                i=i+1
            timestamps.append(len(gt_semantic_masks[scene_idx]))
        else:
            num_slice = min(len(gt_semantic_masks[scene_idx]),num_slice)
            for i in range(1,num_slice):
                timestamps.append(i*(len(gt_semantic_masks[scene_idx])//num_slice))
            timestamps.append(len(gt_semantic_masks[scene_idx]))

        # online evaluation
        for j in range(len(timestamps)):
            ts = timestamps[j]
            point_slice = points[scene_idx][(0 if j==0 else timestamps[j-1]):ts,:,:3].reshape(-1,3)
            gt_semantic_mask_slice = gt_semantic_masks[scene_idx][(0 if j==0 else timestamps[j-1]):ts].reshape(-1)
            gt_instance_mask_slice = gt_instance_masks[scene_idx][(0 if j==0 else timestamps[j-1]):ts].reshape(-1)
            pred_instance_label_slice = pred_instance_labels[scene_idx][j]
            pred_instance_score_slice = pred_instance_scores[scene_idx][j]
            pred_instance_mask_slice = pred_instance_masks[scene_idx][j].permute(1,0)

            if point_slice.shape[0] > 1000000:
                pdb.set_trace()
                choice = np.random.choice(point_slice.shape[0], 1000000, replace=False)
                point_slice = point_slice[choice]
                gt_semantic_mask_slice = gt_semantic_mask_slice[choice]
                gt_instance_mask_slice = gt_instance_mask_slice[choice]
                pred_instance_mask_slice = pred_instance_mask_slice[choice]

            if use_voxel_eval:
                
                t1 = time.time() - t0
                print('stage1 %.1f sec'%t1,end="  ")
                t1 = time.time()
                sparse_tensor_coordinates = (torch.cat((torch.zeros(point_slice.shape[0], 1), (point_slice / voxel_size).floor().int()), dim=1)).contiguous().to(device)
                gt_sparse_feature = torch.cat([gt_instance_mask_slice.unsqueeze(1), gt_semantic_mask_slice.unsqueeze(1)], dim=1).to(device)
                pred_sparse_feature = pred_instance_mask_slice.float().to(device)

                # del gt_instance_mask_slice
                # del gt_semantic_mask_slice
                # del pred_instance_mask_slice
                # del point_slice

                t15 = time.time() - t1
                print('stage1.5 %.1f sec'%t15,end="  ")
                t15 = time.time()
                gt_sparse = ME.SparseTensor(
                    features=gt_sparse_feature,
                    coordinates=sparse_tensor_coordinates,
                    quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
                )
                pred_sparse = ME.SparseTensor(
                    features=pred_sparse_feature,
                    coordinates=sparse_tensor_coordinates,
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM
                )
                print('vNum:%d'%(pred_sparse.coordinates.shape[0]),end="  ")

                # # Visualize
                # np.save('/home/ubuntu/xxw/Online3D/Online3D/work_dirs/vis_voxel_data/points_%s.npy'%(scene_idx), sparse_tensor_coordinates[:,1:].cpu().numpy())
                # np.save('/home/ubuntu/xxw/Online3D/Online3D/work_dirs/vis_voxel_data/gt_semantic_masks_%s.npy'%(scene_idx), gt_sparse_feature[:,1].cpu().numpy())
                # np.save('/home/ubuntu/xxw/Online3D/Online3D/work_dirs/vis_voxel_data/gt_instance_masks_%s.npy'%(scene_idx), gt_sparse_feature[:,0].cpu().numpy())
                # np.save('/home/ubuntu/xxw/Online3D/Online3D/work_dirs/vis_voxel_data/pred_instance_masks_%s.npy'%(scene_idx), pred_sparse_feature.cpu().numpy())

                # del sparse_tensor_coordinates
                # del gt_sparse_feature
                # del pred_sparse_feature
                # torch.cuda.empty_cache()

                t2 = time.time() - t15
                print('stage2 %.1f sec'%t2,end="  ")
                t2 = time.time()
                dic = aggregate_predictions(
                    masks=[(pred_sparse.features_at_coordinates(gt_sparse.coordinates.float())>=1).permute(1,0).cpu()],
                    labels=[pred_instance_label_slice.cpu()],
                    scores=[pred_instance_score_slice.cpu()],
                    valid_class_ids=valid_class_ids)[0]
                new_dic = {}
                for i,key in enumerate(list(dic.keys())):
                    new_dic['%s_%s'%(len(preds),i)] = dic[key]
                preds.append(new_dic)
                gts.append(rename_gt([gt_sparse.features[:,1].cpu()], [gt_sparse.features[:,0].cpu()], valid_class_ids)[0])
                t3 = time.time() - t2
                print('stage3 %.1f sec'%t3)
                t3 = time.time() 
            else:
                # waiting for more 
                dic = aggregate_predictions(
                    masks=[pred_instance_mask_slice.float().permute(1,0).cpu()],
                    labels=[pred_instance_label_slice.cpu()],
                    scores=[pred_instance_score_slice.cpu()],
                    valid_class_ids=valid_class_ids)[0]
                new_dic = {}
                for i,key in enumerate(list(dic.keys())):
                    new_dic['%s_%s'%(len(preds),i)] = dic[key]
                preds.append(new_dic)
                gts.append(rename_gt([gt_semantic_mask_slice], [gt_instance_mask_slice], valid_class_ids)[0])



            # del gt_sparse
            # del pred_sparse
            # torch.cuda.empty_cache()
            
        # del pred_instance_masks[0]
        # del pred_instance_labels[0]
        # del pred_instance_scores[0]
        # del gt_semantic_masks[0]
        # del gt_instance_masks[0]
        # # del points[0]

    
    print('metrics',end="  ")
    tx = time.time()

    metrics = scannet_eval_v2(
        preds=preds,
        gts=gts,
        options=options,
        valid_class_ids=valid_class_ids,
        class_labels=class_labels,
        id_to_label=id_to_label)
    ty = time.time() - tx
    print('stage metrics %.1f sec'%ty,end="  ")
    header = ['classes', 'AP_0.25', 'AP_0.50', 'AP', 'Prec_0.50', 'Rec_0.50']
    rows = []
    for label, data in metrics['classes'].items():
        aps = [data['ap25%'], data['ap50%'], data['ap'], data['prec50%'], data['rec50%']]
        rows.append([label] + [f'{ap:.4f}' for ap in aps])
    aps = metrics['all_ap_25%'], metrics['all_ap_50%'], metrics['all_ap'], metrics['all_prec_50%'], metrics['all_rec_50%']
    footer = ['Overall'] + [f'{ap:.4f}' for ap in aps]
    table = AsciiTable([header] + rows + [footer])
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    return metrics
