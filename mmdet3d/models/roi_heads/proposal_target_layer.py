from turtle import forward
import torch
import numpy as np
import torch.nn as nn
# from mmdet3d.ops.pcdet_nms.pcdet_nms_utils import boxes_iou3d_gpu
from mmcv.ops.iou3d import boxes_iou3d

class ProposalTargetLayer(nn.Module):
    def __init__(self,
                 roi_per_image=128,
                 fg_ratio=0.5,
                 reg_fg_thresh=0.3,
                 cls_fg_thresh=0.55,
                 cls_bg_thresh=0.15,
                 cls_bg_thresh_l0=0.1,
                 hard_bg_ratio=0.8,
                 ):
        super(ProposalTargetLayer,self).__init__()
        self.roi_per_image = roi_per_image
        self.fg_ratio = fg_ratio
        self.reg_fg_thresh = reg_fg_thresh
        self.cls_fg_thresh = cls_fg_thresh
        self.cls_bg_thresh = cls_bg_thresh
        self.cls_bg_thresh_l0 = cls_bg_thresh_l0
        self.hard_bg_ratio = hard_bg_ratio
    
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: b, num_max_rois, 6
                roi_scores:
                roi_labels:
                gt_bboxes_3d: list[tensor(N,6)]
                gt_labels_3d: list[tensor(N)]
        """
        batch_rois, batch_gt_of_rois, batch_gt_label_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict)
        
        # regression valid mask
        reg_valid_mask = (batch_roi_ious > self.reg_fg_thresh).long()

        # classification label
        iou_bg_thresh = self.cls_bg_thresh
        iou_fg_thresh = self.cls_fg_thresh
        fg_mask = batch_roi_ious > iou_fg_thresh
        bg_mask = batch_roi_ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        batch_cls_labels = (fg_mask > 0).float()
        batch_cls_labels[interval_mask] = \
            (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        
        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_label_of_rois': batch_gt_label_of_rois,
                        'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}
                        
        return targets_dict
        
    def sample_rois_for_rcnn(self, batch_dict):
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_bboxes_3d']
        gt_labels = batch_dict['gt_labels_3d']

        code_size = rois.shape[-1] # 7
        gt_code_size = gt_boxes[0].tensor.shape[-1] # 7
        batch_rois = rois.new_zeros(batch_size, self.roi_per_image, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_per_image, gt_code_size) 
        batch_gt_label_of_rois = rois.new_zeros(batch_size, self.roi_per_image) 
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_per_image)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_per_image)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_per_image), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            cur_labels = gt_labels[index]
            cur_gt = torch.cat((cur_gt.gravity_center.clone(), cur_gt.tensor[:, 3:].clone()), dim=1).to(cur_roi.device)
            # TODO: check if there are all zeros gt_boxes
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            # sample roi by each class
            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_labels.long()
                )
            
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
            batch_gt_label_of_rois[index] = cur_labels[gt_assignment[sampled_inds]]
        # TODO: check targets, visulize
        return batch_rois, batch_gt_of_rois, batch_gt_label_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels
    
    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.fg_ratio * self.roi_per_image))
        fg_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.cls_bg_thresh_l0)).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.reg_fg_thresh) &
                (max_overlaps >= self.cls_bg_thresh_l0)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_per_image - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.hard_bg_ratio
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_per_image) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_per_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.hard_bg_ratio
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                #iou3d = boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                iou3d = boxes_iou3d(cur_roi, cur_gt)  # (M, N)
                
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment