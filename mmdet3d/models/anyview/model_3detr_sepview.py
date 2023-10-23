# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from .third_party.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from .utils.pc_util import scale_points, shift_scale_points
from .utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)

from .models.anyview import AnyViewFormer
from .models.helpers import GenericMLP
from .models.position_embedding import PositionEmbeddingCoordsSine
from .models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)

from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from mmcv.cnn import initialize
from mmcv.ops import nms3d, nms3d_normal
import pdb

from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models import DETECTORS
from mmdet3d.core.bbox.structures.depth_box3d import DepthInstance3DBoxes
from mmdet3d.core import bbox3d2result
import time

PPF = PROXY_PER_FRAME = 40
MAX_FRAMES = 50


class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self):
        pass

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / 1
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == 18 + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(self, box_center_unnorm, box_size_unnorm, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size_unnorm, box_angle, box_center_upright)
        return boxes

@DETECTORS.register_module()
class Model3DETR_SepView(Base3DDetector):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder=None,
        encoder=None,
        decoder=None,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        DQ_FPS=True,
        enc_PE=True,
        enc_proj=True,
        evaluator_mode='slice_len_constant',
        num_slice=0,
        len_slice=0,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        pretrained=None
    ):
        super().__init__()

        pre_encoder = build_preencoder()
        encoder = build_encoder()
        decoder = build_decoder()
        self.pre_encoder = nn.ModuleList(pre_encoder)
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        if enc_proj:
            self.enc_projection = GenericMLP(
                input_dim=encoder_dim,
                hidden_dims=[encoder_dim],
                output_dim=encoder_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder_dim = decoder_dim
        self.decoder = decoder
        self.build_mlp_heads(decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.DQ_FPS = DQ_FPS
        self.enc_PE = enc_PE
        self.enc_proj = enc_proj
        self.box_processor = BoxProcessor()
        self.test_cfg = test_cfg
        self.evaluator_mode=evaluator_mode
        self.num_slice=num_slice
        self.len_slice=len_slice

    def view_model_param(self):
        total_param = 0
        print("MODEL DETAILS:\n")
        #print(model)
        for param in self.parameters():
            # print(param.data.size())
            total_param += np.prod(list(param.data.size()))
        # for param in self.pre_encoder.parameters():
        #     # print(param.data.size())
        #     total_param += np.prod(list(param.data.size()))
        # for param in self.encoder.parameters():
        #     # print(param.data.size())
        #     total_param += np.prod(list(param.data.size()))
        # for param in self.encoder_to_decoder_projection.parameters():
        #     # print(param.data.size())
        #     total_param += np.prod(list(param.data.size()))
        # for param in self.enc_projection.parameters():
        #     # print(param.data.size())
        #     total_param += np.prod(list(param.data.size()))
        # for param in self.query_projection.parameters():
        #     # print(param.data.size())
        #     total_param += np.prod(list(param.data.size()))
        # for param in self.decoder.parameters():
        #     # print(param.data.size())
        #     total_param += np.prod(list(param.data.size()))
        # for param in self.mlp_heads.parameters():
        #     # print(param.data.size())
        #     total_param += np.prod(list(param.data.size()))
        print('MODEL/Total parameters:', total_param)
        
        # 假设每个参数是一个 32 位浮点数（4 字节）
        bytes_per_param = 4
        
        # 计算总字节数
        total_bytes = total_param * bytes_per_param
        
        # 转换为兆字节（MB）和千字节（KB）
        total_megabytes = total_bytes / (1024 * 1024)
        total_kilobytes = total_bytes / 1024

        print("Total parameters in MB:", total_megabytes)
        print("Total parameters in KB:", total_kilobytes)

        return total_param


    def build_mlp_heads(self, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=18 + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=1)
        angle_reg_head = mlp_func(output_dim=1)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz.contiguous(), self.num_queries)
        query_inds = query_inds.long()
        xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        query_xyz = gather_operation(xyz_flipped, query_inds.int())
        query_xyz = query_xyz.transpose(1, 2).contiguous()

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds, trans, point_cloud_dims):
        '''
        point_clouds: B T 5000 3+C
        trans: B T 3
        '''
        B, T, _, C = point_clouds.shape
        point_clouds = point_clouds.view(B*T, -1, C)

        # xyz: B*T 5000 3
        # features: B*T C 5000
        xyz, features = self._break_up_pc(point_clouds)
        for pre_enc in self.pre_encoder:
            xyz, features, _ = pre_enc(xyz, features)

        pre_enc_xyz = xyz
        pre_enc_features = features
        # there needs more process
        # throw
        pre_enc_xyz = pre_enc_xyz.view(B, T, PPF, 3)
        valid = (pre_enc_xyz.sum(-1) != 0)
        pre_enc_xyz = pre_enc_xyz + trans.unsqueeze(-2)
        pre_enc_xyz[~valid] = 0
        pre_enc_xyz = pre_enc_xyz.view(B, PPF*T, 3)
        pre_enc_features = pre_enc_features.view(B, T, -1, PPF).transpose(1,2).contiguous().view(B, -1, PPF*T)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # xyz dim B N 3
        # feature dim B C N
        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)
        # feature dim N B C

        enc_pos = None
        if self.enc_PE:
            enc_pos = self.pos_embedding(pre_enc_xyz, input_range=point_cloud_dims)
            if self.enc_proj:
                enc_pos = self.enc_projection(enc_pos)
            enc_pos = enc_pos.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        # to edit self.encoder
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz, pos=enc_pos
        )

        return enc_xyz, enc_features

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def extract_feat(self, points, img, img_metas):
        pass


    def forward_train(self, points, pts_semantic_mask, img, img_metas):
        """Forward of training.

        Returns:
            dict: Loss values.
        """
        pass
    def simple_test(self, points, img_metas, *args, **kwargs):
        # t0 = time.time()
        # print('Frame:%d'%points[0].shape[0])
        inputs = {}
        # all_points = points[0].reshape(-1,points[0].shape[-1])[:,:3]
        # sample = torch.randint(size=(40000,), high=all_points.shape[0], low=0)
        # inputs['point_clouds'] = all_points[sample].unsqueeze(0)
        # inputs['point_cloud_dims_min'] = inputs['point_clouds'].min(dim=1)[0]
        # inputs['point_cloud_dims_max'] = inputs['point_clouds'].max(dim=1)[0]
        
        pcs = points[0][:,:,:3].clone().detach().cpu().numpy()
        valid = (pcs.sum(-1) != 0).reshape(-1, 5000, 1)
        trans = np.nan_to_num((pcs * valid).sum(1) / valid.sum(-1).sum(-1, keepdims=True))
        pcs -= trans.reshape(-1,1,3)
        pcs[~valid.reshape(-1,5000)] = 0
        inputs['pcs'] = torch.from_numpy(pcs.astype(np.float32)).cuda().unsqueeze(0)
        inputs['trans'] = torch.from_numpy(trans.astype(np.float32)).cuda().unsqueeze(0)

                # Benchmark
        timestamps = []
        if self.evaluator_mode == 'slice_len_constant':
            i=1
            while i*self.len_slice<len(points[0]):
                timestamps.append(i*self.len_slice)
                i=i+1
            timestamps.append(len(points[0]))
        else:
            num_slice = min(len(points[0]),self.num_slice)
            for i in range(1,num_slice):
                timestamps.append(i*(len(points[0])//num_slice))
            timestamps.append(len(points[0]))

        # Process
        bbox_results = [[]]

        for i in range(len(timestamps)):
            if i == 0:
                ts_start, ts_end = 0, timestamps[i]
            else:
                ts_start, ts_end = timestamps[i-1], timestamps[i]   

            point_clouds = inputs["pcs"][:,ts_start:ts_end,:,:].contiguous() # B T 5000 3+C
            trans = inputs["trans"][:,ts_start:ts_end,:].contiguous() # B T 3

            points_slice = points[0][ts_start:ts_end,:,:].reshape(-1,points[0].shape[-1])[:,:3]
            if points_slice.shape[0] > 40000: 
                sample = torch.randint(size=(40000,), high=points_slice.shape[0], low=0)
                points_slice = points_slice[sample].unsqueeze(0)
            else:
                points_slice = points_slice.unsqueeze(0)

            point_cloud_dims = [
                points_slice.min(dim=1)[0],
                points_slice.max(dim=1)[0],
            ]
            # pc dim B T 5000 3+C
            enc_xyz, enc_features = self.run_encoder(point_clouds, trans, point_cloud_dims)
            B=enc_xyz.shape[0]
            # feature dim N B C
            enc_features = self.encoder_to_decoder_projection(
                enc_features.permute(1, 2, 0)
            ).permute(2, 0, 1)
            # encoder features: npoints x batch x channel
            # encoder xyz: npoints x batch x 3

            query_xyz, query_embed = self.get_query_embeddings(points_slice if self.DQ_FPS else enc_xyz, point_cloud_dims)
            # query_embed: batch x channel x npoint
            enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

            # decoder expects: npoints x batch x channel
            # just for transformer
            enc_pos = enc_pos.permute(2, 0, 1)
            query_embed = query_embed.permute(2, 0, 1)
            tgt = torch.zeros_like(query_embed)

            # ignore features with coordinate (0,0,0)
            mask = torch.zeros(enc_xyz.shape[0], enc_xyz.shape[1], enc_xyz.shape[1]).to(enc_xyz.get_device())
            zero_index = (enc_xyz.sum(-1) == 0)
            mask = zero_index.unsqueeze(-2).repeat(1,self.num_queries,1)
            bsz, lq, lk = mask.shape
            nhead = self.decoder.layers[0].nhead
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, nhead, 1, 1)
            mask = mask.view(bsz * nhead, lq, lk)
            
            box_features = self.decoder(
                tgt, enc_features, query_pos=query_embed, pos=enc_pos, memory_mask=mask
            )[0]

            # organized data but not final data for final display
            box_predictions = self.get_box_predictions(
                query_xyz, point_cloud_dims, box_features
            )
            bboxes =  torch.cat([box_predictions['outputs']['center_unnormalized'], box_predictions['outputs']['size_unnormalized']],dim=2).squeeze(0)
            scores = box_predictions['outputs']['sem_cls_prob'].squeeze(0)
            bboxes, scores, labels = self._single_scene_multiclass_nms(bboxes, scores)
            ret_res = bbox3d2result(bboxes, scores, labels)
            for j in range(ts_start, ts_end):
                bbox_results[0].append(ret_res)

        # t1 = time.time() - t0
        # print('stage1 %f sec'%t1)
        
        return bbox_results
    
    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
    

    def _single_scene_multiclass_nms(self, bboxes, scores):
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor]: Predicted bboxes, scores and labels.
        """
        if len(scores) > self.test_cfg.nms_pre_merge > 0:
            max_scores, _ = scores.max(dim=1)
            _, ids = max_scores.topk(self.test_cfg.nms_pre_merge)
            bboxes = bboxes[ids]
            scores = scores[ids]

        n_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            class_bboxes = torch.cat(
                (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                dim=1)
            nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr_merge)

            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if with_yaw:
            box_dim = 7
        else:
            box_dim = 6
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = DepthInstance3DBoxes(
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

       
        return nms_bboxes, nms_scores, nms_labels




# independent

def build_preencoder():
    mlp_dims = [3 * 0, 64, 128, 256]
    preencoder1 = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=2048 // 8,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    preencoder2 = PointnetSAModuleVotes(
        radius=0.8,
        nsample=32,
        npoint=PPF,
        mlp=[256, 256, 256, 256],
        normalize_xyz=True,
    )
    preencoder = [preencoder1, preencoder2]
    return preencoder


def build_encoder():
    # there needs more correction
    encoder_layer = TransformerEncoderLayer(
        d_model=256,
        nhead=4,
        dim_feedforward=128,
        dropout=0.1,
        activation='relu',
    )
    
    masking_radius = [math.pow(x, 2) for x in [0.8, 0.8, 1.2]]
    encoder = MaskedTransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=3,
        interim_downsampling=None,
        masking_radius=masking_radius,
    )
    return encoder


def build_decoder():
    decoder_layer = TransformerDecoderLayer(
        d_model=256,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=8, return_intermediate=True
    )
    return decoder



# def build_3detr_sepview(args, dataset_config):
#     pre_encoder = build_preencoder(args)
#     encoder = build_encoder(args)
#     decoder = build_decoder(args)
#     model = Model3DETR_SepView(
#         pre_encoder,
#         encoder,
#         decoder,
#         dataset_config,
#         encoder_dim=args.enc_dim,
#         decoder_dim=args.dec_dim,
#         mlp_dropout=args.mlp_dropout,
#         num_queries=args.nqueries,
#         DQ_FPS=args.DQ_FPS,
#         enc_PE=args.enc_PE,
#         enc_proj=args.enc_proj
#     )
#     output_processor = BoxProcessor(dataset_config)
#     return model, output_processor
