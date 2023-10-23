# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from ..third_party.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from ..utils.pc_util import scale_points, shift_scale_points

from .anyview import AnyViewFormer
from .helpers import GenericMLP
from .position_embedding import PositionEmbeddingCoordsSine
from .transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)

from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from mmcv.cnn import initialize


class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        # 
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
            # every segment
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            # -pi ~ pi
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        # background class
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        # objectness_prob means the probability there is an object
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETR(nn.Module):
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
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        use_image=False,
        no_pcenc=False,
        npoint=1024,
        token_tot_num=2048
    ):
        super().__init__()
        assert (not no_pcenc) or (no_pcenc and use_image)
        self.no_pcenc = no_pcenc
        self.npoint = npoint
        if not no_pcenc:
            self.pre_encoder = pre_encoder
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
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)
        
        # Image Branch
        self.use_image = use_image
        if use_image:
            self.im_backbone = ResNet(depth=34, out_indices=(0,1,2,3), num_stages=4, strides=(1,2,2,2), dilations=(1,1,1,1),
                                      frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=False))
            initialize(self.im_backbone, dict(type='Pretrained', checkpoint='torchvision://resnet34'))
            self.im_neck = FPN(in_channels=[64,128,256,512], out_channels=encoder_dim//4, num_outs=4,
                               add_extra_convs='on_output', relu_before_extra_convs=True)
            self.im_pc_fusion = AnyViewFormer(trans_layers=0, one_side=False, Is_mask=False)
            # self.im_to_decoder_projection = GenericMLP(
            #     input_dim=encoder_dim,
            #     hidden_dims=[encoder_dim],
            #     output_dim=decoder_dim,
            #     norm_fn_name="bn1d",
            #     activation="relu",
            #     use_conv=True,
            #     output_use_activation=True,
            #     output_use_norm=True,
            #     output_use_bias=False,
            # )
        self.token_tot_num = token_tot_num
    # final step
    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # different mlp

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
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
        # query_xyz is for another pos_embeddings
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    # vital 0
    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)

        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # unique transformer
        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long())
        return enc_xyz, enc_features, enc_inds

    # vital 2
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
        # intermediate layers output
        aux_outputs = outputs[:-1]
        # final layer output
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }


    def forward(self, inputs, encoder_only=False):
        point_clouds = inputs["point_clouds"]
        if self.use_image:
            images = inputs["images"] # B T H W 3
            view_mask = inputs["view_mask"] # B T
            point_cloud_inds = inputs["point_cloud_inds"] # B N

        if not self.no_pcenc:
            enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
            enc_features = self.encoder_to_decoder_projection(
                enc_features.permute(1, 2, 0)
            ).permute(2, 0, 1)
            # encoder features: npoints x batch x channel
            # encoder xyz: npoints x batch x 3

            if encoder_only:
                # return: batch x npoints x channels
                return enc_xyz, enc_features.transpose(0, 1)
        else:
            xyz = point_clouds[..., 0:3].contiguous()
            enc_inds = furthest_point_sample(xyz, self.npoint)
            enc_inds = enc_inds.long()
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            enc_xyz = gather_operation(xyz_flipped, enc_inds.int())
            enc_xyz = enc_xyz.transpose(1, 2).contiguous()
            enc_features = torch.zeros_like(enc_xyz).sum(-1).unsqueeze(1).repeat(1,self.decoder_dim,1)
            enc_features = enc_features.permute(2, 0, 1)
            
        
        if self.use_image:
            B = view_mask.shape[0]
            pc_inds = torch.gather(point_cloud_inds, 1, enc_inds.long())
            image_feats = self.im_backbone(images.permute(0,1,4,2,3).contiguous().view(-1, 3, self.H, self.W))
            image_feats_tuple = self.im_neck(image_feats)
            image_feats_list = [image_feats_tuple[0]]
            for i in range(1, len(image_feats_tuple)):
                image_feats_list.append(F.interpolate(image_feats_tuple[i], size=image_feats_tuple[0].shape[-2:]))
            image_feats = torch.cat(image_feats_list, dim=1)
            # image_feats = image_feats_tuple[0]
            # # For DDP...
            # for i in range(1, len(image_feats_tuple)):
            #     image_feats += 0.0 * image_feats_tuple[i].sum(-2, keepdim=True).sum(-1, keepdim=True)
            image_feats = image_feats.view(B, -1, image_feats.shape[1], image_feats.shape[2], image_feats.shape[3])
            enc_features = enc_features.permute(1, 2, 0)
            # feature fusion
            enc_features = self.im_pc_fusion(image_feats, view_mask, enc_features, pc_inds)
            # enc_features = self.im_to_decoder_projection(enc_features)
            enc_features = enc_features.permute(2, 0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        # ?
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        # little trick    query_embed has used pos_embeddings and use another MLP
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]

        # query_xyz, box_features = self.decoder(enc_xyz, enc_features.permute(1, 2, 0))
        # box_features = box_features.permute(2, 0, 1).unsqueeze(0)

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        
        num=torch.ones(B)
        num=num*self.token_tot_num


        box_predictions['outputs']={**box_predictions['outputs'],'token_num':num}
        for i in range(len(box_predictions['aux_outputs'])):
            box_predictions['aux_outputs'][i]={**box_predictions['aux_outputs'][i],'token_num':num}

        return box_predictions


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder


def build_votenet_decoder(args):
    decoder = VoteNetDecoder(
        decoder_feature_dim=args.dec_dim,
    )
    return decoder


def build_3detr(args, dataset_config):
    if not args.no_pcenc:
        pre_encoder = build_preencoder(args)
        encoder = build_encoder(args)
    else:
        pre_encoder, encoder = None, None
    decoder = build_decoder(args)
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        use_image=args.use_image,
        no_pcenc=args.no_pcenc,
        npoint=args.preenc_npoints if args.enc_type == "vanilla" else args.preenc_npoints // 2,
        token_tot_num=args.token_tot_num
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor
