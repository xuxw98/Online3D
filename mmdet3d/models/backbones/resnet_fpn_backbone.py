import torch.nn as nn
import torch
from mmcv.runner import BaseModule, _load_checkpoint_with_prefix, load_state_dict
from mmdet3d.models.builder import BACKBONES, build_neck
from .detectron2_basemodule import ShapeSpec, BasicStem, BasicBlock, BottleneckBlock, \
     ResNet, LastLevelP6P7, LastLevelMaxPool, FPN
import pdb

@BACKBONES.register_module()
class Resnet_FPN_Backbone(BaseModule):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (ont): Number of input channels, 3 for RGB.
        num_stages (int, optional): Resnet stages. Default: 4.
        pool (bool, optional): Add max pooling after first conv if True.
            Default: True.
    """

    def __init__(self, img_memory=None):
        super(Resnet_FPN_Backbone, self).__init__()
        input_shape = ShapeSpec(channels=3)
        
        #bottom_up = build_resnet_backbone(cfg, input_shape)

        norm = 'FrozenBN'
        # input_shape.channels 64
        stem = BasicStem(
            in_channels=input_shape.channels,
            out_channels=64,
            norm=norm,
        )

        # fmt: off
        freeze_at           = 2
        out_features        = ['res2','res3','res4','res5']
        depth               = 50
        num_groups          = 1
        width_per_group     = 64
        bottleneck_channels = num_groups * width_per_group
        in_channels         = 64
        out_channels        = 256
        stride_in_1x1       = True
        res5_dilation       = 1
        deform_on_per_stage = [False, False, False, False]
        deform_modulated    = False
        deform_num_groups   = 1
        # fmt: on
        assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]

        if depth in [18, 34]:
            assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
            assert not any(
                deform_on_per_stage
            ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
            assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
            assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

        stages = []

        for idx, stage_idx in enumerate(range(2, 6)):
            # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
            dilation = res5_dilation if stage_idx == 5 else 1
            first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
            stage_kargs = {
                "num_blocks": num_blocks_per_stage[idx],
                "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "norm": norm,
            }
            # Use BasicBlock for R18 and R34.
            if depth in [18, 34]:
                stage_kargs["block_class"] = BasicBlock
            else:
                stage_kargs["bottleneck_channels"] = bottleneck_channels
                stage_kargs["stride_in_1x1"] = stride_in_1x1
                stage_kargs["dilation"] = dilation
                stage_kargs["num_groups"] = num_groups
                stage_kargs["block_class"] = BottleneckBlock
            blocks = ResNet.make_stage(**stage_kargs)
            in_channels = out_channels
            out_channels *= 2
            bottleneck_channels *= 2
            stages.append(blocks)
        bottom_up = ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
        in_features = ['res2','res3','res4','res5']
        out_channels = 256
        # in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
        self.backbone = FPN(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm='',
            # top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
            top_block=LastLevelMaxPool(),
            fuse_type='sum',
        )
        # if img_memory is not None:
        #     self.img_memory = build_neck(img_memory)

    def init_weights(self):
        ckpt_path = './mmdet3d/models/backbones/img_backbone.pth'
        ckpt = torch.load(ckpt_path)
        load_state_dict(self, ckpt['model'], strict=False)
        # if hasattr(self, 'img_memory'):
        #     self.img_memory.init_weights()

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    # def reset(self):
    #     if hasattr(self, 'img_memory'):
    #         self.img_memory.reset()

    def forward(self, x):
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        x = self.backbone(x)
        # if hasattr(self, 'img_memory'):
        #     x['p2'] = self.img_memory(x['p2'])
        return x
    
