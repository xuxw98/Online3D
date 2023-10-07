from mmcv.cnn.bricks import ConvModule
from torch import nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet3d.ops import PointFPModule
from mmdet.models import HEADS
from abc import ABCMeta, abstractmethod
from mmseg.models.builder import build_loss


class Base3DDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.5.
        conv_cfg (dict|None): Config of conv layers.
            Default: dict(type='Conv1d').
        norm_cfg (dict|None): Config of norm layers.
            Default: dict(type='BN1d').
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
    """

    def __init__(self,
                 channels,
                 num_classes,
                 dropout_ratio=0.5,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 ignore_index=255,
                 init_cfg=None):
        super(Base3DDecodeHead, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index

        self.conv_seg = nn.Conv1d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def init_weights(self):
        """Initialize weights of classification layer."""
        super().init_weights()
        normal_init(self.conv_seg, mean=0, std=0.01)

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, pts_semantic_mask, train_cfg):
        """Forward function for training.

        Args:
            inputs (list[torch.Tensor]): List of multi-level point features.
            img_metas (list[dict]): Meta information of each sample.
            pts_semantic_mask (torch.Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, pts_semantic_mask)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level point features.
            img_metas (list[dict]): Meta information of each sample.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each points."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits \
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of \
                shape [B, N].
        """
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss


@HEADS.register_module()
class PointNet2Head(Base3DDecodeHead):
    r"""PointNet2 decoder head.

    Decoder head used in `PointNet++ <https://arxiv.org/abs/1706.02413>`_.
    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
    """

    def __init__(self,
                 fp_channels=((768, 256, 256), (384, 256, 256),
                              (320, 256, 128), (128, 128, 128, 128)),
                 **kwargs):
        super(PointNet2Head, self).__init__(**kwargs)

        self.num_fp = len(fp_channels)
        self.FP_modules = nn.ModuleList()
        for cur_fp_mlps in fp_channels:
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L40
        self.pre_seg_conv = ConvModule(
            fp_channels[-1][-1],
            self.channels,
            kernel_size=1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            list[torch.Tensor]: Coordinates of multiple levels of points.
            list[torch.Tensor]: Features of multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)

        return sa_xyz, sa_features

    def forward(self, feat_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """
        sa_xyz, sa_features = self._extract_input(feat_dict)

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L24
        sa_features[0] = None

        fp_feature = sa_features[-1]

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                            sa_features[-(i + 2)], fp_feature)
        output = self.pre_seg_conv(fp_feature)
        output = self.cls_seg(output)

        return output