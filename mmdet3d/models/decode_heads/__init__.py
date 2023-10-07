# Copyright (c) OpenMMLab. All rights reserved.
from .td3d_instance_head import TD3DInstanceHead
from .td3d_instance_head_online import TD3DInstanceHead_Online
from .minkunet_sem_head import MinkUnetSemHead
from .pointnet2_head import PointNet2Head

__all__ = ['TD3DInstanceHead', 'TD3DInstanceHead_Online', 'MinkUnetSemHead',
            'PointNet2Head']
