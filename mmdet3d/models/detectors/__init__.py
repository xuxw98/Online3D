# Copyright (c) OpenMMLab. All rights reserved.
import imp
from .base import Base3DDetector
from .single_stage_sparseFF_onlinev3 import SingleStageSparse3DDetectorFF_OnlineV3
# from .single_stage_sparse_onlinev3 import SingleStageSparse3DDetector_OnlineV3
# from .td3d_instance_segmentor import TD3DInstanceSegmentor
from .td3d_instance_segmentorFF import TD3DInstanceSegmentorFF
from .td3d_instance_segmentorFF_online import TD3DInstanceSegmentorFF_Online
# from .single_stage_sparse import SingleStageSparse3DDetector
from .single_stage_sparseFF import SingleStageSparse3DDetectorFF
from .minkunet_semsegFF import MinkUnetSemsegFF, MinkUnetSemseg
from .minkunet_semsegFF_online import MinkUnetSemsegFF_Online


__all__ = [
    'Base3DDetector', 'SingleStageSparse3DDetectorFF_OnlineV3',
    'TD3DInstanceSegmentorFF', 'TD3DInstanceSegmentorFF_Online',
    'SingleStageSparse3DDetectorFF', 'MinkUnetSemsegFF', 'MinkUnetSemseg',
    'MinkUnetSemsegFF_Online'
]
