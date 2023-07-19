# Copyright (c) OpenMMLab. All rights reserved.
import imp
from .base import Base3DDetector
from .mink_online_v3FF import MinkOnline3DDetector_V3FF
from .mink_online_v3 import MinkOnline3DDetector_V3
from .td3d_instance_segmentor import TD3DInstanceSegmentor
from .td3d_instance_segmentorFF import TD3DInstanceSegmentorFF
from .single_stage_sparse import SingleStageSparse3DDetector
from .single_stage_sparseFF import SingleStageSparse3DDetectorFF

__all__ = [
    'Base3DDetector', 'MinkOnline3DDetector_V3FF', 'MinkOnline3DDetector_V3',
    'TD3DInstanceSegmentor', 'TD3DInstanceSegmentorFF',
    'SingleStageSparse3DDetector', 'SingleStageSparse3DDetectorFF'
]
