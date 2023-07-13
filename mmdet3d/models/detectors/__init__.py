# Copyright (c) OpenMMLab. All rights reserved.
import imp
from .base import Base3DDetector
from .mink_online_v3FF import MinkOnline3DDetector_V3FF
from .mink_online_v3 import MinkOnline3DDetector_V3

__all__ = [
    'Base3DDetector', 'MinkOnline3DDetector_V3FF', 'MinkOnline3DDetector_V3'
]
