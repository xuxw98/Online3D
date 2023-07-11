import mmcv
import numpy as np
from random import choice

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import Compose, LoadAnnotations, LoadImageFromFile
from ..builder import PIPELINES


@PIPELINES.register_module()
class MultiImgsAug(object):
    def __init__(self, transforms, img_scales):
        if isinstance(img_scales,list):
            self.img_scale = choice(img_scales)
        else:
            self.img_scale = img_scales
        transforms[0]['img_scale'] = self.img_scale
        self.transforms = Compose(transforms)

    def __call__(self, results):
        imgs = []
        for i in range(len(results['imgs'])):
            _results = dict()
            _results['img'] = results['imgs'][i]
            _results = self.transforms(_results)
            imgs.append(_results['img'])
        for key in _results.keys():
            if key not in ['img', 'img_prefix', 'img_info']:
                results[key] = _results[key]
        results['imgs'] = imgs
        #print('Augument',end=" ")
        #print(type(results['imgs']))
        results['img_fields'] = []
        return results
