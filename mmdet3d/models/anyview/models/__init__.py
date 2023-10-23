# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr
from .model_3detr_sepview import build_3detr_sepview
from .model_3detr_sepview2 import build_3detr_sepview as build_3detr_sepview2

MODEL_FUNCS = {
    "3detr": build_3detr,
    "3detr_sepview": build_3detr_sepview,
    "3detr_sepview2": build_3detr_sepview2,
}

def build_model(args, dataset_config):
    # build_model
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor