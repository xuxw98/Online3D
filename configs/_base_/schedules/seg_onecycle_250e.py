# optimizer
# This schedule is mainly used on ScanNet dataset in segmentation task
optimizer = dict(type='AdamW', lr=0.0008, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='OneCycle', max_lr=optimizer['lr'])
momentum_config = None

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=250)
