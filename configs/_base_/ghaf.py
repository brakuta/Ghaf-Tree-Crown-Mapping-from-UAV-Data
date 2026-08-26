"""Shared dataset, pipeline, schedule and runtime for the Ghaf experiments.

Every value here is transcribed from the archived training configs under
``provenance/``; see ``docs/PROVENANCE.md`` for the run each one came from.

Two properties of the original setup are easy to miss and are preserved
deliberately:

* **Augmentation is horizontal/vertical flip only.** There is no scale jitter,
  no random crop and no photometric distortion. Tiles are fed at their native
  1024x1024.
* **Crop size is 1024, not 512.** Every archived working directory is named
  ``...-512x512``; none of them trained at 512.
"""

# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
dataset_type = 'GhafDataset'
#: Point this at the tile tree, or symlink it to ``data/ghaf``.
#: The original runs used the absolute path ``C:\ghaf``.
data_root = 'data/ghaf'

crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],     # ImageNet statistics
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='training/images', seg_map_path='training/masks'),
        pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='validation/images', seg_map_path='validation/masks'),
        pipeline=test_pipeline),
)

# The reported results are computed on the held-out test set; validation is
# used only for checkpoint selection.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='testing/ghaf26/images',
                         seg_map_path='testing/ghaf26/masks'),
        pipeline=test_pipeline),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=3500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=160000, by_epoch=False),
]

# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
default_scope = 'mmseg'

custom_imports = dict(
    imports=['ghaf.datasets', 'ghaf.models'], allow_failed_imports=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=3500,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'

load_from = None
resume = False

# The original runs set no seed, so exact numerical reproduction is not
# possible. Uncomment to make new runs deterministic; results will then differ
# slightly from the published table.
# randomness = dict(seed=0, deterministic=True)
