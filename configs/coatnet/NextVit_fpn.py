_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/ade20k.py']

custom_imports = dict(imports='mmdet.models', allow_failed_imports=False)
norm_cfg = dict(type='BN', requires_grad=True)

crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
num_classes = 2
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # _delete_=True,
        type='coatnet_small_timm',#'SwinTransformer'
        # embed_dims=96,
        # depths=depths,
        # num_heads=[3, 6, 12, 24],
        # window_size=7,
        # mlp_ratio=4,
        # qkv_bias=True,
        # qk_scale=None,
        # drop_rate=0.,
        # attn_drop_rate=0.,
        # drop_path_rate=0.3,
        # patch_norm=True,
        # out_indices=(0, 1, 2, 3),
        # with_cp=False,
        # frozen_stages=-1,
        # resume=pretrained,
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 336, 768, 1728,2688],
        out_channels=256,
        num_outs=5),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256,256],
        in_index=[0, 1, 2, 3,4],
        feature_strides=[4, 8,8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset config
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(
#         type='RandomChoiceResize',
#         scales=[int(512 * x * 0.1) for x in range(5, 21)],
#         resize_type='ResizeShortestEdge',
#         max_size=2048),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
# train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100000, val_interval=50000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=5000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)
