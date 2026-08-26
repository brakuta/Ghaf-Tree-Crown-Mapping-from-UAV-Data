# model settings
_base_ = ['../_base_/default_runtime.py', '../_base_/datasets/ade20k.py','../_base_/schedules/schedule_40k.py']

norm_cfg = dict(type='SyncBN', requires_grad=True)
pretrained = "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_ma36.pth.tar"
# TODO: delete custom_imports after mmpretrain supports auto import
# please install mmpretrain >= 1.0.0rc7
# import mmpretrain.models to trigger register_module in mmpretrain
# custom_imports = dict(
#     imports=['mmpretrain.models'], allow_failed_imports=False)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(1024,1024),
    test_cfg=dict(size_divisor=32))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='fastvit_small',#'SwinTransformer'
       
        resume=pretrained,
    ),
    neck=dict(
        type='FPN',
        in_channels=[76, 152, 304, 608],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        # _delete_=True,
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
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
   