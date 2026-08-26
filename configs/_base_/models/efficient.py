# model settings
_base_ = ['../default_runtime.py', '../datasets/ade20k.py','../schedules/schedule_40k.py']
norm_cfg = dict(type='SyncBN', requires_grad=True)
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth'  # noqa
# TODO: delete custom_imports after mmpretrain supports auto import
# please install mmpretrain >= 1.0.0rc7
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

image_size = (1024, 1024)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=image_size,
    test_cfg=dict(size_divisor=32))
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # _delete_=True,
        type='mmpretrain.EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
            type='FPN',
            in_channels=[48, 136,  384],
            out_channels=256,
            num_outs=4),
        decode_head=dict(
            type='FPNHead',
            in_channels=[256, 256, 256],
            in_index=[0, 1, 2],
            feature_strides=[4, 8, 16],
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

# custom_imports = dict(
#     imports=['mmpretrain.models'], allow_failed_imports=False)

# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255)
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     backbone=dict(type='EfficientNet', arch='b5'),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=4),
#     decode_head=dict(
#         type='FPNHead',
#         in_channels=[256, 256, 256, 256],
#         in_index=[0, 1, 2, 3],
#         feature_strides=[4, 8, 16, 32],
#         channels=128,
#         dropout_ratio=0.1,
#         num_classes=19,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))
