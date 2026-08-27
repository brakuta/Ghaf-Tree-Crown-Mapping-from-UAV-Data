"""PoolFormer-S36 + FPN.

MetaFormer with pooling as the token mixer: it tests how much of the
transformer benefit comes from the architecture rather than attention itself.

===========  =========================================
backbone     PoolFormer-S36 (``arch='s36'``), 64/128/320/512
neck         FPN, 4 -> 256 channels
decode head  FPNHead, 2 classes
optimiser    AdamW, lr 2e-4, weight decay 1e-4
precision    mixed (``AmpOptimWrapper``)
input        1024 x 1024
===========  =========================================
"""

_base_ = ['../_base_/ghaf.py']

data_preprocessor = {{_base_.data_preprocessor}}

model = dict(
    data_preprocessor=data_preprocessor,
    **{'backbone': {'arch': 's36',
                  'down_pad': 1,
                  'down_patch_size': 3,
                  'down_stride': 2,
                  'drop_path_rate': 0.0,
                  'drop_rate': 0.0,
                  'frozen_stages': 0,
                  'in_pad': 2,
                  'in_patch_size': 7,
                  'in_stride': 4,
                  'init_cfg': {'checkpoint': 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth',
                               'prefix': 'backbone.',
                               'type': 'Pretrained'},
                  'out_indices': (0, 2, 4, 6),
                  'type': 'mmpretrain.PoolFormer'},
     'decode_head': {'align_corners': False,
                     'channels': 128,
                     'dropout_ratio': 0.1,
                     'feature_strides': [4, 8, 16, 32],
                     'in_channels': [256, 256, 256, 256],
                     'in_index': [0, 1, 2, 3],
                     'loss_decode': {'loss_weight': 1.0,
                                     'type': 'CrossEntropyLoss',
                                     'use_sigmoid': False},
                     'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                     'num_classes': 2,
                     'type': 'FPNHead'},
     'neck': {'in_channels': [64, 128, 320, 512],
              'num_outs': 4,
              'out_channels': 256,
              'type': 'FPN'},
     'test_cfg': {'mode': 'whole'},
     'train_cfg': {},
     'type': 'EncoderDecoder'}
)

optim_wrapper = {'optimizer': {'lr': 0.0002, 'type': 'AdamW', 'weight_decay': 0.0001},
     'type': 'AmpOptimWrapper'}
