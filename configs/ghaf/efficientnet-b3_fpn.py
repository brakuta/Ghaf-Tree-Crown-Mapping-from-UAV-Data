"""EfficientNet-B3 + FPN.

The lightest model in the study at 13.7 M parameters, included to establish
what a compound-scaled convolutional encoder achieves at a fraction of the
capacity.

===========  =========================================
backbone     EfficientNet-B3 (``arch='b3'``), 48/136/384
neck         FPN, 3 -> 256 channels
decode head  FPNHead, 2 classes
optimiser    SGD, lr 0.01, momentum 0.9, weight decay 5e-4
precision    fp32
input        1024 x 1024
===========  =========================================
"""

_base_ = ['../_base_/ghaf.py']

data_preprocessor = {{_base_.data_preprocessor}}

model = dict(
    data_preprocessor=data_preprocessor,
    **{'backbone': {'arch': 'b3',
                  'drop_path_rate': 0.2,
                  'frozen_stages': 0,
                  'init_cfg': {'checkpoint': 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth',
                               'prefix': 'backbone',
                               'type': 'Pretrained'},
                  'norm_cfg': {'eps': 0.001,
                               'momentum': 0.01,
                               'requires_grad': True,
                               'type': 'SyncBN'},
                  'norm_eval': False,
                  'out_indices': (3, 4, 5),
                  'type': 'mmpretrain.EfficientNet'},
     'decode_head': {'align_corners': False,
                     'channels': 128,
                     'dropout_ratio': 0.1,
                     'feature_strides': [4, 8, 16],
                     'in_channels': [256, 256, 256],
                     'in_index': [0, 1, 2],
                     'loss_decode': {'loss_weight': 1.0,
                                     'type': 'CrossEntropyLoss',
                                     'use_sigmoid': False},
                     'norm_cfg': {'requires_grad': True, 'type': 'BN'},
                     'num_classes': 2,
                     'type': 'FPNHead'},
     'neck': {'in_channels': [48, 136, 384],
              'num_outs': 4,
              'out_channels': 256,
              'type': 'FPN'},
     'test_cfg': {'mode': 'whole'},
     'train_cfg': {},
     'type': 'EncoderDecoder'}
)

optim_wrapper = {'clip_grad': None,
     'optimizer': {'lr': 0.01,
                   'momentum': 0.9,
                   'type': 'SGD',
                   'weight_decay': 0.0005},
     'type': 'OptimWrapper'}
