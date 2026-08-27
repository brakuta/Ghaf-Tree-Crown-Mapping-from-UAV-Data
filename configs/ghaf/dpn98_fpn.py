"""DPN-98 + FPN.

Dual Path Networks carry a residual path for feature reuse alongside a densely
connected path for new features. The five feature scales feed a feature
pyramid, which is why this configuration's neck takes five inputs where the
others take three or four.

===========  ==============================================
backbone     DPN-98, widths 96/336/768/1728/2688
neck         FPN, 5 -> 256 channels
decode head  FPNHead, 2 classes
optimiser    AdamW, lr 1e-4, weight decay 0.05
precision    fp32
input        1024 x 1024
===========  ==============================================
"""

_base_ = ['../_base_/ghaf.py']

data_preprocessor = {{_base_.data_preprocessor}}

model = dict(
    data_preprocessor=data_preprocessor,
    **{'backbone': {'type': 'DPN98Backbone'},
     'decode_head': {'align_corners': False,
                     'channels': 128,
                     'dropout_ratio': 0.1,
                     'feature_strides': [4, 8, 8, 16, 32],
                     'in_channels': [256, 256, 256, 256, 256],
                     'in_index': [0, 1, 2, 3, 4],
                     'loss_decode': {'loss_weight': 1.0,
                                     'type': 'CrossEntropyLoss',
                                     'use_sigmoid': False},
                     'norm_cfg': {'requires_grad': True, 'type': 'BN'},
                     'num_classes': 2,
                     'type': 'FPNHead'},
     'neck': {'in_channels': [96, 336, 768, 1728, 2688],
              'num_outs': 5,
              'out_channels': 256,
              'type': 'FPN'},
     'test_cfg': {'mode': 'whole'},
     'train_cfg': {},
     'type': 'EncoderDecoder'}
)

optim_wrapper = {'clip_grad': {'max_norm': 0.01, 'norm_type': 2},
     'optimizer': {'betas': (0.9, 0.999),
                   'eps': 1e-08,
                   'lr': 0.0001,
                   'type': 'AdamW',
                   'weight_decay': 0.05},
     'paramwise_cfg': {'custom_keys': {'backbone': {'decay_mult': 1.0,
                                                    'lr_mult': 0.1},
                                       'level_embed': {'decay_mult': 0.0,
                                                       'lr_mult': 1.0},
                                       'query_embed': {'decay_mult': 0.0,
                                                       'lr_mult': 1.0},
                                       'query_feat': {'decay_mult': 0.0,
                                                      'lr_mult': 1.0}},
                       'norm_decay_mult': 0.0},
     'type': 'OptimWrapper'}
