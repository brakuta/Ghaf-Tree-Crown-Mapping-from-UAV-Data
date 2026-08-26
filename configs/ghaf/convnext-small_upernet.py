"""ConvNeXt-S + UPerNet.

A modernised convolutional encoder with a pyramid-pooling decode head.

===========  =========================================
backbone     ConvNeXt-S (``arch='small'``), 96/192/384/768
decode head  UPerNet, 2 classes
optimiser    AdamW, lr 1e-4, weight decay 0.05
precision    mixed (``AmpOptimWrapper``)
input        1024 x 1024
===========  =========================================
"""

_base_ = ['../_base_/ghaf.py']

data_preprocessor = {{_base_.data_preprocessor}}

model = dict(
    data_preprocessor=data_preprocessor,
    **{'auxiliary_head': {'align_corners': False,
                        'channels': 256,
                        'concat_input': False,
                        'dropout_ratio': 0.1,
                        'in_channels': 384,
                        'in_index': 2,
                        'loss_decode': {'loss_weight': 0.4,
                                        'type': 'CrossEntropyLoss',
                                        'use_sigmoid': False},
                        'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                        'num_classes': 2,
                        'num_convs': 1,
                        'type': 'FCNHead'},
     'backbone': {'arch': 'small',
                  'drop_path_rate': 0.3,
                  'gap_before_final_norm': False,
                  'init_cfg': {'checkpoint': 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth',
                               'prefix': 'backbone.',
                               'type': 'Pretrained'},
                  'layer_scale_init_value': 1.0,
                  'out_indices': [0, 1, 2, 3],
                  'type': 'mmpretrain.ConvNeXt'},
     'decode_head': {'align_corners': False,
                     'channels': 512,
                     'dropout_ratio': 0.1,
                     'in_channels': [96, 192, 384, 768],
                     'in_index': [0, 1, 2, 3],
                     'loss_decode': {'loss_weight': 1.0,
                                     'type': 'CrossEntropyLoss',
                                     'use_sigmoid': False},
                     'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                     'num_classes': 2,
                     'pool_scales': (1, 2, 3, 6),
                     'type': 'UPerHead'},
     'pretrained': None,
     'test_cfg': {'crop_size': (1024, 1024),
                  'mode': 'slide',
                  'stride': (341, 341)},
     'train_cfg': {},
     'type': 'EncoderDecoder'}
)

optim_wrapper = {'constructor': 'LearningRateDecayOptimizerConstructor',
     'loss_scale': 'dynamic',
     'optimizer': {'betas': (0.9, 0.999),
                   'lr': 0.0001,
                   'type': 'AdamW',
                   'weight_decay': 0.05},
     'paramwise_cfg': {'decay_rate': 0.9,
                       'decay_type': 'stage_wise',
                       'num_layers': 12},
     'type': 'AmpOptimWrapper'}
