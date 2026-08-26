"""EfficientNet-B3 + FPN.

    ``arch='b3'``, not B0.

    This is the only model trained with **SGD at lr 0.01, wd 5e-4** -- a
    100x larger learning rate than the AdamW 1e-4 used elsewhere, and a
    different optimiser entirely. It is also the weakest result in the table.
    Preserved as-run; see ``docs/KNOWN-ISSUES.md``.

    Transcribed from the archived training run
    ``provenance/efficientefficient/20250309_140341``.
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
