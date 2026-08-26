"""Backbones used by the Ghaf crown-mapping models.

Importing this module registers them with mmseg's ``MODELS`` registry:

=====================  ==========================================  ===============
registry name          architecture                                feature widths
=====================  ==========================================  ===============
``FastViTMA36``        FastViT-MA36, hybrid RepMixer/attention      76/152/304/608
``DPN98Backbone``      DPN-98, dual residual + dense paths          96/336/768/1728/2688
=====================  ==========================================  ===============

The remaining backbones in this study -- ResNet-50, ConvNeXt-S,
PoolFormer-S36 and EfficientNet-B3 -- come from mmsegmentation and
mmpretrain and need no definition here.
"""

from .dpn import DPN98_FEATURE_CHANNELS, DPN98Backbone
from .fastvit import FastViTMA36

#: Stage widths of :class:`FastViTMA36`. A decode head consuming this
#: backbone must declare matching ``in_channels``.
FASTVIT_MA36_EMBED_DIMS = [76, 152, 304, 608]

__all__ = [
    'DPN98Backbone',
    'DPN98_FEATURE_CHANNELS',
    'FASTVIT_MA36_EMBED_DIMS',
    'FastViTMA36',
]
