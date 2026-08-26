"""Custom backbones used by the Ghaf experiments.

Importing this package registers every backbone with mmseg's ``MODELS``
registry. Nothing else needs to be imported at the call site.

Two of the registered names are misnomers inherited from the original code.
They are kept because the archived configs under ``provenance/`` use them, and
each is aliased to an honestly-named class:

===========================  ==================  ==============================
registered name              real architecture   honest alias
===========================  ==================  ==============================
``fastvit_small``            FastViT-MA36        ``FastViTMA36``
``coatnet_small_timm``       DPN-98              ``DPN98Backbone``
===========================  ==================  ==============================
"""

from mmseg.registry import MODELS

from .dpn import DPN98_FEATURE_CHANNELS, DPN98Backbone, coatnet_small_timm
from .fastvit import fastvit_small

#: Widths of the four FastViT-MA36 stages. A config's ``in_channels`` must
#: match this exactly. (FastViT-SA12, which the manuscript names, would be
#: ``[64, 128, 256, 512]`` with ``layers=[2, 2, 6, 2]``.)
FASTVIT_MA36_EMBED_DIMS = [76, 152, 304, 608]


@MODELS.register_module()
class FastViTMA36(fastvit_small):
    """Honestly-named alias for ``fastvit_small``.

    The class named ``fastvit_small`` is built with ``layers=[6, 6, 18, 6]``
    and ``embed_dims=[76, 152, 304, 608]`` -- the FastViT-MA36 configuration.
    The same source file names that exact combination ``fastvit_ma36`` in a
    commented-out definition immediately above it.
    """


__all__ = [
    'DPN98Backbone',
    'DPN98_FEATURE_CHANNELS',
    'FASTVIT_MA36_EMBED_DIMS',
    'FastViTMA36',
    'coatnet_small_timm',
    'fastvit_small',
]
