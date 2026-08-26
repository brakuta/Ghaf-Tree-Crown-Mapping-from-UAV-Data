"""DPN-98 backbone wrapper (the model reported as the ``dual_path`` baseline).

Provenance
----------
The file recovered from the workstation (kept verbatim as
``_original/coatnet_as_recovered.py``) registers this backbone under the name
``coatnet_small_timm``, but by the time it was recovered its only active line
built ``nextvit_base`` -- the CoAtNet and DPN lines had both been commented out
during later, unrelated work.

The checkpoint that produced the published result is unambiguously DPN-98:

* parameter names follow timm's DPN blocks (``features_conv1_1``,
  ``c1x1_w_s1``, ``c1x1_a``, ``c1x1_c``);
* the stem is 96 channels (``dpn98``; ``dpn92`` uses 64);
* the bottleneck widths are 160/320/640/1280, which follow from ``k_r=160``
  (``dpn92`` would give 96/192/384/768);
* the training config's FPN ``in_channels=[96, 336, 768, 1728, 2688]`` are
  exactly timm's ``dpn98`` ``features_only`` channels.

This module therefore builds ``dpn98`` and registers it under BOTH the original
name (so archived configs keep loading) and an honest one.
"""

from mmseg.registry import MODELS
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        'timm is required for the DPN-98 backbone. Install it with '
        '`pip install timm==1.0.19`.'
    ) from exc

#: Feature widths timm reports for ``dpn98`` with ``features_only=True``.
#: A config's FPN ``in_channels`` must match this exactly.
DPN98_FEATURE_CHANNELS = [96, 336, 768, 1728, 2688]


class DPN98(nn.Module):
    """timm ``dpn98`` exposed as a five-scale mmseg backbone.

    Args:
        pretrained: load timm's ``dpn98.mx_in1k`` ImageNet weights. Set to
            ``False`` when the weights are supplied by a checkpoint, so that
            evaluation does not need network access.
        out_indices: which of the five feature maps to return. Defaults to all.
    """

    def __init__(self, pretrained: bool = True, out_indices=(0, 1, 2, 3, 4)):
        super().__init__()
        self.backbone = timm.create_model(
            'dpn98.mx_in1k',
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        self.feature_channels = list(self.backbone.feature_info.channels())

    def forward(self, x):
        return self.backbone(x)


@MODELS.register_module()
class DPN98Backbone(DPN98):
    """Preferred, honestly-named registration."""


@MODELS.register_module()
class coatnet_small_timm(DPN98):
    """Backwards-compatible alias.

    Kept so the archived training configs under ``provenance/`` load unchanged.
    The name is a misnomer inherited from the original code: the model is
    DPN-98, and it was never CoAtNet.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('pretrained', False)
        super().__init__(**kwargs)
