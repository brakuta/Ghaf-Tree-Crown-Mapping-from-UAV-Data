"""DPN-98 backbone.

Dual Path Networks combine a residual path, which reuses features, with a
densely connected path, which explores new ones. DPN-98 is the 98-layer
variant: a 96-channel stem followed by four stages with bottleneck width
``k_r=160``, giving five feature maps for a feature-pyramid decoder.
"""

from mmengine.model import BaseModule
from mmseg.registry import MODELS

try:
    import timm
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        'timm is required for the DPN-98 backbone. Install it with '
        '`pip install timm==1.0.19`.') from exc

#: Feature widths timm reports for ``dpn98`` with ``features_only=True``.
#: A decode head or neck consuming this backbone must match them exactly.
DPN98_FEATURE_CHANNELS = [96, 336, 768, 1728, 2688]


@MODELS.register_module()
class DPN98Backbone(BaseModule):
    """DPN-98 exposed as a five-scale mmseg backbone.

    Feature maps are returned at strides 2, 4, 8, 16 and 32 with widths
    :data:`DPN98_FEATURE_CHANNELS`.

    Args:
        pretrained: load timm's ``dpn98.mx_in1k`` ImageNet weights while the
            backbone is built. Set to ``False`` when weights come from a
            checkpoint, so that evaluation needs no network access.
        out_indices: which of the five feature maps to return.
        init_cfg: mmseg's initialisation config, applied afterwards by
            ``init_weights()``. ImageNet initialisation for this backbone goes
            through timm rather than here, so it is normally left unset; it is
            accepted because every mmseg backbone takes it, and a runner may
            pass one.
    """

    def __init__(self, pretrained: bool = True, out_indices=(0, 1, 2, 3, 4),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.backbone = timm.create_model(
            'dpn98.mx_in1k',
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        self.feature_channels = list(self.backbone.feature_info.channels())

    def forward(self, x):
        return self.backbone(x)
