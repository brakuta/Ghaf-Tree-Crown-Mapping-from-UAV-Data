"""Tests for the parts of the smoke test that do not need the full stack.

``tools/smoke_test.py`` builds every published model, which needs mmcv,
mmsegmentation, mmdet and mmpretrain. The logic that decides *how* to build
them needs none of that, and it is the part that has to be right on a machine
where the models cannot be built at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.smoke_test import _skip_imagenet_weights  # noqa: E402


class OpenMMLabStyle:
    """How mmseg's own backbones are declared: ``pretrained`` is a path."""

    def __init__(self, depth=50, pretrained=None, init_cfg=None):
        pass


class TimmStyle:
    """How DPN-98 is declared: ``pretrained`` is a flag."""

    def __init__(self, pretrained: bool = True, out_indices=(0, 1, 2, 3)):
        pass


class NeitherKnob:
    def __init__(self, arch='small'):
        pass


class CatchAll:
    def __init__(self, arch='small', **kwargs):
        pass


def test_a_path_style_backbone_is_told_none_not_false():
    """mmseg asserts ``pretrained`` is a str or None; ``False`` is rejected."""
    backbone = {'type': 'ResNetV1c', 'depth': 50}
    _skip_imagenet_weights(backbone, OpenMMLabStyle)
    assert backbone['pretrained'] is None
    assert backbone['init_cfg'] is None


def test_a_flag_style_backbone_is_told_false():
    backbone = {'type': 'DPN98Backbone'}
    _skip_imagenet_weights(backbone, TimmStyle)
    assert backbone['pretrained'] is False
    assert 'init_cfg' not in backbone, 'set an argument the class cannot take'


def test_a_backbone_with_neither_knob_is_left_alone():
    backbone = {'type': 'Whatever', 'arch': 'small'}
    _skip_imagenet_weights(backbone, NeitherKnob)
    assert backbone == {'type': 'Whatever', 'arch': 'small'}


def test_a_backbone_taking_kwargs_is_given_both():
    backbone = {'type': 'Whatever'}
    _skip_imagenet_weights(backbone, CatchAll)
    assert backbone['init_cfg'] is None
    assert backbone['pretrained'] is None
