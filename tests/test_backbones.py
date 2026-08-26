"""Build the two custom backbones and check they are the architectures claimed.

These are the only two networks this project defines itself; the other four
come from mmsegmentation and mmpretrain. Their stage widths are quoted in the
README, the model zoo and every config's ``in_channels``, so a silent change
here would make the documentation wrong and stop the released checkpoints
loading.

Only ``mmseg.registry`` is stubbed -- enough to let the module-level
``@MODELS.register_module()`` decorators run -- so the backbones themselves are
built and executed for real. That keeps the check inside CI, which has torch
and timm but not the compiled mmcv extensions mmseg needs.
"""

import sys
import types

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('timm')


@pytest.fixture(scope='module', autouse=True)
def stub_registry():
    """Provide a no-op ``mmseg.registry.MODELS`` for the import to bind to."""
    if 'mmseg.registry' in sys.modules:
        yield                                   # the real thing is installed
        return

    class _Registry:
        def register_module(self, *args, **kwargs):
            return args[0] if args else (lambda cls: cls)

    registry = types.ModuleType('mmseg.registry')
    registry.MODELS = _Registry()
    package = types.ModuleType('mmseg')
    package.registry = registry
    added = {'mmseg': package, 'mmseg.registry': registry}
    sys.modules.update(added)
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


# --------------------------------------------------------------------------
# FastViT-MA36
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fastvit():
    from ghaf.models.fastvit import FastViTMA36
    return FastViTMA36(fork_feat=True).eval()


def test_fastvit_is_ma36_not_sa12(fastvit):
    """MA36 is [6, 6, 18, 6]; SA12 -- a different model -- is [2, 2, 6, 2]."""
    depths = [len(stage) for stage in fastvit.network if hasattr(stage, '__len__')]
    assert depths[:4] == [6, 6, 18, 6], f'stage depths are {depths[:4]}'


def test_fastvit_emits_the_documented_widths(fastvit):
    from ghaf.models import FASTVIT_MA36_EMBED_DIMS

    with torch.no_grad():
        features = fastvit(torch.randn(1, 3, 128, 128))
    assert [f.shape[1] for f in features] == FASTVIT_MA36_EMBED_DIMS
    assert FASTVIT_MA36_EMBED_DIMS == [76, 152, 304, 608]


def test_fastvit_emits_four_scales_at_strides_4_to_32(fastvit):
    size = 128
    with torch.no_grad():
        features = fastvit(torch.randn(1, 3, size, size))
    assert [size // f.shape[-1] for f in features] == [4, 8, 16, 32]


# --------------------------------------------------------------------------
# DPN-98
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def dpn():
    from ghaf.models.dpn import DPN98Backbone
    return DPN98Backbone(pretrained=False).eval()


def test_dpn_reports_the_documented_widths(dpn):
    from ghaf.models.dpn import DPN98_FEATURE_CHANNELS

    assert dpn.feature_channels == DPN98_FEATURE_CHANNELS
    assert DPN98_FEATURE_CHANNELS == [96, 336, 768, 1728, 2688]


def test_dpn_emits_five_scales_at_strides_2_to_32(dpn):
    size = 128
    with torch.no_grad():
        features = dpn(torch.randn(1, 3, size, size))
    assert [f.shape[1] for f in features] == [96, 336, 768, 1728, 2688]
    assert [size // f.shape[-1] for f in features] == [2, 4, 8, 16, 32]


def test_dpn_is_98_not_92(dpn):
    """The bottleneck widths separate the variants: k_r=160 against k_r=96."""
    widths = [
        module.conv.out_channels
        for name, module in dpn.backbone.named_modules()
        if name.endswith('c1x1_a') and hasattr(module, 'conv')
    ]
    first_per_stage = sorted(set(widths))
    assert first_per_stage == [160, 320, 640, 1280], (
        f'found {first_per_stage}; DPN-92 would give [96, 192, 384, 768]')


# --------------------------------------------------------------------------
# the configs must agree with what the backbones emit
# --------------------------------------------------------------------------

@pytest.mark.parametrize('config,expected', [
    ('fastvit-ma36_mask2former', [76, 152, 304, 608]),
    ('dpn98_fpn', [96, 336, 768, 1728, 2688]),
])
def test_configs_consume_exactly_what_the_backbones_produce(config, expected):
    """A mismatch builds a model whose released weights cannot load."""
    from mmengine.config import Config

    from ghaf.release import get

    cfg = Config.fromfile(str(get(config).config_path), import_custom_modules=False)
    consumer = cfg.model.get('neck') or cfg.model.decode_head
    assert list(consumer['in_channels']) == expected


# --------------------------------------------------------------------------
# pretrained-weight loading
# --------------------------------------------------------------------------

def _donor_checkpoint(tmp_path, value=0.25):
    """A checkpoint whose every parameter is a recognisable constant."""
    from ghaf.models.fastvit import FastViTMA36

    donor = FastViTMA36(fork_feat=True)
    with torch.no_grad():
        for parameter in donor.parameters():
            parameter.fill_(value)
    path = tmp_path / 'donor.pth'
    torch.save({'state_dict': donor.state_dict()}, path)
    return path


def test_init_cfg_actually_loads_weights(tmp_path):
    from ghaf.models.fastvit import FastViTMA36

    checkpoint = _donor_checkpoint(tmp_path)
    model = FastViTMA36(
        fork_feat=True,
        init_cfg={'type': 'Pretrained', 'checkpoint': str(checkpoint)})
    loaded = next(model.parameters())
    assert torch.allclose(loaded, torch.full_like(loaded, 0.25)), (
        'init_cfg did not load the checkpoint')


def test_an_unrecognised_keyword_does_not_load_weights(tmp_path):
    """Guards the failure mode: **kwargs swallows it and nothing complains.

    A config using such a keyword trains from scratch while appearing to start
    from ImageNet, which is why ``tests/test_configs.py`` forbids them.
    """
    from ghaf.models.fastvit import FastViTMA36

    checkpoint = _donor_checkpoint(tmp_path)
    model = FastViTMA36(fork_feat=True, resume=str(checkpoint))
    loaded = next(model.parameters())
    assert not torch.allclose(loaded, torch.full_like(loaded, 0.25)), (
        'unexpectedly loaded weights -- this test is no longer meaningful')
