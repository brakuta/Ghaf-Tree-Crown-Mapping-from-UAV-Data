"""The six published configs must keep matching the archived training runs.

These tests need only mmengine (pure Python) -- not mmcv, torch or a GPU -- so
they guard the configs in CI. They check the recipe, not that a model can be
built; ``tools/smoke_test.py`` does that on a machine with the full stack.
"""

from pathlib import Path

import pytest
from mmengine.config import Config

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs' / 'ghaf'

#: backbone, decode head, optimiser, lr, weight decay, optim wrapper.
#: Transcribed from ``provenance/``; see ``docs/PROVENANCE.md``.
EXPECTED = {
    'fastvit-ma36_mask2former': ('FastViTMA36', 'Mask2FormerHead', 'AdamW', 1e-4, 0.05, 'OptimWrapper'),
    'resnet-50_mask2former':    ('ResNet', 'Mask2FormerHead', 'AdamW', 1e-4, 0.05, 'OptimWrapper'),
    'convnext-small_upernet':   ('mmpretrain.ConvNeXt', 'UPerHead', 'AdamW', 1e-4, 0.05, 'AmpOptimWrapper'),
    'dpn98_fpn':                ('DPN98Backbone', 'FPNHead', 'AdamW', 1e-4, 0.05, 'OptimWrapper'),
    'poolformer-s36_fpn':       ('mmpretrain.PoolFormer', 'FPNHead', 'AdamW', 2e-4, 1e-4, 'AmpOptimWrapper'),
    'efficientnet-b3_fpn':      ('mmpretrain.EfficientNet', 'FPNHead', 'SGD', 0.01, 5e-4, 'OptimWrapper'),
}

NAMES = sorted(EXPECTED)


def load(name: str) -> Config:
    return Config.fromfile(str(CONFIG_DIR / f'{name}.py'), import_custom_modules=False)


def test_every_config_is_covered():
    """A new config must not slip in without an expected recipe."""
    on_disk = {p.stem for p in CONFIG_DIR.glob('*.py')}
    assert on_disk == set(EXPECTED), f'untested configs: {on_disk ^ set(EXPECTED)}'


@pytest.mark.parametrize('name', NAMES)
def test_recipe_matches_the_archived_run(name):
    c, (bb, head, opt, lr, wd, wrapper) = load(name), EXPECTED[name]
    assert c.model.backbone.type == bb
    assert c.model.decode_head.type == head
    assert c.optim_wrapper.optimizer.type == opt
    assert c.optim_wrapper.optimizer.lr == pytest.approx(lr)
    assert c.optim_wrapper.optimizer.weight_decay == pytest.approx(wd)
    assert c.optim_wrapper.type == wrapper


@pytest.mark.parametrize('name', NAMES)
def test_binary_task_is_configured_consistently(name):
    """num_classes=2 with reduce_zero_label=False, or the background class dies."""
    c = load(name)
    assert c.model.decode_head.num_classes == 2
    for loader in (c.train_dataloader, c.val_dataloader, c.test_dataloader):
        assert loader.dataset.type == 'GhafDataset'
    for step in c.train_pipeline + c.test_pipeline:
        if step['type'] == 'LoadAnnotations':
            assert step['reduce_zero_label'] is False


@pytest.mark.parametrize('name', NAMES)
def test_crop_size_is_1024_despite_the_filenames(name):
    """Every archived directory says 512x512; none of them trained at 512."""
    c = load(name)
    assert c.crop_size == (1024, 1024)
    assert tuple(c.model.data_preprocessor['size']) == (1024, 1024)


@pytest.mark.parametrize('name', NAMES)
def test_all_models_share_one_evaluation_protocol(name):
    """The comparison is only meaningful if every model is scored identically."""
    c = load(name)
    assert c.test_dataloader.dataset.data_prefix == dict(
        img_path='testing/ghaf26/images', seg_map_path='testing/ghaf26/masks')
    assert c.val_dataloader.dataset.data_prefix == dict(
        img_path='validation/images', seg_map_path='validation/masks')
    assert c.test_evaluator.iou_metrics == ['mIoU', 'mDice', 'mFscore']


@pytest.mark.parametrize('name', NAMES)
def test_custom_backbones_are_importable_by_the_registry(name):
    """Configs must declare custom_imports or the registry lookup fails."""
    c = load(name)
    assert 'ghaf.models' in c.custom_imports['imports']
    assert 'ghaf.datasets' in c.custom_imports['imports']
    assert c.custom_imports['allow_failed_imports'] is False


@pytest.mark.parametrize('name,channels', [
    ('fastvit-ma36_mask2former', [76, 152, 304, 608]),
    ('dpn98_fpn', [96, 336, 768, 1728, 2688]),
])
def test_custom_backbone_widths_match_the_checkpoints(name, channels):
    """These widths identify the architecture; they are the audit's evidence.

    FastViT-SA12 would be [64, 128, 256, 512]; DPN-92 would not produce a
    2688-wide final stage.
    """
    c = load(name)
    node = c.model.get('neck') or c.model.decode_head
    assert list(node['in_channels']) == channels
