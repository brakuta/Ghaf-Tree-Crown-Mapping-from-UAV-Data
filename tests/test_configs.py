"""Guard the training recipe of the six published configs.

These need only mmengine (pure Python) -- not mmcv, torch, a GPU or the
dataset -- so they run in CI on every change. They check that each config still
declares the recipe it is supposed to, and that the six remain mutually
comparable. Building an actual model is ``tools/smoke_test.py``'s job.
"""

from pathlib import Path

import pytest
from mmengine.config import Config

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs' / 'ghaf'

#: backbone, decode head, optimiser, lr, weight decay, optim wrapper.
#: These are the settings each published model was trained with.
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
    """A new config must not slip in without a declared expected recipe."""
    on_disk = {p.stem for p in CONFIG_DIR.glob('*.py')}
    assert on_disk == set(EXPECTED), f'untested configs: {on_disk ^ set(EXPECTED)}'


@pytest.mark.parametrize('name', NAMES)
def test_recipe_is_unchanged(name):
    c, (bb, head, opt, lr, wd, wrapper) = load(name), EXPECTED[name]
    assert c.model.backbone.type == bb
    assert c.model.decode_head.type == head
    assert c.optim_wrapper.optimizer.type == opt
    assert c.optim_wrapper.optimizer.lr == pytest.approx(lr)
    assert c.optim_wrapper.optimizer.weight_decay == pytest.approx(wd)
    assert c.optim_wrapper.type == wrapper


@pytest.mark.parametrize('name', NAMES)
def test_binary_task_is_configured_consistently(name):
    """num_classes=2 with reduce_zero_label=False, or background is dropped."""
    c = load(name)
    assert c.model.decode_head.num_classes == 2
    for loader in (c.train_dataloader, c.val_dataloader, c.test_dataloader):
        assert loader.dataset.type == 'GhafDataset'
    for step in c.train_pipeline + c.test_pipeline:
        if step['type'] == 'LoadAnnotations':
            assert step['reduce_zero_label'] is False


@pytest.mark.parametrize('name', NAMES)
def test_input_size_is_1024(name):
    """Tiles are fed at native resolution; the preprocessor must agree."""
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
    """Configs must declare custom_imports, or the registry lookup fails."""
    c = load(name)
    assert 'ghaf.models' in c.custom_imports['imports']
    assert 'ghaf.datasets' in c.custom_imports['imports']
    assert c.custom_imports['allow_failed_imports'] is False


@pytest.mark.parametrize('name,channels', [
    ('fastvit-ma36_mask2former', [76, 152, 304, 608]),
    ('dpn98_fpn', [96, 336, 768, 1728, 2688]),
])
def test_custom_backbone_widths_match_their_consumers(name, channels):
    """A neck or head must consume exactly the widths its backbone emits.

    A mismatch here builds a model whose weights cannot load, so it is worth
    pinning the numbers rather than trusting them to stay in sync.
    """
    c = load(name)
    node = c.model.get('neck') or c.model.decode_head
    assert list(node['in_channels']) == channels
