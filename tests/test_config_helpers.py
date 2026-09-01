"""Tests for the config edits a command line has to make correctly.

Two traps, both quiet. mmengine accepts `--cfg-options data_root=...` and
reports nothing, while the dataloaders keep the path the config file was
parsed with. And switching off a backbone's ImageNet weights means a different
argument depending on where those weights would come from. Only mmengine is
needed here.
"""

from pathlib import Path

from mmengine.config import Config

from ghaf.config import DATALOADERS, set_data_root, skip_imagenet_weights

CONFIG = (Path(__file__).parent.parent / 'configs' / 'ghaf' /
          'resnet-50_mask2former.py')


def _load() -> Config:
    return Config.fromfile(str(CONFIG), import_custom_modules=False)


def test_every_split_moves_together():
    cfg = _load()
    changed = set_data_root(cfg, '/elsewhere/ghaf')

    assert changed == list(DATALOADERS)
    for name in DATALOADERS:
        assert cfg[name].dataset.data_root == '/elsewhere/ghaf', name
    assert cfg.data_root == '/elsewhere/ghaf'


def test_the_key_mmengine_would_have_set_is_not_enough():
    """The reason --data-root exists rather than a --cfg-options entry."""
    cfg = _load()
    original = cfg.test_dataloader.dataset.data_root

    cfg.merge_from_dict({'data_root': '/elsewhere/ghaf'})

    assert cfg.data_root == '/elsewhere/ghaf'
    assert cfg.test_dataloader.dataset.data_root == original


def test_a_path_object_is_accepted():
    cfg = _load()
    set_data_root(cfg, Path('/elsewhere/ghaf'))
    assert cfg.test_dataloader.dataset.data_root == str(Path('/elsewhere/ghaf'))


def test_a_wrapped_dataset_is_reached_through_its_wrapper():
    """RepeatDataset and friends hold the real dataset one level down."""
    cfg = Config(dict(train_dataloader=dict(
        dataset=dict(type='RepeatDataset', times=2,
                     dataset=dict(type='GhafDataset', data_root='data/ghaf')))))

    assert set_data_root(cfg, '/elsewhere/ghaf') == ['train_dataloader']
    inner = cfg.train_dataloader.dataset.dataset
    assert inner.data_root == '/elsewhere/ghaf'
    assert 'data_root' not in cfg.train_dataloader.dataset, 'wrapper was written to'


def test_a_config_without_a_given_split_is_left_alone():
    cfg = Config(dict(test_dataloader=dict(dataset=dict(data_root='data/ghaf'))))
    assert set_data_root(cfg, '/elsewhere') == ['test_dataloader']


def test_only_the_named_dataloaders_move():
    cfg = _load()
    assert set_data_root(cfg, '/only/test', ['test_dataloader']) == ['test_dataloader']
    assert cfg.test_dataloader.dataset.data_root == '/only/test'
    assert cfg.train_dataloader.dataset.data_root == 'data/ghaf'


# --------------------------------------------------------------------------
# switching off ImageNet initialisation
# --------------------------------------------------------------------------

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
    skip_imagenet_weights(backbone, OpenMMLabStyle)
    assert backbone['pretrained'] is None
    assert backbone['init_cfg'] is None


def test_a_flag_style_backbone_is_told_false():
    backbone = {'type': 'DPN98Backbone'}
    skip_imagenet_weights(backbone, TimmStyle)
    assert backbone['pretrained'] is False
    assert 'init_cfg' not in backbone, 'set an argument the class cannot take'


def test_a_backbone_with_neither_knob_is_left_alone():
    backbone = {'type': 'Whatever', 'arch': 'small'}
    skip_imagenet_weights(backbone, NeitherKnob)
    assert backbone == {'type': 'Whatever', 'arch': 'small'}


def test_a_backbone_taking_kwargs_is_given_both():
    backbone = {'type': 'Whatever'}
    skip_imagenet_weights(backbone, CatchAll)
    assert backbone['init_cfg'] is None
    assert backbone['pretrained'] is None
