"""How the training command line folds into a config.

These cover the assembly only -- no runner is built and no GPU is touched --
because that is where a fine-tuning run is won or lost: the weights have to
reach `load_from`, and the dataset has to reach every split.
"""

import sys
from pathlib import Path

from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.train import apply_args, parse_args  # noqa: E402

CONFIG = (Path(__file__).parent.parent / 'configs' / 'ghaf' /
          'resnet-50_mask2former.py')


def assemble(*argv) -> Config:
    args = parse_args([str(CONFIG), *argv])
    return apply_args(Config.fromfile(str(CONFIG), import_custom_modules=False),
                      args)


def test_a_plain_run_starts_from_the_configs_own_weights():
    cfg = assemble()
    assert cfg.get('load_from') is None
    assert cfg.resume is False


def test_fine_tuning_weights_reach_load_from(tmp_path):
    weights = tmp_path / 'best_mIoU_iter_3500.pth'
    cfg = assemble('--load-from', str(weights))
    assert cfg.load_from == str(weights)
    assert cfg.resume is False, 'fine-tuning starts a fresh schedule'


def test_resuming_is_not_the_same_as_loading():
    """--resume continues a run; --load-from starts one from given weights."""
    assert assemble('--resume').resume is True
    assert assemble('--resume').get('load_from') is None


def test_both_together_are_allowed_for_an_interrupted_fine_tune(tmp_path):
    cfg = assemble('--load-from', str(tmp_path / 'w.pth'), '--resume')
    assert cfg.load_from == str(tmp_path / 'w.pth')
    assert cfg.resume is True


def test_the_dataset_reaches_every_split(tmp_path):
    cfg = assemble('--data-root', str(tmp_path))
    for loader in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        assert cfg[loader].dataset.data_root == str(tmp_path), loader


def test_the_work_directory_defaults_to_the_config_name():
    assert Path(assemble().work_dir).name == CONFIG.stem


def test_a_work_directory_can_be_named(tmp_path):
    assert assemble('--work-dir', str(tmp_path)).work_dir == str(tmp_path)


def test_amp_switches_the_optimiser_wrapper():
    cfg = assemble('--amp')
    assert cfg.optim_wrapper.type == 'AmpOptimWrapper'
    assert cfg.optim_wrapper.loss_scale == 'dynamic'


def test_without_amp_the_wrapper_is_left_alone():
    assert assemble().optim_wrapper.type == 'OptimWrapper'


def test_cfg_options_still_apply(tmp_path):
    cfg = assemble('--cfg-options', 'train_cfg.max_iters=4000')
    assert cfg.train_cfg.max_iters == 4000
