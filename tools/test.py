#!/usr/bin/env python
"""Evaluate a checkpoint on the held-out test set.

    python tools/test.py configs/ghaf/fastvit-ma36_mask2former.py \
        checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \
        --data-root /path/to/ghaf

Reports mIoU, mDice and mFscore over the held-out ``testing/ghaf26`` split --
the protocol all published results use.

The backbone's ImageNet initialisation is switched off: the checkpoint
supplies every tensor and is loaded afterwards, so fetching those weights
would download a few hundred megabytes only to overwrite them. Evaluation
therefore needs no network access.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

import ghaf  # noqa: E402
from ghaf.config import set_data_root, skip_imagenet_weights  # noqa: E402
from ghaf.environment import require_stack  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('config', help='config file path')
    p.add_argument('checkpoint', help='checkpoint file')
    p.add_argument('--data-root',
                   help='dataset root to be evaluated, overriding the config')
    p.add_argument('--work-dir', help='directory for logs')
    p.add_argument('--show-dir', help='directory for prediction visualisations')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    p.add_argument('--launcher', default='none',
                   choices=['none', 'pytorch', 'slurm', 'mpi'])
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args(argv)


def _skip_backbone_download(cfg) -> None:
    """Build the backbone without its ImageNet weights.

    ``load_from`` replaces every tensor a moment later, so the download is
    wasted bandwidth at best; on a machine that cannot reach the host, it is
    the difference between an evaluation that runs and one that does not.
    """
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    init_default_scope(cfg.get('default_scope', 'mmseg'))
    backbone = cfg.model.backbone
    skip_imagenet_weights(backbone, MODELS.get(backbone['type']))


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        require_stack()
    except ModuleNotFoundError as exc:
        print(exc)
        return 1

    ghaf.register_all()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.data_root:
        set_data_root(cfg, args.data_root)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    _skip_backbone_download(cfg)
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.work_dir or str(
        Path('work_dirs') / Path(args.config).stem)

    if args.show_dir:
        hook = cfg.default_hooks.visualization
        hook.update(draw=True, show=False, img_shape=None)
        cfg.visualizer.setdefault('save_dir', args.show_dir)

    Runner.from_cfg(cfg).test()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
