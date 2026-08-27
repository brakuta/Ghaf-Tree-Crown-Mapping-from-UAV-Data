#!/usr/bin/env python
"""Evaluate a checkpoint on the held-out test set.

    python tools/test.py configs/ghaf/fastvit-ma36_mask2former.py \
        checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth

Reports mIoU, mDice and mFscore over the held-out ``testing/ghaf26`` split --
the protocol all published results use.
"""

import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import ghaf
from ghaf.config import set_data_root


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


def main(argv=None) -> int:
    args = parse_args(argv)
    ghaf.register_all()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.data_root:
        set_data_root(cfg, args.data_root)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
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
