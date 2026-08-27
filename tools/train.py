#!/usr/bin/env python
"""Train a Ghaf segmentation model.

    python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py

The configs declare ``custom_imports``, so mmengine registers this project's
dataset and backbones itself; ``register_all`` below is belt-and-braces for
configs that omit it.
"""

import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import ghaf


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('config', help='config file path')
    p.add_argument('--work-dir', help='directory for logs and checkpoints')
    p.add_argument('--resume', action='store_true',
                   help='resume from the latest checkpoint in --work-dir')
    p.add_argument('--amp', action='store_true',
                   help='enable mixed precision; a no-op for configs that '
                        'already set AmpOptimWrapper')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='override config entries, e.g. data_root=/data/ghaf')
    p.add_argument('--launcher', default='none',
                   choices=['none', 'pytorch', 'slurm', 'mpi'])
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ghaf.register_all()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir or str(
        Path('work_dirs') / Path(args.config).stem)

    if args.amp and cfg.optim_wrapper.type == 'OptimWrapper':
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
    cfg.resume = args.resume

    Runner.from_cfg(cfg).train()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
