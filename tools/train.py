#!/usr/bin/env python
"""Train or fine-tune a Ghaf segmentation model.

Training from ImageNet weights, which is how the published models were made::

    python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py \
        --data-root /path/to/ghaf

Fine-tuning from a released checkpoint on new labels -- usually far fewer
iterations and a smaller learning rate::

    python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py \
        --data-root /path/to/new-site \
        --load-from best_mIoU_iter_3500.pth \
        --cfg-options train_cfg.max_iters=4000 optim_wrapper.optimizer.lr=1e-5

``--load-from`` and ``--resume`` are different things: ``--load-from`` takes
the weights and starts a fresh run, which is what fine-tuning means, while
``--resume`` also restores the optimiser state and iteration count to continue
an interrupted run.

The configs declare ``custom_imports``, so mmengine registers this project's
dataset and backbones itself; ``register_all`` below is belt-and-braces for
configs that omit it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

import ghaf  # noqa: E402
from ghaf import init_weights
from ghaf.config import set_data_root
from ghaf.environment import require_stack


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('config', help='config file path')
    p.add_argument('--data-root',
                   help='dataset root to be trained on, overriding the config')
    p.add_argument('--work-dir', help='directory for logs and checkpoints')
    p.add_argument('--resume', action='store_true',
                   help='continue an interrupted run from the latest '
                        'checkpoint in --work-dir, optimiser state included')
    p.add_argument('--load-from', metavar='CHECKPOINT',
                   help='start from these weights on a fresh schedule, for '
                        'fine-tuning on new labels')
    p.add_argument('--init-weights', metavar='DIR',
                   help='folder of ImageNet initialisation weights collected '
                        'by tools/fetch_init_weights.py, so training needs no '
                        'internet access')
    p.add_argument('--amp', action='store_true',
                   help='enable mixed precision; a no-op for configs that '
                        'already set AmpOptimWrapper')
    p.add_argument('--cfg-options', nargs='+', action=DictAction,
                   help='override config entries, e.g. randomness.seed=0')
    p.add_argument('--launcher', default='none',
                   choices=['none', 'pytorch', 'slurm', 'mpi'])
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args(argv)


def apply_args(cfg: Config, args) -> Config:
    """Fold the command line into a parsed config.

    Separated from :func:`main` so the assembly can be checked without
    building a runner or touching a GPU.
    """
    cfg.launcher = args.launcher
    if args.data_root:
        set_data_root(cfg, args.data_root)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir or str(
        Path('work_dirs') / Path(args.config).stem)

    if args.amp and cfg.optim_wrapper.type == 'OptimWrapper':
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.load_from:
        cfg.load_from = str(args.load_from)
    cfg.resume = args.resume
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        require_stack()
    except ModuleNotFoundError as exc:
        print(exc)
        return 1


    # Before register_all, which imports timm: Hugging Face reads its cache
    # variables when the library is imported, not when it downloads.
    if args.init_weights:
        for name, value in init_weights.use(args.init_weights).items():
            print(f'{name}={value}')

    ghaf.register_all()

    cfg = apply_args(Config.fromfile(args.config), args)
    Runner.from_cfg(cfg).train()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
