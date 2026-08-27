#!/usr/bin/env python
"""Verify every published config builds and matches its released model.

Run this first on any new machine::

    python tools/smoke_test.py            # build + forward pass
    python tools/smoke_test.py --strict   # exit non-zero on any mismatch

For each config it constructs the model from scratch, runs one forward pass on
a random batch, and checks its tensor total against the released model's. The
comparison is over ``state_dict`` on both sides -- that is how
``ghaf.release`` measures a checkpoint, and it counts the normalisation
running statistics that ``parameters()`` leaves out. A mismatch means the code
and the weights have diverged -- investigate before training or evaluating.

Unlike ``tests/``, this needs the full stack: torch, mmcv, mmsegmentation,
mmdet (for Mask2Former) and mmpretrain. It needs no dataset and no GPU, though
a GPU makes it considerably faster.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.release import RELEASED_MODELS, get, iter_models  # noqa: E402


def _skip_imagenet_weights(backbone: dict, cls) -> None:
    """Build the backbone without fetching ImageNet weights.

    Every tensor is the same shape either way, so the totals below are
    unaffected -- and the check stays offline and quick.

    Which knob to turn depends on where the weights would come from: the
    OpenMMLab backbones take mmseg's ``init_cfg``, while DPN-98 loads through
    timm and takes ``pretrained``. Only the arguments a backbone actually
    accepts are set, so neither is forced on a class that has no use for it.
    """
    parameters = inspect.signature(cls.__init__).parameters
    catch_all = any(p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in parameters.values())
    for name, off in (('init_cfg', None), ('pretrained', False)):
        if name in parameters or catch_all:
            backbone[name] = off


def build(key: str, device: str):
    """Construct one published model with randomly initialised weights."""
    from mmengine.config import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    import ghaf
    ghaf.register_all()
    init_default_scope('mmseg')

    cfg = Config.fromfile(str(get(key).config_path))
    _skip_imagenet_weights(cfg.model.backbone,
                           MODELS.get(cfg.model.backbone['type']))
    net = MODELS.build(cfg.model)

    # The configs specify SyncBN, which is what multi-GPU training used. It
    # needs a process group, so a single-process check swaps in plain
    # BatchNorm -- identical in shape, and identical in output at eval time.
    return revert_sync_batchnorm(net).to(device).eval()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--size', type=int, default=1024,
                        help='forward-pass input size')
    parser.add_argument('--strict', action='store_true',
                        help='fail on any tensor-count mismatch')
    parser.add_argument('--only', nargs='*', metavar='KEY',
                        choices=sorted(RELEASED_MODELS), help='limit to these models')
    args = parser.parse_args(argv)

    import torch

    models = [m for m in iter_models() if not args.only or m.key in set(args.only)]
    failures = []

    print(f'{"model":30s} {"built":>12s} {"released":>12s} {"delta":>8s} '
          f'{"trainable":>12s}  backbone output')
    print('-' * 97)

    for model in models:
        try:
            net = build(model.key, args.device)
            # state_dict, not parameters(): running statistics are part of a
            # checkpoint, and the released totals count them.
            built = sum(t.numel() for t in net.state_dict().values())
            trainable = sum(t.numel() for t in net.parameters() if t.requires_grad)
            delta = built - model.parameters

            x = torch.randn(1, 3, args.size, args.size, device=args.device)
            with torch.no_grad():
                feats = net.backbone(x)
            shape = (f'{len(feats)} scales, widths '
                     f'{[f.shape[1] for f in feats]}'
                     if isinstance(feats, (list, tuple)) else 'single tensor')

            if delta and args.strict:
                failures.append(f'{model.key}: built {built:,}, '
                                f'released {model.parameters:,}')
            flag = '' if delta == 0 else '  <-- MISMATCH'
            print(f'{model.key:30s} {built:12,d} {model.parameters:12,d} '
                  f'{delta:+8d} {trainable:12,d}  {shape}{flag}')
        except Exception as exc:                                # noqa: BLE001
            failures.append(f'{model.key}: {type(exc).__name__}: {exc}')
            print(f'{model.key:30s} {"-":>12s} {model.parameters:12,d} {"-":>8s} '
                  f'{"-":>12s}  FAILED: {type(exc).__name__}: {exc}')

    print()
    if failures:
        print(f'{len(failures)} failure(s):')
        for failure in failures:
            print('  -', failure)
        return 1
    print(f'all {len(models)} model(s) built and ran a forward pass')
    return 0


if __name__ == '__main__':
    sys.exit(main())
