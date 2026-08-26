#!/usr/bin/env python
"""Verify every published config builds and matches its released model.

Run this first on any new machine::

    python tools/smoke_test.py            # build + forward pass
    python tools/smoke_test.py --strict   # exit non-zero on any mismatch

For each config it constructs the model from scratch, runs one forward pass on
a random batch, and checks the parameter count against the released model's.
A mismatch means the code and the weights have diverged -- investigate before
training or evaluating.

Unlike ``tests/``, this needs the full stack: torch, mmcv, mmsegmentation,
mmdet (for Mask2Former) and mmpretrain. It needs no dataset and no GPU, though
a GPU makes it considerably faster.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.release import RELEASED_MODELS, get, iter_models  # noqa: E402

#: Backbones are built without downloading ImageNet weights: the parameter
#: count is identical either way, and this keeps the check offline.
NO_PRETRAINED = dict(model=dict(backbone=dict(init_cfg=None)))


def build(key: str, device: str):
    """Construct one published model with randomly initialised weights."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    import ghaf
    ghaf.register_all()
    init_default_scope('mmseg')

    cfg = Config.fromfile(str(get(key).config_path))
    cfg.merge_from_dict(NO_PRETRAINED)
    return MODELS.build(cfg.model).to(device).eval()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--size', type=int, default=1024,
                        help='forward-pass input size')
    parser.add_argument('--strict', action='store_true',
                        help='fail on any parameter-count mismatch')
    parser.add_argument('--only', nargs='*', metavar='KEY',
                        choices=sorted(RELEASED_MODELS), help='limit to these models')
    args = parser.parse_args(argv)

    import torch

    models = [m for m in iter_models() if not args.only or m.key in set(args.only)]
    failures = []

    print(f'{"model":30s} {"built":>12s} {"released":>12s} {"delta":>8s}  backbone output')
    print('-' * 84)

    for model in models:
        try:
            net = build(model.key, args.device)
            built = sum(t.numel() for t in net.parameters())
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
                  f'{delta:+8d}  {shape}{flag}')
        except Exception as exc:                                # noqa: BLE001
            failures.append(f'{model.key}: {type(exc).__name__}: {exc}')
            print(f'{model.key:30s} {"-":>12s} {model.parameters:12,d} {"-":>8s}  '
                  f'FAILED: {type(exc).__name__}: {exc}')

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
