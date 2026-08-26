#!/usr/bin/env python
"""Verify every published config builds and matches its checkpoint.

Run this first on any new machine::

    python tools/smoke_test.py            # build + forward pass, no data needed
    python tools/smoke_test.py --strict   # also require exact parameter counts

For each config it constructs the model from scratch, runs one forward pass on
a random 1024x1024 batch, and compares the parameter count against the count
measured directly from the published checkpoint's tensors.

Unlike ``tests/``, this needs the full stack: torch, mmcv, mmsegmentation,
mmdet (for Mask2Former) and mmpretrain. It needs no dataset and no GPU,
though a GPU makes it much faster.
"""

import argparse
import sys

#: Parameter counts summed over each published checkpoint's ``state_dict``.
#: See ``docs/PROVENANCE.md`` for how these were measured.
EXPECTED_PARAMS = {
    'fastvit-ma36_mask2former': 62_549_115,
    'resnet-50_mask2former':    44_056_504,
    'convnext-small_upernet':   81_776_049,
    'dpn98_fpn':                65_346_639,
    'poolformer-s36_fpn':       34_600_137,   # PoolFormer-S36 + FPN
    'efficientnet-b3_fpn':      13_734_524,
}

#: Checkpoints hold no ImageNet-download side effects, so backbones are built
#: without pretrained weights: the count must match regardless.
NO_PRETRAINED = dict(model=dict(backbone=dict(init_cfg=None)))


def build(name: str, device: str):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    import ghaf
    ghaf.register_all()
    init_default_scope('mmseg')

    cfg = Config.fromfile(f'configs/ghaf/{name}.py')
    cfg.merge_from_dict(NO_PRETRAINED)
    return MODELS.build(cfg.model).to(device).eval()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--device', default='cpu')
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--strict', action='store_true',
                   help='fail on any parameter-count mismatch')
    p.add_argument('--only', nargs='*', help='limit to these config names')
    args = p.parse_args(argv)

    import torch

    names = args.only or sorted(EXPECTED_PARAMS)
    failures = []
    print(f'{"config":30s} {"params":>12s} {"expected":>12s}  {"delta":>8s}  forward')
    print('-' * 78)

    for name in names:
        try:
            model = build(name, args.device)
            n = sum(t.numel() for t in model.parameters())
            expected = EXPECTED_PARAMS[name]
            delta = n - expected

            x = torch.randn(1, 3, args.size, args.size, device=args.device)
            with torch.no_grad():
                feats = model.backbone(x)
            shapes = 'OK (%d scales)' % len(feats) if isinstance(
                feats, (list, tuple)) else 'OK'

            flag = '' if delta == 0 else '  <-- MISMATCH'
            if delta and args.strict:
                failures.append(f'{name}: {n} != {expected}')
            print(f'{name:30s} {n:12,d} {expected:12,d}  {delta:+8d}  {shapes}{flag}')
        except Exception as exc:                       # noqa: BLE001
            failures.append(f'{name}: {type(exc).__name__}: {exc}')
            print(f'{name:30s} {"-":>12s} {EXPECTED_PARAMS[name]:12,d} '
                  f'{"-":>8s}  FAILED: {type(exc).__name__}: {exc}')

    print()
    if failures:
        print(f'{len(failures)} failure(s):')
        for f in failures:
            print('  -', f)
        return 1
    print(f'all {len(names)} config(s) built and ran a forward pass')
    return 0


if __name__ == '__main__':
    sys.exit(main())
