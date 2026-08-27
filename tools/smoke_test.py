#!/usr/bin/env python
"""Verify every published config builds and matches its released model.

Run this first on any new machine::

    python tools/smoke_test.py            # build + forward pass
    python tools/smoke_test.py --strict   # exit non-zero on any mismatch

Given a folder of released checkpoints it also loads each one into the model
built from its config and runs a real prediction, which is the check to make
before handing the weights on::

    python tools/smoke_test.py --checkpoints G:\\ghaf-handover\\checkpoints

For each config it constructs the model from scratch, runs one forward pass on
a random batch, and checks its tensor total against the released model's. The
comparison is over ``state_dict`` on both sides -- that is how
``ghaf.release`` measures a checkpoint, and it counts the normalisation
running statistics that ``parameters()`` leaves out. A mismatch means the code
and the weights have diverged -- investigate before training or evaluating.

Unlike ``tests/``, this needs the full stack: torch, mmcv, mmsegmentation,
mmdet (for Mask2Former) and mmpretrain. It needs no dataset and no GPU, though
a GPU makes it considerably faster -- on CPU, allow a few minutes for the
whole set at the default input size.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.inference.large_image import GHAF_CLASS_INDEX  # noqa: E402
from ghaf.release import RELEASED_MODELS, get, iter_models  # noqa: E402


def _skip_imagenet_weights(backbone: dict, cls) -> None:
    """Build the backbone without fetching ImageNet weights.

    Every tensor is the same shape either way, so the totals below are
    unaffected -- and the check stays offline and quick.

    Which knob to turn depends on where the weights would come from, and the
    two conventions in play disagree about the type of ``pretrained``:

    * mmseg's backbones take ``init_cfg``, and their ``pretrained`` is a
      checkpoint path, so "no weights" is ``None`` -- ``False`` is rejected;
    * DPN-98 loads through timm, whose ``pretrained`` is a flag, so "no
      weights" is ``False``.

    The default declared by the class settles which it is, and only arguments
    the backbone actually accepts are set.
    """
    parameters = inspect.signature(cls.__init__).parameters
    catch_all = any(p.kind is inspect.Parameter.VAR_KEYWORD
                    for p in parameters.values())

    if 'init_cfg' in parameters or catch_all:
        backbone['init_cfg'] = None

    declared = parameters.get('pretrained')
    if declared is not None or catch_all:
        default = getattr(declared, 'default', None)
        backbone['pretrained'] = False if isinstance(default, bool) else None


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


def load_released_weights(net, path: Path, device: str):
    """Load a released checkpoint into a freshly built model.

    Returns:
        ``(missing, unexpected)``: the keys the model expected and the
        checkpoint did not carry, and the keys the checkpoint carried and the
        model has no place for. Both empty means the code and the weights
        describe exactly the same network.
    """
    from mmengine.runner import CheckpointLoader

    checkpoint = CheckpointLoader.load_checkpoint(str(path), map_location=device)
    state = checkpoint.get('state_dict', checkpoint)
    report = net.load_state_dict(state, strict=False)
    return list(report.missing_keys), list(report.unexpected_keys)


def predict_once(net, size: int, device: str) -> float:
    """Run mmseg's real prediction path once on a random image.

    This is the path ``ghaf.inference`` uses -- data preprocessor, backbone,
    neck, decode head, and the resize back to the input size -- so a model
    that gets here has been exercised end to end rather than built and
    counted.

    Returns:
        The share of pixels predicted as ghaf. On random noise the value
        carries no meaning beyond "the model produced a segmentation".
    """
    import torch
    from mmseg.structures import SegDataSample

    sample = SegDataSample()
    sample.set_metainfo(dict(img_shape=(size, size), ori_shape=(size, size),
                             pad_shape=(size, size), scale_factor=(1.0, 1.0)))
    image = torch.randint(0, 256, (3, size, size), dtype=torch.uint8,
                          device=device)
    with torch.no_grad():
        results = net.test_step(
            {'inputs': [image], 'data_samples': [sample]})
    prediction = results[0].pred_sem_seg.data
    return float((prediction == GHAF_CLASS_INDEX).float().mean())


def check_checkpoint(model, net, args, failures) -> str:
    """Verify one released checkpoint against the model built from its config.

    Three things in order, each a prerequisite for the next: the file is the
    released one (size and SHA-256), its tensors are exactly the tensors the
    model has -- none missing, none left over -- and the loaded model produces
    a segmentation.

    Returns:
        One formatted table row. Anything wrong is also appended to
        ``failures``, so the exit status reflects it. This never raises: a
        problem with one checkpoint should not stop the other five from being
        checked.
    """
    row = f'{model.key:30s} '
    path = model.find_checkpoint(args.checkpoints)
    if path is None:
        failures.append(
            f'{model.key}: no {model.checkpoint} under {args.checkpoints}')
        return row + f'{"absent":>8s} {"-":>26s}  not found'

    try:
        model.verify(path)
    except (OSError, ValueError) as exc:
        failures.append(f'{model.key}: {exc}')
        return row + f'{"BAD":>8s} {"-":>26s}  digest or size differs'

    try:
        missing, unexpected = load_released_weights(net, path, args.device)
        if missing or unexpected:
            example = (missing + unexpected)[0]
            failures.append(
                f'{model.key}: {len(missing)} missing and {len(unexpected)} '
                f'unexpected tensor(s), e.g. {example}')
            weights = f'{len(missing)} missing, {len(unexpected)} extra'
            return row + f'{"ok":>8s} {weights:>26s}  not run'

        weights = f'all {len(net.state_dict()):,} matched'
        share = predict_once(net, args.size, args.device)
    except Exception as exc:                                    # noqa: BLE001
        failures.append(f'{model.key}: {type(exc).__name__}: {exc}')
        return row + (f'{"ok":>8s} {"-":>26s}  '
                      f'FAILED: {type(exc).__name__}: {exc}')

    return row + f'{"ok":>8s} {weights:>26s}  {share:7.2%} ghaf'


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
    parser.add_argument('--checkpoints', type=Path, metavar='DIR',
                        help='folder of released checkpoints: verify each '
                             'digest, load the weights and predict once')
    args = parser.parse_args(argv)

    import torch

    models = [m for m in iter_models() if not args.only or m.key in set(args.only)]
    failures = []
    loaded = []

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

            if args.checkpoints:
                loaded.append(
                    check_checkpoint(model, net, args, failures))
        except Exception as exc:                                # noqa: BLE001
            failures.append(f'{model.key}: {type(exc).__name__}: {exc}')
            print(f'{model.key:30s} {"-":>12s} {model.parameters:12,d} {"-":>8s} '
                  f'{"-":>12s}  FAILED: {type(exc).__name__}: {exc}')

    if loaded:
        print()
        print(f'{"model":30s} {"digest":>8s} {"weights":>26s}  prediction')
        print('-' * 80)
        for line in loaded:
            print(line)

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
