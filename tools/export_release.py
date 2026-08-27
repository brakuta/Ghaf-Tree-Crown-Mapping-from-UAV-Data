#!/usr/bin/env python
"""Assemble the shareable model bundle.

Given a folder of released checkpoints, this writes one self-contained
directory per model, laid out the way mmsegmentation lays out a working
directory, so each can be used on its own::

    <output>/
      README.md
      MODELS.json
      fastvit-ma36_mask2former/
        fastvit-ma36_mask2former.py     resolved config, no _base_ needed
        best_mIoU_iter_3500.pth         weights
        metadata.json                   digest, size, parameters, scores
      poolformer-s36_fpn/
        ...

The config written into each folder is **fully resolved**: mmengine's
inheritance is flattened, so the file carries the complete recipe and does not
depend on ``configs/_base_/`` being present. That is the same form
mmsegmentation dumps to ``<work_dir>/vis_data/config.py``, and it is what makes
a folder portable.

Every checkpoint is verified against its published SHA-256 before being
copied, and again after, so a bundle can never contain a silently corrupted
file.

Usage::

    python tools/export_release.py --checkpoints G:\\ghaf-handover\\checkpoints \\
                                   --output      G:\\ghaf-release

    python tools/export_release.py --checkpoints ... --output ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.release import ReleasedModel, iter_models, sha256_of  # noqa: E402

LOGGER = logging.getLogger('export_release')


def find_checkpoint(root: Path, model: ReleasedModel) -> Optional[Path]:
    """Locate a model's checkpoint under ``root``.

    Tries the tidy layout first (``<root>/<key>/<checkpoint>``), then the
    checkpoint name anywhere beneath ``root``, so a folder that has not been
    reorganised still works.
    """
    direct = root / model.key / model.checkpoint
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(model.checkpoint))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        LOGGER.warning('%s: %d files named %s under %s; using none',
                       model.key, len(matches), model.checkpoint, root)
    return None


def resolve_config(model: ReleasedModel) -> str:
    """Flatten a config's inheritance into a single self-contained file."""
    from mmengine.config import Config

    cfg = Config.fromfile(str(model.config_path), import_custom_modules=False)
    header = (
        f'# {model.backbone} + {model.decode_head} -- resolved configuration.\n'
        f'#\n'
        f'# Generated from configs/ghaf/{model.key}.py with its _base_\n'
        f'# inheritance flattened, so this file is self-contained.\n'
        f'#\n'
        f'# Test split: mIoU {model.miou:.2f}, F1 {model.fscore:.2f}, '
        f'{model.parameters:,} parameters.\n\n')
    return header + cfg.pretty_text


def export_model(model: ReleasedModel, source: Path, out_dir: Path,
                 dry_run: bool) -> dict:
    """Write one model's folder and return its ``MODELS.json`` entry."""
    LOGGER.info('%s', model.key)
    LOGGER.info('    verifying %s (%.0f MB)', source.name, source.stat().st_size / 1e6)
    model.verify(source)

    folder = out_dir / model.key
    checkpoint_out = folder / model.checkpoint
    config_out = folder / f'{model.key}.py'

    if dry_run:
        LOGGER.info('    would write %s', config_out)
        LOGGER.info('    would copy  %s', checkpoint_out)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        config_out.write_text(resolve_config(model), encoding='utf-8')
        LOGGER.info('    wrote %s', config_out.name)

        shutil.copy2(source, checkpoint_out)
        copied = sha256_of(checkpoint_out)
        if copied != model.sha256:
            checkpoint_out.unlink(missing_ok=True)
            raise RuntimeError(
                f'{model.key}: the copy of {model.checkpoint} does not match '
                f'its source digest; the destination may be failing. '
                f'Removed the bad copy.')
        LOGGER.info('    copied %s, digest verified', model.checkpoint)

        (folder / 'metadata.json').write_text(json.dumps({
            'key': model.key,
            'backbone': model.backbone,
            'neck': model.neck,
            'decode_head': model.decode_head,
            'config': config_out.name,
            'checkpoint': model.checkpoint,
            'sha256': model.sha256,
            'size_bytes': model.size_bytes,
            'parameters': model.parameters,
            'test_miou': model.miou,
            'test_fscore': model.fscore,
            'classes': ['background', 'ghaf'],
            'input_size': [1024, 1024],
        }, indent=2) + '\n', encoding='utf-8')

    return {
        'key': model.key, 'backbone': model.backbone,
        'decode_head': model.decode_head, 'neck': model.neck,
        'config': f'{model.key}/{model.key}.py',
        'checkpoint': f'{model.key}/{model.checkpoint}',
        'sha256': model.sha256, 'size_bytes': model.size_bytes,
        'parameters': model.parameters,
        'test_miou': model.miou, 'test_fscore': model.fscore,
    }


def write_index(out_dir: Path, entries: List[dict]) -> None:
    """Write ``MODELS.json`` and a README for whoever receives the bundle."""
    (out_dir / 'MODELS.json').write_text(json.dumps({
        'generated': date.today().isoformat(),
        'classes': ['background', 'ghaf'],
        'input_size': [1024, 1024],
        'test_split': 'testing/ghaf26 (767 tiles)',
        'models': entries,
    }, indent=2) + '\n', encoding='utf-8')

    rows = '\n'.join(
        f"| `{e['key']}` | {e['backbone']} | {e['decode_head']} | "
        f"{e['parameters']:,} | {e['test_miou']:.2f} | {e['test_fscore']:.2f} |"
        for e in entries)

    (out_dir / 'README.md').write_text(f"""# Ghaf crown-mapping models

Six trained semantic-segmentation models for delineating Ghaf
(*Prosopis cineraria*) tree crowns in UAV orthomosaics.

| Model | Backbone | Decode head | Params | mIoU | F1 |
|---|---|---|---:|---:|---:|
{rows}

Results are on the held-out test split (767 tiles). All six models share one
dataset, one augmentation pipeline, one schedule and one metric implementation.

## What each folder contains

```
<model>/
├── <model>.py       resolved configuration -- self-contained, no imports needed
├── <checkpoint>.pth trained weights
└── metadata.json    digest, size, parameter count, scores
```

The configuration is flattened, so it can be passed straight to
mmsegmentation without the rest of the repository.

## Using a model

```bash
python tools/test.py <model>/<model>.py <model>/<checkpoint>.pth

python -m ghaf.inference.large_image \\
    <model>/<model>.py <model>/<checkpoint>.pth mosaic.tif \\
    --out-mask crowns.tif --out-polygons crowns.gpkg
```

Both need the `ghaf` package installed; see the source repository for
installation and the two custom backbones.

## Verifying the download

Digests are in `MODELS.json` and each `metadata.json`.

```bash
sha256sum <model>/<checkpoint>.pth          # Linux, macOS
certutil -hashfile <checkpoint>.pth SHA256  # Windows
```

## Classes

Masks are single-band, one value per pixel: `0` background, `1` ghaf.
Models expect 8-bit RGB input at 1024 x 1024.
""", encoding='utf-8')
    LOGGER.info('wrote MODELS.json and README.md')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Checkpoints are verified against their published SHA-256 '
               'before and after copying.')
    parser.add_argument('--checkpoints', type=Path, required=True,
                        help='folder holding the released .pth files')
    parser.add_argument('--output', type=Path, required=True,
                        help='folder to build the bundle in')
    parser.add_argument('--only', nargs='*', metavar='KEY',
                        help='export only these models')
    parser.add_argument('--dry-run', action='store_true',
                        help='verify and report without writing anything')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if not args.checkpoints.is_dir():
        parser.error(f'--checkpoints is not a directory: {args.checkpoints}')

    models = [m for m in iter_models()
              if not args.only or m.key in set(args.only)]
    if not models:
        parser.error(f'no models matched --only {args.only}')

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

    entries, missing, failed = [], [], []
    for model in models:
        source = find_checkpoint(args.checkpoints, model)
        if source is None:
            LOGGER.error('%s: %s not found under %s',
                         model.key, model.checkpoint, args.checkpoints)
            missing.append(model.key)
            continue
        try:
            entries.append(export_model(model, source, args.output, args.dry_run))
        except (ValueError, RuntimeError, OSError) as exc:
            LOGGER.error('%s: %s', model.key, exc)
            failed.append(model.key)

    if entries and not args.dry_run:
        write_index(args.output, entries)

    total = sum(e['size_bytes'] for e in entries)
    LOGGER.info('')
    LOGGER.info('%d of %d model(s) exported, %.2f GB%s',
                len(entries), len(models), total / 1e9,
                ' (dry run, nothing written)' if args.dry_run else '')
    if missing:
        LOGGER.error('missing: %s', ', '.join(missing))
    if failed:
        LOGGER.error('failed verification: %s', ', '.join(failed))
    return 1 if (missing or failed) else 0


if __name__ == '__main__':
    sys.exit(main())
