#!/usr/bin/env python
"""Assemble everything a recipient needs into one folder.

The public repository carries the code. Everything else -- the trained
weights, the ImageNet initialisation weights, the labelled tiles, a sample
orthomosaic -- is shared privately, and this puts all of it in one place, in a
layout the documented commands already expect::

    <output>/
      README.md            what this is and where to start
      MANIFEST.json        every part, its size, and what was verified
      code/                the repository
      models/              the six trained checkpoints, one folder each
      init-weights/        ImageNet weights, so training needs no internet
      data/ghaf/           training, validation and test tiles
      samples/             orthomosaics to run inference on
      predictions/         per-tile predictions, if you made them

Every part is optional: pass the ones you have. What is missing is reported
rather than assumed, so a partial bundle is obvious at a glance instead of
discovered later by whoever received it.

``--data`` names the root of a tile tree, and only the dataset's own splits
are taken from it -- whatever else the working directory has accumulated over
a project's life stays where it is, and the count left behind is reported.

    python tools/build_handover.py --output D:\\ghaf-project ^
        --code . ^
        --checkpoints D:\\handover\\checkpoints ^
        --init-weights D:\\handover\\init-weights ^
        --data D:\\handover\\data\\ghaf ^
        --samples D:\\handover\\samples

Add ``--dry-run`` to see what would be copied and how large it is. On one
volume, ``--link`` hard-links the big folders instead of copying them, which
is immediate and costs no extra space; copying the bundle to another drive
later turns the links back into ordinary files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf import init_weights  # noqa: E402
from ghaf.release import iter_models  # noqa: E402
from ghaf.splits import directories  # noqa: E402

#: Files never worth copying into a handover.
EXCLUDED = ('.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'work_dirs',
            '*.pyc', '.ipynb_checkpoints')


@dataclass
class Part:
    """One folder of the bundle, and what became of it."""

    name: str
    source: Optional[Path]
    destination: Path
    files: int = 0
    bytes: int = 0
    notes: List[str] = field(default_factory=list)
    ok: bool = True

    @property
    def status(self) -> str:
        if self.source is None:
            return 'not given'
        if not self.ok:
            return 'PROBLEM'
        return 'copied'

    def as_dict(self) -> dict:
        return {
            'part': self.name,
            'source': str(self.source) if self.source else None,
            'files': self.files, 'bytes': self.bytes,
            'ok': self.ok, 'notes': self.notes,
        }


def measure(root: Path) -> tuple:
    """``(file count, total bytes)`` beneath ``root``."""
    files = [p for p in root.rglob('*') if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def copy_tree(source: Path, destination: Path, link: bool, dry_run: bool) -> None:
    """Copy or hard-link a folder, skipping what a handover does not need."""
    if dry_run:
        return
    ignore = shutil.ignore_patterns(*EXCLUDED)
    if link:
        try:
            shutil.copytree(source, destination, ignore=ignore,
                            copy_function=_hardlink, dirs_exist_ok=True)
            return
        except OSError:
            # Different volumes, or a filesystem without hard links.
            shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination, ignore=ignore, dirs_exist_ok=True)


def _hardlink(source, destination) -> None:
    """Link rather than copy, replacing whatever is already there.

    ``os.link`` rather than ``Path.hardlink_to``: the latter arrived in
    Python 3.10 and this project supports 3.9. It also refuses an existing
    destination, which would make a second run of a part-built bundle fail,
    so the old name is removed first.
    """
    destination = Path(destination)
    if destination.exists():
        destination.unlink()
    os.link(source, destination)


def add_part(name: str, source: Optional[Path], destination: Path,
             link: bool, dry_run: bool) -> Part:
    """Copy one part of the bundle and report what happened."""
    part = Part(name, source, destination)
    if source is None:
        part.notes.append('not given on the command line')
        return part
    if not source.exists():
        part.ok = False
        part.notes.append(f'{source} does not exist')
        return part

    part.files, part.bytes = measure(source)
    if part.files == 0:
        part.ok = False
        part.notes.append(f'{source} holds no files')
        return part

    copy_tree(source, destination, link, dry_run)
    return part


def add_dataset(name: str, source: Optional[Path], destination: Path,
                link: bool, dry_run: bool) -> Part:
    """Copy the dataset splits, and only those.

    A tile tree normally lives inside a working directory that has collected
    other things over a project's life -- earlier surveys, scratch output,
    notes, an ``inference_errors`` folder. Copying the folder wholesale would
    hand all of it to the recipient, unexamined and unexplained, and swamp
    the tiles the documented commands actually read. So each split's image
    and mask directory is copied by name, and anything else beside them is
    left where it is and reported.
    """
    part = Part(name, source, destination)
    if source is None:
        part.notes.append('not given on the command line')
        return part
    if not source.exists():
        part.ok = False
        part.notes.append(f'{source} does not exist')
        return part

    for relative in directories():
        folder = source / relative
        if not folder.is_dir():
            part.ok = False
            part.notes.append(f'{relative} is missing')
            continue
        files, size = measure(folder)
        if files == 0:
            part.ok = False
            part.notes.append(f'{relative} holds no tiles')
            continue
        part.files += files
        part.bytes += size
        copy_tree(folder, destination / relative, link, dry_run)

    if part.ok:
        beside = measure(source)[0] - part.files
        if beside > 0:
            part.notes.append(
                f'{beside:,} file(s) beside the splits were left behind')
    return part


def verify_models(models_dir: Path) -> List[str]:
    """Check every released checkpoint that reached the bundle."""
    problems = []
    for model in iter_models():
        found = model.find_checkpoint(models_dir)
        if found is None:
            problems.append(f'{model.key}: {model.checkpoint} is not in the bundle')
            continue
        try:
            model.verify(found)
        except (OSError, ValueError) as exc:
            problems.append(f'{model.key}: {exc}')
    return problems


def write_readme(output: Path, parts: List[Part]) -> Path:
    """A first page for whoever opens the folder."""
    rows = '\n'.join(
        f'| `{p.name}/` | {p.status} | {p.files:,} files | '
        f'{p.bytes / 1e9:.2f} GB |' for p in parts)
    text = f"""# Ghaf tree-crown mapping — project handover

Assembled {date.today().isoformat()}.

Everything needed to reproduce, apply and extend the models from
**"Hybrid Vision–CNN Architecture for Mapping *Prosopis cineraria* from
Area-wide UAV-based Images."**

| Folder | State | Files | Size |
|---|---|---:|---:|
{rows}

## Start here

Read `code/docs/GETTING_STARTED.md`. It walks through installing the software
and running each task one command at a time.

The short version, once installed:

```
cd code
python tools\\smoke_test.py --checkpoints ..\\models
python tools\\check_dataset.py ..\\data\\ghaf
```

Those two confirm the models and the tiles arrived intact. Then, to map crowns
in a sample orthomosaic — start with the smallest file in `samples/`, which is
a clip cut from the full mosaic and runs in minutes rather than hours:

```
python -m ghaf.inference.large_image ..\\models\\fastvit-ma36_mask2former\\fastvit-ma36_mask2former.py ..\\models\\fastvit-ma36_mask2former\\best_mIoU_iter_3500.pth ..\\samples\\<mosaic>.tif --out-mask ..\\output\\crowns.tif --out-polygons ..\\output\\crowns.gpkg
```

## What is in each folder

| Folder | Contents |
|---|---|
| `code/` | The repository, also public on GitHub. Start with its README |
| `models/` | The six trained models. Each folder holds the weights, a self-contained config, and a metadata file with its digest and scores |
| `init-weights/` | ImageNet weights the backbones start from. Needed only for training or fine-tuning, and only so that neither needs internet access: pass the folder as `--init-weights` |
| `data/ghaf/` | The labelled tiles: 7 005 training, 869 validation, 767 test. Paired 1024 × 1024 PNGs, masks holding `0` for background and `1` for a crown. These splits alone -- other material from the working directory they were prepared in is not here. The test tiles carry world files, so their predictions open in GIS already placed |
| `samples/` | Orthomosaics for trying inference end to end. The small clip first, then the full survey mosaic |
| `predictions/` | Per-tile model output, if it was included |

`MANIFEST.json` lists every part with its size and the checks that were run.

## Verifying what you received

```
cd code
python tools\\smoke_test.py --checkpoints ..\\models
```

Each model should read `ok`, `all N matched`, and a prediction. That confirms
the file is the released one by its SHA-256, and that the code in `code/`
builds exactly the network the weights describe.
"""
    path = output / 'README.md'
    path.write_text(text, encoding='utf-8')
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', type=Path, required=True,
                        help='folder to assemble the bundle in')
    parser.add_argument('--code', type=Path, help='this repository')
    parser.add_argument('--checkpoints', type=Path, help='trained weights')
    parser.add_argument('--init-weights', type=Path,
                        help='ImageNet weights from tools/fetch_init_weights.py')
    parser.add_argument('--data', type=Path, help='the tile tree, i.e. .../ghaf')
    parser.add_argument('--samples', type=Path, help='orthomosaics for inference')
    parser.add_argument('--predictions', type=Path, help='per-tile predictions')
    parser.add_argument('--link', action='store_true',
                        help='hard-link instead of copying, where the volume allows')
    parser.add_argument('--dry-run', action='store_true',
                        help='report what would be copied, and write nothing')
    args = parser.parse_args(argv)

    output = args.output
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    wanted = [
        ('code', args.code, output / 'code'),
        ('models', args.checkpoints, output / 'models'),
        ('init-weights', args.init_weights, output / 'init-weights'),
        ('samples', args.samples, output / 'samples'),
        ('predictions', args.predictions, output / 'predictions'),
    ]
    parts = [add_part(name, source, destination, args.link, args.dry_run)
             for name, source, destination in wanted]
    # The dataset is not a folder to copy but a set of named splits: see
    # add_dataset for why the difference matters.
    parts.insert(3, add_dataset('data', args.data, output / 'data' / 'ghaf',
                                args.link, args.dry_run))

    if args.checkpoints and not args.dry_run:
        problems = verify_models(output / 'models')
        models = next(p for p in parts if p.name == 'models')
        models.notes.extend(problems)
        models.ok = models.ok and not problems

    if args.init_weights and not args.dry_run:
        collected = init_weights.stored_weights(output / 'init-weights')
        weights = next(p for p in parts if p.name == 'init-weights')
        weights.notes.append(f'{len(collected)} weight file(s)')
        if not collected:
            weights.ok = False
            weights.notes.append('no weight files found; was the fetch run?')

    print(f'{"part":14s} {"state":10s} {"files":>9s} {"size":>10s}  notes')
    print('-' * 78)
    for part in parts:
        print(f'{part.name:14s} {part.status:10s} {part.files:9,d} '
              f'{part.bytes / 1e9:9.2f}G  {"; ".join(part.notes)}')

    total = sum(p.bytes for p in parts)
    print(f'\n{total / 1e9:.2f} GB in total')

    if args.dry_run:
        print('dry run: nothing was written')
        return 0

    readme = write_readme(output, parts)
    manifest = output / 'MANIFEST.json'
    manifest.write_text(json.dumps({
        'assembled': date.today().isoformat(),
        'parts': [p.as_dict() for p in parts],
        'total_bytes': total,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {readme}\nwrote {manifest}')

    missing = [p.name for p in parts if p.source is None]
    if missing:
        print(f'\nnot included: {", ".join(missing)}')
    broken = [p for p in parts if not p.ok]
    if broken:
        print(f'\n{len(broken)} part(s) need attention:')
        for part in broken:
            print(f'  - {part.name}: {"; ".join(part.notes)}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
