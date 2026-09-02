#!/usr/bin/env python
"""Assemble everything a recipient needs into one folder.

The public repository carries the code. Everything else -- the trained
weights, the ImageNet initialisation weights, the labelled tiles, a sample
orthomosaic -- is shared privately, and this puts all of it in one place, in a
layout the documented commands already expect::

    <output>/
      README.md            what this is and where to start
      MANIFEST.json        every part, its size, and what was verified
      code/                the repository, as git tracks it, at a named commit
      models/              the six trained checkpoints, one folder each
      init-weights/        ImageNet weights, so training needs no internet
      data/ghaf/           training, validation and test tiles
      samples/             orthomosaics to run inference on
      predictions/         per-tile predictions, if you made them

Every part is optional: pass the ones you have. What is missing is reported
rather than assumed, so a partial bundle is obvious at a glance instead of
discovered later by whoever received it.

``--code`` names a checkout of this repository, and only the files git tracks
are taken from it: a working copy accumulates local checkpoints, scratch
output and the remains of earlier layouts, none of which is the code. The
commit is recorded in the manifest and the README, so a recipient can name
exactly which version they have. Outside a git checkout the folder is copied
whole, minus caches.

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
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf import init_weights  # noqa: E402
from ghaf.release import iter_models  # noqa: E402
from ghaf.splits import directories  # noqa: E402

#: Files never worth copying into a handover, for a folder git does not
#: describe. A checkout is copied by what git tracks instead: see `add_code`.
EXCLUDED = ('.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'work_dirs',
            '*.pyc', '*.egg-info', '.ipynb_checkpoints')

#: Where the tracked code is public, so the bundle can name the same commit.
REPOSITORY_URL = 'https://github.com/brakuta/Ghaf-Tree-Crown-Mapping-from-UAV-Data'


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
    revision: Optional[Dict[str, object]] = None
    """For the code: the commit it was taken at, and whether the tree was clean."""

    @property
    def status(self) -> str:
        if self.source is None:
            return 'not given'
        if not self.ok:
            return 'PROBLEM'
        return 'copied'

    def as_dict(self) -> dict:
        entry = {
            'part': self.name,
            'source': str(self.source) if self.source else None,
            'files': self.files, 'bytes': self.bytes,
            'ok': self.ok, 'notes': self.notes,
        }
        if self.revision is not None:
            entry['revision'] = self.revision
        return entry


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


def _git(source: Path, *args: str) -> Optional[str]:
    """Run one git command in ``source``; ``None`` if git or the repo is absent."""
    try:
        done = subprocess.run(['git', '-C', str(source), *args],
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout


def tracked_files(source: Path) -> Optional[List[Path]]:
    """The files git tracks under ``source``, or ``None`` outside a checkout.

    Paths git lists but that are no longer on disk -- deleted and not yet
    committed -- are left out, since there is nothing to copy.
    """
    listing = _git(source, 'ls-files', '-z')
    if listing is None:
        return None
    paths = [source / name for name in listing.split('\0') if name]
    return [path for path in paths if path.is_file()]


def code_revision(source: Path) -> Optional[Dict[str, object]]:
    """Which commit a checkout is at, and whether anything is uncommitted."""
    commit = _git(source, 'rev-parse', 'HEAD')
    status = _git(source, 'status', '--porcelain')
    if commit is None or status is None:
        return None
    return {'commit': commit.strip(), 'uncommitted_changes': bool(status.strip())}


def add_code(name: str, source: Optional[Path], destination: Path,
             link: bool, dry_run: bool) -> Part:
    """Copy the repository as git describes it, not as the folder has grown.

    A working copy collects things that are not the code: checkpoints put
    beside the configs for convenience, an editable-install egg-info, the
    folders of a layout since abandoned, all ignored by git and invisible in
    a diff. Copying the folder would hand those on as though they were part
    of the repository. So inside a checkout only tracked files are taken, the
    commit is recorded, and the number of files left behind is reported --
    if any of them belong in the handover, commit them and run again.

    Outside a checkout -- no git, or a folder that is only a copy of one --
    the folder is copied whole minus caches, and the note says so.
    """
    if source is None or not source.exists():
        return add_part(name, source, destination, link, dry_run)

    files = tracked_files(source)
    if files is None:
        part = add_part(name, source, destination, link, dry_run)
        part.notes.append('not a git checkout: copied whole, minus caches')
        return part

    part = Part(name, source, destination)
    part.revision = code_revision(source)
    if not files:
        part.ok = False
        part.notes.append(f'{source} is a git checkout that tracks no files')
        return part

    part.files = len(files)
    part.bytes = sum(path.stat().st_size for path in files)
    if not dry_run:
        for path in files:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(path, target, link)

    if part.revision:
        short = str(part.revision['commit'])[:12]
        dirty = ' with uncommitted changes' if part.revision['uncommitted_changes'] else ''
        part.notes.append(f'commit {short}{dirty}')

    present = sum(1 for p in source.rglob('*')
                  if p.is_file() and '.git' not in p.relative_to(source).parts)
    untracked = present - part.files
    if untracked > 0:
        part.notes.append(
            f'{untracked:,} file(s) git does not track were left behind')
    return part


def _copy_file(source: Path, destination: Path, link: bool) -> None:
    """One file, linked where asked and possible, copied otherwise."""
    if link:
        try:
            _hardlink(source, destination)
            return
        except OSError:
            pass                    # different volumes, or no hard links here
    shutil.copy2(source, destination)


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


def describe_code(parts: List[Part]) -> str:
    """One sentence naming the commit in ``code/``, when it is known."""
    code = next((p for p in parts if p.name == 'code'), None)
    if code is None or not code.revision:
        return ''
    commit = str(code.revision['commit'])
    text = (f'`code/` is the repository at commit `{commit[:12]}`, which is '
            f'public at {REPOSITORY_URL}/tree/{commit}.')
    if code.revision['uncommitted_changes']:
        text += (' The checkout carried uncommitted changes when the bundle '
                 'was assembled, so it may differ from that commit.')
    return text


def write_readme(output: Path, parts: List[Part]) -> Path:
    """A first page for whoever opens the folder."""
    rows = '\n'.join(
        f'| `{p.name}/` | {p.status} | {p.files:,} files | '
        f'{p.bytes / 1e9:.2f} GB |' for p in parts)
    text = f"""# Ghaf tree-crown mapping — project handover

Assembled {date.today().isoformat()}. {describe_code(parts)}

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
| `code/` | The repository, exactly the files git tracks at the commit named above. Start with its README |
| `models/` | The six trained models. Each folder holds the weights, a self-contained config, and a metadata file with its digest and scores |
| `init-weights/` | ImageNet weights the backbones start from. Needed only for training or fine-tuning, and only so that neither needs internet access: pass the folder as `--init-weights` |
| `data/ghaf/` | The labelled training, validation and test tiles. Paired 1024 × 1024 PNGs, masks holding `0` for background and `1` for a crown. These splits alone -- other material from the working directory they were prepared in is not here. The test tiles carry world files, so their predictions open in GIS already placed |
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
        ('models', args.checkpoints, output / 'models'),
        ('init-weights', args.init_weights, output / 'init-weights'),
        ('samples', args.samples, output / 'samples'),
        ('predictions', args.predictions, output / 'predictions'),
    ]
    parts = [add_part(name, source, destination, args.link, args.dry_run)
             for name, source, destination in wanted]
    # Two parts are not folders to copy: the code is what git tracks, and the
    # dataset is a set of named splits. See add_code and add_dataset.
    parts.insert(0, add_code('code', args.code, output / 'code',
                             args.link, args.dry_run))
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
