#!/usr/bin/env python
"""Validate a Ghaf tile tree before training, evaluating or handing it on.

Checks that the splits exist, that every image has a mask and vice versa, that
paired files agree on size, and that masks contain only the two class indices.
Faults here surface during training as confusing loss curves or a silently
one-class problem, so it is worth a few minutes up front.

    python tools/check_dataset.py data/ghaf
    python tools/check_dataset.py data/ghaf --full      # every tile, not a sample
    python tools/check_dataset.py data/ghaf --json report.json

Exits 0 when the tree is usable, 1 when it is not. Needs rasterio; falls back
to Pillow if rasterio is unavailable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: split name -> (images dir, masks dir), relative to the dataset root.
SPLITS: Dict[str, Tuple[str, str]] = {
    'training': ('training/images', 'training/masks'),
    'validation': ('validation/images', 'validation/masks'),
    'testing': ('testing/ghaf26/images', 'testing/ghaf26/masks'),
}

#: Tile counts of the published dataset. A different count is not an error --
#: it is reported so a substitution is noticed rather than assumed.
PUBLISHED_COUNTS = {'training': 7005, 'validation': 869, 'testing': 767}

#: The only pixel values a mask may contain.
CLASS_VALUES = {0, 1}

SUFFIXES = ('.tif', '.tiff', '.TIF', '.TIFF')


@dataclass
class SplitReport:
    """What was found in one split."""

    name: str
    images: int = 0
    masks: int = 0
    paired: int = 0
    images_without_masks: List[str] = field(default_factory=list)
    masks_without_images: List[str] = field(default_factory=list)
    size_mismatches: List[str] = field(default_factory=list)
    bad_label_values: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    inspected: int = 0
    missing_directories: List[str] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return (len(self.images_without_masks) + len(self.masks_without_images)
                + len(self.size_mismatches) + len(self.bad_label_values)
                + len(self.unreadable) + len(self.missing_directories))

    def as_dict(self) -> dict:
        return {
            'split': self.name, 'images': self.images, 'masks': self.masks,
            'paired': self.paired, 'inspected': self.inspected,
            'problems': self.problems,
            'missing_directories': self.missing_directories,
            'images_without_masks': self.images_without_masks[:50],
            'masks_without_images': self.masks_without_images[:50],
            'size_mismatches': self.size_mismatches[:50],
            'bad_label_values': self.bad_label_values[:50],
            'unreadable': self.unreadable[:50],
        }


def _read_raster(path: Path):
    """Return ``(width, height, first_band)`` using whichever reader is present."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.width, src.height, src.read(1)
    except ImportError:
        pass
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as img:
            array = np.array(img)
        height, width = array.shape[:2]
        return width, height, array if array.ndim == 2 else array[:, :, 0]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            'reading tiles needs rasterio or Pillow: '
            '`pip install rasterio` (preferred) or `pip install pillow`') from exc


def stems(directory: Path) -> Dict[str, Path]:
    """Map file stem -> path for every raster in a directory."""
    return {p.stem: p for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix in SUFFIXES}


def check_split(root: Path, name: str, image_rel: str, mask_rel: str,
                sample: Optional[int], seed: int) -> SplitReport:
    """Check one split, inspecting either a sample of tiles or all of them."""
    report = SplitReport(name)
    image_dir, mask_dir = root / image_rel, root / mask_rel

    for directory in (image_dir, mask_dir):
        if not directory.is_dir():
            report.missing_directories.append(str(directory.relative_to(root)))
    if report.missing_directories:
        return report

    images, masks = stems(image_dir), stems(mask_dir)
    report.images, report.masks = len(images), len(masks)

    shared = sorted(set(images) & set(masks))
    report.paired = len(shared)
    report.images_without_masks = sorted(set(images) - set(masks))
    report.masks_without_images = sorted(set(masks) - set(images))

    chosen: Sequence[str] = shared
    if sample is not None and len(shared) > sample:
        chosen = random.Random(seed).sample(shared, sample)

    for stem in chosen:
        try:
            iw, ih, _ = _read_raster(images[stem])
            mw, mh, labels = _read_raster(masks[stem])
        except Exception as exc:                                # noqa: BLE001
            report.unreadable.append(f'{stem}: {type(exc).__name__}: {exc}')
            continue

        report.inspected += 1
        if (iw, ih) != (mw, mh):
            report.size_mismatches.append(
                f'{stem}: image {iw}x{ih}, mask {mw}x{mh}')

        values = set(map(int, np.unique(labels)))
        if not values <= CLASS_VALUES:
            report.bad_label_values.append(
                f'{stem}: found {sorted(values)}, expected {sorted(CLASS_VALUES)}')

    return report


def render(reports: List[SplitReport]) -> bool:
    """Print a human-readable summary. Returns True when the tree is usable."""
    print(f'{"split":12s} {"images":>8s} {"masks":>8s} {"paired":>8s} '
          f'{"checked":>8s} {"published":>10s}  status')
    print('-' * 72)

    healthy = True
    for report in reports:
        expected = PUBLISHED_COUNTS.get(report.name)
        note = ''
        if report.missing_directories:
            status, note = 'MISSING', ', '.join(report.missing_directories)
        elif report.problems:
            status = f'{report.problems} PROBLEM(S)'
        elif expected is not None and report.paired != expected:
            status = 'ok (count differs)'
        else:
            status = 'ok'
        healthy &= report.problems == 0
        print(f'{report.name:12s} {report.images:8d} {report.masks:8d} '
              f'{report.paired:8d} {report.inspected:8d} '
              f'{expected if expected else "-":>10}  {status} {note}')

    for report in reports:
        if not report.problems:
            continue
        print(f'\n{report.name}:')
        for label, items in (
                ('missing directories', report.missing_directories),
                ('images with no mask', report.images_without_masks),
                ('masks with no image', report.masks_without_images),
                ('image/mask size mismatch', report.size_mismatches),
                ('unexpected label values', report.bad_label_values),
                ('unreadable', report.unreadable)):
            if items:
                print(f'  {label} ({len(items)}):')
                for item in items[:10]:
                    print(f'    - {item}')
                if len(items) > 10:
                    print(f'    ... and {len(items) - 10} more')
    return healthy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('root', type=Path, help='dataset root, e.g. data/ghaf')
    parser.add_argument('--sample', type=int, default=200,
                        help='tiles to open per split; 0 checks pairing only')
    parser.add_argument('--full', action='store_true',
                        help='open every tile (slow, but exhaustive)')
    parser.add_argument('--seed', type=int, default=0,
                        help='sampling seed, so a run is repeatable')
    parser.add_argument('--json', type=Path, help='also write a JSON report here')
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        parser.error(f'not a directory: {args.root}')

    sample = None if args.full else (args.sample or 0)
    reports = [check_split(args.root, name, image_rel, mask_rel, sample, args.seed)
               for name, (image_rel, mask_rel) in SPLITS.items()]

    healthy = render(reports)

    total = sum(r.paired for r in reports)
    print(f'\n{total:,} paired tile(s) across {len(reports)} split(s)')
    print('dataset looks usable' if healthy else 'dataset has problems (see above)')

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            'root': str(args.root), 'healthy': healthy,
            'splits': [r.as_dict() for r in reports],
        }, indent=2) + '\n', encoding='utf-8')
        print(f'wrote {args.json}')

    return 0 if healthy else 1


if __name__ == '__main__':
    sys.exit(main())
