#!/usr/bin/env python
"""Write per-tile predictions for a dataset split.

Runs a trained model over every image in a split and saves the predicted mask
as a GeoTIFF beside it, carrying the source tile's CRS and geotransform so the
predictions drop straight into GIS.

    python tools/predict_split.py \\
        configs/ghaf/fastvit-ma36_mask2former.py \\
        checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \\
        --data-root data/ghaf --split testing --out-dir predictions/testing

Masks are single-band uint8, ``0`` background and ``1`` ghaf -- the same
encoding as the ground-truth masks, so the two can be differenced directly.
``--save-probability`` additionally writes the float32 P(ghaf) map per tile.

Tiles are already the model's input size, so no windowing is involved; for a
whole orthomosaic use ``ghaf.inference.large_image`` instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.inference.large_image import _foreground_probability, _import  # noqa: E402
from ghaf.inference.tiling import iter_batches  # noqa: E402

LOGGER = logging.getLogger('predict_split')

#: split name -> images directory, relative to the dataset root.
SPLIT_IMAGES = {
    'training': 'training/images',
    'validation': 'validation/images',
    'testing': 'testing/ghaf26/images',
}

SUFFIXES = ('.tif', '.tiff', '.TIF', '.TIFF')


def list_tiles(root: Path, split: str) -> List[Path]:
    """Every image tile in a split, in a stable order."""
    directory = root / SPLIT_IMAGES[split]
    if not directory.is_dir():
        raise FileNotFoundError(f'no such split directory: {directory}')
    tiles = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix in SUFFIXES)
    if not tiles:
        raise FileNotFoundError(f'no {"/".join(SUFFIXES)} tiles in {directory}')
    return tiles


def _read_tile(path: Path, bands: Sequence[int]):
    """Read one tile as BGR uint8, with the profile needed to write beside it."""
    rasterio = _import('rasterio', 'rasterio')
    with rasterio.open(path) as src:
        if max(bands) > src.count:
            raise ValueError(
                f'{path.name}: raster has {src.count} band(s), '
                f'requested {list(bands)}')
        data = src.read(indexes=list(bands))
        profile = src.profile.copy()
    if data.dtype != np.uint8:
        raise ValueError(
            f'{path.name}: expected an 8-bit tile, got {data.dtype}')
    rgb = np.transpose(data, (1, 2, 0))
    return rgb[:, :, ::-1].copy(), profile


def _write(path: Path, array: np.ndarray, profile: dict, dtype: str,
           nodata) -> None:
    rasterio = _import('rasterio', 'rasterio')
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {**profile, 'driver': 'GTiff', 'count': 1, 'dtype': dtype,
               'compress': 'deflate',
               'predictor': 3 if dtype == 'float32' else 2}
    if nodata is None:
        profile.pop('nodata', None)
    else:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(dtype), 1)


def predict_split(model, tiles: List[Path], out_dir: Path,
                  threshold: float = 0.5, bands: Sequence[int] = (1, 2, 3),
                  batch_size: int = 1, save_probability: bool = False,
                  progress: bool = True) -> dict:
    """Predict every tile and write the masks. Returns a summary dict."""
    inference_model = _import('mmseg.apis', 'mmsegmentation').inference_model

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f'threshold must be in [0, 1], got {threshold}')
    if batch_size < 1:
        raise ValueError(f'batch_size must be at least 1, got {batch_size}')

    out_dir = Path(out_dir)
    masks_dir = out_dir / 'masks'
    probs_dir = out_dir / 'probability' if save_probability else None

    batches = list(iter_batches(tiles, batch_size))
    if progress:
        try:
            from tqdm import tqdm
            batches = tqdm(batches, unit='batch', desc=out_dir.name)
        except ImportError:
            LOGGER.info('tqdm not installed; running without a progress bar')

    canopy = total = 0
    for batch in batches:
        payload = [_read_tile(path, bands) for path in batch]
        images = [image for image, _ in payload]

        results = inference_model(model, images)
        if not isinstance(results, (list, tuple)):
            results = [results]
        if len(results) != len(batch):
            raise RuntimeError(
                f'model returned {len(results)} result(s) for a batch of '
                f'{len(batch)}')

        for path, (image, profile), result in zip(batch, payload, results):
            probability = _foreground_probability(result, image.shape[0])
            mask = (probability >= threshold).astype(np.uint8)
            canopy += int(mask.sum())
            total += mask.size

            _write(masks_dir / f'{path.stem}.tif', mask, profile, 'uint8', None)
            if probs_dir is not None:
                _write(probs_dir / f'{path.stem}.tif', probability, profile,
                       'float32', None)

    return {
        'tiles': len(tiles),
        'threshold': threshold,
        'canopy_pixels': canopy,
        'total_pixels': total,
        'canopy_fraction': canopy / total if total else 0.0,
        'masks': str(masks_dir),
        'probability': str(probs_dir) if probs_dir else None,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('data/ghaf'))
    parser.add_argument('--split', default='testing', choices=sorted(SPLIT_IMAGES))
    parser.add_argument('--out-dir', type=Path,
                        help='default: predictions/<split>')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--bands', type=int, nargs=3, default=(1, 2, 3),
                        metavar=('R', 'G', 'B'))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-probability', action='store_true',
                        help='also write the float32 P(ghaf) map per tile')
    parser.add_argument('--limit', type=int,
                        help='stop after this many tiles (for a quick check)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = parse_args(argv)
    out_dir = args.out_dir or Path('predictions') / args.split

    import ghaf
    ghaf.register_all()
    from mmseg.apis import init_model

    tiles = list_tiles(args.data_root, args.split)
    if args.limit:
        tiles = tiles[:args.limit]
    LOGGER.info('%s: %d tile(s) -> %s', args.split, len(tiles), out_dir)

    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    summary = predict_split(
        model, tiles, out_dir, threshold=args.threshold,
        bands=tuple(args.bands), batch_size=args.batch_size,
        save_probability=args.save_probability, progress=not args.no_progress)

    summary.update(split=args.split, config=str(args.config),
                   checkpoint=str(args.checkpoint), generated=date.today().isoformat())
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n',
                                          encoding='utf-8')
    LOGGER.info('canopy: %.2f%% of pixels across %d tile(s)',
                100 * summary['canopy_fraction'], summary['tiles'])
    LOGGER.info('wrote %s', out_dir / 'summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
