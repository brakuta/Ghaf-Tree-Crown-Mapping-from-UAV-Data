#!/usr/bin/env python
"""Map every image in a folder, one run over the lot.

Point this at a directory and it predicts each image in turn, writing a
georeferenced mask -- and, on request, the probability map and crown polygons
-- for each one, then a single ``summary.json`` covering the batch. It is the
step between the two existing ones: ``predict_split`` handles a dataset split
of same-sized tiles, ``ghaf.inference.large_image`` handles one orthomosaic,
and this handles the ordinary case of a folder holding neither -- a season's
clips, a set of survey plots, the frames from one flight.

    python tools/predict_folder.py \\
        configs/ghaf/fastvit-ma36_mask2former.py \\
        checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \\
        images/ --out-dir predictions/plots --polygons

Each image goes through the same windowed inference as a full mosaic, so the
images in a folder need not share a size and none of them needs to fit in
memory: a 900 px plot and a 40 000 px mosaic can sit side by side.

Outputs mirror the input's folder structure, so subdirectories stay apart and
two images with the same name in different folders cannot overwrite each
other::

    <out-dir>/
      masks/<subfolder>/<image>.tif          uint8, 0 background, 1 ghaf
      probability/<subfolder>/<image>.tif    float32 P(ghaf), with --save-probability
      polygons/<subfolder>/<image>.gpkg      crown polygons, with --polygons
      summary.json                           per image, and the totals

A batch is long, so a failure on one image is reported and the run carries on
to the next; the exit status is non-zero if any image failed, and
``summary.json`` names each failure. ``--skip-existing`` picks an interrupted
run back up where it stopped.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.environment import quiet_repeated_warnings, require_stack  # noqa: E402
from ghaf.inference.large_image import predict_large_image  # noqa: E402

LOGGER = logging.getLogger('predict_folder')

#: Image extensions, matched without regard to case. GDAL reads all of them;
#: the georeferencing each carries is another matter, and images that have
#: none produce ungeoreferenced outputs rather than an error.
SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.jp2', '.vrt')


def list_images(directory: Path, pattern: Optional[str] = None,
                recursive: bool = False,
                exclude: Optional[Path] = None) -> List[Path]:
    """Every image in a folder, in a stable order.

    Args:
        directory: the folder to read.
        pattern: an optional shell-style glob (``*_rgb.tif``) matched against
            the file name, without regard to case.
        recursive: descend into subdirectories.
        exclude: a directory whose contents are skipped. The output folder is
            passed here, so that a run whose outputs land inside the input
            folder does not, on the next run, predict its own predictions.

    Raises:
        FileNotFoundError: if the folder does not exist or holds no image.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'no such folder: {directory}')

    candidates = directory.rglob('*') if recursive else directory.iterdir()
    excluded = _resolve(exclude) if exclude is not None else None

    images = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        if pattern and not fnmatch.fnmatch(path.name.lower(), pattern.lower()):
            continue
        if excluded is not None and _is_within(path, excluded):
            continue
        images.append(path)

    if not images:
        where = f'{directory} (recursively)' if recursive else str(directory)
        what = f' matching {pattern}' if pattern else ''
        raise FileNotFoundError(f'no image{what} in {where}')
    return sorted(images)


def _resolve(path: Path) -> Path:
    """``Path.resolve`` that also works for a path which does not exist yet."""
    return Path(path).expanduser().absolute()


def _is_within(path: Path, directory: Path) -> bool:
    """Is ``path`` inside ``directory``?"""
    try:
        _resolve(path).relative_to(directory)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class Outputs:
    """Where one image's results are written."""

    mask: Path
    probability: Optional[Path] = None
    polygons: Optional[Path] = None

    def exist(self) -> bool:
        """Is every requested output already on disk?"""
        return all(p.exists() for p in (self.mask, self.probability,
                                        self.polygons) if p is not None)


def output_paths(image: Path, root: Path, out_dir: Path,
                 save_probability: bool = False,
                 polygons: bool = False,
                 polygon_suffix: str = '.gpkg') -> Outputs:
    """Where one image's outputs go, mirroring its place under ``root``.

    An image at ``root/plot-3/north.tif`` writes its mask to
    ``out_dir/masks/plot-3/north.tif``. Flattening the tree instead would let
    two images of the same name in different folders overwrite each other,
    silently and only in the second run.
    """
    try:
        relative = Path(image).relative_to(root).parent
    except ValueError:
        # An image named outside the folder that was listed: keep it, but put
        # it at the top rather than climbing out of the output directory.
        relative = Path('.')
    stem = Path(image).stem
    return Outputs(
        mask=out_dir / 'masks' / relative / f'{stem}.tif',
        probability=(out_dir / 'probability' / relative / f'{stem}.tif'
                     if save_probability else None),
        polygons=(out_dir / 'polygons' / relative / f'{stem}{polygon_suffix}'
                  if polygons else None),
    )


def predict_folder(model, images: Sequence[Path], root: Path, out_dir: Path,
                   threshold: float = 0.5, bands: Sequence[int] = (1, 2, 3),
                   batch_size: int = 1, tile: int = 1024, overlap: int = 512,
                   sigma: float = 0.4, min_area: float = 0.0,
                   save_probability: bool = False, polygons: bool = False,
                   polygon_suffix: str = '.gpkg',
                   scratch_dir: Optional[Path] = None,
                   skip_existing: bool = False, progress: bool = True,
                   predict: Callable = predict_large_image) -> dict:
    """Predict every image and write its outputs. Returns a summary dict.

    Args:
        model: a segmentor from :func:`mmseg.apis.init_model`.
        images: the images to predict, as returned by :func:`list_images`.
        root: the folder they were listed from; output paths mirror it.
        out_dir: where ``masks/``, ``probability/`` and ``polygons/`` go.
        predict: the per-image engine. Injectable so the loop -- skipping,
            failure handling and the totals -- can be tested without a model.

    Every other argument is passed straight through to
    :func:`ghaf.inference.large_image.predict_large_image`; see it for their
    meaning.

    A failure on one image does not stop the batch: it is logged, recorded in
    the returned summary under ``error``, and the run moves to the next image.
    """
    out_dir = Path(out_dir)
    rows: List[dict] = []
    canopy = valid = failed = skipped = 0

    for number, image in enumerate(images, start=1):
        outputs = output_paths(image, root, out_dir, save_probability,
                               polygons, polygon_suffix)
        name = _relative_name(image, root)

        if skip_existing and outputs.exist():
            LOGGER.info('[%d/%d] %s: already done, skipping', number,
                        len(images), name)
            rows.append({'image': name, 'skipped': True})
            skipped += 1
            continue

        LOGGER.info('[%d/%d] %s', number, len(images), name)
        for path in (outputs.mask, outputs.probability, outputs.polygons):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)

        try:
            summary = predict(
                model, Path(image),
                out_prob=outputs.probability,
                out_mask=outputs.mask,
                out_polygons=outputs.polygons,
                tile=tile, overlap=overlap, sigma=sigma, threshold=threshold,
                bands=tuple(bands), batch_size=batch_size, min_area=min_area,
                scratch_dir=scratch_dir, progress=progress)
        except Exception as exc:                     # noqa: BLE001 -- reported
            LOGGER.error('[%d/%d] %s failed: %s', number, len(images), name, exc)
            rows.append({'image': name, 'error': f'{type(exc).__name__}: {exc}'})
            failed += 1
            continue

        canopy += summary.canopy_pixels
        valid += summary.valid_pixels
        rows.append({
            'image': name,
            'width': summary.width,
            'height': summary.height,
            'windows': summary.windows,
            'canopy_pixels': summary.canopy_pixels,
            'valid_pixels': summary.valid_pixels,
            'canopy_fraction': summary.canopy_fraction,
            'outputs': [str(p) for p in summary.outputs],
        })

    return {
        'images': len(images),
        'predicted': len(images) - failed - skipped,
        'skipped': skipped,
        'failed': failed,
        'threshold': threshold,
        'canopy_pixels': canopy,
        'valid_pixels': valid,
        'canopy_fraction': canopy / valid if valid else 0.0,
        'out_dir': str(out_dir),
        'results': rows,
    }


def _relative_name(image: Path, root: Path) -> str:
    """The image's path as the user typed it: relative to the folder listed."""
    try:
        return Path(image).relative_to(root).as_posix()
    except ValueError:
        return Path(image).name


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('images', type=Path, help='folder of images to predict')
    parser.add_argument('--out-dir', type=Path, default=Path('predictions/folder'))
    parser.add_argument('--pattern',
                        help='only images whose name matches this glob, '
                             'e.g. "*_rgb.tif"')
    parser.add_argument('--recursive', action='store_true',
                        help='descend into subfolders')
    parser.add_argument('--save-probability', action='store_true',
                        help='also write the float32 P(ghaf) map per image')
    parser.add_argument('--polygons', action='store_true',
                        help='also write crown polygons per image')
    parser.add_argument('--polygon-suffix', default='.gpkg',
                        help='vector format for --polygons, by extension')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-area', type=float, default=0.0,
                        help='drop crown polygons smaller than this many m2')
    parser.add_argument('--bands', type=int, nargs=3, default=(1, 2, 3),
                        metavar=('R', 'G', 'B'))
    parser.add_argument('--tile', type=int, default=1024,
                        help='window size; match the training crop size')
    parser.add_argument('--overlap', type=int, default=512)
    parser.add_argument('--sigma', type=float, default=0.4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--scratch-dir', type=Path,
                        help='where the temporary accumulators go')
    parser.add_argument('--skip-existing', action='store_true',
                        help='leave images whose outputs are already written')
    parser.add_argument('--limit', type=int,
                        help='stop after this many images (for a quick check)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-progress', action='store_true')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = parse_args(argv)

    try:
        require_stack()
    except ModuleNotFoundError as exc:
        print(exc)
        return 1

    try:
        images = list_images(args.images, args.pattern, args.recursive,
                             exclude=args.out_dir)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    if args.limit:
        images = images[:args.limit]
    LOGGER.info('%d image(s) in %s -> %s', len(images), args.images, args.out_dir)

    import ghaf
    ghaf.register_all()
    quiet_repeated_warnings()
    from mmseg.apis import init_model

    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    summary = predict_folder(
        model, images, args.images, args.out_dir,
        threshold=args.threshold, bands=tuple(args.bands),
        batch_size=args.batch_size, tile=args.tile, overlap=args.overlap,
        sigma=args.sigma, min_area=args.min_area,
        save_probability=args.save_probability, polygons=args.polygons,
        polygon_suffix=args.polygon_suffix, scratch_dir=args.scratch_dir,
        skip_existing=args.skip_existing, progress=not args.no_progress)

    summary.update(config=str(args.config), checkpoint=str(args.checkpoint),
                   source=str(args.images), generated=date.today().isoformat())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = args.out_dir / 'summary.json'
    report.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    LOGGER.info('%d predicted, %d skipped, %d failed; canopy %.2f%% of valid px',
                summary['predicted'], summary['skipped'], summary['failed'],
                100 * summary['canopy_fraction'])
    LOGGER.info('wrote %s', report)
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
