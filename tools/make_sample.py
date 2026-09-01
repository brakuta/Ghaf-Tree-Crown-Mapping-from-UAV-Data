#!/usr/bin/env python
"""Cut a small, self-contained sample out of a large orthomosaic.

A survey mosaic is often billions of pixels: running the area-wide inference
over one takes hours and tens of gigabytes of scratch space, which is a poor
first thing to hand somebody. This clips a georeferenced window out of it, so
there is a sample that runs in a minute or two and proves the whole pipeline
works before the full mosaic is attempted.

    python tools/make_sample.py G:\\samples\\Kalba26.tif ^
        --output G:\\samples\\Kalba26_subset.tif --size 8192

The clip keeps the source CRS and pixel grid, so its predictions land in the
right place on a map and can be compared with predictions from the full
mosaic. By default it takes the three RGB bands (an orthomosaic's fourth band
is usually transparency, which the model does not read) and writes a tiled,
compressed GeoTIFF.

The window is centred unless ``--origin COL ROW`` names its top-left corner.
The reported share of valid pixels says whether the window landed on imagery
or on the transparent border: re-run with a different origin if it is low.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghaf.inference.large_image import _import  # noqa: E402

LOGGER = logging.getLogger('make_sample')

#: Written large enough to exercise the tiling, small enough to run quickly.
DEFAULT_SIZE = 8192


@dataclass(frozen=True)
class Clip:
    """The window to cut, in source pixel coordinates."""

    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def plan_clip(src_width: int, src_height: int, size: Sequence[int],
              origin: Optional[Sequence[int]] = None) -> Clip:
    """Decide which window to cut, clamped to the source raster.

    Args:
        src_width, src_height: size of the source raster in pixels.
        size: requested (width, height) of the clip.
        origin: (col, row) of the clip's top-left corner; centred if omitted.

    Returns:
        The window to read, never reaching outside the source.

    Raises:
        ValueError: if the requested size is not positive, or the origin lies
            outside the source raster.
    """
    want_w, want_h = int(size[0]), int(size[1])
    if want_w < 1 or want_h < 1:
        raise ValueError(f'clip size must be positive, got {want_w}x{want_h}')

    width = min(want_w, src_width)
    height = min(want_h, src_height)

    if origin is None:
        col_off = (src_width - width) // 2
        row_off = (src_height - height) // 2
    else:
        col_off, row_off = int(origin[0]), int(origin[1])
        if not 0 <= col_off < src_width or not 0 <= row_off < src_height:
            raise ValueError(
                f'origin ({col_off}, {row_off}) lies outside a '
                f'{src_width}x{src_height} raster')
        width = min(width, src_width - col_off)
        height = min(height, src_height - row_off)

    return Clip(col_off, row_off, width, height)


def _output_profile(src, clip: Clip, bands: Sequence[int], compress: str) -> dict:
    """Build the clip's profile from scratch rather than editing the source's.

    The source may carry an alpha band, overviews, or a block layout that does
    not suit a small file; naming every field keeps the output predictable.
    """
    _import('rasterio', 'rasterio')
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    transform = window_transform(
        Window(clip.col_off, clip.row_off, clip.width, clip.height),
        src.transform)
    profile = {
        'driver': 'GTiff',
        'width': clip.width,
        'height': clip.height,
        'count': len(bands),
        'dtype': src.dtypes[bands[0] - 1],
        'crs': src.crs,
        'transform': transform,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': compress,
    }
    if compress.lower() in ('deflate', 'lzw'):
        profile['predictor'] = 2
    return profile


def make_sample(source: Path, output: Path, size: Sequence[int] = (DEFAULT_SIZE,) * 2,
                origin: Optional[Sequence[int]] = None,
                bands: Sequence[int] = (1, 2, 3),
                compress: str = 'deflate') -> Clip:
    """Write a georeferenced clip of ``source`` to ``output``.

    Args:
        source: the orthomosaic to cut from.
        output: the GeoTIFF to write.
        size: requested (width, height) of the clip in pixels.
        origin: (col, row) of the top-left corner; centred if omitted.
        bands: 1-based band indices to keep, in output order.
        compress: GeoTIFF compression, e.g. ``deflate``, ``lzw``, ``none``.

    Returns:
        The window that was written.

    Raises:
        FileNotFoundError: if the source does not exist.
        ValueError: on an unusable band selection or window.
    """
    rasterio = _import('rasterio', 'rasterio')
    from rasterio.windows import Window

    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not bands:
        raise ValueError('keep at least one band')

    with rasterio.open(source) as src:
        if max(bands) > src.count:
            raise ValueError(
                f'raster has {src.count} band(s); requested {list(bands)}')

        clip = plan_clip(src.width, src.height, size, origin)
        window = Window(clip.col_off, clip.row_off,
                        clip.width, clip.height)
        LOGGER.info('%s: %d x %d px at (%d, %d) of %d x %d', source.name,
                    clip.width, clip.height, clip.col_off, clip.row_off,
                    src.width, src.height)

        data = src.read(indexes=list(bands), window=window)
        valid = src.read_masks(bands[0], window=window) > 0

        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, 'w',
                           **_output_profile(src, clip, bands, compress)) as dst:
            dst.write(data)

    share = float(valid.mean()) if valid.size else 0.0
    LOGGER.info('valid imagery: %.1f%% of the clip', 100 * share)
    if share < 0.5:
        LOGGER.warning(
            'over half the clip is outside the imagery; pass --origin COL ROW '
            'to cut somewhere else')
    LOGGER.info('wrote %s (%.1f MB)', output, output.stat().st_size / 1e6)
    return clip


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='orthomosaic to cut from')
    parser.add_argument('--output', type=Path, required=True,
                        help='GeoTIFF to write')
    parser.add_argument('--size', type=int, nargs='+', default=[DEFAULT_SIZE],
                        metavar='PX',
                        help='clip size: one number for a square, or width '
                             f'and height (default {DEFAULT_SIZE})')
    parser.add_argument('--origin', type=int, nargs=2, metavar=('COL', 'ROW'),
                        help='top-left corner of the clip (default: centred)')
    parser.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3],
                        metavar='N', help='1-based bands to keep (default 1 2 3)')
    parser.add_argument('--compress', default='deflate',
                        help='GeoTIFF compression (default deflate)')
    args = parser.parse_args(argv)

    if len(args.size) == 1:
        args.size = args.size * 2
    elif len(args.size) != 2:
        parser.error('--size takes one or two numbers')
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args(argv)
    make_sample(args.source, args.output, args.size, args.origin,
                tuple(args.bands), args.compress)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
