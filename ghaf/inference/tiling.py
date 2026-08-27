"""Sliding-window planning and overlap-add blending.

Pure NumPy: no mmseg, no rasterio, no torch. Keeping the geometry separate
from the inference script means it can be tested without a GPU or a model, and
``tests/test_tiling.py`` exercises every branch here.

The scheme is overlap-add with Gaussian weights. Each tile's prediction is
multiplied by a bell-shaped weight that decays toward the tile edge, summed
into a numerator accumulator, and the weights are summed into a denominator.
Dividing at the end gives a weighted mean per pixel, so seams between
neighbouring tiles are blended instead of stitched.
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Tuple

import numpy as np


class Window(NamedTuple):
    """Pixel window into a raster. ``col_off``/``row_off`` are top-left."""

    col_off: int
    row_off: int
    width: int
    height: int


def plan_windows(width: int, height: int, tile: int, overlap: int) -> List[Window]:
    """Tile a raster, guaranteeing full coverage including the right/bottom edges.

    Windows advance by ``tile - overlap``. The final window in each axis is
    clamped so that it ends exactly at the raster edge, which means the last
    step may be shorter than the others -- that is deliberate: it keeps every
    window the full ``tile`` size (so the network always sees its training
    resolution) while still covering the edge.

    Rasters smaller than one tile yield a single window covering the whole
    raster, which will be padded by the caller.

    Args:
        width: raster width in pixels.
        height: raster height in pixels.
        tile: window size in pixels (square).
        overlap: number of pixels shared between neighbouring windows.

    Returns:
        Windows in row-major order, each at most ``tile`` on a side.

    Raises:
        ValueError: if the arguments cannot produce a covering set.
    """
    if tile <= 0:
        raise ValueError(f'tile must be positive, got {tile}')
    if not 0 <= overlap < tile:
        raise ValueError(
            f'overlap must satisfy 0 <= overlap < tile, got {overlap} with tile={tile}')
    if width <= 0 or height <= 0:
        raise ValueError(f'raster must be non-empty, got {width}x{height}')

    step = tile - overlap

    def offsets(extent: int) -> List[int]:
        if extent <= tile:
            return [0]
        last = extent - tile
        out = list(range(0, last, step))
        out.append(last)          # clamp the final window to the edge
        return out

    return [
        Window(col, row, min(tile, width - col), min(tile, height - row))
        for row in offsets(height)
        for col in offsets(width)
    ]


def gaussian_weights(tile: int, sigma: float = 0.4, dtype=np.float32) -> np.ndarray:
    """Bell-shaped tile weights, peaking at the centre.

    ``sigma`` is expressed in units of half-tiles: the coordinate grid runs
    from -1 to +1 across the tile, so ``sigma=0.4`` puts the tile edge 2.5
    standard deviations out. Weights are strictly positive everywhere, so the
    blending denominator can never be zero.

    Args:
        tile: window size in pixels (square).
        sigma: standard deviation in half-tile units. Must be positive.
        dtype: output dtype.

    Returns:
        ``(tile, tile)`` array of weights, normalised to peak at 1.0.
    """
    if tile <= 0:
        raise ValueError(f'tile must be positive, got {tile}')
    if sigma <= 0:
        raise ValueError(f'sigma must be positive, got {sigma}')

    axis = np.linspace(-1.0, 1.0, tile, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    w = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (w / w.max()).astype(dtype)


class Accumulator:
    """Overlap-add accumulator for one raster.

    Keeps a weighted-sum numerator and a weight denominator, then divides.
    Both are ``float32``; for a raster of ``H*W`` pixels this needs ``8*H*W``
    bytes, so callers working on very large mosaics should pass memory-mapped
    arrays via ``numerator``/``denominator``.
    """

    def __init__(self, height: int, width: int, numerator=None, denominator=None):
        if height <= 0 or width <= 0:
            raise ValueError(f'invalid shape {height}x{width}')
        self.height, self.width = height, width
        self.numerator = (
            np.zeros((height, width), np.float32) if numerator is None else numerator)
        self.denominator = (
            np.zeros((height, width), np.float32) if denominator is None else denominator)
        for name, arr in (('numerator', self.numerator),
                          ('denominator', self.denominator)):
            if arr.shape != (height, width):
                raise ValueError(
                    f'{name} has shape {arr.shape}, expected {(height, width)}')

    def add(self, window: Window, values: np.ndarray, weights: np.ndarray) -> None:
        """Accumulate one tile.

        ``values`` and ``weights`` may be larger than the window (the caller
        works at full tile size even at the raster edge); the top-left
        ``window.height x window.width`` corner is used.
        """
        h, w = window.height, window.width
        if values.shape[:2] < (h, w) or weights.shape[:2] < (h, w):
            raise ValueError(
                f'tile arrays {values.shape} / {weights.shape} are smaller '
                f'than window {(h, w)}')
        v = values[:h, :w].astype(np.float32, copy=False)
        k = weights[:h, :w].astype(np.float32, copy=False)
        rs, cs = window.row_off, window.col_off
        self.numerator[rs:rs + h, cs:cs + w] += v * k
        self.denominator[rs:rs + h, cs:cs + w] += k

    def _check_covered(self, rows: slice) -> None:
        block = self.denominator[rows]
        if not np.all(block > 0):
            missed = int((block <= 0).sum())
            raise RuntimeError(
                f'{missed} pixel(s) in rows {rows.start}:{rows.stop} received '
                f'no tile coverage; the window plan is incomplete')

    def blocks(self, rows: int = 1024) -> Iterator[Tuple[slice, np.ndarray]]:
        """Yield the weighted mean a horizontal stripe at a time.

        Streaming keeps peak memory at ``rows * width * 4`` bytes regardless of
        raster size, which is the whole point of the memory-mapped
        accumulators -- materialising the full result would undo it.

        Args:
            rows: stripe height in pixels.

        Yields:
            ``(row_slice, values)`` where ``values`` is float32 and has shape
            ``(row_slice.stop - row_slice.start, width)``.

        Raises:
            ValueError: if ``rows`` is not positive.
            RuntimeError: if any pixel in a stripe received no weight, which
                means the window plan did not cover the raster -- a bug, so it
                is loud rather than silently filled.
        """
        if rows <= 0:
            raise ValueError(f'rows must be positive, got {rows}')
        for start in range(0, self.height, rows):
            stop = min(start + rows, self.height)
            band = slice(start, stop)
            self._check_covered(band)
            yield band, (self.numerator[band] / self.denominator[band]).astype(
                np.float32, copy=False)

    def result(self) -> np.ndarray:
        """The weighted mean over the whole raster, in memory.

        Convenient for tests and small rasters. For anything large, iterate
        :meth:`blocks` instead -- this allocates ``height * width * 4`` bytes.

        Raises:
            RuntimeError: if any pixel received no tile coverage.
        """
        out = np.empty((self.height, self.width), np.float32)
        for rows, values in self.blocks():
            out[rows] = values
        return out


def iter_batches(items: List[Window], size: int) -> Iterator[List[Window]]:
    """Yield ``items`` in chunks of at most ``size``.

    Used to group windows into inference batches: one forward pass over several
    tiles amortises both the GPU launch overhead and mmseg's per-call
    construction of the test pipeline.

    Raises:
        ValueError: if ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError(f'batch size must be positive, got {size}')
    for start in range(0, len(items), size):
        yield items[start:start + size]
