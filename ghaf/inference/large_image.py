"""Area-wide inference over a UAV orthomosaic.

Runs a trained segmentor across a georeferenced raster far larger than the
network's input size, using overlapping windows blended with Gaussian weights
(see :mod:`ghaf.inference.tiling`), and writes georeferenced outputs.

Outputs, all sharing the source CRS and geotransform:

``--out-prob``
    ``float32`` GeoTIFF of P(ghaf) in [0, 1].
``--out-mask``
    ``uint8`` GeoTIFF, 1 where P(ghaf) >= ``--threshold``, else 0.
``--out-polygons``
    optional vector layer of the delineated crowns.

Memory is bounded regardless of mosaic size. Windows are read one at a time,
the blending accumulators are memory-mapped to a temporary directory, and the
results are written back a stripe at a time, so peak resident memory is a few
stripes rather than a multiple of the raster. The cost is disk: roughly nine
bytes of scratch per source pixel while a run is in progress.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import shutil
import tempfile
import time
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .tiling import Accumulator, Window, gaussian_weights, iter_batches, plan_windows

LOGGER = logging.getLogger(__name__)

#: Index of the foreground class in :class:`ghaf.datasets.GhafDataset`.
GHAF_CLASS_INDEX = 1

#: Rows per write when streaming results back to disk.
_STRIPE_ROWS = 1024

#: Scratch bytes per source pixel: two float32 accumulators plus a uint8
#: validity plane. Reported up front so a run fails on a full disk with an
#: explanation rather than a partial file.
SCRATCH_BYTES_PER_PIXEL = 4 + 4 + 1


@dataclass(frozen=True)
class PredictionSummary:
    """What a run produced. Returned instead of the raster itself.

    Returning the array would force it into memory and undo the streaming, so
    callers that want pixels read the written GeoTIFF back.
    """

    width: int
    height: int
    windows: int
    canopy_pixels: int
    valid_pixels: int
    outputs: Tuple[Path, ...]

    @property
    def canopy_fraction(self) -> float:
        """Share of valid (non-nodata) pixels classified as ghaf."""
        return self.canopy_pixels / self.valid_pixels if self.valid_pixels else 0.0


def check_scratch_space(directory: Path, height: int, width: int,
                        margin: float = 1.1) -> None:
    """Fail before a long run rather than part-way through it.

    Args:
        directory: where the accumulators will be written.
        height, width: source raster dimensions.
        margin: headroom multiplier over the bare requirement.

    Raises:
        OSError: if the filesystem holding ``directory`` has too little free
            space, naming what is needed and what is available.
    """
    needed = int(height * width * SCRATCH_BYTES_PER_PIXEL * margin)
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise OSError(
            f'not enough scratch space in {directory}: need '
            f'{needed / 1e9:.1f} GB, {free / 1e9:.1f} GB free. Point '
            f'--scratch-dir at a larger filesystem.')


class _ScratchSpace:
    """A temporary directory holding the memory-mapped accumulators.

    Exists because a mapped file cannot be deleted on Windows while any
    mapping of it is open, so the maps have to be released before the
    directory is removed. Releasing them is normally just a matter of
    dropping the last reference; when a run fails, the traceback keeps the
    frame -- and therefore the arrays -- alive, so the mapping is closed
    explicitly as a second step.

    Cleanup never raises. A run that failed should surface its own error, not
    an error from tidying up after it.
    """

    def __init__(self, parent: Optional[Path] = None):
        self.path = Path(tempfile.mkdtemp(prefix='ghaf-infer-', dir=parent))
        self._maps: List[np.memmap] = []
        self._weak: List[weakref.ref] = []

    def array(self, name: str, dtype, shape: Tuple[int, int]) -> np.memmap:
        """Create a zeroed memory-mapped array inside the scratch directory."""
        arr = np.memmap(self.path / name, dtype=dtype, mode='w+', shape=shape)
        self._maps.append(arr)
        self._weak.append(weakref.ref(arr))
        return arr

    def close(self) -> None:
        """Flush the maps, release them, and remove the directory."""
        maps = self._maps
        self._maps = []
        while maps:                       # flush and drop one at a time, so
            arr = maps.pop()              # no reference outlives the loop
            with contextlib.suppress(AttributeError, ValueError):
                arr.flush()               # already closed: nothing to write
            del arr
        gc.collect()
        if self._remove():
            self._weak = []
            return

        # Something outside this object still holds a map. The run is over and
        # nothing reads the accumulators again, so close the mapping by hand.
        for ref in self._weak:
            arr = ref()
            handle = getattr(arr, '_mmap', None) if arr is not None else None
            if handle is not None and not handle.closed:
                with contextlib.suppress(BufferError, ValueError):
                    handle.close()
        self._weak = []
        if not self._remove():
            LOGGER.warning('could not remove the scratch directory %s; '
                           'delete it manually to reclaim the space', self.path)

    def _remove(self, attempts: int = 3) -> bool:
        """Remove the directory, retrying briefly. True if it is gone."""
        for attempt in range(attempts):
            try:
                shutil.rmtree(self.path)
                return True
            except FileNotFoundError:
                return True
            except OSError:
                if attempt + 1 < attempts:
                    time.sleep(0.05 * 2 ** attempt)
        return False

    def __enter__(self) -> _ScratchSpace:
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False


def _import(module: str, package: str):
    """Import an optional dependency, or explain how to install it."""
    try:
        return __import__(module, fromlist=['_'])
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            f'{module} is required for area-wide inference. '
            f'Install it with `pip install {package}`.') from exc


def _read_window(src, window: Window, bands: Sequence[int],
                 tile: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read one window as a zero-padded BGR tile plus its validity mask.

    mmseg's ``inference_model`` treats a raw array as BGR, so the RGB bands are
    reversed on the way out.

    Returns:
        ``(tile_bgr, valid)`` -- ``(tile, tile, 3)`` uint8 and ``(h, w)`` bool
        covering only the window's real extent.
    """
    from rasterio.windows import Window as RioWindow

    box = RioWindow(window.col_off, window.row_off, window.width, window.height)
    data = src.read(indexes=list(bands), window=box)          # (bands, h, w)
    if data.dtype != np.uint8:
        raise ValueError(
            f'expected an 8-bit raster, got dtype {data.dtype}. Convert the '
            f'mosaic first, e.g. `gdal_translate -ot Byte -scale in.tif out.tif`.')

    valid = src.read_masks(bands[0], window=box) > 0

    padded = np.zeros((len(bands), tile, tile), np.uint8)
    padded[:, :window.height, :window.width] = data
    rgb = np.transpose(padded, (1, 2, 0))
    return rgb[:, :, ::-1].copy(), valid


def _to_numpy(torch, tensor) -> np.ndarray:
    """Bring a tensor across to NumPy, naming the usual cause of failure."""
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError as exc:
        raise RuntimeError(
            f'could not convert a prediction to a NumPy array ({exc}). '
            f'PyTorch {torch.__version__} was built against NumPy 1.x and '
            f'cannot interoperate with NumPy 2; install a matching NumPy '
            f'with `python -m pip install "numpy<2"`.') from exc


def _foreground_probability(result, tile: int) -> np.ndarray:
    """Extract P(foreground) from an mmseg ``SegDataSample``."""
    torch = _import('torch', 'torch')

    logits = result.seg_logits.data                            # (classes, h, w)
    if logits.ndim != 3:
        raise RuntimeError(f'unexpected seg_logits shape {tuple(logits.shape)}')
    if logits.shape[0] <= GHAF_CLASS_INDEX:
        raise RuntimeError(
            f'model predicts {logits.shape[0]} class(es); this pipeline needs '
            f'at least {GHAF_CLASS_INDEX + 1} (background, ghaf)')

    prob = torch.softmax(logits.float(), dim=0)[GHAF_CLASS_INDEX]
    prob = _to_numpy(torch, prob).astype(np.float32)
    if prob.shape != (tile, tile):
        raise RuntimeError(
            f'model returned {prob.shape} for a {tile}x{tile} tile; the config '
            f'crop size and --tile must agree')
    return prob


def _progress(batches, enabled: bool, label: str, batch_size: int):
    if not enabled:
        return batches
    try:
        from tqdm import tqdm
        unit = 'batch' if batch_size > 1 else 'tile'
        return tqdm(batches, unit=unit, desc=label)
    except ImportError:
        LOGGER.info('tqdm not installed; running without a progress bar')
        return batches


def predict_large_image(
    model,
    src_path: Path,
    out_prob: Optional[Path] = None,
    out_mask: Optional[Path] = None,
    out_polygons: Optional[Path] = None,
    tile: int = 1024,
    overlap: int = 512,
    sigma: float = 0.4,
    threshold: float = 0.5,
    bands: Sequence[int] = (1, 2, 3),
    batch_size: int = 1,
    min_area: float = 0.0,
    scratch_dir: Optional[Path] = None,
    progress: bool = True,
) -> PredictionSummary:
    """Predict over a whole orthomosaic and write georeferenced outputs.

    Args:
        model: a segmentor from :func:`mmseg.apis.init_model`.
        src_path: georeferenced 8-bit orthomosaic.
        out_prob: where to write the float32 probability GeoTIFF.
        out_mask: where to write the uint8 binary mask GeoTIFF.
        out_polygons: where to write crown polygons; the OGR driver is inferred
            from the suffix. Requires geopandas, and reads the finished mask
            back into memory (one byte per pixel).
        tile: window size; must match the crop size the model was trained at.
        overlap: pixels shared between neighbouring windows.
        sigma: Gaussian blending width, in half-tile units.
        threshold: probability at or above which a pixel is called ghaf.
        bands: 1-based band indices to read as R, G, B.
        batch_size: tiles per forward pass. Larger batches use the GPU better
            and amortise mmseg's per-call construction of the test pipeline,
            at the cost of VRAM. Raise it until memory becomes the limit.
        min_area: drop crown polygons smaller than this many square metres.
            At ``0`` the polygon layer matches the mask raster exactly; a
            small value removes the single-pixel specks that any threshold
            leaves behind, at the cost of that correspondence.
        scratch_dir: where to place the memory-mapped accumulators. Defaults to
            the system temporary directory, which on Windows is usually on the
            system drive and may be far smaller than a run needs.
        progress: show a progress bar if tqdm is installed.

    Returns:
        A :class:`PredictionSummary`. The probability raster is not returned;
        read ``out_prob`` back if you need the pixels.

    Raises:
        ValueError: on unusable geometry, dtype, band selection or threshold.
        FileNotFoundError: if ``src_path`` does not exist.
        RuntimeError: if the model output does not match the requested tile.
    """
    rasterio = _import('rasterio', 'rasterio')
    inference_model = _import('mmseg.apis', 'mmsegmentation').inference_model

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f'threshold must be in [0, 1], got {threshold}')
    if len(bands) != 3:
        raise ValueError(f'expected exactly 3 band indices, got {list(bands)}')
    if batch_size < 1:
        raise ValueError(f'batch_size must be at least 1, got {batch_size}')
    src_path = Path(src_path)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    if not (out_prob or out_mask or out_polygons):
        raise ValueError('nothing to write: pass at least one output path')

    with rasterio.open(src_path) as src:
        height, width = src.height, src.width
        profile, crs, transform = src.profile.copy(), src.crs, src.transform
        if max(bands) > src.count:
            raise ValueError(
                f'raster has {src.count} band(s); requested {list(bands)}')
        if crs is None:
            LOGGER.warning('%s has no CRS; outputs will not be georeferenced',
                           src_path)

        windows = plan_windows(width, height, tile, overlap)
        weights = gaussian_weights(tile, sigma)
        scratch_gb = height * width * SCRATCH_BYTES_PER_PIXEL / 1e9
        LOGGER.info('%s: %d x %d px, %d window(s) of %d px (overlap %d), '
                    'batch %d, %.1f GB scratch', src_path.name, width, height,
                    len(windows), tile, overlap, batch_size, scratch_gb)

        if scratch_dir is not None:
            scratch_dir = Path(scratch_dir)
            scratch_dir.mkdir(parents=True, exist_ok=True)
        check_scratch_space(scratch_dir or Path(tempfile.gettempdir()),
                            height, width)

        acc = valid_plane = None
        with _ScratchSpace(scratch_dir) as scratch:
            shape = (height, width)
            try:
                acc = Accumulator(
                    height, width,
                    numerator=scratch.array('num.dat', np.float32, shape),
                    denominator=scratch.array('den.dat', np.float32, shape))
                valid_plane = scratch.array('valid.dat', np.uint8, shape)

                batches = list(iter_batches(windows, batch_size))
                for batch in _progress(batches, progress, src_path.stem,
                                       batch_size):
                    patches, valids = zip(
                        *(_read_window(src, w, bands, tile) for w in batch))

                    # A list in, a list out: mmseg batches when given a sequence.
                    results = inference_model(model, list(patches))
                    if not isinstance(results, (list, tuple)):
                        results = [results]
                    if len(results) != len(batch):
                        raise RuntimeError(
                            f'model returned {len(results)} result(s) for a '
                            f'batch of {len(batch)}')

                    for window, result, window_valid in zip(batch, results,
                                                            valids):
                        acc.add(window, _foreground_probability(result, tile),
                                weights)
                        rows = slice(window.row_off,
                                     window.row_off + window.height)
                        cols = slice(window.col_off,
                                     window.col_off + window.width)
                        np.maximum(valid_plane[rows, cols],
                                   window_valid.astype(np.uint8),
                                   out=valid_plane[rows, cols])

                summary = _write_outputs(
                    acc, valid_plane, threshold, profile, crs, transform,
                    out_prob, out_mask, len(windows))
            finally:
                # Drop the accumulators before the scratch directory goes:
                # a mapped file cannot be deleted while it is still mapped.
                acc = valid_plane = None

    if out_polygons:
        _write_polygons(out_polygons, out_mask, out_prob, transform, crs,
                        threshold, min_area)
        summary = replace(summary, outputs=summary.outputs + (Path(out_polygons),))

    LOGGER.info('canopy: %d of %d valid px (%.2f%%)', summary.canopy_pixels,
                summary.valid_pixels, 100 * summary.canopy_fraction)
    return summary


def _write_outputs(acc: Accumulator, valid_plane, threshold: float,
                   profile: dict, crs, transform,
                   out_prob: Optional[Path], out_mask: Optional[Path],
                   window_count: int) -> PredictionSummary:
    """Stream the blended result into the output rasters, stripe by stripe."""
    _import('rasterio', 'rasterio')
    from rasterio.windows import Window as RioWindow

    written, canopy, valid_total = [], 0, 0
    prob_dst = _open_raster(out_prob, profile, crs, transform, 'float32', None)
    mask_dst = _open_raster(out_mask, profile, crs, transform, 'uint8', 0)

    try:
        for rows, values in acc.blocks(_STRIPE_ROWS):
            valid = valid_plane[rows].astype(bool)
            # Nodata in the source is never canopy: zero it before thresholding
            # so unsurveyed ground cannot be reported as crown.
            values = np.where(valid, values, 0.0).astype(np.float32, copy=False)
            binary = ((values >= threshold) & valid).astype(np.uint8)

            valid_total += int(valid.sum())
            canopy += int(binary.sum())

            band_window = RioWindow(
                0, rows.start, values.shape[1], values.shape[0])
            if prob_dst is not None:
                prob_dst.write(values, 1, window=band_window)
            if mask_dst is not None:
                mask_dst.write(binary, 1, window=band_window)
    finally:
        for dst, path in ((prob_dst, out_prob), (mask_dst, out_mask)):
            if dst is not None:
                dst.close()
                written.append(Path(path))
                LOGGER.info('wrote %s', path)

    return PredictionSummary(
        width=acc.width, height=acc.height, windows=window_count,
        canopy_pixels=canopy, valid_pixels=valid_total,
        outputs=tuple(written))


def _open_raster(path: Optional[Path], profile: dict, crs, transform,
                 dtype: str, nodata):
    """Open a single-band GeoTIFF for windowed writing, or return ``None``."""
    if path is None:
        return None
    rasterio = _import('rasterio', 'rasterio')

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        **profile,
        'driver': 'GTiff', 'count': 1, 'dtype': dtype,
        'crs': crs, 'transform': transform,
        'compress': 'deflate',
        # Predictor 3 is GDAL's floating-point predictor; 2 is horizontal
        # differencing and applies to integer data only.
        'predictor': 3 if dtype == 'float32' else 2,
        'tiled': True, 'blockxsize': 512, 'blockysize': 512,
        'BIGTIFF': 'IF_SAFER',
    }
    if nodata is None:
        profile.pop('nodata', None)
    else:
        profile['nodata'] = nodata
    return rasterio.open(path, 'w', **profile)


def _in_square_metres(crs) -> bool:
    """Whether polygon areas in this CRS are square metres."""
    if crs is None or not crs.is_projected:
        return False
    units = (crs.linear_units or '').lower()
    return units in ('metre', 'meter', 'm')


def _write_polygons(path: Path, mask_path: Optional[Path],
                    prob_path: Optional[Path], transform, crs,
                    threshold: float, min_area: float = 0.0) -> None:
    """Vectorise the crown mask.

    Reads a finished raster back rather than holding one during inference. The
    mask is preferred (one byte per pixel); if only a probability raster was
    written, it is thresholded on the way in.

    Every polygon carries its area, so crowns can be counted, measured and
    filtered in GIS without further work. ``min_area`` drops polygons smaller
    than that many square metres before writing; at the default of ``0`` the
    layer corresponds exactly to the mask raster, speck for speck.
    """
    rasterio = _import('rasterio', 'rasterio')
    gpd = _import('geopandas', 'geopandas')
    from rasterio.features import shapes
    from shapely.geometry import shape

    source = mask_path or prob_path
    if source is None:
        raise ValueError(
            'polygon output needs --out-mask or --out-prob to vectorise from')

    with rasterio.open(source) as src:
        band = src.read(1)
    mask = band.astype(np.uint8) if mask_path else (band >= threshold).astype(np.uint8)

    geoms = [
        shape(geom)
        for geom, value in shapes(mask, mask=mask.astype(bool), transform=transform)
        if value == 1
    ]

    metric = _in_square_metres(crs)
    if min_area > 0 and not metric:
        LOGGER.warning(
            'min-area needs a projected CRS in metres; %s is not one, so no '
            'polygons were dropped', crs)
    elif min_area > 0:
        kept = [g for g in geoms if g.area >= min_area]
        LOGGER.info('dropped %d polygon(s) under %g m2', len(geoms) - len(kept),
                    min_area)
        geoms = kept

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    column = 'area_m2' if metric else 'area_crs_units'
    frame = gpd.GeoDataFrame(
        {'class': ['ghaf'] * len(geoms), column: [g.area for g in geoms]},
        geometry=geoms, crs=crs)
    if frame.empty:
        LOGGER.warning('no crowns at or above the threshold; writing an empty layer')
    frame.to_file(path)
    LOGGER.info('wrote %s (%d polygon(s))', path, len(frame))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=Path, help='model config file')
    parser.add_argument('checkpoint', type=Path, help='checkpoint .pth')
    parser.add_argument('image', type=Path, help='8-bit georeferenced orthomosaic')
    parser.add_argument('--out-prob', type=Path, help='float32 probability GeoTIFF')
    parser.add_argument('--out-mask', type=Path, help='uint8 binary mask GeoTIFF')
    parser.add_argument('--out-polygons', type=Path,
                        help='crown polygons, e.g. crowns.gpkg')
    parser.add_argument('--tile', type=int, default=1024,
                        help='window size; must match the training crop size')
    parser.add_argument('--overlap', type=int, default=512)
    parser.add_argument('--sigma', type=float, default=0.4,
                        help='Gaussian blending width, in half-tile units')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--bands', type=int, nargs=3, default=(1, 2, 3),
                        metavar=('R', 'G', 'B'))
    parser.add_argument('--batch-size', type=int, default=1,
                        help='tiles per forward pass; raise until VRAM limits it')
    parser.add_argument('--min-area', type=float, default=0.0, metavar='M2',
                        help='drop crown polygons smaller than this many '
                             'square metres; 0 keeps every speck the mask has')
    parser.add_argument('--scratch-dir', type=Path,
                        help='where to put the temporary accumulators '
                             '(default: the system temporary directory)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args(argv)
    if not (args.out_prob or args.out_mask or args.out_polygons):
        parser.error('nothing to do: pass at least one of --out-prob, '
                     '--out-mask, --out-polygons')
    if args.out_polygons and not (args.out_mask or args.out_prob):
        parser.error('--out-polygons needs --out-mask or --out-prob to '
                     'vectorise from')
    return args


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = parse_args(argv)

    import ghaf
    from ghaf.environment import quiet_repeated_warnings

    ghaf.register_all()
    quiet_repeated_warnings()
    from mmseg.apis import init_model

    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    predict_large_image(
        model, args.image,
        out_prob=args.out_prob, out_mask=args.out_mask,
        out_polygons=args.out_polygons,
        tile=args.tile, overlap=args.overlap, sigma=args.sigma,
        threshold=args.threshold, bands=tuple(args.bands),
        batch_size=args.batch_size, min_area=args.min_area,
        scratch_dir=args.scratch_dir,
        progress=not args.no_progress,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
