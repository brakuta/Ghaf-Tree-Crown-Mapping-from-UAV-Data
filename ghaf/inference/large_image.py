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
    optional vector file of the crown polygons.

This replaces ``tools/analysis_tools/large_infernce.py`` from the original
code, which hard-coded its paths, held the whole mosaic in RAM, and dropped
pixels in the final row and column of windows. Its behaviour is otherwise
preserved, including the Gaussian blending and its default sigma of 0.4.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .tiling import Accumulator, Window, gaussian_weights, plan_windows

LOGGER = logging.getLogger(__name__)

#: Index of the foreground class in :class:`ghaf.datasets.GhafDataset`.
GHAF_CLASS_INDEX = 1


def _require(module: str, package: str):
    try:
        return __import__(module, fromlist=['_'])
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            f'{module} is required for area-wide inference. '
            f'Install it with `pip install {package}`.') from exc


def _read_window(src, window: Window, bands: Sequence[int], tile: int) -> np.ndarray:
    """Read one window as an ``(tile, tile, 3)`` uint8 array, zero-padded.

    mmseg's preprocessing expects BGR, which is the order ``inference_model``
    assumes for a raw array, so the RGB bands are reversed on the way out.
    """
    rasterio = _require('rasterio', 'rasterio')
    from rasterio.windows import Window as RioWindow

    data = src.read(
        indexes=list(bands),
        window=RioWindow(window.col_off, window.row_off, window.width, window.height),
    )                                                    # (bands, h, w)
    if data.dtype != np.uint8:
        raise ValueError(
            f'expected an 8-bit raster, got dtype {data.dtype}. Convert the '
            f'mosaic to Byte first (e.g. gdal_translate -ot Byte -scale).')

    chw = np.zeros((len(bands), tile, tile), np.uint8)
    chw[:, :window.height, :window.width] = data
    rgb = np.transpose(chw, (1, 2, 0))                   # (tile, tile, bands)
    return rgb[:, :, ::-1].copy()                        # RGB -> BGR


def _foreground_probability(result, tile: int) -> np.ndarray:
    """Extract P(foreground) from an mmseg ``SegDataSample``."""
    torch = _require('torch', 'torch')

    logits = result.seg_logits.data                      # (num_classes, h, w)
    if logits.ndim != 3:
        raise RuntimeError(f'unexpected seg_logits shape {tuple(logits.shape)}')
    if logits.shape[0] <= GHAF_CLASS_INDEX:
        raise RuntimeError(
            f'model predicts {logits.shape[0]} class(es); this pipeline needs '
            f'at least {GHAF_CLASS_INDEX + 1} (background, ghaf)')

    prob = torch.softmax(logits.float(), dim=0)[GHAF_CLASS_INDEX]
    prob = prob.detach().cpu().numpy().astype(np.float32)
    if prob.shape != (tile, tile):
        raise RuntimeError(
            f'model returned {prob.shape} for a {tile}x{tile} tile; the config '
            f'crop size and --tile must agree')
    return prob


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
    progress: bool = True,
) -> np.ndarray:
    """Predict over a whole orthomosaic and write georeferenced outputs.

    Args:
        model: a segmentor from :func:`mmseg.apis.init_model`.
        src_path: georeferenced 8-bit orthomosaic.
        out_prob: where to write the float32 probability GeoTIFF.
        out_mask: where to write the uint8 binary mask GeoTIFF.
        out_polygons: where to write crown polygons (any OGR-writable format;
            the driver is inferred from the suffix). Requires geopandas.
        tile: window size; must match the crop size the model was trained at.
        overlap: pixels shared between neighbouring windows.
        sigma: Gaussian blending width, in half-tile units.
        threshold: probability at or above which a pixel is called ghaf.
        bands: 1-based band indices to read as R, G, B.
        progress: show a progress bar if tqdm is installed.

    Returns:
        The probability array, shape ``(height, width)``, dtype float32.

    Raises:
        ValueError: on unusable geometry, dtype, or threshold.
        RuntimeError: if the model output does not match the requested tile.
    """
    rasterio = _require('rasterio', 'rasterio')

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f'threshold must be in [0, 1], got {threshold}')
    if len(bands) != 3:
        raise ValueError(f'expected exactly 3 band indices, got {list(bands)}')
    src_path = Path(src_path)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    inference_model = _require('mmseg.apis', 'mmsegmentation').inference_model

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
        LOGGER.info('%s: %dx%d px, %d window(s) of %d px (overlap %d)',
                    src_path.name, width, height, len(windows), tile, overlap)

        # Accumulate out of core: an in-memory float32 pair costs 8 bytes/px,
        # which a large mosaic will not fit in RAM.
        with tempfile.TemporaryDirectory(prefix='ghaf-infer-') as tmp:
            num = np.memmap(Path(tmp) / 'num.dat', np.float32, 'w+',
                            shape=(height, width))
            den = np.memmap(Path(tmp) / 'den.dat', np.float32, 'w+',
                            shape=(height, width))
            acc = Accumulator(height, width, numerator=num, denominator=den)

            iterator = windows
            if progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(windows, unit='tile', desc=src_path.stem)
                except ImportError:
                    LOGGER.info('tqdm not installed; running without progress')

            for window in iterator:
                patch = _read_window(src, window, bands, tile)
                prob = _foreground_probability(inference_model(model, patch), tile)
                acc.add(window, prob, weights)

            probability = np.array(acc.result(), np.float32)   # into RAM once
            del num, den

    # Mask out source nodata so it cannot be reported as canopy.
    with rasterio.open(src_path) as src:
        valid = src.read_masks(bands[0]) > 0
    if not valid.all():
        LOGGER.info('masking %d nodata pixel(s)', int((~valid).sum()))
        probability[~valid] = 0.0

    mask = (probability >= threshold).astype(np.uint8)

    if out_prob:
        _write_raster(out_prob, probability, profile, crs, transform,
                      dtype='float32', nodata=None)
    if out_mask:
        _write_raster(out_mask, mask, profile, crs, transform,
                      dtype='uint8', nodata=0)
    if out_polygons:
        _write_polygons(out_polygons, mask, transform, crs)

    return probability


def _write_raster(path: Path, array: np.ndarray, profile: dict, crs, transform,
                  dtype: str, nodata) -> None:
    rasterio = _require('rasterio', 'rasterio')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        **profile,
        'driver': 'GTiff', 'count': 1, 'dtype': dtype,
        'crs': crs, 'transform': transform,
        'compress': 'deflate', 'predictor': 2 if dtype == 'float32' else 1,
        'tiled': True, 'blockxsize': 512, 'blockysize': 512, 'BIGTIFF': 'IF_SAFER',
    }
    if nodata is None:
        profile.pop('nodata', None)
    else:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array.astype(dtype), 1)
    LOGGER.info('wrote %s', path)


def _write_polygons(path: Path, mask: np.ndarray, transform, crs) -> None:
    rasterio = _require('rasterio', 'rasterio')
    gpd = _require('geopandas', 'geopandas')
    from rasterio.features import shapes
    from shapely.geometry import shape

    geoms = [
        shape(geom)
        for geom, value in shapes(mask, mask=mask.astype(bool), transform=transform)
        if value == 1
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame({'class': ['ghaf'] * len(geoms)},
                             geometry=geoms, crs=crs)
    if frame.empty:
        LOGGER.warning('no crowns above threshold; writing an empty layer')
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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args(argv)
    if not (args.out_prob or args.out_mask or args.out_polygons):
        parser.error('nothing to do: pass at least one of --out-prob, '
                     '--out-mask, --out-polygons')
    return args


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = parse_args(argv)

    import ghaf
    ghaf.register_all()
    from mmseg.apis import init_model

    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    predict_large_image(
        model, args.image,
        out_prob=args.out_prob, out_mask=args.out_mask,
        out_polygons=args.out_polygons,
        tile=args.tile, overlap=args.overlap, sigma=args.sigma,
        threshold=args.threshold, bands=tuple(args.bands),
        progress=not args.no_progress,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
