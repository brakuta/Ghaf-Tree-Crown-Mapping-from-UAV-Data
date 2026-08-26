# Area-wide inference

A UAV orthomosaic covering a survey area is far larger than the 1024 × 1024
input a segmentation network accepts. Mapping one therefore means predicting
over a grid of overlapping windows and merging the results — and how that merge
is done determines whether the output is a usable canopy map or a mosaic of
visible seams.

## The problem with tiling

Predictions are least reliable near a tile's edge, where the network sees only
part of a crown's context. Two naive approaches both fail:

- **Non-overlapping tiles** put the least reliable predictions directly against
  each other at every tile boundary, producing a visible grid of artefacts and
  splitting crowns that straddle a boundary.
- **Overlapping tiles with the last write winning** discards good central
  predictions in favour of whichever tile happened to be processed last.

## Gaussian-weighted overlap-add

Each window's prediction is weighted by a bell-shaped kernel that peaks at the
tile centre and decays toward its edges:

```
w(x, y) = exp( −(x² + y²) / 2σ² ),    x, y ∈ [−1, 1] across the tile
```

with σ = 0.4 in half-tile units, placing the tile edge 2.5 standard deviations
from the centre. Weighted predictions accumulate into a numerator, the weights
themselves into a denominator, and dividing gives a weighted mean per pixel:

```
P(x, y) = Σᵢ wᵢ(x, y) · pᵢ(x, y)  /  Σᵢ wᵢ(x, y)
```

Every pixel is therefore a smooth, confidence-weighted blend of every
prediction covering it, dominated by whichever tile saw it most centrally. The
weights are strictly positive everywhere, so the denominator can never vanish.

With the default 1024 px window and 512 px stride, interior pixels are covered
by four windows.

## Coverage at the edges

Windows advance by `tile − overlap`. The final window along each axis is
clamped so it ends exactly at the raster edge, which means its step may be
shorter than the others. That is deliberate: it keeps every window at the full
tile size, so the network always sees the resolution it was trained at, while
still covering the last row and column of pixels.

`tests/test_tiling.py` asserts full coverage across six raster shapes,
including sizes that are not multiples of the stride, rasters smaller than one
tile, and extreme aspect ratios.

## Memory

A float32 numerator and denominator over the whole raster costs 8 bytes per
source pixel, which exceeds RAM for a large survey. Both accumulators are
memory-mapped to temporary files, so the limit is free disk rather than memory,
and windows are read from the source with rasterio's windowed reads rather than
by loading the mosaic.

## Outputs

All three carry the source CRS and geotransform:

| Flag | Content |
|---|---|
| `--out-prob` | `float32` GeoTIFF of P(ghaf) ∈ [0, 1] |
| `--out-mask` | `uint8` GeoTIFF, 1 where P ≥ `--threshold` |
| `--out-polygons` | crown polygons, any OGR-writable format |

Rasters are written tiled and Deflate-compressed, with `BIGTIFF=IF_SAFER` so
large survey areas do not overflow the classic TIFF limit. Pixels that are
nodata in the source are forced to zero, so unsurveyed ground is never reported
as canopy.

## Usage

```bash
python -m ghaf.inference.large_image \
    configs/ghaf/fastvit-ma36_mask2former.py \
    checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \
    mosaic.tif \
    --out-prob probability.tif \
    --out-mask crowns.tif \
    --out-polygons crowns.gpkg
```

| Option | Default | Notes |
|---|---|---|
| `--tile` | 1024 | must match the training input size |
| `--overlap` | 512 | larger is smoother and slower |
| `--sigma` | 0.4 | blending width, in half-tile units |
| `--threshold` | 0.5 | probability at which a pixel is called ghaf |
| `--bands` | 1 2 3 | 1-based source band indices read as R, G, B |
| `--device` | `cuda:0` | |

The mosaic must be 8-bit; convert first if necessary:

```bash
gdal_translate -ot Byte -scale input.tif mosaic.tif
```

## Calling it from Python

```python
import ghaf
from mmseg.apis import init_model
from ghaf.inference.large_image import predict_large_image

ghaf.register_all()
model = init_model('configs/ghaf/fastvit-ma36_mask2former.py',
                   'checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth')

probability = predict_large_image(model, 'mosaic.tif', out_mask='crowns.tif')
```

The tiling and blending are importable on their own — `ghaf.inference.tiling`
depends on nothing but NumPy — if you need the same scheme for a different
model or task.
