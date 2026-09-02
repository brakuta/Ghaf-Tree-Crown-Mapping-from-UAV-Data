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

Peak memory does not grow with the mosaic. Three things make that true:

- **Windows are read one at a time** with rasterio's windowed reads, never by
  loading the mosaic.
- **The accumulators are memory-mapped** to a temporary directory rather than
  allocated. They cost 4 bytes per pixel each, plus a 1-byte validity plane —
  **9 bytes of scratch disk per source pixel** while a run is in progress,
  reported at startup so a full disk fails with an explanation.
- **Results are written back a stripe at a time.** The blended output is never
  materialised whole; `Accumulator.blocks()` yields 1024-row stripes that go
  straight into the output GeoTIFF.

The one exception is `--out-polygons`, which reads the finished mask raster
back to vectorise it — one byte per pixel, and only when requested.

Free space is checked before a run begins, so an undersized filesystem fails
immediately with the figure it needs rather than part-way through with
`ENOSPC`. `--scratch-dir` moves the accumulators off the system drive, which on
Windows is usually where the temporary directory lives and is often the
smallest volume on the machine.

The scratch directory is removed when the run ends, successfully or not. The
maps are flushed and released first, because Windows keeps a file locked while
any mapping of it is open; if a lock outlives the run the directory is named in
a warning rather than raised as an error, so a failing run still reports its own
cause.

## Throughput

One tile per forward pass leaves the GPU idle between tiles, and mmseg rebuilds
its test pipeline on every call — upstream flags this in `_preprare_data` with
`TODO: Consider using the singleton pattern`. `--batch-size` addresses both:
tiles are grouped into one call, which amortises the pipeline construction and
keeps the device busy.

A survey mosaic runs to thousands of tiles — a 50,000 x 50,000 px mosaic at the
default stride is about 9,500 — so the difference is measured in hours. Raise
`--batch-size` until VRAM becomes the limit; the default of 1 is the safe
choice, not the fast one.

Batch size is a throughput knob only. `tests/test_large_image.py` asserts that
batch sizes of 1, 2, 5 and 64 produce identical output.

## Outputs

All three carry the source CRS and geotransform:

| Flag | Content |
|---|---|
| `--out-prob` | `float32` GeoTIFF of P(ghaf) ∈ [0, 1] |
| `--out-mask` | `uint8` GeoTIFF, 1 where P ≥ `--threshold` |
| `--out-polygons` | crown polygons, any OGR-writable format, each carrying its area |

Rasters are written tiled and Deflate-compressed, with `BIGTIFF=IF_SAFER` so
large survey areas do not overflow the classic TIFF limit. Pixels that are
nodata in the source are forced to zero, so unsurveyed ground is never reported
as canopy.

Each polygon carries an `area_m2` attribute — its area in square metres, taken
from the geometry, so crowns can be counted and measured in GIS with no further
work. When the source CRS is not projected in metres the column is named
`area_crs_units` instead, because that is what the number then is.

Vectorising a threshold leaves specks: isolated pixels that scrape past the
cut-off and become polygons of a few square centimetres. They add nothing to
the mapped area but do inflate a crown count. `--min-area` drops polygons below
a given size in square metres; on 2.7 cm imagery `--min-area 1` removes the
specks and keeps every real crown. It is off by default, so the polygon layer
corresponds exactly to the mask raster unless you ask for otherwise.

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
| `--batch-size` | 1 | tiles per forward pass; raise until VRAM limits it |
| `--min-area` | 0 | drop crown polygons below this many m²; 0 keeps every speck |
| `--scratch-dir` | system temp | where the accumulators go |
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

summary = predict_large_image(model, 'mosaic.tif', out_mask='crowns.tif')
print(summary.canopy_fraction, summary.outputs)
```

`predict_large_image` returns a small `PredictionSummary` — raster dimensions,
window count, canopy and valid pixel counts, and the paths written — rather
than the array. Returning pixels would force the whole raster into memory and
undo the streaming; read the written GeoTIFF back if you need them.

The tiling and blending are importable on their own — `ghaf.inference.tiling`
depends on nothing but NumPy — if you need the same scheme for a different
model or task.
