# Ghaf Tree-Crown Mapping from Area-Wide UAV Imagery

Code and configurations for **"Hybrid Vision–CNN Architecture for Mapping
*Prosopis cineraria* from Area-wide UAV-based Images."**

*Prosopis cineraria* — the Ghaf — is the national tree of the United Arab
Emirates and a keystone species of its arid ecosystems. This repository trains
and evaluates six semantic-segmentation models that delineate individual Ghaf
crowns in UAV orthomosaics, and applies them across complete mosaics to produce
georeferenced canopy maps.

The proposed model pairs a **FastViT-MA36** hybrid backbone — RepMixer token
mixing in the early stages, self-attention in the last — with a **Mask2Former**
decode head, and is compared against five alternatives spanning convolutional,
transformer and hybrid designs at 13 M to 82 M parameters.

This is an **overlay on stock mmsegmentation**: it adds a dataset, two
backbones, six configurations and an area-wide inference pipeline. Nothing
inside `mmseg` is modified, so `pip install mmsegmentation` is the whole
framework setup.

[![tests](https://github.com/brakuta/Ghaf-Tree-Crown-Mapping-from-UAV-Data/actions/workflows/tests.yml/badge.svg)](https://github.com/brakuta/Ghaf-Tree-Crown-Mapping-from-UAV-Data/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**[Results](#results)** · **[Install](#installation)** · **[Data](#data)** ·
**[Usage](#usage)** · **[Layout](#repository-layout)** ·
**[Testing](#testing)** · **[Limits](#scope-and-limits)** ·
[Model zoo](docs/MODEL_ZOO.md) · [Area-wide inference](docs/AREA_WIDE_INFERENCE.md) ·
[Release bundle](docs/RELEASE_BUNDLE.md)

---

## Results

Held-out test split (`testing/ghaf26`, 767 tiles). All six models share one
dataset, one augmentation pipeline, one schedule and one metric implementation,
so the differences below are attributable to the encoder and decode head.

| Backbone | Decode head | Params | mIoU | F1 |
|---|---|---:|---:|---:|
| **FastViT-MA36** | Mask2Former | 62.5 M | **79.32** | **87.22** |
| PoolFormer-S36 | FPN | 34.6 M | 78.65 | 86.72 |
| DPN-98 | FPN | 65.3 M | 78.19 | 86.35 |
| ConvNeXt-S | UPerNet | 81.8 M | 78.02 | 86.20 |
| ResNet-50 | Mask2Former | 44.1 M | 77.69 | 85.98 |
| EfficientNet-B3 | FPN | 13.7 M | 70.77 | 80.29 |

On the 869-tile validation split, FastViT-MA36 reaches **80.14 mIoU / 87.87 F1**.

Two comparisons are worth drawing out. FastViT-MA36 and ResNet-50 share an
identical Mask2Former head, optimiser and schedule, so the 1.63-point mIoU gap
between them isolates the backbone. And FastViT-MA36 outperforms ConvNeXt-S
while using 24 % fewer parameters, so the gain is not simply capacity.

Per-model details, checkpoint hashes and training settings are in
[`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md).

## Installation

```bash
conda env create -f environment.yml
conda activate ghaf
pip install -e .
python tools/smoke_test.py
```

Or step by step:

```bash
conda create -n ghaf python=3.9 -y && conda activate ghaf
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install mmengine==0.10.7 "mmcv>=2.0.0rc4,<2.2.0"
pip install mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0
pip install -r requirements.txt
pip install -e .
```

`mmdet` supplies the Mask2Former head; `mmpretrain` supplies the ConvNeXt,
PoolFormer and EfficientNet backbones; `timm` supplies DPN-98 and the FastViT
building blocks.

`tools/smoke_test.py` builds all six models, runs a forward pass and checks
each parameter count against the published model. Run it before training.

## Data

```
data/ghaf/
├── training/{images,masks}/         7 005 tiles
├── validation/{images,masks}/         869 tiles
└── testing/ghaf26/{images,masks}/     767 tiles
```

Paired 1024 × 1024 GeoTIFFs sharing a filename stem. Masks are single-band with
pixel values equal to the class index: `0` background, `1` ghaf. Place the tree
at `data/ghaf`, symlink it, or override `data_root`:

```bash
python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py \
    --cfg-options data_root=/path/to/ghaf
```

## Usage

**Check the data first**

```bash
python tools/check_dataset.py data/ghaf
```

Verifies that every image has a mask, that pairs agree on size, and that masks
contain only the two class indices. A fault here shows up during training as a
confusing loss curve rather than an error, so it is worth the minute. Add
`--full` to open every tile instead of a sample.

**Train**

```bash
python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py
```

Checkpoints and logs are written to `work_dirs/<config-name>/`. Validation runs
every 3 500 iterations and the best-scoring checkpoint is kept as
`best_mIoU_iter_*.pth`.

**Evaluate**

```bash
python tools/test.py configs/ghaf/fastvit-ma36_mask2former.py \
    checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth
```

Reports mIoU, mDice and mFscore over the held-out test split.

**Package the models for sharing**

```bash
python tools/export_release.py --checkpoints /path/to/checkpoints \
                               --output      /path/to/ghaf-release
```

Writes one self-contained folder per model — a resolved config beside its
weights, in mmsegmentation's working-directory layout — with every checkpoint
hashed before and after the copy. See
[`docs/RELEASE_BUNDLE.md`](docs/RELEASE_BUNDLE.md).

**Map a whole orthomosaic**

```bash
python -m ghaf.inference.large_image \
    configs/ghaf/fastvit-ma36_mask2former.py \
    checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \
    mosaic.tif \
    --out-prob probability.tif \
    --out-mask crowns.tif \
    --out-polygons crowns.gpkg \
    --batch-size 4 \
    --scratch-dir /fast/disk
```

`--batch-size` trades VRAM for throughput; raise it until memory is the limit.
`--scratch-dir` places the temporary accumulators, which need **9 bytes per
source pixel** — worth pointing at a large disk, since the system temporary
directory is often on a small system drive. Free space is checked before a run
starts rather than part-way through.

Slides a 1024 px window with 512 px overlap, blends overlapping predictions
with Gaussian weights, and writes georeferenced probability, mask and polygon
outputs in the source CRS. Peak memory does not grow with the mosaic — windows
are read one at a time, the accumulators are memory-mapped, and results stream
back to disk a stripe at a time. See
[`docs/AREA_WIDE_INFERENCE.md`](docs/AREA_WIDE_INFERENCE.md).

## Repository layout

```
ghaf/
├── datasets.py            GhafDataset — the two-class tile dataset
├── release.py             the published models: digests, params, scores
├── models/
│   ├── fastvit.py         FastViT-MA36 backbone
│   ├── dpn.py             DPN-98 backbone
│   └── modules/           MobileOne and RepLKNet blocks used by FastViT
└── inference/
    ├── tiling.py          window planning and Gaussian blending (pure NumPy)
    └── large_image.py     orthomosaic inference and georeferenced output

configs/
├── _base_/ghaf.py         dataset, pipeline, schedule and runtime, shared
└── ghaf/                  the six model configurations

tools/
├── check_dataset.py       validate a tile tree before using it
├── train.py · test.py     thin wrappers over the mmengine runner
├── smoke_test.py          build every model, check parameter counts
└── export_release.py      assemble the shareable bundle
tests/                     153 tests; mmengine and NumPy, plus torch, timm,
                           rasterio and geopandas for the end-to-end ones
docs/                      model zoo, area-wide inference, release bundle
```

## Testing

```bash
pytest tests/ -q            # 153 tests: no GPU, mmcv or dataset required
python tools/smoke_test.py  # builds all six models and checks parameter counts
```

The suite covers the tiling geometry (every pixel of a raster is covered, at
six raster shapes and four overlap settings), the blending mathematics (a
constant field reconstructs exactly; overlap-add equals a direct weighted
mean), the streaming path (stripes are contiguous and reconstruct the whole
result), the configurations (each declares its intended recipe, the binary task
stays consistently specified, all six share one evaluation protocol), and the
released-model registry (digests well-formed and unique, integrity checks
reject wrong sizes and wrong contents).

The two custom backbones are built and executed: FastViT-MA36 is checked to
have stage depths `[6, 6, 18, 6]` and widths `[76, 152, 304, 608]`, DPN-98 to
emit five scales at `[96, 336, 768, 1728, 2688]`, each config to consume
exactly what its backbone produces, and pretrained weights to load through
`init_cfg`.

Area-wide inference is tested end to end against real GeoTIFFs with a stubbed
segmentor: outputs are georeferenced to the source CRS, a uniform prediction
survives blending unchanged, awkward raster sizes leave no pixel unwritten,
nodata is never reported as canopy, crown polygons carry the source CRS, and
each failure mode — 16-bit input, a missing band, a single-class model, a tile
size the model disagrees with — is rejected with a message that says what to
do. Scratch directories are asserted to be cleaned up even when a run fails.

`ghaf/release.py` is the single source of truth for what was published — the
digests, parameter counts and scores that `tools/smoke_test.py` checks built
models against and `tools/export_release.py` verifies copies against. A model
therefore cannot be documented one way and shipped another.

## Scope and limits

Stated plainly, so nobody discovers them the hard way.

- **Reproducing the published numbers means evaluating the published
  checkpoints.** The configs fix no random seed by default, so retraining lands
  close but not identical — expect the third significant figure to move. Set
  `randomness` in `configs/_base_/ghaf.py` for deterministic runs.
- **Area-wide inference needs 8-bit input.** Convert first with
  `gdal_translate -ot Byte -scale`. 16-bit rasters are rejected with that
  remedy rather than silently rescaled.
- **`--tile` must match the training input size (1024).** A model given a
  different tile size raises rather than resizing, because resizing would
  change the result without saying so.
- **Scratch disk, not RAM, is the binding constraint** on how large a mosaic
  can be processed: 9 bytes per source pixel.
- **Crown polygons are raster boundaries**, one polygon per connected
  component. Touching crowns merge; the pipeline does not separate instances.
- **The two custom backbones are Python, not configuration.** A released model
  folder is self-contained as a config, but still needs the `ghaf` package
  installed.
- **Windows is the platform these models were trained on**, and the tools are
  path-agnostic, but CI runs on Linux only.

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub's
"Cite this repository" button reads it.

```bibtex
@article{ghaf_uav_segmentation,
  title   = {Hybrid Vision--CNN Architecture for Mapping Prosopis cineraria
             from Area-wide UAV-based Images},
  author  = {Gibril, Mohamed Barakat A. and others},
  journal = {},
  year    = {2025}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE), with third-party attributions in
[`NOTICE`](NOTICE).

`ghaf/models/fastvit.py` and `ghaf/models/modules/` derive from
[Apple's FastViT](https://github.com/apple/ml-fastvit) and retain its copyright
notice. DPN-98 and several baseline backbones are obtained through
[timm](https://github.com/huggingface/pytorch-image-models) and
[mmpretrain](https://github.com/open-mmlab/mmpretrain).
