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
**[Testing](#testing)** ·
[Getting started](docs/GETTING_STARTED.md) · [Model zoo](docs/MODEL_ZOO.md) ·
[Area-wide inference](docs/AREA_WIDE_INFERENCE.md) ·
[Release bundle](docs/RELEASE_BUNDLE.md)

New to the project? [**docs/GETTING_STARTED.md**](docs/GETTING_STARTED.md)
walks through installing, mapping crowns in an orthomosaic, scoring a model,
training and fine-tuning, one command at a time.

## Availability

This repository holds the code, the configurations and the documentation. The
trained weights, the UAV imagery and the labelled tiles are **not distributed
here**; they are available from the corresponding author on reasonable request.

What is here is enough to check them. `ghaf/release.py` records every released
checkpoint's SHA-256, byte size, parameter count and test scores, so a copy
received by any route can be verified against what was published:

```bash
python tools/smoke_test.py --checkpoints /path/to/checkpoints
```

That confirms each file is the released one, loads it into the model built
from the config in this repository, reports any tensor the two do not share,
and runs a prediction through it.

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
```

Every command below uses `python -m pip` rather than a bare `pip`, for the
reason given under **Verify**.

Conda supplies only the interpreter; the stack is installed with pip and
openmim, which fetches the mmcv wheel matched to your PyTorch and CUDA
versions rather than building it from source.

**PyTorch** — the published models were trained with 1.12.1 on CUDA 11.3:

```bash
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

Substitute the build for your driver, or for a machine without a GPU:

```bash
python -m pip install torch==1.12.1 torchvision==0.13.1
```

**OpenMMLab** — mmcv must stay below 2.2.0, which mmsegmentation 1.2.2 asserts
at import time:

```bash
python -m pip install -U openmim
python -m mim install mmengine==0.10.7 "mmcv>=2.0.0rc4,<2.2.0"
python -m pip install mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0
python -m pip install ftfy regex
```

`mmdet` supplies the Mask2Former head; `mmpretrain` supplies the ConvNeXt,
PoolFormer and EfficientNet backbones. `ftfy` and `regex` back the CLIP
tokenizer that `mmseg.utils` imports when the package is loaded, so they are
needed for any use of mmsegmentation 1.2.2; `requirements.txt` installs them
too, whichever route you take.

**This project** — `timm` supplies DPN-98 and the FastViT building blocks, and
`rasterio` brings its own GDAL, so no system GDAL install is required:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

The `[test]` extra adds pytest. Use `-e .` alone if you do not intend to run
the suite.

NumPy is held below 2.0 because PyTorch 1.12.1 is built against the NumPy 1.x
binary interface; with NumPy 2 installed, moving a tensor to an array fails.
If the environment already has NumPy 2, `python -m pip install "numpy<2"`
brings it back into line. pip may then report that the `opencv-python` version
mmcv installed declares `numpy>=2`; the wheel is built against the NumPy 2
headers, which stay binary-compatible with NumPy 1.x, so it keeps working and
the message can be ignored.

**Verify**:

```bash
python -m pytest tests/ -q   # 275 tests
python tools/smoke_test.py   # builds all six models, checks parameter counts
```

Both are invoked through `python -m` deliberately. A bare `pytest` runs
whichever launcher is first on PATH, which on a machine with a base Anaconda
install is often that one rather than the active environment's — the tests then
fail with `ModuleNotFoundError: No module named 'mmengine'` while the
environment is perfectly healthy. `python -m` always uses the interpreter you
have activated. If in doubt:

```bash
python -c "import sys; print(sys.executable)"
```

Run `smoke_test.py` before training or evaluating: it constructs every model
from its config and compares its tensor total -- summed over `state_dict`, as
the published counts are -- against the published one, so a mismatch between
the code and the weights surfaces immediately. It also reports each model's
trainable parameter count.

Point it at a folder of checkpoints and it goes further, which is the check to
run before passing the weights on:

```bash
python tools/smoke_test.py --checkpoints /path/to/checkpoints
```

For each model it confirms the file is the released one (size and SHA-256),
loads the weights into the model built from the config and reports any tensor
the two do not share, then runs one real prediction through mmseg's inference
path. A model that reaches the last column has been exercised end to end.

## Data

```
data/ghaf/
├── training/{images,masks}/         7 005 tiles
├── validation/{images,masks}/         869 tiles
└── testing/ghaf26/{images,masks}/     767 tiles
```

Paired 1024 × 1024 PNG tiles sharing a filename stem. Masks are single-band
with pixel values equal to the class index: `0` background, `1` ghaf. Place the
tree at `data/ghaf`, symlink it, or name it on the command line:

```bash
python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py \
    --data-root /path/to/ghaf
```

`--data-root` moves every split together. It is a flag of its own rather than
a `--cfg-options` entry because the config's `data_root` is a module-level
variable, copied into each dataloader as the file is parsed — setting it
afterwards would change a key nothing reads.

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

**Predict a whole split**

```bash
python tools/predict_split.py \
    configs/ghaf/fastvit-ma36_mask2former.py \
    checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \
    --data-root data/ghaf --split testing --out-dir predictions/testing
```

Writes one predicted mask per tile, carrying each tile's CRS and geotransform,
encoded the same way as the ground truth (`0` background, `1` ghaf) so the two
can be differenced directly. `--save-probability` adds the float32 P(ghaf) map.

**Package the models for sharing**

```bash
python tools/export_release.py --checkpoints /path/to/checkpoints \
                               --output      /path/to/ghaf-release
```

Writes one self-contained folder per model — a resolved config beside its
weights, in mmsegmentation's working-directory layout — with every checkpoint
hashed before and after the copy. See
[`docs/RELEASE_BUNDLE.md`](docs/RELEASE_BUNDLE.md).

**Cut a quick sample out of a large mosaic**

```bash
python tools/make_sample.py mosaic.tif --output sample.tif --size 8192
```

A survey mosaic can be billions of pixels; the clip keeps the source CRS and
pixel grid, so it runs in a minute or two and proves the pipeline before the
full mosaic is attempted. The window is centred unless `--origin COL ROW`
names its top-left corner, and the share of valid imagery in it is reported so
a window that landed on the transparent border is obvious.

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
├── config.py              point a config at a dataset root; skip ImageNet init
├── environment.py         check the stack is in this Python, once
├── init_weights.py        keep the ImageNet weights with the project
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
├── predict_split.py       per-tile predictions for a dataset split
├── make_sample.py         cut a small georeferenced sample from a mosaic
├── smoke_test.py          build every model, check parameter counts
├── export_release.py      assemble the shareable model bundle
├── fetch_init_weights.py  collect the ImageNet weights, once, while online
└── build_handover.py      assemble the whole handover folder
tests/                     275 tests; mmengine and NumPy, plus torch, timm,
                           rasterio and geopandas for the end-to-end ones
docs/                      getting started, model zoo, area-wide inference,
                           release bundle
```

## Testing

```bash
python -m pytest tests/ -q            # 275 tests: no GPU, mmcv or dataset required
python tools/smoke_test.py  # builds all six models and checks parameter counts
python tools/smoke_test.py --checkpoints DIR   # also loads and runs the weights
```

The suite covers the tiling geometry (every pixel of a raster is covered, at
six raster shapes and four overlap settings), the blending mathematics (a
constant field reconstructs exactly; overlap-add equals a direct weighted
mean), the streaming path (stripes are contiguous and reconstruct the whole
result), the configurations (each declares its intended recipe, the binary task
stays consistently specified, all six share one evaluation protocol, and
`--data-root` moves every split where `--cfg-options` would not), and the
released-model registry (digests well-formed and unique, integrity checks
reject wrong sizes and wrong contents, a checkpoint is found wherever it sits
and two candidates are refused rather than guessed at).

Four further tests state the dataset contract -- PNG tiles, two classes,
background supervised rather than ignored -- and run wherever mmsegmentation
is installed.

The dataset validator is held to the standard that matters for a check: it
fails a tree it cannot use. Directories that exist but hold no tiles are
reported as empty, along with the extensions they hold instead, rather than
passing every pairing test by having nothing to pair.

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
