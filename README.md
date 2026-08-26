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
    --out-polygons crowns.gpkg
```

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

tools/                     train.py · test.py · smoke_test.py · export_release.py
tests/                     120 tests; mmengine and NumPy, plus
                           rasterio and geopandas for the end-to-end ones
docs/                      model zoo, area-wide inference, release bundle
```

## Testing

```bash
pytest tests/ -q            # 120 tests: no GPU, mmcv or dataset required
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

Released under the Apache License 2.0, following mmsegmentation.
`ghaf/models/fastvit.py` and `ghaf/models/modules/` derive from
[Apple's FastViT](https://github.com/apple/ml-fastvit) and retain its copyright
notice. DPN-98 and several baseline backbones are obtained through
[timm](https://github.com/huggingface/pytorch-image-models) and
[mmpretrain](https://github.com/open-mmlab/mmpretrain).
