# Ghaf Tree-Crown Mapping from Area-Wide UAV Imagery

Code, configurations and provenance for **"Hybrid Vision–CNN Architecture for
Mapping *Prosopis cineraria* from Area-wide UAV-based Images."**

Six semantic-segmentation models were trained to delineate Ghaf tree crowns in
UAV orthomosaics over the United Arab Emirates, and applied across whole
mosaics with overlapping-window inference.

This repository is an **overlay on stock mmsegmentation**: it adds a dataset,
two custom backbones, six configs and an area-wide inference pipeline. No file
inside `mmseg` is patched, so `pip install mmsegmentation` is all the
framework setup required.

---

## Results

Held-out test set, `testing/ghaf26`, 767 tiles. Every model was scored with an
identical protocol on identical ground truth on 18 March 2025; the evaluation
logs are preserved under [`provenance/`](provenance/).

| Backbone | Decode head | Params | mIoU | F1 |
|---|---|---:|---:|---:|
| **FastViT-MA36** | Mask2Former | 62.5 M | **79.32** | **87.22** |
| PoolFormer-S36 | FPN | 34.6 M | 78.65 | 86.72 |
| DPN-98 | FPN | 65.3 M | 78.19 | 86.35 |
| ConvNeXt-S | UPerNet | 81.8 M | 78.02 | 86.20 |
| ResNet-50 | Mask2Former | 44.1 M | 77.69 | 85.98 |
| EfficientNet-B3 | FPN | 13.7 M | 70.77 | 80.29 |

FastViT-MA36 also scores **80.14 mIoU / 87.87 F1** on the 869-tile validation
split.

> **Three numbers in the manuscript do not match these logs**: PoolFormer's
> mIoU (78.45 → **78.65**) and F1 (86.62 → **86.72**), and FastViT's F1
> (87.30 → **87.22**). The first two also appear in the abstract, and they
> narrow the proposed model's margin from 0.87 to **0.67** points. See
> [`docs/MANUSCRIPT-CORRECTIONS.md`](docs/MANUSCRIPT-CORRECTIONS.md).

Parameter counts are summed directly over each checkpoint's tensors, not
quoted from the paper. `tools/smoke_test.py` re-checks them on any machine.

## Model naming

Several names in the code are misleading, and five of the six model names in
the manuscript are wrong. Every claim below is verified against the
checkpoints' own weights — see
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the evidence.

| Registered / directory name | What it actually is | Manuscript says |
|---|---|---|
| `fastvit_small` | FastViT-**MA36** (`layers=[6,6,18,6]`, `dims=[76,152,304,608]`) | FastViT-SA12 |
| `coatnet_small_timm`, `dual_path` | **DPN-98** — never CoAtNet | DPN-92 |
| `mask2former_swin-t_…` | FastViT + Mask2Former — no Swin involved | — |
| `ADE20KDataset` | the 2-class Ghaf dataset | — |
| `…_ade20k-512x512` | trained at **1024×1024** | — |
| ConvNeXt / PoolFormer / EfficientNet | **S / S36 / B3** | T / S12 / B0 |

This repository registers honest names (`FastViTMA36`, `DPN98Backbone`,
`GhafDataset`) and keeps the originals as working aliases so archived configs
still load.

## Install

```bash
conda create -n ghaf python=3.9 -y && conda activate ghaf
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu113
pip install -U openmim && mim install mmengine==0.10.7 "mmcv>=2.0.0rc4,<2.2.0"
pip install mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0 timm==1.0.19
pip install -r requirements.txt        # rasterio, geopandas, shapely, tqdm
pip install -e .
python tools/smoke_test.py             # builds all six models
```

`mmdet` is required by Mask2Former; `mmpretrain` supplies the ConvNeXt,
PoolFormer and EfficientNet backbones. The exact environment used for training
is recorded in [`environment/`](environment/) — it ran mmcv 2.2.0 with
mmseg's version guard disabled; see
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Data

```
data/ghaf/
├── training/{images,masks}/       7 005 tiles
├── validation/{images,masks}/       869 tiles
└── testing/ghaf26/{images,masks}/   767 tiles
```

Paired 1024×1024 GeoTIFFs sharing a stem. Masks are single-band, `0` =
background, `1` = ghaf. Point `data_root` at this tree or symlink it.

## Usage

```bash
# train
python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py

# evaluate on the test set
python tools/test.py configs/ghaf/fastvit-ma36_mask2former.py \
    checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth

# map a whole orthomosaic
python -m ghaf.inference.large_image \
    configs/ghaf/fastvit-ma36_mask2former.py \
    checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \
    mosaic.tif --out-mask crowns.tif --out-polygons crowns.gpkg
```

Area-wide inference slides a 1024 px window with 512 px overlap and blends
overlapping predictions with Gaussian weights, then writes georeferenced
probability, mask and polygon outputs. Accumulators are memory-mapped, so
mosaic size is bounded by disk rather than RAM.

## Layout

```
ghaf/                     the overlay package
├── datasets.py           GhafDataset
├── models/               FastViT-MA36, DPN-98 (+ originals, verbatim)
└── inference/
    ├── tiling.py         window planning and Gaussian blending (pure NumPy)
    └── large_image.py    orthomosaic inference and georeferenced output
configs/ghaf/             the six published configs
tools/                    train, test, smoke_test
tests/                    67 tests; need only mmengine + NumPy
docs/                     reproduction, provenance, corrections, known issues
provenance/               archived training and evaluation logs and configs
environment/              the workstation environment as captured
```

## Verification

```bash
pytest tests/ -q            # 67 tests, no GPU/mmcv/dataset needed
python tools/smoke_test.py  # builds all six models, checks parameter counts
```

The tests assert that the six configs still match the archived runs, that the
binary task stays consistently configured, that all models share one
evaluation protocol, and that the tiling scheme covers every pixel.

## Provenance and honesty

Everything published here was reconstructed by auditing checkpoints and logs
after the original author left. Rather than presenting a tidied story, the
audit is part of the repository:

- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — checkpoint → training run →
  reported score, with SHA-256 for every file
- [`docs/KNOWN-ISSUES.md`](docs/KNOWN-ISSUES.md) — asymmetries between the
  proposed model and the baselines that reviewers should know about
- [`docs/MANUSCRIPT-CORRECTIONS.md`](docs/MANUSCRIPT-CORRECTIONS.md) — every
  correction the text requires
- Tag `archive/full-source-fork` — the complete 1 154-file mmsegmentation fork
  exactly as recovered from the workstation, before any restructuring

## Citation

```bibtex
@article{ghaf_uav_segmentation,
  title  = {Hybrid Vision--CNN Architecture for Mapping Prosopis cineraria
            from Area-wide UAV-based Images},
  author = {Gibril, Mohamed Barakat A. and others},
  year   = {2025}
}
```

## License

Code released under the Apache License 2.0, following mmsegmentation.
`ghaf/models/fastvit.py` is derived from Apple's FastViT
([LICENSE](https://github.com/apple/ml-fastvit)) and retains its notice.
