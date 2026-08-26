# Reproducing the results

## The environment actually used

Captured from the training workstation and preserved in
[`environment/`](../environment/):

| Component | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 1.12.1 (CUDA 11.3) |
| mmengine | 0.10.7 |
| mmcv | 2.2.0 |
| mmsegmentation | 1.2.2 (source tree, `pip install -e .`) |
| timm | 1.0.19 |
| GDAL | 3.9.2 |

Two things about that environment are worth knowing before you copy it.

**mmcv 2.2.0 is outside mmseg 1.2.2's supported range.** mmseg declares
`MMCV_MAX = '2.2.0'` and asserts `mmcv_version < mmcv_max_version`, which
2.2.0 fails. The original tree got past this by commenting the assertion out
and raising `MMCV_MAX` to `'2.3.0'`. This repository does not patch mmseg;
install **`mmcv>=2.0.0rc4,<2.2.0`** instead, which satisfies the guard as
written. If you must match the training environment exactly, use mmcv 2.2.0
and install mmsegmentation from source with the guard relaxed — and say so in
any write-up.

**`mmseg` was never installed.** Training ran with the source tree as the
working directory, so Python picked up `mmseg/` and `models/` from the cwd.
That is why every archived config records
`work_dir = G:\experiments\mmsegmentation\work_dirs\...`.

## Fresh install

```bash
conda create -n ghaf python=3.9 -y && conda activate ghaf
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install mmengine==0.10.7 "mmcv>=2.0.0rc4,<2.2.0"
pip install mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0 timm==1.0.19
pip install -r requirements.txt
pip install -e .
```

Verify before touching data:

```bash
pytest tests/ -q            # 67 tests: configs, tiling geometry, blending
python tools/smoke_test.py  # builds all six models, checks parameter counts
```

`smoke_test.py` compares each model's parameter count against the count
measured from its published checkpoint. If a count differs, the code and the
checkpoint have diverged — stop and investigate rather than training.

## Data layout

```
data/ghaf/
├── training/{images,masks}/         7 005 tiles
├── validation/{images,masks}/         869 tiles
└── testing/ghaf26/{images,masks}/     767 tiles
```

Paired 1024×1024 GeoTIFFs sharing a stem; masks single-band, `0` background,
`1` ghaf. Either place the tree at `data/ghaf` or override:

```bash
python tools/train.py configs/ghaf/fastvit-ma36_mask2former.py \
    --cfg-options data_root=/path/to/ghaf
```

Two further label variants exist in the original data and are **not** used by
these configs: `validation/masks_refined` and `validation/masks_ref`. They were
used for checkpoint selection on the FastViT runs only — see
[`KNOWN-ISSUES.md`](KNOWN-ISSUES.md) §2.

## Training

```bash
python tools/train.py configs/ghaf/<name>.py
```

Checkpoints and logs land in `work_dirs/<name>/`. The configs set
`max_iters=160000`, but no published run approached it — checkpoints were
selected on validation mIoU between 3,500 and 38,500 iterations. Expect to
stop early and pick `best_mIoU_iter_*.pth`.

## Evaluation

```bash
python tools/test.py configs/ghaf/<name>.py <checkpoint>.pth
```

This reproduces the published protocol: the 767-tile `testing/ghaf26` split,
scored with `IoUMetric` for mIoU, mDice and mFscore.

## Expected differences from the published numbers

**Reruns will not match exactly.** No seed was set in any archived run, so
expect variation in the third significant figure. To make new runs
deterministic, uncomment in `configs/_base_/ghaf.py`:

```python
randomness = dict(seed=0, deterministic=True)
```

Reproducing the *published* numbers exactly is only possible by evaluating the
published checkpoints — which is what `tools/test.py` with the hashes in
[`PROVENANCE.md`](PROVENANCE.md) is for.

## Area-wide mapping

```bash
python -m ghaf.inference.large_image \
    configs/ghaf/fastvit-ma36_mask2former.py \
    checkpoints/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \
    mosaic.tif \
    --out-prob prob.tif --out-mask crowns.tif --out-polygons crowns.gpkg
```

The mosaic must be 8-bit and georeferenced. `--tile` must match the training
crop size (1024). Accumulators are memory-mapped, so the limit is free disk
(8 bytes per source pixel) rather than RAM.
