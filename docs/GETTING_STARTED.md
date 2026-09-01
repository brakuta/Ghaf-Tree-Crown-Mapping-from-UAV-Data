# Getting started

A walkthrough for running this project end to end, written for someone who is
comfortable following instructions but does not write code. Every step is a
command you copy, paste and run. Each one says what it does, roughly how long
it takes, and what you should see when it works.

By the end you will be able to map Ghaf crowns in a new UAV orthomosaic,
reproduce the published scores, train a model, and fine-tune one on new
labels.

Commands are written for the Windows Command Prompt. On macOS or Linux they
are identical apart from the path separator (`/` instead of `\`).

> **Paste one command at a time.** Some terminals join multi-line pastes into
> a single line, which produces confusing errors.

**Contents**

1. [What you were given](#1-what-you-were-given)
2. [Install, once](#2-install-once)
3. [Check that it works](#3-check-that-it-works)
4. [Map trees in a new orthomosaic](#4-map-trees-in-a-new-orthomosaic)
5. [Predict every tile in a split](#5-predict-every-tile-in-a-split)
6. [Score a model against the labelled test set](#6-score-a-model-against-the-labelled-test-set)
7. [Train a model](#7-train-a-model)
8. [Fine-tune on new labels](#8-fine-tune-on-new-labels)
9. [When something goes wrong](#9-when-something-goes-wrong)

---

## 1. What you were given

Three things, which live in separate places for a reason.

| | What it is | Where it comes from |
|---|---|---|
| **The code** | This repository — configurations, tools, documentation | Public on GitHub |
| **The models** | Six trained checkpoints, one folder each | Shared with you directly; not on GitHub |
| **The data** | Labelled tiles, and a sample orthomosaic to try inference on | Shared with you directly; not on GitHub |

A convenient layout, assumed by the examples below. Yours may differ; adjust
the paths and nothing else changes.

```
D:\ghaf-project\
├── code\                              this repository
├── models\
│   ├── fastvit-ma36_mask2former\
│   │   ├── fastvit-ma36_mask2former.py    the complete recipe, self-contained
│   │   ├── best_mIoU_iter_3500.pth        the weights
│   │   └── metadata.json                  digest, size, scores
│   └── ...                                five more
├── data\ghaf\
│   ├── training\{images,masks}\           7 005 tiles
│   ├── validation\{images,masks}\           869 tiles
│   └── testing\ghaf26\{images,masks}\       767 tiles
└── samples\
    ├── Kalba26_sample.tif                 a small clip: start here, minutes
    └── Kalba26.tif                        the full UAV orthomosaic, hours
```

Tiles are 1024 × 1024 PNG pairs: an image and a mask sharing a filename. In a
mask, `0` is background and `1` is a Ghaf crown.

---

## 2. Install, once

You need [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
and, for anything beyond a small test, an NVIDIA GPU.

**Step 1 — create the environment and enter it.**

```
conda create -n ghaf python=3.9 -y
```

```
conda activate ghaf
```

Every command from here on assumes you have run `conda activate ghaf` in that
window. If you open a new window, run it again — the prompt shows `(ghaf)` when
you are in the right place.

**Step 2 — install PyTorch.** This is the version the published models were
trained with.

```
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

No GPU? Use `python -m pip install torch==1.12.1 torchvision==0.13.1` instead.
Everything still runs, just slowly.

**Step 3 — install the OpenMMLab framework.**

```
python -m pip install -U openmim
```

```
python -m mim install mmengine==0.10.7 "mmcv>=2.0.0rc4,<2.2.0"
```

```
python -m pip install mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0
```

**Step 4 — install this project.**

```
cd /d D:\ghaf-project\code
```

```
python -m pip install -r requirements.txt
```

```
python -m pip install -e ".[test]"
```

You may see a message that `opencv-python` wants a newer NumPy. It is safe to
ignore — see [section 9](#9-when-something-goes-wrong).

---

## 3. Check that it works

**Step 1 — the test suite.** No GPU, no data, about a minute.

```
python -m pytest tests\ -q
```

Every test should pass. If you see `No module named 'mmengine'`, you are almost
certainly running a different Python than the one you installed into; section 9
explains.

**Step 2 — the models.** This builds all six from their configurations, loads
the weights you were given, checks each file against its published fingerprint,
and runs a prediction through each one.

```
python tools\smoke_test.py --checkpoints D:\ghaf-project\models
```

Look for six rows reading `ok`, `all N matched`, and `+0` in the delta column.
That means the code and the weights describe exactly the same model — nothing
was corrupted or mismatched in transit. Allow a few minutes on a GPU.

**Step 3 — the data.**

```
python tools\check_dataset.py D:\ghaf-project\data\ghaf
```

Three rows reading `ok`, and `dataset looks usable` at the end. Add `--full` to
open every tile rather than a sample of 200 — slower, and worth doing once.

---

## 4. Map trees in a new orthomosaic

This is the main thing the models are for: give it a UAV orthomosaic, get back
a canopy map. The mosaic can be far larger than memory — it is processed in
overlapping windows and blended, so there are no visible seams.

**Start with the small clip.** `Kalba26_sample.tif` is a piece cut out of the
full mosaic; it runs in a minute or two and produces exactly the same kind of
output, so it confirms the installation before you commit to a long run. Only
once that works is the full `Kalba26.tif` worth starting — see *How long the
full mosaic takes* below.

```
python -m ghaf.inference.large_image D:\ghaf-project\models\fastvit-ma36_mask2former\fastvit-ma36_mask2former.py D:\ghaf-project\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth D:\ghaf-project\samples\sample_mosaic.tif --out-mask D:\ghaf-project\output\crowns.tif --out-prob D:\ghaf-project\output\probability.tif --out-polygons D:\ghaf-project\output\crowns.gpkg
```

That is one long command on one line. It writes three files:

| File | What it is | How to use it |
|---|---|---|
| `crowns.tif` | Where the trees are: `1` for crown, `0` for background | Drag into QGIS or ArcGIS; it lands in the right place on the map |
| `probability.tif` | How confident the model is, from 0 to 1 | Useful for choosing a different cut-off than the default 0.5 |
| `crowns.gpkg` | The crowns as polygons you can count, measure and attribute | Open in QGIS; each polygon is one delineated crown |

All three carry the coordinate system of the input mosaic, so they line up with
your other GIS layers without any manual placement.

**Useful adjustments**

| Option | Effect |
|---|---|
| `--threshold 0.6` | Stricter: fewer, more confident crowns. `0.4` is more inclusive |
| `--batch-size 4` | Faster on a GPU with spare memory. Lower it if you run out |
| `--device cpu` | Run without a GPU |
| `--scratch-dir E:\scratch` | Put the temporary working files on a bigger drive |

A run needs about **9 bytes of free disk per pixel** of the mosaic while it
works, released when it finishes. It checks before starting and tells you if
there is not enough, rather than failing part-way through.

**How long the full mosaic takes**

`Kalba26.tif` is 84 072 × 103 691 pixels — 8.7 billion of them. At 9 bytes per
pixel that is about **79 GB of scratch space**, and roughly 33 500 windows to
predict, which is a run of hours rather than minutes on one GPU. Point
`--scratch-dir` at a drive with room to spare, leave it running, and check the
small clip worked first.

**Cutting your own sample**

To try a different part of a mosaic — or to make a quick sample from a new
survey — cut one out:

```
python tools\make_sample.py D:\ghaf-project\samples\Kalba26.tif --output D:\ghaf-project\samples\my_sample.tif --size 8192 --origin 30000 40000
```

`--size` is the clip in pixels (one number for a square). `--origin` is the
top-left corner; leave it out and the clip is taken from the centre. The tool
reports how much of the clip is actual imagery rather than the transparent
border around the survey, so a badly placed window is obvious immediately.

---

## 5. Predict every tile in a split

For producing predictions over the labelled tiles — to compare against the
ground truth, or to hand on as a result set.

```
python tools\predict_split.py D:\ghaf-project\models\fastvit-ma36_mask2former\fastvit-ma36_mask2former.py D:\ghaf-project\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth --data-root D:\ghaf-project\data\ghaf --split testing --out-dir D:\ghaf-project\output\predictions --save-probability
```

One predicted mask per tile, named after the tile it came from, in the same
`0`/`1` encoding as the ground-truth masks — so the two can be compared
directly. `--limit 20` does a quick partial run first, if you want to see the
output before committing to all 767.

---

## 6. Score a model against the labelled test set

This reproduces the published numbers. About three minutes on a GPU.

```
python tools\test.py D:\ghaf-project\models\fastvit-ma36_mask2former\fastvit-ma36_mask2former.py D:\ghaf-project\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth --data-root D:\ghaf-project\data\ghaf
```

The last line reports `mIoU`, `mDice` and `mFscore`, and a table above it gives
the two classes separately. For the leading model expect **mIoU 79.32** and
**mFscore 87.22**; the [model zoo](MODEL_ZOO.md) lists all six, and the
per-class breakdown behind those means.

Swap the two paths to score any of the other five.

---

## 7. Train a model

Only worth doing if you are changing something — the trained models are
already provided. Training the published configuration takes many hours on one
GPU.

```
python tools\train.py D:\ghaf-project\code\configs\ghaf\fastvit-ma36_mask2former.py --data-root D:\ghaf-project\data\ghaf
```

Checkpoints and logs go to `work_dirs\<config-name>\`. The model is scored on
the validation split every 3 500 iterations and the best one is kept as
`best_mIoU_iter_*.pth`.

If a run is interrupted, continue it — optimiser state and all:

```
python tools\train.py D:\ghaf-project\code\configs\ghaf\fastvit-ma36_mask2former.py --data-root D:\ghaf-project\data\ghaf --resume
```

---

## 8. Fine-tune on new labels

If you have labelled tiles from a new site, starting from a released checkpoint
is much cheaper than training from scratch, and usually better.

Prepare the new tiles in the same layout — `training/{images,masks}`,
`validation/{images,masks}`, 1024 × 1024 PNG pairs, masks containing only `0`
and `1` — and check them:

```
python tools\check_dataset.py D:\ghaf-project\data\new-site --full
```

Then fine-tune, with a shorter schedule and a smaller learning rate than a
run from scratch:

```
python tools\train.py D:\ghaf-project\code\configs\ghaf\fastvit-ma36_mask2former.py --data-root D:\ghaf-project\data\new-site --load-from D:\ghaf-project\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth --cfg-options train_cfg.max_iters=4000 optim_wrapper.optimizer.lr=1e-5
```

`--load-from` takes the weights and starts a fresh schedule, which is what
fine-tuning means. `--resume` is the different thing described in section 7.

Score the result the same way as section 6, pointing `--data-root` at the new
site and the checkpoint at your new `work_dirs\...\best_mIoU_iter_*.pth`.

---

## 9. When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `mmseg ... is not installed in conda environment "X"` | You are in the wrong environment, or on a machine where the stack was never installed | The message names the interpreter it asked. `conda activate ghaf`, then run the command again |
| `No module named 'mmengine'`, or `No module named 'pytest'` | A different Python is running than the one you installed into — usually a base Anaconda install ahead of the environment on `PATH` | Check with `python -c "import sys; print(sys.executable)"`. The path should contain `envs\ghaf`. Run `conda activate ghaf`, and always start commands with `python -m` |
| `No module named 'ftfy'` | mmsegmentation imports a tokenizer that needs it, though its own metadata does not say so | `python -m pip install ftfy regex` |
| `RuntimeError: Numpy is not available` | NumPy 2 alongside a PyTorch built for NumPy 1 | `python -m pip install "numpy<2"` |
| `opencv-python ... requires numpy>=2` during install | A message from pip's resolver, not an error | Ignore it. OpenCV works with either |
| `CUDA out of memory` | The GPU ran out of room | Lower `--batch-size`, or add `--device cpu` to run without a GPU |
| `not enough scratch space` | The drive holding temporary files is too small for the mosaic | Point `--scratch-dir` at a larger drive |
| `NO TILES` from `check_dataset.py` | The folders are there but hold no `.png` or `.tif` tiles | The row names what it found instead; usually the tiles are one folder deeper, or in another format |
| `SHA-256 mismatch` from `smoke_test.py` | A checkpoint file does not match the one that was released | The copy is damaged. Copy it again from the original |
| The command does something odd after pasting several lines | The terminal joined them into one line | Paste and run one command at a time |

Anything not listed here: keep the full text of the error. The last few lines
usually name the file and the line that stopped, which is enough to act on.

---

## Where to read more

| Document | Covers |
|---|---|
| [README](../README.md) | The project, the results, the repository layout |
| [Model zoo](MODEL_ZOO.md) | All six models, per-class scores, training settings |
| [Area-wide inference](AREA_WIDE_INFERENCE.md) | How the mosaic is tiled and blended, and how to tune it |
| [Release bundle](RELEASE_BUNDLE.md) | How the models folder is assembled and verified |
