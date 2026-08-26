# Building the shareable bundle

The trained weights and the dataset are distributed alongside the code rather
than inside it. `tools/export_release.py` assembles the weights half into
self-contained per-model folders, laid out the way mmsegmentation lays out a
working directory.

## Build it

```bash
python tools/export_release.py \
    --checkpoints /path/to/checkpoints \
    --output      /path/to/ghaf-release
```

Add `--dry-run` to verify every checkpoint and report what would be written
without touching the destination. `--only <key> [<key> ...]` limits the run to
particular models.

Checkpoints are located either at `<checkpoints>/<model-key>/<file>.pth` or
anywhere beneath the folder by file name, so a directory that has not been
reorganised still works.

## What it produces

```
ghaf-release/
├── README.md                          for whoever receives the bundle
├── MODELS.json                        index: paths, digests, scores
├── fastvit-ma36_mask2former/
│   ├── fastvit-ma36_mask2former.py    resolved config
│   ├── best_mIoU_iter_3500.pth        weights
│   └── metadata.json                  digest, size, parameters, scores
├── poolformer-s36_fpn/
├── dpn98_fpn/
├── convnext-small_upernet/
├── resnet-50_mask2former/
└── efficientnet-b3_fpn/
```

Total is about 2.07 GB across the six models.

## Why the config is resolved

The file written into each folder is **flattened**: mmengine's `_base_`
inheritance is expanded so the config carries the complete recipe — model,
dataset, pipeline, schedule, runtime — with nothing to import. That is the same
form mmsegmentation dumps to `<work_dir>/vis_data/config.py`, and it is what
lets a folder be moved, archived or handed on without the rest of the
repository.

A folder can be used directly:

```bash
python tools/test.py \
    ghaf-release/dpn98_fpn/dpn98_fpn.py \
    ghaf-release/dpn98_fpn/best_mIoU_iter_14000.pth
```

The one thing a folder does not carry is the two custom backbones, which are
Python rather than configuration. Its config declares them through
`custom_imports`, so the `ghaf` package must be installed — `pip install -e .`
from the source repository.

## Integrity

Every checkpoint is hashed against its published SHA-256 **before** the copy
and again **after**. A source file that does not match is reported and skipped;
a copy that does not match is deleted and the run fails. A completed bundle
cannot contain a silently corrupted checkpoint.

The published digests live in `ghaf/release.py`, which is also what
`tools/smoke_test.py` checks built models against and what
`tests/test_release.py` keeps in step with the configs. One source of truth, so
a model cannot be documented one way and shipped another.

Recipients verify with:

```bash
sha256sum <model>/<checkpoint>.pth           # Linux, macOS
certutil -hashfile <checkpoint>.pth SHA256   # Windows
```

## The dataset half

The tiles are copied separately — they are far larger than the weights and do
not change. Alongside the bundle:

```
ghaf-project/
├── ghaf-release/                      built by the command above, ~2.07 GB
└── data/ghaf/                         ~60 GB
    ├── training/{images,masks}/       7 005 tiles
    ├── validation/{images,masks}/       869 tiles
    └── testing/ghaf26/{images,masks}/   767 tiles
```

Point a config at the tiles with `--cfg-options data_root=/path/to/data/ghaf`,
or place them at `data/ghaf` relative to the working directory, which is the
default.
