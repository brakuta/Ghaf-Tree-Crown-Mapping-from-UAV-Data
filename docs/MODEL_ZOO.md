# Model zoo

Six models, one protocol. Every entry below was trained on the same training
tiles, validated on the same split for checkpoint selection, and scored on the
same held-out test split with the same metric implementation.

## Results

Held-out test split, `testing/ghaf26`.

| Config | Backbone | Neck | Decode head | Params | mIoU | F1 |
|---|---|---|---|---:|---:|---:|
| `fastvit-ma36_mask2former` | FastViT-MA36 | — | Mask2Former | 62 549 115 | **79.32** | **87.22** |
| `poolformer-s36_fpn` | PoolFormer-S36 | FPN | FPNHead | 34 600 137 | 78.65 | 86.72 |
| `dpn98_fpn` | DPN-98 | FPN | FPNHead | 65 346 639 | 78.19 | 86.35 |
| `convnext-small_upernet` | ConvNeXt-S | — | UPerNet | 81 776 049 | 78.02 | 86.20 |
| `resnet-50_mask2former` | ResNet-50 | — | Mask2Former | 44 056 504 | 77.69 | 85.98 |
| `efficientnet-b3_fpn` | EfficientNet-B3 | FPN | FPNHead | 13 734 524 | 70.77 | 80.29 |

Parameter counts are summed over every tensor in each model's checkpoint,
so they include the normalisation running statistics that `parameters()`
leaves out. `tools/smoke_test.py` prints both totals for a built model.

mIoU and F1 are means over both classes, background included -- the
convention of mmsegmentation's `IoUMetric`, which is what produced them. For
the leading model the two classes read:

| Class | IoU | Acc | Dice | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| background | 98.17 | 99.25 | 99.08 | 99.08 | 98.91 | 99.25 |
| ghaf | 60.47 | 72.11 | 75.37 | 75.37 | 78.93 | 72.11 |

Overall accuracy is 98.22.

## Reproducing a score

```bash
python tools/test.py configs/ghaf/fastvit-ma36_mask2former.py \
    <checkpoints>/fastvit-ma36_mask2former/best_mIoU_iter_3500.pth \
    --data-root <data>/ghaf
```

That command, on a clean clone with the released checkpoint, returns the
79.32 / 87.22 in the table above.

## Training settings

Shared by all six: 1024 x 1024 input, batch size 2, `PolyLR` with power 0.9,
horizontal/vertical flipping as the only augmentation, validation every 3 500
iterations with the best-mIoU checkpoint retained.

| Config | Optimiser | LR | Weight decay | Precision |
|---|---|---:|---:|---|
| `fastvit-ma36_mask2former` | AdamW | 1e-4 | 0.05 | fp32 |
| `resnet-50_mask2former` | AdamW | 1e-4 | 0.05 | fp32 |
| `convnext-small_upernet` | AdamW | 1e-4 | 0.05 | mixed |
| `dpn98_fpn` | AdamW | 1e-4 | 0.05 | fp32 |
| `poolformer-s36_fpn` | AdamW | 2e-4 | 1e-4 | mixed |
| `efficientnet-b3_fpn` | SGD | 0.01 | 5e-4 | fp32 |

All six backbones start from ImageNet weights, declared in each config as
`init_cfg=dict(type='Pretrained', checkpoint=...)`:

| Backbone | Weights |
|---|---|
| FastViT-MA36 | Apple's `fastvit_ma36.pth.tar` |
| ResNet-50 | `torchvision://resnet50` |
| ConvNeXt-S | `convnext-small_3rdparty_32xb128-noema_in1k` (mmpretrain) |
| PoolFormer-S36 | `poolformer-s36_3rdparty_32xb128_in1k` (mmpretrain) |
| EfficientNet-B3 | `efficientnet-b3_3rdparty_8xb32-aa_in1k` (mmpretrain) |
| DPN-98 | `dpn98.mx_in1k`, fetched by timm |

`init_cfg` is the only keyword mmsegmentation acts on. Anything else is
absorbed by a backbone's `**kwargs` and silently ignored, so
`tests/test_configs.py` asserts that every config declares its weights this
way and that none uses a keyword that would be dropped.

## Feature widths

A neck or decode head must consume exactly the widths its backbone emits.
These are pinned by `tests/test_configs.py`.

| Backbone | Strides | Widths |
|---|---|---|
| FastViT-MA36 | 4, 8, 16, 32 | 76, 152, 304, 608 |
| ResNet-50 | 4, 8, 16, 32 | 256, 512, 1024, 2048 |
| ConvNeXt-S | 4, 8, 16, 32 | 96, 192, 384, 768 |
| DPN-98 | 2, 4, 8, 16, 32 | 96, 336, 768, 1728, 2688 |
| PoolFormer-S36 | 4, 8, 16, 32 | 64, 128, 320, 512 |
| EfficientNet-B3 | 8, 16, 32 | 48, 136, 384 |

DPN-98 is the only backbone here that exposes five scales, which is why its FPN
takes five inputs where the others take three or four.

## Checkpoints

The trained weights are not distributed in this repository. They are shared
alongside the dataset, and are available from the corresponding author on
reasonable request.

Every released checkpoint's digest, size, parameter count and scores are
recorded in `ghaf/release.py`, so a copy can be checked against what was
published however it reached you:

```bash
python tools/smoke_test.py --checkpoints /path/to/checkpoints
```

For the file alone, `sha256sum <file>` on Linux or macOS, or
`certutil -hashfile <file> SHA256` on Windows, against the digests above.

| Model | File | SHA-256 | Bytes |
|---|---|---|---:|
| FastViT-MA36 + Mask2Former | `best_mIoU_iter_3500.pth` | `f26cd5257b55058f81c52349d51c888a382ca54a924da550bd0a711bcfafa84a` | 252 650 755 |
| PoolFormer-S36 + FPN | `iter_10200.pth` | `59683e3788548494f50206dffe7b3ce2a91610a7e1039e4e7f5999401b0156e0` | 416 437 419 |
| DPN-98 + FPN | `best_mIoU_iter_14000.pth` | `e292f2262f132f57f0a81e9dc8be169b7a2696d397ab0a5b1a898dff28c964cf` | 263 385 401 |
| ConvNeXt-S + UPerNet | `iter_14000.pth` | `8435410e2514054f53a68302b2b70c60b86dfdf1ab50ec385e6664f65d7b1036` | 983 511 196 |
| ResNet-50 + Mask2Former | `best_mIoU_iter_38500.pth` | `5a79f618902be032d8a38c27b18c3e5ed7bd0605297b272626ae0cbd56fdf5b2` | 196 545 149 |
| EfficientNet-B3 + FPN | `iter_6800.pth` | `9d6131e21a1ec5f38f36adf8c570d0c09fdd8d067e73ef57509459b970a5aa60` | 108 104 299 |

Arrange them as `checkpoints/<config-name>/<file>.pth` to match the commands in
the README, or build a self-contained bundle with a config beside each
checkpoint — see [`RELEASE_BUNDLE.md`](RELEASE_BUNDLE.md).

These digests come from `ghaf/release.py`, the single source of truth that
`tools/smoke_test.py` and `tools/export_release.py` also read.

## Reproducing

```bash
python tools/test.py configs/ghaf/<config>.py checkpoints/<config>/<file>.pth
```

Evaluating a published checkpoint reproduces its published score exactly.
Retraining will land close but not identical: the configs do not fix a random
seed by default, so runs vary in the third significant figure. For
bit-reproducible training, uncomment in `configs/_base_/ghaf.py`:

```python
randomness = dict(seed=0, deterministic=True)
```
