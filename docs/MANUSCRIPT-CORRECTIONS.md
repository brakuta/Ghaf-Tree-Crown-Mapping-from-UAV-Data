# Corrections required in the manuscript

Every item is verified against the checkpoints or the evaluation logs. Sources
are named so each can be re-checked independently.

## §2.8.2 — model names

Five of six are wrong. Verified from each checkpoint's `state_dict` tensor
shapes and from the `type=` strings in the archived configs.

| Manuscript | Correct | Evidence |
|---|---|---|
| FastViT-SA12 | **FastViT-MA36** | `layers=[6,6,18,6]`, `embed_dims=[76,152,304,608]`; the source file names this exact combination `fastvit_ma36` in a commented-out definition directly above `fastvit_small` |
| ConvNeXt-T | **ConvNeXt-S** | config: `arch='small'` |
| PoolFormer-S12 | **PoolFormer-S36** | config: `arch='s36'` |
| EfficientNet-B0 | **EfficientNet-B3** | config: `arch='b3'` |
| DPN-92 | **DPN-98** | stem 96 ch (DPN-92 uses 64); bottleneck widths 160/320/640/1280 from `k_r=160` (DPN-92 gives 96/192/384/768); FPN `in_channels=[96,336,768,1728,2688]` are timm's `dpn98` feature channels |
| ResNet-50 | ResNet-50 ✓ | `type='ResNet', depth=50` — correct as written |

## §2.10 — training protocol

The manuscript describes one protocol; three were used.

- **Decode heads**: Mask2Former (FastViT, ResNet-50), UPerNet (ConvNeXt),
  FPN (DPN-98, PoolFormer, EfficientNet).
- **Optimisers**: AdamW 1e-4/0.05 for four models; AdamW 2e-4/1e-4 for
  PoolFormer; **SGD 0.01/5e-4** for EfficientNet.
- **Mixed precision**: ConvNeXt and PoolFormer only.
- **Crop size**: **1024×1024**, not 512×512, despite every directory name.
- **Augmentation**: horizontal/vertical flip only.
- **Schedule**: `max_iters=160000` with `PolyLR(power=0.9)` in all six configs,
  but no run reached it — the reported checkpoints are at 3,500–38,500
  iterations, selected on validation mIoU.

## §2.7 — dataset sizes

Counted from the tile directories:

| Split | Tiles |
|---|---:|
| training | **7 005** |
| validation | **869** |
| testing (`ghaf26`) | **767** |

The training figure confirms the "(7005)" already in the text; the other two
fill placeholders.

## Tables 1–2 and the abstract — three transcription errors

From the 18 March 2025 evaluation logs, all `Iter(test) [767/767]`:

| Model | Field | Manuscript | Logs |
|---|---|---:|---:|
| PoolFormer-S36 | mIoU | 78.45 | **78.65** |
| PoolFormer-S36 | F1 | 86.62 | **86.72** |
| FastViT-MA36 | F1 | 87.30 | **87.22** |

The two PoolFormer values also appear in the abstract. Correcting them narrows
the proposed model's margin over the runner-up from 0.87 to **0.67** points —
still a win, but the text should not overstate it.

Parameter counts in Table 1 are all within 0.65 % of the values measured from
the checkpoints, so that column is sound.

## Additions the paper should make

1. **Area-wide inference** (title claim, currently undescribed): sliding
   1024 px window, 512 px overlap, **Gaussian-weighted blending** (σ = 0.4 in
   half-tile units), then georeferenced raster and polygon output. Implemented
   in `ghaf/inference/large_image.py`.
2. **Disclosures** from `docs/KNOWN-ISSUES.md`: FastViT's warm start (§1), its
   different validation labels (§2), non-uniform AMP (§3) and optimisers (§4).
3. **The annotation ablation** (§9): ~9 mIoU points from the active-learning
   workflow, measured and currently unclaimed.
