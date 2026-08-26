# Provenance

Every published result traced from its config, through the checkpoint that
produced it, to the evaluation log that reported it. Nothing here is quoted
from the manuscript; all of it was measured from the artefacts.

## How this was reconstructed

The original author left before the manuscript was finalised. Rather than
trust the file names — which turned out to be wrong in six different ways —
each model was identified by measurement:

1. **Parameter counts** summed over every tensor in the checkpoint's
   `state_dict`, then compared with Table 1.
2. **Per-stage channel widths** read from 4-D convolution kernels, which
   identify an architecture variant uniquely.
3. **The embedded config**: mmengine stores the fully resolved config in
   `meta['cfg']`, so each checkpoint carries the exact recipe that made it,
   including `load_from`.
4. **Evaluation logs** matched to checkpoints via the `load_from` recorded in
   each evaluation run.

`tools/provenance/` holds the scripts used; they run on any machine with only
the standard library plus torch.

## The chain

| Config | Checkpoint | Training run | Evaluated | mIoU | F1 |
|---|---|---|---|---:|---:|
| `fastvit-ma36_mask2former` | `fastvit-ma36_mask2former` | `20250227_142643` | `20250318_103534` | 79.32 | 87.22 |
| `poolformer-s36_fpn` | `poolformer-s36_fpn` | `20250309_005445` | `20250318_111018` | 78.65 | 86.72 |
| `dpn98_fpn` | `coatnet-small_fpn` | `20250313_125512` | `20250318_110202` | 78.19 | 86.35 |
| `convnext-small_upernet` | `convnext-small_upernet` | `20250306_142034` | `20250318_104309` | 78.02 | 86.20 |
| `resnet-50_mask2former` | `resnet-50_mask2former` | `20250311_230401` | `20250318_104913` | 77.69 | 85.98 |
| `efficientnet-b3_fpn` | `efficientnet-b3_fpn` | `20250309_140341` | `20250318_112006` | 70.77 | 80.29 |

## Parameter counts

Summed over each checkpoint's tensors and compared with Table 1 of the manuscript.

| Model | Measured | Table 1 | Difference |
|---|---:|---:|---:|
| `fastvit-ma36_mask2former` | 62,549,115 | 62.454 M | +0.15 % |
| `poolformer-s36_fpn` | 34,600,137 | 34.598 M | +0.01 % |
| `dpn98_fpn` | 65,346,639 | 65.185 M | +0.25 % |
| `convnext-small_upernet` | 81,776,049 | 81.763 M | +0.02 % |
| `resnet-50_mask2former` | 44,056,504 | 44.003 M | +0.12 % |
| `efficientnet-b3_fpn` | 13,734,524 | 13.646 M | +0.65 % |

All six agree to within 0.65 %, so Table 1's parameter column is sound.

## Checkpoint hashes

SHA-256 over the `.pth` files as copied from the workstation. Verify with
`sha256sum <file>` (or `certutil -hashfile <file> SHA256` on Windows).

| Checkpoint | SHA-256 | Bytes |
|---|---|---:|
| `coatnet-small_fpn/best_mIoU_iter_14000.pth` | `e292f2262f132f57f0a81e9dc8be169b7a2696d397ab0a5b1a898dff28c964cf` | 263,385,401 |
| `convnext-small_upernet/iter_14000.pth` | `8435410e2514054f53a68302b2b70c60b86dfdf1ab50ec385e6664f65d7b1036` | 983,511,196 |
| `efficientnet-b3_fpn/iter_6800.pth` | `9d6131e21a1ec5f38f36adf8c570d0c09fdd8d067e73ef57509459b970a5aa60` | 108,104,299 |
| `fastvit-ma36_mask2former/best_mIoU_iter_3500.pth` | `f26cd5257b55058f81c52349d51c888a382ca54a924da550bd0a711bcfafa84a` | 252,650,755 |
| `poolformer-s36_fpn/iter_10200.pth` | `59683e3788548494f50206dffe7b3ce2a91610a7e1039e4e7f5999401b0156e0` | 416,437,419 |
| `resnet-50_mask2former/best_mIoU_iter_38500.pth` | `5a79f618902be032d8a38c27b18c3e5ed7bd0605297b272626ae0cbd56fdf5b2` | 196,545,149 |

## Directory names versus reality

Recorded so that anyone reading the archived logs can navigate them.

| Working directory | What it contains |
|---|---|
| `mask2former_swin-t_…fast_all_data` | FastViT-MA36 + Mask2Former (no Swin) |
| `dual_path` | DPN-98 + FPN |
| `efficientefficient` | EfficientNet-B3 + FPN |
| `…_ade20k-512x512` | the Ghaf dataset at 1024×1024 (no ADE20K, not 512) |
| `fastvit_olddata` | FastViT on pre-active-learning labels (ablation) |

## What is archived here

- `provenance/` — training and evaluation logs, and the resolved config for
  every run that contributed to the paper
- `provenance/checkpoint-reports/` — for each checkpoint: recovered config,
  parameter shapes, metric history
- `environment/` — the conda environment and source-built dependency commits
  as captured from the workstation
- Tag `archive/full-source-fork` — the complete 1 154-file mmsegmentation
  fork exactly as recovered, before restructuring

The checkpoints themselves (2.07 GB) are not in git. They are held with the
project data; the hashes above identify them unambiguously.

