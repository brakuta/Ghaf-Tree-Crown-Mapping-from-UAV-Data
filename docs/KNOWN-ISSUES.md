# Known issues and caveats

Findings from auditing the checkpoints and logs. Each is stated with the
evidence that establishes it, so a reviewer can check it independently. None
of them invalidates the benchmark; several must be disclosed in the paper.

## 1. The proposed model was warm-started; the baselines were not

The published FastViT checkpoint records, in its own embedded config:

```
load_from = '...\fast_all_data\iter_7000.pth'      time = 20250227_153109
```

Tracing the chain back through the archived logs:

```
_fastvit_23_25/best_mIoU_iter_4000.pth
  └─ run 20250227_120057   saves iter 3500, 7000
       └─ run 20250227_142643   ← warm-started from iter_7000
            └─ best_mIoU_iter_3500.pth   ← the published checkpoint
```

So "3,500 iterations" is roughly **14,500 cumulative**. All five baselines have
`load_from = None` and start from ImageNet weights only.

**Assessment.** The cumulative total lands close to ConvNeXt and DPN-98 (14,000
each) and well under ResNet-50 (38,500), so this is not a large-budget
advantage. But citing 3,500 without the chain would misrepresent it, and a
reviewer who found the `load_from` independently would be entitled to be
harsh. State the cumulative figure.

## 2. The proposed model was selected on different labels

Checkpoint selection (`save_best='mIoU'`) ran against:

| Model | validation labels |
|---|---|
| FastViT-MA36 | `validation/masks_refined` |
| all five baselines | `validation/masks` |

The two label sets differ on **199 of 869 tiles (4.4 %)**.

**Assessment.** Test-time evaluation used common ground truth
(`testing/ghaf26/masks`) for every model, so the reported comparison is fair.
Only *which iteration was chosen* differed. Disclose it; the configs in this
repository use `validation/masks` for all six so that new runs are directly
comparable.

## 3. Mixed precision was not applied uniformly

`AmpOptimWrapper` was used for **ConvNeXt-S** and **PoolFormer-S36** only; the
other four trained in fp32. Mixed precision can move segmentation metrics by a
few hundredths — the same order as the FastViT-to-PoolFormer margin (0.67).
Worth one sentence in the methods.

## 4. The optimiser was not uniform either

| Model | optimiser | lr | weight decay |
|---|---|---|---|
| FastViT, ResNet-50, ConvNeXt, DPN-98 | AdamW | 1e-4 | 0.05 |
| PoolFormer-S36 | AdamW | 2e-4 | 1e-4 |
| **EfficientNet-B3** | **SGD** | **0.01** | **5e-4** |

EfficientNet is the only model trained with SGD, at a learning rate 100× the
AdamW setting used elsewhere, and it is also the weakest result by 7 points.
Its poor showing is at least partly a tuning artefact and should not be
presented as evidence about the architecture.

## 5. Augmentation is flip-only

The training pipeline is `LoadImageFromFile → LoadAnnotations → RandomFlip(0.5)
→ PackSegInputs`. There is no scale jitter, no random crop, no photometric
distortion. If §2.10 describes a richer pipeline, it describes something that
did not run.

## 6. No random seed was set

`randomness` is unset in every archived config, so exact numerical reproduction
is not possible — reruns will differ in the third significant figure. The
configs here ship a commented-out `randomness = dict(seed=0, deterministic=True)`
for anyone who wants determinism going forward.

## 7. GFLOPs, if reported, are unreliable

`tools/analysis_tools/get_flops.py` in the original tree had upstream's guard
commented out:

```python
# if cfg.model.decode_head.type in ['MaskFormerHead', 'Mask2FormerHead']:
#     raise NotImplementedError('MaskFormer and Mask2Former are not supported yet.')
```

Upstream refuses to compute FLOPs for these heads because the result is wrong;
disabling the check does not make it right. The default input shape was also
changed from 2048×1024 to 1024×1024. **Parameter counts are unaffected** — those
were measured directly from checkpoint tensors and are trustworthy. If Table 1
reports GFLOPs for the Mask2Former models, drop the column or recompute it with
a method that supports mask-classification heads.

## 8. Two source files could not rebuild their checkpoints

Both were left mid-edit when the author moved to unrelated work:

- **`coatnet.py`** built `nextvit_base`; the `dpn98` line was commented out.
  Fixed in `ghaf/models/dpn.py`; original kept at
  `ghaf/models/_original/coatnet_as_recovered.py`.
- **`tools/train.py`** had `fastvit` and `coatnet` imports commented out, so
  training the paper's own models raised
  `KeyError: 'fastvit_small is not in the model registry'`. Fixed by
  `custom_imports` in the configs.

Anyone who had cloned the original tree and run it would have silently trained
a different architecture. This is the strongest argument for the audit trail
in `docs/PROVENANCE.md`.

## 9. An unreported ablation quantifies the annotation workflow

`fastvit_olddata` — the same architecture on pre-active-learning data — peaks at
**72.09 %** against 81–83 % for the refined data. That is a controlled ~9-point
measurement of what the §2.5 annotation workflow bought. The manuscript
describes the workflow but never quantifies it. This is a result worth
claiming, not a problem.
