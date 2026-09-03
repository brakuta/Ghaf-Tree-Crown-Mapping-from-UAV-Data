#!/usr/bin/env python3
"""Chapters 1 and 2 of the technical manual.

Every value quoted here is traceable to docs/handover/FACTS.yml, to a file in
the repository, or to a command run while the manual was written. Where a
value was never established the text says so; it does not supply a plausible
one.
"""

from typeset import (bullets, callout, chapter, code, glue, para, section,
                     sub, table)


def story():
    s = []
    s += chapter_1()
    s += chapter_2()
    return s


# ==========================================================================
def chapter_1():
    s = chapter(
        'What this system does, and what you were given',
        'The system takes a UAV orthomosaic and returns the Ghaf crowns in '
        'it as GIS polygons. This chapter says what the models were trained '
        'to do, what arrived with them, and what the system cannot do — the '
        'last of which is the part worth reading twice.')

    s.append(glue(section('The task, exactly'), None))
    s.append(para(
        'Two classes and nothing else: `0` for background and `1` for a Ghaf '
        'crown. The models are semantic segmentors, so they label pixels, not '
        'objects; the crown polygons in the output are traced from the '
        'labelled pixels afterwards. Training tiles are 1024 × 1024 pixel PNG '
        'pairs, an image and a mask sharing a filename stem, and the '
        'inference window is the same 1024 pixels for the same reason.'))
    s.append(para(
        'A mask is a label image, not a picture. Opened in an image viewer it '
        'is almost black, because its two values are 0 and 1 out of a '
        'possible 255. That is correct, and it is the first thing that makes '
        'a newcomer think the data is broken.'))
    s.extend(table([
        ['Property', 'Value', 'Where it is fixed'],
        ['Classes', '`background`, `ghaf`', '`ghaf/datasets.py`'],
        ['Mask encoding', '0 background, 1 ghaf, single band',
         '`ghaf/datasets.py`'],
        ['`reduce_zero_label`', '`False` — background is supervised, not ignored',
         '`ghaf/datasets.py`'],
        ['Tile and crop size', '1024 × 1024 px', '`configs/_base_/ghaf.py`'],
        ['Tile format', 'PNG, image and mask sharing a stem',
         '`ghaf/datasets.py`'],
        ['Coordinate system', 'EPSG:32640 (UTM zone 40 N)',
         'the imagery itself'],
    ], widths=[100, 190, None], size=8.6,
        caption='The dataset contract. Changing any row of it invalidates '
                'every published score.'))

    s.append(glue(section('The six models'), None))
    s.append(para(
        'All six share one dataset, one augmentation pipeline, one schedule '
        'and one metric implementation, so the differences between them are '
        'attributable to the encoder and the decode head. Scores are on the '
        'held-out `testing/ghaf26` split. Use FastViT-MA36 unless you have a '
        'reason not to; the other five are the comparison the paper reports, '
        'and every procedure in this manual works with any of them by '
        'substituting two paths.'))
    s.extend(table([
        ['Model key', 'Backbone', 'Head', 'Params', 'mIoU', 'F1'],
        ['`fastvit-ma36_mask2former`', 'FastViT-MA36', 'Mask2Former',
         '62,549,115', '**79.32**', '**87.22**'],
        ['`poolformer-s36_fpn`', 'PoolFormer-S36', 'FPNHead',
         '34,600,137', '78.65', '86.72'],
        ['`dpn98_fpn`', 'DPN-98', 'FPNHead', '65,346,639', '78.19', '86.35'],
        ['`convnext-small_upernet`', 'ConvNeXt-S', 'UPerHead',
         '81,776,049', '78.02', '86.20'],
        ['`resnet-50_mask2former`', 'ResNet-50', 'Mask2Former',
         '44,056,504', '77.69', '85.98'],
        ['`efficientnet-b3_fpn`', 'EfficientNet-B3', 'FPNHead',
         '13,734,524', '70.77', '80.29'],
    ], widths=[132, 78, 66, 62, 40, None], size=8.4,
        caption='Parameter counts are exact sums over each `state_dict`, '
                'measured from the checkpoint rather than transcribed '
                '(`ghaf/release.py`).'))
    s.append(para(
        'Only FastViT-MA36 has a recorded validation score, 80.14 mIoU and '
        '87.87 F1. The validation figures for the other five were never '
        'established, and this manual does not estimate them.'))

    s.append(glue(section('What arrived, and what did not'), None))
    s.append(para(
        'The code is public. The weights, the imagery and the labelled tiles '
        'are not: they travel separately, and the bundle below is what a '
        'recipient is sent. Sizes are those of the bundle as it was built, '
        'from commit `fe081a7`.'))
    s.extend(table([
        ['Part', 'Size', 'Contents'],
        ['`code/`', '63 files',
         'Exactly what `git ls-files` reported at that commit. No working '
         'files, no local checkpoints'],
        ['`models/`', '2.22 GB',
         'Six folders, each a resolved configuration beside its weights and a '
         '`metadata.json`'],
        ['`init-weights/`', '0.90 GB',
         'ImageNet initialisation weights for the six backbones. Needed only '
         'to train without internet access; not used for inference'],
        ['`data/ghaf/`', '15.64 GB',
         '19,583 files. The three named splits only — the working tree it was '
         'cut from held 49,490 files more'],
        ['`samples/`', '0.17 GB',
         'Two files: the 8192 × 8192 clip and its overview. **The full '
         'orthomosaic is not in the bundle**'],
        ['`predictions/testing/`', '768 files',
         '767 predicted masks for the test split, and one `summary.json`'],
        ['`MANIFEST.json`', '—',
         'Every part, its size, its file count, and the commit the code came '
         'from. This is how a result is traced back to a version'],
    ], widths=[92, 52, None], size=8.4,
        caption='The handover bundle, 18.93 GB in total.'))

    s.append(glue(section('What it cannot do'), None))
    s.append(para(
        'Four limits, stated because each one has cost somebody a day.'))
    s.extend(bullets([
        '**Two classes.** The models separate Ghaf from not-Ghaf. They do not '
        'distinguish one tree species from another, and a second species in '
        'the imagery will be labelled background or crown according to how '
        'much it resembles a Ghaf, with no signal that it happened.',
        '**Touching crowns merge.** Semantic segmentation labels pixels, so '
        'two crowns whose canopies overlap are one connected region and '
        'become one polygon. The crown count is therefore a lower bound in '
        'dense stands. It was never quantified on this imagery.',
        '**8-bit RGB only.** The pipeline reads three bands as red, green and '
        'blue. Other band orders need `--bands`; 16-bit input is rejected '
        'outright, which at least fails loudly.',
        '**Runs are not reproducible.** `randomness` is commented out in '
        '`configs/_base_/ghaf.py`, so no seed is set. Two training runs of the '
        'same configuration will not agree exactly, and the seed mmengine '
        'chose exists only in the run log.',
    ]))
    s.extend(callout('The full survey mosaic has never been run end to end', [
        'Every timing in this manual for area-wide inference comes from the '
        '8192 × 8192 clip. The full `Kalba26.tif` is 84,072 × 103,691 pixels, '
        'which is 8.7 billion; the 78.5 GB of scratch space and the 33,128 '
        'windows quoted in chapter 5 are arithmetic on that size, not a '
        'measurement. Nobody has watched it finish, and nothing in the '
        'pipeline reports a partial result — a run that dies at window 30,000 '
        'leaves scratch files and no output.',
    ]))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'You know which of the six models you are using, and why.',
        'You know that a mask looks black and that this is correct.',
        'You have `MANIFEST.json`, so any result can be traced to a commit.',
        'You have read the four limits above before promising anybody a tree '
        'count.',
    ]))
    return s


# ==========================================================================
def chapter_2():
    s = chapter(
        'The repository, folder by folder',
        'Sixty-seven tracked files in four directories. This chapter is the '
        'map: what each file is for, and which of them you will ever open. '
        'Read it when you need to find something, not before.')

    s.append(glue(section('`ghaf/` — the library'), None))
    s.append(para(
        'The importable package. Nothing here is run directly except '
        '`ghaf.inference.large_image`, which is a module rather than a script '
        'in `tools/` because it is part of the library that other code calls.'))
    s.extend(table([
        ['File', 'What it holds'],
        ['`config.py`',
         'Points a configuration at a dataset root, and disables ImageNet '
         'initialisation when a checkpoint is being loaded anyway'],
        ['`datasets.py`',
         '`GhafDataset` — the two-class tile dataset, and the guard that '
         'refuses `reduce_zero_label=True`'],
        ['`splits.py`',
         'Where each split lives, declared once. `check_dataset`, '
         '`predict_split` and `build_handover` all read it, so the layout '
         'cannot drift between them'],
        ['`environment.py`',
         'Confirms the framework is installed in the interpreter that is '
         'running, and quietens the repeated framework warnings'],
        ['`init_weights.py`',
         'Finds the ImageNet initialisation weights that ship with the bundle'],
        ['`release.py`',
         'The published models: digests, byte sizes, parameter counts and '
         'scores. The single source of truth that `smoke_test` and '
         '`export_release` both check against'],
        ['`models/fastvit.py`, `models/dpn.py`',
         'The two backbones that mmpretrain does not supply. '
         '`models/modules/` holds the MobileOne and RepLKNet blocks FastViT '
         'is built from'],
        ['`inference/tiling.py`',
         'Window planning and Gaussian blending, in pure NumPy. No mmseg, no '
         'torch, no rasterio — which is why the geometry can be tested '
         'without a GPU'],
        ['`inference/large_image.py`',
         'Orthomosaic inference: windowed reads, memory-mapped accumulators, '
         'georeferenced output'],
    ], widths=[118, None], size=8.5))

    s.append(glue(section('`configs/` — the recipes'), None))
    s.append(para(
        '`_base_/ghaf.py` holds the dataset, the augmentation pipeline, the '
        'schedule and the runtime. The six files in `configs/ghaf/` inherit '
        'it and differ only in backbone, neck and decode head. That is what '
        'makes the comparison in chapter 1 a comparison of architectures '
        'rather than of training budgets.'))
    s.extend(callout('Editing a base config changes results already published', [
        'Every one of the six models inherits `configs/_base_/ghaf.py`. A '
        'change to the crop size, the evaluator or the schedule in that file '
        'silently redefines what the published scores mean, and nothing in '
        'the pipeline will warn you. Copy it to a new file and edit the copy.',
    ]))

    s.append(glue(section('`tools/` — the programs you run'), None))
    s.extend(table([
        ['Program', 'What it does', 'Chapter'],
        ['`check_dataset.py`',
         'Verifies pairing, sizes and mask values before a tree is used', '4'],
        ['`smoke_test.py`',
         'Builds all six models, verifies each checkpoint against its '
         'published digest, runs one prediction through each', '3'],
        ['`predict_folder.py`',
         'Maps every image in a folder, one GeoPackage of crowns per image',
         '6'],
        ['`predict_split.py`',
         'One predicted mask per tile for a labelled split', '8'],
        ['`test.py`', 'Scores a model against a labelled split', '8'],
        ['`train.py`', 'Trains a model, or fine-tunes one', '9, 10'],
        ['`make_sample.py`',
         'Cuts a small georeferenced clip out of a large mosaic', '5'],
        ['`export_release.py`',
         'Assembles the models folder for sharing, verifying every checkpoint '
         'before and after the copy', '11'],
        ['`fetch_init_weights.py`',
         'Collects the ImageNet weights once, while online', '9'],
        ['`build_handover.py`',
         'Assembles the whole bundle described in chapter 1', '11'],
    ], widths=[112, None, 44], size=8.5))

    s.append(glue(section('`tests/` and `docs/`'), None))
    s.append(para(
        'Seventeen test files, 325 tests, one skipped. They run without a '
        'GPU, without mmcv and without the dataset, which is what makes them '
        'worth running on a machine that has just been set up: a pass proves '
        'the code is intact even before the weights arrive. `docs/` holds '
        'this manual and its sources, the quick reference, and four reference '
        'documents inherited from the repository.'))

    s.append(glue(section('Where results are written'), None))
    s.append(para(
        'No program writes into `data/`, `models/` or `code/`. Every one of '
        'them takes an output path on the command line, and the examples '
        'throughout this manual write to `output/`. Training is the exception '
        'worth knowing: `train.py` writes to `work_dirs/<config-name>/` '
        'relative to the working directory, and that is where the loss curves '
        'and the seed of a run survive.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'You can name the file that defines the dataset layout.',
        'You know that the six configs differ only in architecture.',
        'You know not to edit anything under `configs/_base_/`.',
        'You know where a training run leaves its logs.',
    ]))
    return s
