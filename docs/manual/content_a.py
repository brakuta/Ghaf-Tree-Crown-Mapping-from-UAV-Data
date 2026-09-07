#!/usr/bin/env python3
"""The first half of the technical manual: what to read before running anything.

Every value quoted here is traceable to docs/handover/FACTS.yml, to a file in
the repository, or to a command run while the manual was written. Where a
value was never established the text says so; it does not supply a plausible
one.
"""

import typeset as T
from order import Ch, ch, emitted
from typeset import bullets, callout, chapter, code, glue, para, section, sub, table


def story():
    s = []
    for key, build in (('what', what),
                       ('repository', repository),
                       ('installing', installing),
                       ('verifying', verifying),
                       ('data', data),
                       ('configuration', configuration)):
        s += build()
        emitted(T, key)
    return s


# ==========================================================================
def what():
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
        'from commit `71a07db`.'))
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
        f'windows quoted in {ch("data")} are arithmetic on that size, not a '
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
def repository():
    s = chapter(
        'The repository, folder by folder',
        'Fifty-four files in four directories — '
        '`git ls-files ghaf configs tools tests` counts them. This chapter '
        'is the map: what each file is for, and which of them you will ever '
        'open. Read it when you need to find something, not before.')

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
        ['`paths.py`',
         'One order for a folder listing on every operating system. Windows '
         'sorts file names without regard to case and Linux does not, so '
         'sorting the platform\'s way would give `--limit` a different set '
         'of images on each machine'],
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
        f'makes the comparison in {ch("what")} a comparison of architectures '
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
         'Verifies pairing, sizes and mask values before a tree is used',
         Ch('verifying')],
        ['`smoke_test.py`',
         'Builds all six models, verifies each checkpoint against its '
         'published digest, runs one prediction through each',
         Ch('verifying')],
        ['`train.py`', 'Trains a model, or fine-tunes one',
         f'{Ch("training")}, {Ch("adapting")}'],
        ['`test.py`', 'Scores a model against a labelled split',
         Ch('evaluating')],
        ['`predict_split.py`',
         'One predicted mask per tile for a labelled split', Ch('tiles')],
        ['`predict_folder.py`',
         'Maps every image in a folder, one GeoPackage of crowns per image',
         Ch('folder')],
        ['`make_sample.py`',
         'Cuts a small georeferenced clip out of a large mosaic', Ch('data')],
        ['`fetch_init_weights.py`',
         'Collects the ImageNet weights once, while online', Ch('training')],
        ['`export_release.py`',
         'Assembles the models folder for sharing, verifying every checkpoint '
         'before and after the copy', Ch('handover')],
        ['`build_handover.py`',
         f'Assembles the whole bundle described in {ch("what")}',
         Ch('handover')],
    ], widths=[112, None, 44], size=8.5,
        caption='Listed in the order the manual runs them. '
                '`ghaf.inference.large_image` is the one that maps a whole '
                f'orthomosaic ({ch("orthomosaic")}); it is a module rather '
                'than a script here.'))

    s.append(glue(section('`tests/` and `docs/`'), None))
    s.append(para(
        'Eighteen test files and 345 tests. They run without a '
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


# ==========================================================================
def installing():
    s = chapter(
        'Installing the environment',
        'Twenty minutes, once per machine, and the version numbers are not '
        'suggestions. This chapter installs the stack the six models were '
        f'trained against; {ch("verifying")} proves the installation before anything '
        'depends on it.')

    s.append(glue(section('What the machine needs'), None))
    s.extend(table([
        ['Item', 'Requirement', 'Notes'],
        ['Operating system', 'Windows 10 or 11, macOS, or Linux',
         'Commands here are for the Windows Command Prompt. On macOS and '
         'Linux the only difference is `/` for `\\`'],
        ['GPU', 'NVIDIA, 8 GB or more',
         'The workstation the models were trained on is an RTX A5000. Without '
         'a GPU every procedure still runs with `--device cpu`, far more '
         'slowly; the factor was never measured'],
        ['Disk, bundle', '20 GB', 'Code, models, initialisation weights, '
         'tiles and the sample clip'],
        ['Disk, one run', '9 bytes per pixel of the image',
         'The 8192 × 8192 clip needs 0.6 GB. The full mosaic needs 78.5 GB. '
         'Released when the run ends, and checked before it starts'],
        ['Python', '3.9', 'Installed by conda below. No system Python is '
         'involved'],
        ['Prerequisite', 'Miniconda or Anaconda', 'From docs.conda.io'],
    ], widths=[76, 96, None], size=8.5, caption='System requirements.'))

    s.append(glue(section('Conventions in this manual'), None))
    s.append(para(
        'A shaded block is a command. Enter it, press Enter, and wait for the '
        'prompt before the next one. A caret at the end of a line continues '
        'that command onto the next in the Command Prompt; the continuation '
        'starts hard against the left margin on purpose, because leading '
        'spaces would otherwise be carried into the argument. On macOS or '
        'Linux the continuation character is a backslash instead.'))
    s.append(para(
        'The paths are the ones on the machine this was written on. '
        '`D:\\ghaf-project\\` is where the bundle was unpacked there; '
        'substitute wherever it was unpacked here, and every command works '
        'unchanged. No program looks for that name, so `no such folder` or '
        '`no checkpoint under` naming `D:\\ghaf-project` is the example path '
        'being run as though it were real, not a broken installation.'))
    s.extend(callout('Paste one command at a time', [
        'Several terminals join a multi-line paste into a single line and run '
        'something nobody typed. It usually fails loudly. It does not always: '
        'a joined `cd` followed by a `python` call can run the right program '
        'in the wrong folder, which produces a result rather than an error.',
    ]))

    s.append(glue(section('Step 1 — create the environment'), None))
    s.extend(code(
        'conda create -n ghaf python=3.9 -y\n'
        'conda activate ghaf'))
    s.append(para(
        'The prompt now begins with `(ghaf)`. Every command in this manual '
        'assumes it. A new terminal window opens outside the environment, so '
        'this is run again in each new window; forgetting it is the first '
        'entry in the error catalogue for a reason.'))

    s.append(glue(section('Step 2 — PyTorch'), None))
    s.append(para(
        'The version the released models were trained with. About 2 GB.'))
    s.extend(code(
        'python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 ^\n'
        '--extra-index-url https://download.pytorch.org/whl/cu113'))
    s.append(para('Without an NVIDIA GPU, install the CPU build instead.'))
    s.extend(code('python -m pip install torch==1.12.1 torchvision==0.13.1'))

    s.append(glue(section('Step 3 — the OpenMMLab stack'), None))
    s.append(para(
        'In this order. `mmcv` must stay below 2.2.0, which mmsegmentation '
        '1.2.2 asserts at import time, and `mim` fetches the prebuilt wheel '
        'matched to the installed PyTorch and CUDA rather than compiling it.'))
    s.extend(code(
        'python -m pip install -U openmim\n'
        'python -m mim install mmengine==0.10.7 "mmcv>=2.0.0rc4,<2.2.0"\n'
        'python -m pip install mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0'))
    s.append(para(
        '`mmdet` supplies the Mask2Former head and `mmpretrain` the ConvNeXt, '
        'PoolFormer and EfficientNet backbones. The second command is the '
        'slow one; several minutes is normal.'))

    s.append(glue(section('Step 4 — this project'), None))
    s.extend(code(
        'cd /d D:\\ghaf-project\\code\n'
        'python -m pip install -r requirements.txt\n'
        'python -m pip install -e ".[test]"'))
    s.append(para(
        '`requirements.txt` adds `timm` 1.0.19 for the FastViT blocks and '
        'DPN-98, `rasterio` and `geopandas` for the georeferenced input and '
        'output, and `ftfy` and `regex`, which mmsegmentation 1.2.2 needs to '
        'import at all — its own metadata does not say so. NumPy is held '
        'below 2.0 because PyTorch 1.12.1 is built against the NumPy 1.x '
        'binary interface.'))
    s.extend(callout('What the last step prints, and what none of it means', [
        'pip may report that `opencv-python` requires `numpy>=2`, and may '
        'then replace a newer OpenCV with 4.11.0.86 — a line reading '
        '`Attempting uninstall: opencv-python`. Both are the NumPy pin doing '
        'its work: 4.11.0.86 is the newest OpenCV built against NumPy 1.x, '
        'and it is the only kind this project can use. Nothing here imports '
        'OpenCV directly; mmengine reads images with it and is content with '
        'any version from 3 upwards. pip may also list conflicts among '
        'packages this project does not import. Only a line beginning '
        '`ERROR:` that stops the install matters.',
    ]))

    s.append(glue(section('Paths with spaces, and paths with an ampersand'), None))
    s.append(para(
        'Quote any path containing a space. Quote a path containing `&` '
        'without exception: the Command Prompt reads `&` as the end of one '
        'command and the start of another, so an unquoted path fails with '
        '"The system cannot find the path specified" printed twice, once for '
        'each half.'))
    s.extend(code(
        'cd /d "Z:\\Survey Data\\Cineraria_Data & Model\\ghaf-project\\code"'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'The prompt shows `(ghaf)`.',
        '`python -c "import sys; print(sys.executable)"` prints a path '
        'containing `envs\\ghaf`.',
        'No line beginning `ERROR:` stopped any install step.',
        f'You are in `code\\`, and {ch("verifying")} is next.',
    ]))
    return s


# ==========================================================================
def verifying():
    s = chapter(
        'Verifying the installation',
        'Three checks, five minutes. They establish that the code, the '
        'weights and the tiles arrived intact and agree with one another. '
        'Run them after installing, and again on any day when a result looks '
        'wrong.')

    s.append(glue(section('Check 1 — the code'), None))
    s.append(para(
        'No GPU, no weights and no dataset are needed, which is what makes '
        'this worth running on a machine that has only just been set up.'))
    s.extend(code('python -m pytest tests\\ -q'))
    s.extend(code(
        '.........................................................\n'
        '345 passed, 1 skipped in 28.69s'))
    s.append(para(
        'Any tally of passes with no `F` in the progress output is a pass. '
        'The number skipped is not fixed and does not need to match: a test '
        'that needs `geopandas`, `torch` or `git` skips itself where they are '
        'absent, so a machine without them reports more skips than this '
        'one, and is not failing. `No module named mmengine` here means a different Python is '
        f'running than the one the packages went into; {ch("errors")} has the '
        'remedy.'))

    s.append(glue(section('Check 2 — the weights'), None))
    s.append(para(
        'This builds all six models from their configurations, compares each '
        'checkpoint against its published SHA-256, loads the weights into the '
        'model, names any tensor the two do not share, and runs one '
        'prediction through each. About 90 seconds on a GPU.'))
    s.extend(code(
        'python tools\\smoke_test.py --checkpoints D:\\ghaf-project\\models'))
    s.extend(code(
        'model                            digest             weights  prediction\n'
        '----------------------------------------------------------------------\n'
        'fastvit-ma36_mask2former             ok    all 1,558 matched  0.00% ghaf\n'
        'poolformer-s36_fpn                   ok      all 436 matched  0.00% ghaf\n'
        'dpn98_fpn                            ok      all 676 matched  0.00% ghaf\n'
        'convnext-small_upernet               ok      all 430 matched  0.00% ghaf\n'
        'resnet-50_mask2former                ok      all 610 matched  0.00% ghaf\n'
        'efficientnet-b3_fpn                  ok      all 610 matched  0.00% ghaf\n'
        '\n'
        'all 6 model(s) built and ran a forward pass'))
    s.extend(table([
        ['Column', 'What it establishes'],
        ['`digest` reads `ok`',
         'The file is byte-for-byte the one that was released. Nothing was '
         'corrupted in transit or in the copy'],
        ['`all N matched`',
         'Every tensor in the checkpoint found its place in the model built '
         'from the configuration. Nothing missing, nothing left over'],
        ['`+0` in the delta column',
         'The parameter count equals the published figure exactly'],
        ['`0.00% ghaf`',
         '**Expected.** The forward pass runs on a blank synthetic tile, not '
         'on imagery, so finding no trees is the correct answer here'],
    ], widths=[110, None], size=8.5))
    s.extend(callout('Screens of UserWarning are not a fault', [
        'Every command that loads a model prints warnings from inside PyTorch '
        'and mmsegmentation about `__floordiv__`, `torch.meshgrid`, binary '
        'segmentation and `build_loss`. They appear on a correct run. The '
        'result is the table printed after them, not the absence of warnings.',
    ]))
    s.append(para(
        '`--only fastvit-ma36_mask2former` limits the check to one model, and '
        '`--strict` turns a tensor-count mismatch from a report into a '
        'non-zero exit, which is what a scripted check wants.'))

    s.append(glue(section('Check 3 — the tiles'), None))
    s.extend(code(
        'python tools\\check_dataset.py D:\\ghaf-project\\data\\ghaf'))
    s.extend(code(
        'split          images    masks   paired  checked  status\n'
        '--------------------------------------------------------\n'
        'training         7005     7005     7005      200  ok\n'
        'validation        869      869      869      200  ok\n'
        'testing           767      767      767      200  ok\n'
        '\n'
        '8,641 paired tile(s) across 3 split(s)\n'
        'dataset looks usable'))
    s.extend(table([
        ['Variant', 'What it does', 'When'],
        ['`--sample 0`',
         'Pairing only. Opens no file at all', 'Seconds, even over a network '
         'drive. Use it first after any copy'],
        ['`--sample 25`', 'Opens 25 tiles per split',
         'A minute. Enough to catch a truncated copy'],
        ['`--full`', 'Opens all 8,641 tiles',
         'Slow. Worth doing once, on the machine that will train'],
        ['`--json report.json`', 'Writes the same result as JSON',
         'When the check runs unattended'],
    ], widths=[76, 120, None], size=8.5))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'The test suite passed.',
        'All six digests read `ok`, and every tensor matched.',
        'You understand why the prediction column reads `0.00% ghaf`.',
        'The three splits report 7005, 869 and 767 pairs.',
    ]))
    return s


# ==========================================================================
def data():
    s = chapter(
        'The data',
        'What the tiles are, how they are laid out, and the two properties '
        'that break a dataset silently. Read this before building a dataset '
        'of your own; skip it if you are only running the supplied models.')

    s.append(glue(section('Layout'), None))
    s.append(para(
        'Three splits, declared once in `ghaf/splits.py` and read from there '
        'by `check_dataset`, `predict_split` and `build_handover` alike, so '
        'the layout cannot drift between the programs that depend on it.'))
    s.extend(code(
        'data\\ghaf\\\n'
        '  training\\images\\      7005 PNG tiles, 1024 x 1024, 8-bit RGB\n'
        '  training\\masks\\       7005 PNG masks, one per image\n'
        '  validation\\images\\     869 tiles\n'
        '  validation\\masks\\      869 masks\n'
        '  testing\\ghaf26\\images\\  767 tiles, with .pgw and .png.aux.xml\n'
        '  testing\\ghaf26\\masks\\   767 masks'))
    s.extend(table([
        ['Split', 'Pairs', 'What it is for'],
        ['`training`', '7,005', 'Fitting the model'],
        ['`validation`', '869',
         'Scored every 3,500 iterations during training; the best checkpoint '
         'is chosen on it'],
        ['`testing/ghaf26`', '767',
         'Held out entirely. The source of every published score'],
    ], widths=[86, 44, None], size=8.6,
        caption='8,641 pairs in total, counted by `check_dataset.py`.'))

    s.append(glue(section('The two things that fail silently'), None))
    s.append(para(
        'Neither raises anything. Both produce a model that trains to a '
        'respectable number and predicts nothing, which is why they are the '
        'first thing to check when a result looks impossible.'))
    s.extend(callout('A mask with the wrong values trains without complaint', [
        'The mask pixel value **is** the class index: 0 for background, 1 for '
        'ghaf. A mask exported as 0 and 255, which is what most annotation '
        'tools produce by default, is not rejected — mmseg reads 255 as class '
        '255, the loss ignores it, and the model learns that nothing is a '
        'crown. The loss curve looks unremarkable. `check_dataset.py` is the '
        'only thing that catches this, and it is why it exists.',
        '`reduce_zero_label` must stay `False`. With it set, background is '
        'ignored rather than supervised, which silently converts the task '
        'into something the published scores no longer describe. '
        '`GhafDataset` refuses the argument rather than accepting it.',
    ]))

    s.append(glue(section('Georeferencing, and which tiles carry it'), None))
    s.append(para(
        'The test tiles have `.pgw`, `.png.aux.xml` and `.ovr` companion '
        'files beside them. A PNG cannot hold a coordinate system, so the '
        'world file carries the transform and GDAL keeps the CRS in the '
        '`.aux.xml`. That is why a prediction made from a test tile opens in '
        'QGIS in the right place, and a prediction from a training tile does '
        'not. The training and validation tiles have no companion files, '
        'which affects neither training nor scoring. Keep the companions '
        'beside the tiles they belong to; moving the PNG alone silently '
        'strips the position.'))

    s.append(glue(section('Building a dataset of your own'), None))
    s.append(para(
        'Match the layout above: 1024 × 1024 PNG pairs sharing a stem, masks '
        'containing only 0 and 1, one folder per split. Then verify it before '
        'training rather than after.'))
    s.extend(code(
        'python tools\\check_dataset.py D:\\ghaf-project\\data\\new-site --full'))
    s.append(para(
        'A fault reported here costs a minute. The same fault found after a '
        'training run costs the run, and it will not be obvious which fault '
        'it was, because a model trained on empty masks converges to '
        'predicting nothing and reports a high mIoU while doing it — the '
        'background class alone is 96 per cent of the pixels.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'Every image has a mask with the same stem.',
        'Masks contain only 0 and 1, verified rather than assumed.',
        'Companion files travelled with the tiles they belong to.',
        '`check_dataset.py` reports `dataset looks usable`.',
    ]))
    return s


# ==========================================================================
def configuration():
    s = chapter(
        'The configuration files',
        'Every number the training and scoring commands obey lives in two '
        'files: one shared by all six models, and one per model. This '
        'chapter reads the shared file field by field, says which fields are '
        'safe to change and which invalidate the published scores, and gives '
        'the rule that keeps a change from reaching the six models at once.')

    s.append(glue(section('Two files, and what each one owns'), None))
    s.append(para(
        'A model configuration is assembled from `configs/_base_/ghaf.py` '
        'and the file that names it. The model file begins '
        '`_base_ = [\'../_base_/ghaf.py\']`, which reads the shared file '
        'first; anything the model file then defines replaces what it found. '
        f'That division is what makes the comparison in {ch("what")} a '
        'comparison of architectures: the six models differ only in the half '
        'they own.'))
    s.extend(table([
        ['File', 'Owns', 'Fields'],
        ['`configs/_base_/ghaf.py`',
         'Everything the six models share',
         'dataset, pipelines, preprocessor, the three dataloaders, the '
         'evaluator, the schedule, the hooks, the runtime'],
        ['`configs/ghaf/<model>.py`',
         'The architecture, and how it is optimised',
         '`model` (backbone, neck, decode head, losses) and `optim_wrapper` '
         '(the optimiser, its learning rate, and the per-parameter '
         'multipliers)'],
    ], widths=[112, 96, None], size=8.5,
        caption='The optimiser sits with the model, not in the shared file: '
                'a Mask2Former head and an FPN head do not want the same '
                'learning rate.'))
    s.extend(callout('Never edit the shared file in place', [
        'All six configurations inherit it. A change to the crop size, the '
        'evaluator or the schedule there silently redefines what every '
        'published score means, and nothing in the pipeline warns you. To '
        'try something different, copy the file, change the copy, and point '
        'a new model config at it. Section 6.8 does exactly that.',
    ]))

    s.append(glue(section('The dataset block'), None))
    s.extend(code(
        "dataset_type = 'GhafDataset'\n"
        "data_root    = 'data/ghaf'\n"
        "crop_size    = (1024, 1024)"))
    s.extend(table([
        ['Field', 'Value', 'What it decides'],
        ['`dataset_type`', '`GhafDataset`',
         'The two-class reader in `ghaf/datasets.py`. It is what refuses '
         '`reduce_zero_label=True`'],
        ['`data_root`', '`data/ghaf`',
         '**Relative**, and read while the file is parsed. Resolved against '
         'the directory the command is typed in, which is `code\\`. Override '
         'it with `--data-root`, never with `--cfg-options`'],
        ['`crop_size`', '1024 × 1024',
         'The size the preprocessor pads to, and the size the models were '
         'trained at. Changing it invalidates every published score'],
    ], widths=[74, 74, None], size=8.5))
    s.append(para(
        '`--cfg-options data_root=...` is accepted and does nothing useful: '
        '`data_root` is a plain module-level variable, copied into each '
        'dataloader as the file is read, so setting the key afterwards '
        'changes something nothing looks at. `ghaf/config.py` makes the '
        'substitution the way it has to be made, once per dataset, which is '
        'what `--data-root` calls.'))

    s.append(glue(section('The two pipelines'), None))
    s.append(para(
        'A pipeline is the list of transforms applied to a tile between the '
        'file and the model. There are two, and the whole difference between '
        'them is one line.'))
    s.extend(code(
        "train_pipeline = [\n"
        "    dict(type='LoadImageFromFile'),\n"
        "    dict(type='LoadAnnotations', reduce_zero_label=False),\n"
        "    dict(type='RandomFlip', prob=0.5),\n"
        "    dict(type='PackSegInputs'),\n"
        "]\n"
        "\n"
        "test_pipeline = [                       # the same, without the flip\n"
        "    dict(type='LoadImageFromFile'),\n"
        "    dict(type='LoadAnnotations', reduce_zero_label=False),\n"
        "    dict(type='PackSegInputs'),\n"
        "]"))
    s.extend(table([
        ['Transform', 'What it does'],
        ['`LoadImageFromFile`', 'Reads the tile as 8-bit BGR'],
        ['`LoadAnnotations`',
         'Reads the mask as class indices. `reduce_zero_label=False` keeps '
         'background as a supervised class rather than an ignored one — the '
         f'setting {ch("data")} says must not change'],
        ['`RandomFlip`',
         'Horizontal flip, half the time. **The only augmentation in the '
         'study**: no scale jitter, no random crop, no colour distortion. '
         'Tiles are fed at native resolution'],
        ['`PackSegInputs`',
         'Packs the image and mask into the structure the model expects'],
    ], widths=[104, None], size=8.5))
    s.append(para(
        'Adding augmentation is a legitimate experiment and a change to what '
        'the numbers mean. Do it in a copy of the file, and score the result '
        'against the same test split before believing it.'))

    s.append(glue(section('The preprocessor'), None))
    s.append(para(
        'What happens to a tile between the pipeline and the model: '
        'normalisation, channel order, and the padding that makes an '
        'awkward size fit.'))
    s.extend(code(
        "data_preprocessor = dict(\n"
        "    type='SegDataPreProcessor',\n"
        "    mean=[123.675, 116.28, 103.53],     # ImageNet statistics\n"
        "    std=[58.395, 57.12, 57.375],\n"
        "    bgr_to_rgb=True,\n"
        "    pad_val=0, seg_pad_val=255,\n"
        "    size=crop_size,\n"
        "    test_cfg=dict(size_divisor=32))"))
    s.append(para(
        'Normalisation uses ImageNet statistics because every backbone was '
        'pre-trained on ImageNet; `bgr_to_rgb` undoes the channel order the '
        'loader produced. `seg_pad_val=255` matters: padding in the mask is '
        'labelled 255, which the loss ignores, so a padded edge contributes '
        'nothing to the gradient. `size_divisor=32` rounds an odd-sized test '
        'image up to something the strides divide.'))

    s.append(glue(section('The three dataloaders'), None))
    s.append(para(
        'One per split, identical but for the folder, the sampler and the '
        'batch size. Each names its folders relative to `data_root`, which '
        'is why `--data-root` moves all three together.'))
    s.extend(table([
        ['', '`train_dataloader`', '`val_dataloader`', '`test_dataloader`'],
        ['`img_path`', '`training/images`', '`validation/images`',
         '`testing/ghaf26/images`'],
        ['`seg_map_path`', '`training/masks`', '`validation/masks`',
         '`testing/ghaf26/masks`'],
        ['`pipeline`', '`train_pipeline`', '`test_pipeline`',
         '`test_pipeline`'],
        ['`batch_size`', '2', '1', '1'],
        ['`num_workers`', '2', '2', '2'],
        ['`sampler`', '`InfiniteSampler`, shuffled',
         '`DefaultSampler`', '`DefaultSampler`'],
    ], widths=[70, 96, 88, None], size=8.3,
        caption='`InfiniteSampler` is what an iteration-based schedule '
                'needs: it never runs out, so training stops at an iteration '
                'count rather than at the end of an epoch.'))
    s.append(para(
        f'`batch_size=2` is what the workstation in {ch("installing")} fitted at 1024 '
        'pixels; it is a memory limit, not a tuned value. Raising it changes '
        'the effective batch and therefore the result. Scoring uses 1 so '
        'that no tile is padded to match another. `num_workers=2` is the '
        'number of loader processes, and is safe to change: it affects '
        'speed only.'))

    s.append(glue(section('The schedule'), None))
    s.extend(code(
        "train_cfg = dict(type='IterBasedTrainLoop',\n"
        "                 max_iters=160000, val_interval=3500)\n"
        "param_scheduler = [dict(type='PolyLR', eta_min=0, power=0.9,\n"
        "                        begin=0, end=160000, by_epoch=False)]"))
    s.extend(table([
        ['Field', 'Value', 'Meaning'],
        ['`max_iters`', '160,000',
         'The length of the schedule. Not the length of a run: the released '
         'checkpoints stopped far earlier, at 3,500 to 38,500 iterations'],
        ['`val_interval`', '3,500',
         'Score the validation split this often. The checkpoint hook uses '
         'the same interval, which is why the released filenames carry those '
         'numbers'],
        ['`PolyLR`, `power=0.9`', '—',
         'Polynomial decay of the learning rate from its initial value to '
         '`eta_min`'],
        ['`end=160000`', '—',
         '**Must match `max_iters`.** The decay is computed against `end`, '
         'so shortening the schedule without changing both leaves the rate '
         'decaying on the old curve'],
    ], widths=[86, 54, None], size=8.5))
    s.append(para(
        'The optimiser is in the model file, not here. For FastViT-MA36 it '
        'is AdamW at a learning rate of 1e-4 with weight decay 0.05, '
        'gradients clipped at a norm of 0.01, and a `paramwise_cfg` that '
        'gives the backbone a tenth of the learning rate and exempts every '
        'normalisation layer from decay.'))

    s.append(glue(section('The hooks, and what each leaves on disk'), None))
    s.append(para(
        'A hook is a callback the runner fires at a fixed point — every '
        'iteration, every validation, the end of a run. These six are on by '
        'default, and between them they account for everything a training '
        'run writes.'))
    s.extend(table([
        ['Hook', 'Setting', 'What you see'],
        ['`LoggerHook`', '`interval=50`',
         'A line every 50 iterations, with loss and learning rate, to the '
         'console and to `work_dirs/<config>/<timestamp>/`'],
        ['`CheckpointHook`', '`interval=3500`, `save_best=\'mIoU\'`',
         '`iter_3500.pth` and so on, plus `best_mIoU_iter_N.pth`, which is '
         'rewritten whenever the validation mIoU improves'],
        ['`ParamSchedulerHook`', '—', 'Steps `PolyLR`. Nothing on disk'],
        ['`IterTimerHook`', '—', 'The `time:` and `eta:` fields in the log'],
        ['`DistSamplerSeedHook`', '—',
         'Reseeds the sampler each epoch under distributed training'],
        ['`SegVisualizationHook`', '`draw=False`',
         'Nothing, until `--show-dir` turns it on. It says so at startup, '
         'and that warning is not a fault'],
    ], widths=[92, 92, None], size=8.4))
    s.append(para(
        '`best_mIoU_iter_N.pth` is the file the released models are: the '
        'checkpoint that scored best on the validation split, not the last '
        'one written. Two hooks together decide which iterations can be '
        'chosen — `val_interval` and the checkpoint interval — and both are '
        '3,500.'))

    s.append(glue(section('The runtime block'), None))
    s.extend(table([
        ['Field', 'Value', 'Why it is there'],
        ['`custom_imports`', '`ghaf.datasets`, `ghaf.models`',
         '**Load-bearing.** Without it `GhafDataset`, `FastViTMA36` and '
         '`DPN98` are not in the registry and the config fails to build'],
        ['`default_scope`', '`mmseg`',
         'Which registry a bare type name is looked up in. Types written '
         '`mmdet.DiceLoss` reach across to mmdet'],
        ['`cudnn_benchmark`', '`True`',
         'Lets cuDNN pick algorithms for a fixed input size. Fine here — '
         'every tile is 1024 × 1024'],
        ['`mp_start_method`', '`fork`',
         'Ignored on Windows: mmengine applies it only on other systems, so '
         'it is inert on the machine most readers use'],
        ['`dist_cfg`', '`nccl`', 'Multi-GPU only. Unused in a single-GPU run'],
        ['`load_from` / `resume`', '`None` / `False`',
         'Set by `--load-from` and `--resume` on the command line. '
         'Chapter 13 explains why the two are not interchangeable'],
        ['`randomness`', 'commented out',
         'So no seed is set, and two runs of one config do not agree exactly '
         f'({ch("what")}). Uncommenting it costs throughput and changes results '
         'slightly'],
    ], widths=[80, 74, None], size=8.4))

    s.append(glue(section('What to change first, and how'), None))
    s.append(para(
        'After the folders are in place, these are the fields worth setting '
        'before a first run of your own. Everything in the last column is '
        'preferred: a command-line flag leaves the file — and therefore the '
        'published scores — untouched.'))
    s.extend(table([
        ['Want to', 'Field', 'Do it with'],
        ['Point at your tiles', '`data_root`',
         '`--data-root ..\\data\\ghaf` on `train.py` and `test.py`'],
        ['Train for less than 160,000 iterations',
         '`max_iters`, and `end` in `param_scheduler`',
         'A copied config. Both, or the decay curve is wrong'],
        ['Validate and checkpoint more often',
         '`val_interval`, and the checkpoint hook\'s `interval`',
         '`--cfg-options train_cfg.val_interval=1000` — these are read at '
         'run time, so the flag works'],
        ['Fit a smaller GPU', '`batch_size`, in `train_dataloader`',
         '`--cfg-options train_dataloader.batch_size=1`. It changes the '
         'result; say so when reporting'],
        ['Start from a released checkpoint', '`load_from`',
         f'`--load-from`, never `--resume` ({ch("errors")})'],
        ['Make a run repeatable', '`randomness`',
         'Uncomment it in a copied config'],
        ['Put logs elsewhere', '`work_dir`', '`--work-dir`'],
    ], widths=[96, 104, None], size=8.4))

    s.append(glue(sub('Making a variant properly'), None))
    s.append(para(
        'Copy the shared file, change the copy, and point a new model config '
        'at it. The six delivered configs go on meaning what they meant.'))
    s.extend(code(
        'copy configs\\_base_\\ghaf.py configs\\_base_\\ghaf-new-site.py\n'
        'copy configs\\ghaf\\fastvit-ma36_mask2former.py ^\n'
        'configs\\ghaf\\fastvit-ma36_new-site.py'))
    s.append(para(
        'Then edit one line at the top of the new model config, so it reads '
        '`_base_ = [\'../_base_/ghaf-new-site.py\']`, and make your changes '
        'in `ghaf-new-site.py`. Confirm the result before training with it: '
        '`python -c "from mmengine.config import Config; '
        'print(Config.fromfile(r\'configs\\ghaf\\fastvit-ma36_new-site.py\')'
        '.train_dataloader.batch_size)"` prints what the runner will '
        'actually use, which is not always what the file appears to say.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'You know which of the two files owns the field you want to change.',
        'You have not edited anything under `configs/_base_/` in place.',
        '`max_iters` and `param_scheduler[0].end` still agree.',
        'A changed batch size or crop size is recorded wherever the results '
        'are reported.',
    ]))
    return s
