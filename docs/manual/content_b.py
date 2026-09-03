#!/usr/bin/env python3
"""Chapters 6 to 14 of the technical manual.

Commands were checked against the argument parsers they invoke before being
written here: a flag that appears in this file exists in the code that reads
it. Values are traceable to docs/handover/FACTS.yml.
"""

from typeset import bullets, callout, chapter, code, glue, para, section, sub, table


def story():
    s = []
    s += chapter_6()
    s += chapter_7()
    s += chapter_8()
    s += chapter_9()
    s += chapter_10()
    s += chapter_11()
    s += chapter_12()
    s += chapter_13()
    s += chapter_14()
    return s


# ==========================================================================
def chapter_6():
    s = chapter(
        'Mapping one orthomosaic',
        'The main job. An orthomosaic goes in, crown polygons come out, and '
        'the image may be far larger than the memory of the machine. Start '
        'with the sample clip: it proves the whole chain in a couple of '
        'minutes rather than a couple of hours.')

    s.append(glue(section('How it works, in one paragraph'), None))
    s.append(para(
        'The mosaic is read in 1024-pixel windows that overlap by 512. Each '
        'window is predicted independently, and the predictions are combined '
        'with Gaussian weights that fall off toward the window edge, so a '
        'pixel covered by four windows takes a weighted mean rather than '
        'whichever window happened to be written last. That is what leaves no '
        'seams. The accumulators are memory-mapped files rather than arrays, '
        'so peak memory does not grow with the mosaic — but disk does, at 9 '
        'bytes per source pixel.'))

    s.append(glue(section('The command'), None))
    s.append(para(
        'Work from `code\\`, which keeps the paths short enough to read. The '
        'caret continues the line in the Command Prompt.'))
    s.extend(code(
        'cd /d D:\\ghaf-project\\code\n'
        'set MODEL=..\\models\\fastvit-ma36_mask2former\n'
        'python -m ghaf.inference.large_image ^\n'
        '%MODEL%\\fastvit-ma36_mask2former.py ^\n'
        '%MODEL%\\best_mIoU_iter_3500.pth ^\n'
        '..\\samples\\Kalba26_sample.tif ^\n'
        '--out-mask ..\\output\\crowns.tif ^\n'
        '--out-prob ..\\output\\probability.tif ^\n'
        '--out-polygons ..\\output\\crowns.gpkg ^\n'
        '--batch-size 4 --min-area 1'))
    s.append(para(
        'The three positional arguments are the configuration, the weights '
        'and the image, in that order. At least one output path is required, '
        'and `--out-polygons` additionally requires `--out-mask` or '
        '`--out-prob`, because the polygons are traced from a written raster '
        'rather than from memory. Passing polygons alone is refused by the '
        'argument parser, not discovered later.'))
    s.extend(code(
        'INFO Kalba26_sample.tif: 8192 x 8192 px, 225 window(s) of 1024 px\n'
        '     (overlap 512), batch 4, 0.6 GB scratch\n'
        'Kalba26_sample: 100%|#############| 57/57 [00:40<00:00, 1.42batch/s]\n'
        'INFO Created 27 records\n'
        'INFO canopy: 650055 of 67108864 valid px (0.97%)'))

    s.append(glue(section('The three outputs'), None))
    s.extend(table([
        ['File', 'What it holds', 'What it is for'],
        ['`crowns.gpkg`',
         'The crowns as polygons, with `area_m2` per feature',
         'Counting, measuring, joining to other layers. The primary result'],
        ['`crowns.tif`',
         'One band, 1 for crown and 0 for background, the size of the input',
         'Overlaying, canopy cover, differencing against a labelled mask'],
        ['`probability.tif`',
         'One band of float32, the confidence per pixel from 0 to 1',
         'Re-thresholding after the run, and seeing where the model was '
         'unsure'],
    ], widths=[74, 140, None], size=8.5,
        caption='All three carry the CRS of the input, so they land in place '
                'with no manual positioning.'))

    s.append(glue(section('Whether the answer is plausible'), None))
    s.extend(table([
        ['Quantity', 'On the sample clip', 'If it is far from that'],
        ['Canopy share', '0.9687 per cent of valid pixels',
         '`0.00` means nothing was found; above 50 per cent means everything '
         'was. Both point upstream — usually the wrong checkpoint, or bands '
         'that are not red, green, blue'],
        ['Polygons', '27 at the defaults, 16 with `--min-area 1`',
         'A much larger count is single-pixel fragments, not trees'],
        ['Crown area', '2.4 to 112 m²',
         'Hundreds of square metres means crowns merged, or the imagery is '
         'not near the 2.68 cm ground sampling distance the models were '
         'trained at'],
    ], widths=[62, 128, None], size=8.5))
    s.append(para(
        'The polygon areas on the sample clip sum to 469 m², against 467 m² '
        'computed from the mask pixels. The two should agree to about that; a '
        'wide gap means the polygons and the raster came from different runs.'))

    s.append(glue(section('Options that change the answer'), None))
    s.extend(table([
        ['Option', 'Default', 'Effect'],
        ['`--threshold`', '0.5',
         'The confidence at which a pixel becomes a crown. Higher gives fewer '
         'and more certain crowns'],
        ['`--min-area`', '0.0',
         'Discards polygons below this many square metres. At the default the '
         'polygon layer matches the mask exactly, fragments included'],
        ['`--batch-size`', '1',
         'Windows per forward pass. Raise it until VRAM is the limit; 4 was '
         'used for every timing here'],
        ['`--tile` / `--overlap`', '1024 / 512',
         '**Do not change `--tile`.** It must match the crop size the model '
         'was trained at'],
        ['`--bands`', '1 2 3',
         'Which bands are red, green and blue'],
        ['`--scratch-dir`', 'system temp',
         'Where the accumulators go. Point it at a large local disk'],
        ['`--device`', '`cuda:0`', '`cpu` runs without a GPU'],
    ], widths=[86, 60, None], size=8.5,
        caption='Defaults are the argparse defaults in '
                '`ghaf/inference/large_image.py`, not recommendations.'))
    s.extend(callout('Changing the tile size succeeds and gives a worse answer', [
        'The model was trained on 1024-pixel crops. Run it at 512 and it '
        'still runs, still writes all three outputs, and still reports a '
        'canopy percentage — one produced by a model looking at the imagery '
        'at the wrong scale. Nothing in the output records the tile size that '
        'produced it. If you change it, write it down beside the result.',
    ]))

    s.append(glue(section('The full survey mosaic'), None))
    s.append(para(
        '`Kalba26.tif` is 84,072 × 103,691 pixels, or 8.7 billion. At 9 bytes '
        'per pixel the run needs **78.5 GB** of scratch space and plans '
        '**33,128** windows. Both are arithmetic on the raster size, not '
        'measurements: the full mosaic has never been run end to end, and no '
        'timing for it exists. Free space is checked before the run starts '
        'rather than part way through, which is the one thing that will not '
        'surprise you.'))

    s.append(glue(section('Cutting a clip to try first'), None))
    s.extend(code(
        'python tools\\make_sample.py ..\\samples\\Kalba26.tif ^\n'
        '--output ..\\samples\\my_sample.tif --size 8192 --origin 30000 40000'))
    s.append(para(
        '`--size` takes one number for a square or two for a rectangle. '
        '`--origin` is the top-left corner in pixels; omitted, the clip is '
        'taken from the centre. The tool reports what share of the clip is '
        'imagery rather than the transparent border around a survey, so a '
        'window that landed on nothing is obvious before the model runs.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'The sample clip ran, and reported about 0.97 per cent canopy.',
        'You passed `--out-mask` or `--out-prob` alongside `--out-polygons`.',
        'You did not change `--tile`.',
        '`--scratch-dir` points at a disk with room for 9 bytes per pixel.',
    ]))
    return s


# ==========================================================================
def chapter_7():
    s = chapter(
        'Mapping every image in a folder',
        'A survey usually arrives as many images rather than one mosaic. This '
        'maps all of them in a single run and writes one GeoPackage of crowns '
        'per image. Each image is windowed exactly as a full mosaic is, so '
        'they need not share a size.')

    s.extend(code(
        'cd /d D:\\ghaf-project\\code\n'
        'set MODEL=..\\models\\fastvit-ma36_mask2former\n'
        'python tools\\predict_folder.py ^\n'
        '%MODEL%\\fastvit-ma36_mask2former.py ^\n'
        '%MODEL%\\best_mIoU_iter_3500.pth ^\n'
        'D:\\my-images --out-dir ..\\output\\folder ^\n'
        '--batch-size 4 --min-area 1'))

    s.append(glue(section('What it writes'), None))
    s.extend(code(
        'folder\\\n'
        '  polygons\\      one .gpkg of crowns per image, with area_m2\n'
        '  masks\\         the 0/1 raster, only with --save-mask\n'
        '  probability\\   the confidence raster, only with --save-probability\n'
        '  summary.json   every image, its canopy share, and any failures'))
    s.append(para(
        'Crowns are the output and the rasters are opt-in. A GeoPackage is a '
        'few hundred kilobytes where the mask it was traced from is hundreds '
        'of megabytes, and counting, measuring and drawing all work from the '
        'polygons. The mask is still produced, because the polygons are '
        'traced from it, but it goes to the scratch directory and is deleted '
        'when that image finishes. `--save-mask` keeps it.'))
    s.append(para(
        'Subfolders in the input are reproduced in the output, so two images '
        'of the same name in different folders cannot overwrite one another. '
        'The output folder is excluded from a recursive listing, so a second '
        'run does not predict the first run\'s own rasters.'))

    s.append(glue(section('Options for a long run'), None))
    s.extend(table([
        ['Option', 'Effect'],
        ['`--limit 3`',
         'Stop after three images. Do this first on a large folder: it proves '
         'the settings before several hundred images are committed to'],
        ['`--recursive`', 'Include images in subfolders'],
        ['`--pattern "*_rgb.tif"`',
         'Only files whose name matches. Useful where a folder mixes imagery '
         'with elevation models'],
        ['`--skip-existing`',
         'Resume an interrupted run instead of repeating finished images'],
        ['`--save-mask`, `--save-probability`', 'Keep the rasters as well'],
        ['`--min-area 1`', 'Discard crowns under one square metre'],
    ], widths=[112, None], size=8.5))
    s.extend(callout('Read summary.json after any long run', [
        'One unreadable file does not stop the batch. The image is reported, '
        'the run continues, and the failure is recorded in `summary.json` '
        'with the exception that caused it. The exit status is non-zero if '
        'anything failed — but an operator watching a progress bar scroll '
        'past will not see it, and a folder of 400 images that produced 397 '
        'GeoPackages looks complete in a file browser.',
    ]))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        '`--limit 3` was run first, and the crowns looked right in QGIS.',
        '`summary.json` reports `0 failed`, or you have read the failures.',
        'The canopy share per image is in the range chapter 6 gives.',
    ]))
    return s


# ==========================================================================
def chapter_8():
    s = chapter(
        'Reviewing the results',
        'Opening the outputs, and the numbers to check them against. Every '
        'file the pipeline writes carries the coordinate system of the image '
        'it came from, so nothing needs positioning by hand.')

    s.append(glue(section('The crowns in QGIS'), None))
    s.extend(bullets([
        '**Layer > Add Layer > Add Vector Layer**, choose `crowns.gpkg`, '
        '**Add**.',
        'Add the imagery through **Add Raster Layer** and drag it below the '
        'crowns in the Layers panel.',
        'Right-click the crowns > **Properties > Symbology**. Set the fill to '
        '*No brush* and pick a bright outline, so the imagery stays visible '
        'inside each crown.',
        'Right-click > **Open Attribute Table**. One row per crown, with '
        '`area_m2`. The row count is the crown count.',
        '**Vector > Analysis Tools > Basic Statistics for Fields** on '
        '`area_m2` gives the count, sum, mean, minimum and maximum.',
    ]))

    s.append(glue(section('The rasters in QGIS'), None))
    s.append(para(
        'Both open through **Add Raster Layer**. The mask appears black, '
        'because its values are 0 and 1 while QGIS stretches the display over '
        '0 to 255; set the render type to **Paletted/Unique values** and '
        'click **Classify** to get one colour per class. For the probability '
        'raster use **Singleband pseudocolor** from 0 to 1.'))

    s.append(glue(section('Figures to check against'), None))
    s.extend(table([
        ['Quantity', 'Value', 'Where it came from'],
        ['Canopy share, sample clip', '0.9687 per cent of valid pixels',
         'Measured: 650,055 of 67,108,864 px'],
        ['Canopy share, test split', '3.44 per cent over 767 tiles',
         'Measured by `predict_split.py`'],
        ['Crowns, sample clip', '27 raw, 16 after `--min-area 1`',
         'Measured'],
        ['Crown area, sample clip', '2.4 to 112 m²', 'Measured'],
        ['Total crown area, sample clip', '469 m² from polygons, 467 m² from '
         'pixels', 'Measured'],
        ['Ground sampling distance', '2.68 cm per pixel',
         'From the transform of the sample clip'],
        ['Tile ground size', '27.44 m square, 0.075 ha',
         'Derived: 1024 × 2.68 cm'],
    ], widths=[104, 116, None], size=8.4,
        caption='All measured on the delivered models and the sample clip. '
                'None of them is a target — they are what this imagery '
                'produced.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'The crowns sit on the imagery without being moved.',
        'The canopy share is within an order of magnitude of the table above.',
        'The crown count came from the attribute table, not from an estimate.',
    ]))
    return s


# ==========================================================================
def chapter_9():
    s = chapter(
        'Evaluating a model',
        'Scoring a model against the labelled test split reproduces the '
        'published figures. It is the strongest single check that an '
        'installation, a checkpoint and a dataset are all the ones they claim '
        'to be. Three minutes on a GPU.')

    s.extend(code(
        'cd /d D:\\ghaf-project\\code\n'
        'set MODEL=..\\models\\fastvit-ma36_mask2former\n'
        'python tools\\test.py ^\n'
        '%MODEL%\\fastvit-ma36_mask2former.py ^\n'
        '%MODEL%\\best_mIoU_iter_3500.pth ^\n'
        '--data-root ..\\data\\ghaf'))
    s.append(para(
        'The last line reports mIoU, mDice and mFscore, with a per-class '
        'table above it. For FastViT-MA36 expect **79.32** and **87.22**; the '
        'other five are in chapter 1. A departure larger than a rounding '
        'difference has two usual causes, and chapter 4 settles the second: '
        'a data root pointing somewhere unintended, or a checkpoint that is '
        'not the one it is taken for.'))
    s.append(para(
        '`--show-dir` writes prediction visualisations, and `--work-dir` '
        'places the log somewhere other than the default.'))

    s.append(glue(section('Per-tile predictions over a split'), None))
    s.append(para(
        'Scoring reduces a split to three numbers. The maps behind them are '
        'what error analysis and figures are made from, and they show where '
        'the model is wrong rather than by how much.'))
    s.extend(code(
        'python tools\\predict_split.py ^\n'
        '%MODEL%\\fastvit-ma36_mask2former.py ^\n'
        '%MODEL%\\best_mIoU_iter_3500.pth ^\n'
        '--data-root ..\\data\\ghaf --split testing ^\n'
        '--out-dir ..\\output\\predictions --save-probability'))
    s.append(para(
        'One mask per tile, encoded exactly as the ground truth is, so a '
        'prediction and its label can be subtracted directly. `--limit 20` '
        'runs a partial pass first. The canopy fraction over the test split '
        'is 3.44 per cent; the bundle already contains these 767 predictions, '
        'so this only needs running for a model other than FastViT-MA36.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'The reported mIoU matches the published figure for that model.',
        '`--data-root` pointed where you thought it did.',
        'If it did not match, chapter 4 check 2 was run before anything else.',
    ]))
    return s


# ==========================================================================
def chapter_10():
    s = chapter(
        'Training',
        'Only worth doing if something is to be changed: six trained models '
        'are supplied. This chapter covers a run from scratch, what it writes, '
        'and the one setting that makes a run impossible to reproduce.')

    s.append(glue(section('The run'), None))
    s.extend(code(
        'cd /d D:\\ghaf-project\\code\n'
        'python tools\\train.py configs\\ghaf\\fastvit-ma36_mask2former.py ^\n'
        '--data-root ..\\data\\ghaf ^\n'
        '--init-weights ..\\init-weights'))
    s.append(para(
        '`--init-weights` points at the ImageNet weights in the bundle, which '
        'is what lets a machine with no internet access start a run. Without '
        'it the backbones try to download their initialisation.'))
    s.extend(table([
        ['Setting', 'Value', 'Declared in'],
        ['Iterations', '160,000', '`configs/_base_/ghaf.py`'],
        ['Validation interval', 'every 3,500 iterations', 'same'],
        ['Checkpoint interval', 'every 3,500 iterations, best on mIoU', 'same'],
        ['Schedule', 'PolyLR, power 0.9', 'same'],
        ['Batch size', '2 train, 1 validation, 1 test', 'same'],
        ['Optimiser (FastViT)', 'AdamW, lr 1e-4, weight decay 0.05, grad clip '
         '0.01', '`configs/ghaf/fastvit-ma36_mask2former.py`'],
    ], widths=[92, 152, None], size=8.5))
    s.append(para(
        'Checkpoints and logs go to `work_dirs\\<config-name>\\`, relative to '
        'the working directory. An interrupted run continues with `--resume`, '
        'which restores the optimiser state and the iteration count. '
        '`--amp` enables mixed precision where the configuration supports it. '
        'How long a full run takes on the A5000 was never recorded.'))

    s.extend(callout('Nothing records the seed, so no run can be repeated', [
        'The line `randomness = dict(seed=0, deterministic=True)` is '
        'commented out in `configs/_base_/ghaf.py`. mmengine therefore picks '
        'a seed at random and writes it into the run log — and nowhere else. '
        'Two runs of the same configuration on the same data will differ, and '
        'if `work_dirs\\` is deleted the seed that produced a published '
        'number is gone for good. Keep the logs, or uncomment the line before '
        'starting anything you will need to defend.',
    ]))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        '`check_dataset.py --full` passed on the data first.',
        '`--init-weights` was passed, or the machine has internet access.',
        '`work_dirs\\` is on a disk with room, and will not be deleted.',
        'The seed in the run log was copied somewhere it will survive.',
    ]))
    return s


# ==========================================================================
def chapter_11():
    s = chapter(
        'Adapting to a new site',
        'Fine-tuning from a released checkpoint costs far less than training '
        'from scratch and usually scores better. It is also the operation '
        'most likely to produce a model that looks fine and is worse than the '
        'one it started from.')

    s.append(glue(section('Prepare and check the new tiles'), None))
    s.append(para(
        'Same layout as chapter 5: 1024 × 1024 PNG pairs, masks containing '
        'only 0 and 1, `training` and `validation` folders. Check them before '
        'the run, not after.'))
    s.extend(code(
        'python tools\\check_dataset.py D:\\ghaf-project\\data\\new-site --full'))

    s.append(glue(section('The run'), None))
    s.extend(code(
        'set MODEL=..\\models\\fastvit-ma36_mask2former\n'
        'python tools\\train.py configs\\ghaf\\fastvit-ma36_mask2former.py ^\n'
        '--data-root ..\\data\\new-site ^\n'
        '--load-from %MODEL%\\best_mIoU_iter_3500.pth ^\n'
        '--cfg-options train_cfg.max_iters=4000 ^\n'
        'optim_wrapper.optimizer.lr=1e-5'))
    s.extend(table([
        ['Argument', 'What it does'],
        ['`--load-from`',
         'Takes the weights and starts a fresh schedule. This is what '
         'fine-tuning means'],
        ['`--resume`',
         'Continues an interrupted run of your own, keeping the optimiser '
         'state and iteration count. A different operation entirely, and '
         'confusing the two silently restarts a schedule'],
        ['`max_iters=4000`',
         'A short schedule. The full 160,000 would overwrite what the model '
         'already knows'],
        ['`lr=1e-5`',
         'A tenth of the training rate, so the model adapts rather than '
         'forgets'],
    ], widths=[86, None], size=8.5))

    s.extend(callout('A fine-tuned model can score well and be worse', [
        'Fine-tuning on a small site teaches the model that site. Score the '
        'result on the **original** test split as well as the new one before '
        'replacing anything: a model that scores 0.88 on the new site and has '
        'lost four points on `testing/ghaf26` is not an improvement, and '
        'nothing in the training log will say so. No fine-tuning run has been '
        'performed on this project, so there is no measured example of how '
        'far it drifts.',
    ]))
    s.append(para(
        'Score the result exactly as in chapter 9, with `--data-root` at the '
        'new site and the checkpoint at the new '
        '`work_dirs\\...\\best_mIoU_iter_*.pth`, then again with '
        '`--data-root ..\\data\\ghaf` to see what it cost.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        'The new tiles passed `check_dataset.py --full`.',
        '`--load-from` was used, not `--resume`.',
        'The result was scored on both the new site and the original test '
        'split.',
        'The original checkpoint is still where it was; nothing overwrote it.',
    ]))
    return s


# ==========================================================================
def chapter_12():
    s = chapter(
        'Handing the system on',
        'How the bundle in chapter 1 is assembled, and how a recipient checks '
        'what they were sent. Two programs, both of which verify rather than '
        'trust.')

    s.append(glue(section('Packaging the models'), None))
    s.extend(code(
        'python tools\\export_release.py ^\n'
        '--checkpoints D:\\ghaf-project\\checkpoints ^\n'
        '--output D:\\ghaf-release'))
    s.append(para(
        'One self-contained folder per model: a resolved configuration beside '
        'its weights and a `metadata.json`. The configuration is flattened, '
        'so it carries the whole recipe and does not need '
        '`configs/_base_/`. Every checkpoint is verified against the SHA-256 '
        'in `ghaf/release.py` before the copy and again after, which means '
        'the command accepts only the six released checkpoints and a bundle '
        'cannot carry a silently corrupted file. `--dry-run` verifies and '
        'reports without writing; `--only <key>` limits it to one model.'))

    s.append(glue(section('Assembling the whole bundle'), None))
    s.extend(code(
        'python tools\\build_handover.py --output D:\\ghaf-project ^\n'
        '--code D:\\ghaf-public --checkpoints D:\\checkpoints ^\n'
        '--init-weights D:\\init-weights --data D:\\tiles\\ghaf ^\n'
        '--samples D:\\samples --predictions D:\\predictions'))
    s.append(para(
        'Inside a git checkout, `--code` copies exactly what `git ls-files` '
        'reports and records the commit in `MANIFEST.json`; files that are '
        'not tracked are counted and reported rather than copied. `--data` '
        'copies the three named splits and nothing else, which on the run '
        'that produced this bundle meant 19,583 files copied and 49,490 left '
        'behind. `--dry-run` reports the sizes without writing, and is worth '
        'running first: the first attempt on this project reported 103 GB '
        'because the tile tree was a working directory.'))

    s.append(glue(section('What a recipient runs'), None))
    s.append(para(
        'The check that matters is chapter 4 check 2, pointed at the models '
        'folder they were sent. It confirms every digest, loads every '
        'checkpoint into the model built from the configuration, and runs a '
        'prediction through each.'))
    s.extend(code(
        'python tools\\smoke_test.py --checkpoints ..\\models\n'
        'python tools\\check_dataset.py ..\\data\\ghaf --sample 0'))
    s.append(para(
        '`--sample 0` checks pairing without opening a file, which is what '
        'makes it usable over a network share; a full check of 8,641 tiles '
        'across a mounted drive takes long enough that it is usually '
        'abandoned half way, which is worse than not starting it.'))

    s.append(glue(sub('Checklist'), None))
    s.extend(bullets([
        '`build_handover.py --dry-run` was read before the real run.',
        '`MANIFEST.json` records the commit the code came from.',
        'The recipient ran `smoke_test.py --checkpoints` and got six `ok`.',
        'The full orthomosaic was handled separately; it is not in the '
        'bundle.',
    ]))
    return s


# ==========================================================================
def chapter_13():
    s = chapter(
        'Error catalogue',
        'Keyed on what appears on screen. Find the message, read across. '
        'Everything here has actually happened on this project or is raised '
        'explicitly by its code; nothing is a hypothetical.')

    s.append(glue(section('How to read a failure'), None))
    s.append(para(
        'Python prints a record of what it was doing and then, on the last '
        'line, what went wrong. The last line is the error; everything above '
        'it is context. The word before the colon is the category. Copy the '
        'text before running anything else — closing the window loses it, and '
        'an error nobody can quote is an error nobody can diagnose.'))
    s.append(para(
        'Three questions resolve most of it before any searching. Does the '
        'prompt read `(ghaf)`? Does '
        '`python -c "import sys; print(sys.executable)"` print a path '
        'containing `envs\\ghaf`? Are you in `code\\`?'))

    s.append(glue(section('Messages'), None))
    s.extend(table([
        ['What you see', 'What it means, and what to do'],
        ['`No module named mmengine`, `pytest` or `ghaf`',
         'A different Python is running than the one the packages went into. '
         '`conda activate ghaf`, then check `sys.executable`. Always start '
         'commands with `python -m`'],
        ['`mmseg ... is not installed in conda environment "X"`',
         'Wrong environment, or the stack was never installed on this '
         'machine. The message names the interpreter it asked'],
        ['`No module named ftfy`',
         'mmsegmentation 1.2.2 imports a tokenizer needing it, though its own '
         'metadata does not declare it. `python -m pip install ftfy regex`'],
        ['`RuntimeError: Numpy is not available`',
         'NumPy 2 beside a PyTorch built for 1.x. '
         '`python -m pip install "numpy<2"`'],
        ['`opencv-python ... requires numpy>=2` while installing',
         'Not an error. The wheel is built against the NumPy 2 headers and '
         'works with either'],
        ['`CUDA out of memory`',
         'Lower `--batch-size` to 2 or 1, close other GPU programs, or use '
         '`--device cpu`. Do not lower `--overlap` to save memory; it changes '
         'the result'],
        ['`not enough scratch space`',
         'The drive holding temporary files is too small. `--scratch-dir` at '
         'a larger disk. An image needs 9 bytes per pixel'],
        ['`nothing to do: pass at least one of --out-prob, --out-mask, '
         '--out-polygons`',
         'The parser refusing a run with no output. Add one'],
        ['`--out-polygons needs --out-mask or --out-prob to vectorise from`',
         'Polygons are traced from a written raster. Add `--out-mask`'],
        ['`FileNotFoundError` naming a path',
         'A typo, a missing `..\\`, or a path with spaces that needs quotes'],
        ['`The system cannot find the path specified`, printed twice',
         'The path contains `&`, which the Command Prompt read as two '
         'commands. Quote the whole path'],
        ['`The process cannot access the file because it is being used by '
         'another process`',
         'QGIS or another program holds the file open. When deleting a '
         'folder, first `cd` out of it: a terminal inside a folder holds it'],
        ['`NO TILES` from `check_dataset.py`',
         'The folders exist and hold no PNG or TIF. The row names what it '
         'found instead; the tiles are usually one level deeper'],
        ['`SHA-256 mismatch` from `smoke_test.py`',
         'That checkpoint is not the released file. The copy is damaged; copy '
         'it again from the original'],
        ['`0.00% ghaf` from `smoke_test.py`',
         'Correct. That forward pass runs on a blank synthetic tile'],
        ['`KeyboardInterrupt`',
         'Ctrl+C, or the window closed. Nothing is damaged. Long runs resume '
         'with `--skip-existing` or `--resume`'],
        ['Screens of `UserWarning` about `__floordiv__` or `meshgrid`',
         'Deprecation notices from inside PyTorch and mmsegmentation. They '
         'appear on a correct run'],
    ], widths=[142, None], size=8.4))

    s.append(glue(section('Failures that produce a result'), None))
    s.append(para(
        'The messages above announce themselves. These do not: each one '
        'finishes cleanly, writes output, and reports a number that is '
        'wrong.'))
    s.extend(table([
        ['What you see', 'What it means, and what to do'],
        ['`0.00%` canopy on real imagery',
         'The run found nothing and said so calmly. Usually the wrong '
         'checkpoint, or bands that are not red, green, blue. Check both, '
         'then try `--bands`'],
        ['Canopy above 50 per cent',
         'The opposite failure, same causes'],
        ['A crown count far above the range in chapter 6',
         'Single-pixel fragments counted as trees. `--min-area 1` removed 11 '
         'of 27 on the sample clip'],
        ['A model that trains to a high mIoU and predicts nothing',
         'Masks encoded 0 and 255 rather than 0 and 1. Background alone is '
         'about 96 per cent of the pixels, so predicting nothing scores well. '
         '`check_dataset.py` is the only thing that catches it'],
        ['Numbers that differ from a previous run of the same command',
         'No seed is set (chapter 10). Also check the tile size, which '
         'nothing records in the output'],
        ['A folder run that produced fewer files than there were images',
         'One or more images failed and the batch continued. `summary.json` '
         'names them; the exit status was non-zero'],
    ], widths=[142, None], size=8.4))

    s.append(glue(section('Searching for anything else'), None))
    s.append(para(
        'Take the last line of the error and strip what is specific to your '
        'machine — user name, drive letters, file names, long numbers. None '
        'of it appears in anyone else\'s error. Add the library that raised '
        'it. Search that. Errors raised inside the framework belong on the '
        'mmsegmentation issue tracker, closed issues included, since a closed '
        'issue usually holds the fix; the address is in chapter 14.'))
    s.append(para(
        'Two rules when judging what you find. Check the versions: an answer '
        'written for mmsegmentation 0.x does not apply to 1.2.2. And do not '
        'upgrade a package on general advice — the versions in chapter 3 were '
        'chosen to work together and to match the trained weights, and '
        '`pip install --upgrade` will replace one and break four. Change one '
        'thing, then re-run chapter 4.'))
    s.append(para(
        'If an installation reaches a state that cannot be explained, delete '
        'the environment and rebuild it from chapter 3. Twenty minutes, and '
        'it touches no data: `conda deactivate`, `conda env remove -n ghaf`, '
        'then step 1.'))

    s.append(glue(section('Reporting it to somebody else'), None))
    s.extend(bullets([
        'The exact command, copied rather than retyped.',
        'The complete output, as text, from the command to the last line.',
        'What you expected instead.',
        'The output of `python -c "import sys; print(sys.executable)"`.',
        'The `torch`, `mmcv`, `mmsegmentation`, `mmdet` and `numpy` lines '
        'from `python -m pip list`.',
        'Whether chapter 4 passes now, and whether it ever did on this '
        'machine.',
    ]))
    return s


# ==========================================================================
def chapter_14():
    s = chapter(
        'Reference',
        'Every routine command in one place, the released models by digest, '
        'and what was never established. The quick reference sheet carries a '
        'shorter version of the first table.')

    s.append(glue(section('Commands'), None))
    s.append(para(
        'All assume `conda activate ghaf` and `cd /d D:\\ghaf-project\\code`, '
        'with `MODEL` set as in chapter 6.'))
    s.extend(code(
        'python -m pytest tests\\ -q\n'
        'python tools\\smoke_test.py --checkpoints ..\\models\n'
        'python tools\\check_dataset.py ..\\data\\ghaf --sample 0\n'
        'python tools\\check_dataset.py ..\\data\\ghaf --full\n'
        '\n'
        'python -m ghaf.inference.large_image %MODEL%\\CONFIG %MODEL%\\WEIGHTS ^\n'
        'image.tif --out-mask crowns.tif --out-polygons crowns.gpkg ^\n'
        '--batch-size 4 --min-area 1\n'
        '\n'
        'python tools\\predict_folder.py %MODEL%\\CONFIG %MODEL%\\WEIGHTS ^\n'
        'D:\\my-images --out-dir ..\\output\\folder --batch-size 4\n'
        '\n'
        'python tools\\make_sample.py mosaic.tif --output clip.tif --size 8192\n'
        'python tools\\test.py %MODEL%\\CONFIG %MODEL%\\WEIGHTS ^\n'
        '--data-root ..\\data\\ghaf\n'
        'python tools\\train.py configs\\ghaf\\CONFIG --data-root ..\\data\\ghaf'))

    s.append(glue(section('The released models'), None))
    s.extend(table([
        ['Model', 'Checkpoint', 'Bytes', 'SHA-256, first 16'],
        ['`fastvit-ma36_mask2former`', '`best_mIoU_iter_3500.pth`',
         '252,650,755', '`f26cd5257b55058f`'],
        ['`poolformer-s36_fpn`', '`iter_10200.pth`',
         '416,437,419', '`59683e3788548494`'],
        ['`dpn98_fpn`', '`best_mIoU_iter_14000.pth`',
         '263,385,401', '`e292f2262f132f57`'],
        ['`convnext-small_upernet`', '`iter_14000.pth`',
         '983,511,196', '`8435410e25140545`'],
        ['`resnet-50_mask2former`', '`best_mIoU_iter_38500.pth`',
         '196,545,149', '`5a79f618902be032`'],
        ['`efficientnet-b3_fpn`', '`iter_6800.pth`',
         '108,104,299', '`9d6131e21a1ec5f3`'],
    ], widths=[124, 116, 62, None], size=8.3,
        caption='Identify a checkpoint by its digest, not by its filename. '
                'The full digests are in `ghaf/release.py` and in each '
                '`metadata.json`.'))

    s.append(glue(section('What was never established'), None))
    s.append(para(
        'Stated so that nobody has to work out whether a number is missing or '
        'merely not repeated here. None of the following was measured, and '
        'this manual does not estimate them.'))
    s.extend(bullets([
        'Wall-clock time for a full-mosaic inference run; the mosaic has '
        'never been run end to end.',
        'Wall-clock time for training or for fine-tuning, on any hardware.',
        'Validation scores for the five models other than FastViT-MA36.',
        'Inference throughput per model.',
        'GPU memory actually consumed at batch size 4, which is why chapter 6 '
        'gives a starting point rather than a recommendation.',
        'The date each released checkpoint was trained, and which run in '
        '`work_dirs` produced it; the logs were not inspected.',
        'Any fine-tuning result on a second site, and therefore how far a '
        'fine-tuned model drifts from the original.',
        'How often touching crowns merge into one polygon on this imagery.',
    ]))
    s.append(glue(section('Where to look when it is not here'), None))
    s.extend(table([
        ['Source', 'For'],
        ['`docs/handover/FACTS.yml`',
         'Every value this manual prints, with its provenance'],
        ['github.com/open-mmlab/mmsegmentation',
         'Errors raised inside the framework. Search closed issues too'],
        ['`python tools/<program>.py --help`',
         'What an option is called and what it does'],
        ['`docs/AREA_WIDE_INFERENCE.md`',
         'How a mosaic is windowed and blended, in more detail than chapter 6'],
        ['`docs/MODEL_ZOO.md`', 'Per-class scores and training settings'],
    ], widths=[132, None], size=8.5))

    s.append(para(
        'The authority for every value in this manual is '
        '`docs/handover/FACTS.yml`. Each entry there carries its provenance: '
        'read from a file in the repository, measured by a command that was '
        'run, derived arithmetically, or marked as never established. Where '
        'this manual and that file could disagree, the file wins.'))
    return s
