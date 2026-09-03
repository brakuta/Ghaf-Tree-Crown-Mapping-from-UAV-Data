#!/usr/bin/env python3
"""Build the two-page quick reference.

A separate document on purpose. The manual is read once and consulted
occasionally; this is the sheet that sits next to the machine, and mixing the
two would mean printing twenty-two pages to get at two.

    python docs/manual/quickref.py [output.pdf]
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import typeset as T  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.platypus import Paragraph, Spacer  # noqa: E402


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        HERE / 'Ghaf-Crown-Mapping-Quick-Reference.pdf')

    T.register_fonts()
    T.SS = T.stylesheet()
    from typeset import HRule, callout, code, glue, para, table

    s = []
    s.append(Paragraph('Ghaf crown mapping', ParagraphStyle(
        'qt', parent=T.SS['TitleBig'], fontSize=17, leading=21, spaceAfter=2)))
    s.append(Paragraph('Quick reference', ParagraphStyle(
        'qs', parent=T.SS['TitleSub'], fontSize=11, spaceAfter=5)))
    s.append(HRule(thickness=1.0, colour=T.INK, space=3))
    s.append(Spacer(1, 8))
    s.append(para(
        'Everything here is expanded in the technical manual; the numbers in '
        'the right-hand column point at its chapters. Values come from '
        '`docs/handover/FACTS.yml`, which wins over this sheet if they ever '
        'disagree. Every command below assumes the `ghaf` environment is '
        'active, the working directory is the code folder, and `MODEL` names '
        'the FastViT model folder.'))

    s.append(glue(Paragraph('Prove it works — in this order', T.SS['Section']),
                  table([
        ['Command', 'Passes when', 'Ch.'],
        ['`python -m pytest tests\\ -q`',
         '`325 passed, 1 skipped` — the code is intact', '4'],
        ['`python tools\\smoke_test.py --checkpoints ..\\models`',
         'six rows of `ok`, every tensor matched', '4'],
        ['`python tools\\check_dataset.py ..\\data\\ghaf --sample 0`',
         '7005 / 869 / 767 paired', '4'],
    ], widths=[228, 160, None], size=8.4)))
    s.append(para(
        '`0.00% ghaf` in the prediction column of the second command is '
        'correct: that forward pass runs on a blank synthetic tile, not on '
        'imagery.'))

    s.append(Paragraph('Map one orthomosaic', T.SS['Section']))
    s.extend(code(
        'python -m ghaf.inference.large_image ^\n'
        '%MODEL%\\fastvit-ma36_mask2former.py ^\n'
        '%MODEL%\\best_mIoU_iter_3500.pth ^\n'
        '..\\samples\\Kalba26_sample.tif ^\n'
        '--out-mask ..\\output\\crowns.tif ^\n'
        '--out-polygons ..\\output\\crowns.gpkg ^\n'
        '--batch-size 4 --min-area 1'))
    s.extend(callout('The defaults are not the settings that produced the results', [
        'Batch size defaults to 1, and `--min-area` to 0.0 — which keeps every '
        'single-pixel fragment as a crown and inflated the count on the sample '
        'clip from 16 to 27. `--out-polygons` will not run without '
        '`--out-mask` or `--out-prob`, and **`--tile` must stay 1024**: change '
        'it and the run still succeeds, still writes all three outputs, and '
        'reports a canopy figure produced at the wrong scale. *(Manual §6.5)*',
    ]))

    s.append(Paragraph('Map a folder of images', T.SS['Section']))
    s.extend(code(
        'python tools\\predict_folder.py ^\n'
        '%MODEL%\\fastvit-ma36_mask2former.py ^\n'
        '%MODEL%\\best_mIoU_iter_3500.pth ^\n'
        'D:\\my-images --out-dir ..\\output\\folder ^\n'
        '--batch-size 4 --min-area 1 --limit 3'))
    s.append(para(
        'Drop `--limit 3` once three images have come out right. One '
        'unreadable file does not stop the run: it is recorded in '
        '`summary.json` and the exit status is non-zero, neither of which is '
        'visible to somebody watching the progress bar. *(Manual §7.2)*'))

    s.append(glue(Paragraph('The numbers', T.SS['Section']), table([
        ['', 'Value', '', 'Value'],
        ['Tile', '1024 px', 'Trained GSD', '2.68 cm/px'],
        ['Overlap', '512 px', 'Tile on the ground', '27.44 m (0.075 ha)'],
        ['Threshold', '0.5', 'Scratch needed', '9 bytes per source pixel'],
        ['Batch used here', '4', 'Sample clip', '8192 × 8192, 225 windows'],
        ['Canopy, clip', '0.9687 %', 'Canopy, test split', '3.44 %'],
        ['Crowns, clip', '27 raw, 16 filtered', 'Crown area', '2.4 to 112 m²'],
    ], widths=[92, 88, 105, None], size=8.6)))
    s.append(para(
        'Deployed model mIoU **79.32**, F1 **87.22** on `testing/ghaf26`. '
        'Identify a checkpoint by its digest, not its filename: '
        '`f26cd5257b55058f…` for `best_mIoU_iter_3500.pth`, 252,650,755 bytes. '
        'The full digests are in `ghaf/release.py`.'))

    s.append(glue(Paragraph('Error → cause', T.SS['Section']), table([
        ['What you see', 'Actually means', 'Ch.'],
        ['`No module named mmengine`', 'wrong Python; `conda activate ghaf`', '13'],
        ['`mmseg ... is not installed in conda environment`',
         'wrong environment; the message names the interpreter', '13'],
        ['`No module named ftfy`', '`pip install ftfy regex`', '13'],
        ['`Numpy is not available`', 'NumPy 2 beside torch 1.12; pin `<2`', '13'],
        ['`CUDA out of memory`',
         'lower `--batch-size`; **never** `--overlap`', '13'],
        ['`not enough scratch space`', '`--scratch-dir` at a bigger disk', '13'],
        ['`--out-polygons needs --out-mask or --out-prob`',
         'polygons are traced from a written raster', '13'],
        ['`path specified` printed twice', 'an `&` in the path; quote it', '13'],
        ['`SHA-256 mismatch`', 'that checkpoint is not the released file', '13'],
        ['`0.00% ghaf` from `smoke_test.py`', 'correct; a blank tile', '4'],
        ['`0.00%` canopy on real imagery',
         'wrong checkpoint, or bands not RGB', '13'],
        ['a crown count far too high', 'fragments; use `--min-area 1`', '13'],
        ['high mIoU, predicts nothing', 'masks encoded 0 and 255, not 0 and 1', '13'],
    ], widths=[196, 250, None], size=8.4)))

    s.append(glue(Paragraph('Rules that are not negotiable', T.SS['Section']),
                  table([
        ['Rule', 'Because'],
        ['Never edit `configs/_base_/ghaf.py`',
         'all six models inherit it; a change silently redefines what every '
         'published score means'],
        ['Keep `work_dirs` logs',
         'no seed is set, so the seed of a run exists only in its log'],
        ['`--load-from` to fine-tune, never `--resume`',
         '`--resume` continues a schedule; `--load-from` starts a new one'],
        ['Score a fine-tuned model on the old split too',
         'it can gain on the new site and lose on the original, and nothing '
         'reports it'],
        ['Keep companion files beside their tiles',
         'moving a PNG without its `.pgw` strips the position silently'],
        ['`check_dataset.py` before any training run',
         'masks encoded 0 and 255 train to a high mIoU and predict nothing'],
    ], widths=[186, None], size=8.6)))

    s.append(Spacer(1, 4))
    s.append(para(
        '**Never established, so do not quote it:** how long a full-mosaic '
        'run takes — the 84,072 × 103,691 mosaic has never been run end to '
        'end, and its 78.5 GB of scratch and 33,128 windows are arithmetic. '
        'Also unmeasured: training time, validation scores for the five '
        'models other than FastViT-MA36, and how often touching crowns merge '
        'into one polygon. *(Manual §14.3)*'))

    for size, n, line in T.CODE_OVERLONG:
        print(f'[WARN] code line of {n} chars needs {size} pt: {line}')

    doc = T.Manual(
        str(out),
        title='Ghaf crown mapping — quick reference',
        running='Ghaf crown mapping · quick reference',
        cover_pages=0)
    doc.build(s)
    print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB)')
    return out


if __name__ == '__main__':
    main()
