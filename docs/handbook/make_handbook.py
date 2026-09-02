#!/usr/bin/env python
"""Build the Ghaf crown-mapping operator handbook as a PDF.

Everything the document says is drawn from the repository's own documentation
and from runs that were actually performed, so the expected outputs printed
here are the ones the reader will see.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# The PDF belongs beside the other documents, one level up.
OUT = Path(__file__).resolve().parent.parent / 'Ghaf-Crown-Mapping-Handbook.pdf'

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm

INK = colors.HexColor('#1a1a1a')
MUTED = colors.HexColor('#5b6670')
RULE = colors.HexColor('#c9d1d9')
GREEN = colors.HexColor('#1f6f43')
CODE_BG = colors.HexColor('#f4f6f8')
NOTE_BG = colors.HexColor('#eef4fb')
WARN_BG = colors.HexColor('#fdf3e7')
HEAD_BG = colors.HexColor('#eaeef2')

TITLE = 'Mapping Ghaf Tree Crowns from UAV Imagery'
SUBTITLE = 'A step-by-step operating handbook'

# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------

ss = getSampleStyleSheet()

S = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=26,
                            leading=31, textColor=INK, alignment=TA_CENTER,
                            spaceAfter=8),
    'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=14,
                               leading=19, textColor=MUTED, alignment=TA_CENTER,
                               spaceAfter=26),
    'cover': ParagraphStyle('cover', fontName='Helvetica', fontSize=10.5,
                            leading=16, textColor=INK, alignment=TA_CENTER),
    'part': ParagraphStyle('part', fontName='Helvetica-Bold', fontSize=11,
                           leading=14, textColor=GREEN, spaceBefore=2,
                           spaceAfter=2),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=17,
                         leading=21, textColor=INK, spaceBefore=4,
                         spaceAfter=10),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12.5,
                         leading=16, textColor=INK, spaceBefore=14,
                         spaceAfter=5),
    'h3': ParagraphStyle('h3', fontName='Helvetica-BoldOblique', fontSize=10.5,
                         leading=14, textColor=INK, spaceBefore=10,
                         spaceAfter=3),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9.8,
                           leading=14.2, textColor=INK, spaceAfter=7,
                           alignment=TA_LEFT),
    'bullet': ParagraphStyle('bullet', fontName='Helvetica', fontSize=9.8,
                             leading=14.2, textColor=INK, spaceAfter=3,
                             leftIndent=12, bulletIndent=2),
    'code': ParagraphStyle('code', fontName='Courier', fontSize=8.2,
                           leading=11.2, textColor=INK),
    'out': ParagraphStyle('out', fontName='Courier', fontSize=7.6,
                          leading=10.2, textColor=colors.HexColor('#333333')),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.6,
                           leading=11.6, textColor=INK),
    'cellb': ParagraphStyle('cellb', fontName='Helvetica-Bold', fontSize=8.6,
                            leading=11.6, textColor=INK),
    'cellc': ParagraphStyle('cellc', fontName='Courier', fontSize=7.8,
                            leading=11, textColor=INK),
    'caption': ParagraphStyle('caption', fontName='Helvetica-Oblique',
                              fontSize=8.4, leading=11.5, textColor=MUTED,
                              spaceAfter=8),
    'toc1': ParagraphStyle('toc1', fontName='Helvetica-Bold', fontSize=10,
                           leading=16, textColor=INK),
    'toc2': ParagraphStyle('toc2', fontName='Helvetica', fontSize=9.4,
                           leading=13.5, leftIndent=14, textColor=INK),
}

CONTENT_W = PAGE_W - 2 * MARGIN
CODE_COLS = 96          # Courier 8.2pt fits comfortably inside the frame


# --------------------------------------------------------------------------
# flowable helpers
# --------------------------------------------------------------------------

def esc(text: str) -> str:
    return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def md(text: str) -> str:
    """Bold with **, monospace with `` -- after escaping."""
    out = esc(textwrap.dedent(text).strip())
    while '**' in out:
        out = out.replace('**', '<b>', 1).replace('**', '</b>', 1)
    while '`' in out:
        out = out.replace('`', '<font face="Courier" size="9">', 1)
        out = out.replace('`', '</font>', 1)
    return out


def P(text, style='body'):
    return Paragraph(md(text), S[style])


def H1(text, story):
    story.append(Paragraph(md(text), S['h1']))


def bullets(items, story, numbered=False):
    for i, item in enumerate(items, start=1):
        mark = f'{i}.' if numbered else '\u2022'
        story.append(Paragraph(md(item), S['bullet'], bulletText=mark))
    story.append(Spacer(1, 5))


def _wrap(text: str, cols: int) -> str:
    lines = []
    for raw in textwrap.dedent(text).strip('\n').split('\n'):
        if len(raw) <= cols:
            lines.append(raw)
            continue
        # Wrap on spaces, indenting the continuation so it reads as one command.
        indent = '      '
        current = ''
        for word in raw.split(' '):
            candidate = word if not current else f'{current} {word}'
            if len(candidate) > cols and current:
                lines.append(current)
                current = indent + word
            else:
                current = candidate
        lines.append(current)
    return '\n'.join(lines)


def code(text, story, label=None):
    """A command to type, in a shaded box."""
    body = Preformatted(_wrap(text, CODE_COLS), S['code'])
    rows = [[body]]
    t = Table(rows, colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CODE_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    block = [t, Spacer(1, 8)]
    if label:
        block.insert(0, Paragraph(md(label), S['h3']))
    story.append(KeepTogether(block))


def output(text, story, caption='What you should see'):
    body = Preformatted(_wrap(text, 100), S['out'])
    t = Table([[body]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, RULE),
        ('LINEBEFORE', (0, 0), (0, -1), 2.2, GREEN),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    block = [Paragraph(md(caption), S['caption']), t, Spacer(1, 9)]
    story.append(KeepTogether(block))


def box(text, story, kind='note', title=None):
    bg = NOTE_BG if kind == 'note' else WARN_BG
    edge = colors.HexColor('#4a7fb5') if kind == 'note' else colors.HexColor('#c9821f')
    inner = []
    if title:
        inner.append(Paragraph(md(title), ParagraphStyle(
            'boxt', parent=S['body'], fontName='Helvetica-Bold', spaceAfter=3)))
    inner.append(Paragraph(md(text), ParagraphStyle(
        'boxb', parent=S['body'], spaceAfter=0)))
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('LINEBEFORE', (0, 0), (0, -1), 2.5, edge),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(KeepTogether([t, Spacer(1, 9)]))


def table(header, rows, story, widths=None, mono_first=False):
    def cell(text, style):
        return Paragraph(md(str(text)), S[style])

    data = [[cell(h, 'cellb') for h in header]]
    for row in rows:
        first = 'cellc' if mono_first else 'cell'
        data.append([cell(row[0], first)] + [cell(c, 'cell') for c in row[1:]])

    if widths is None:
        widths = [CONTENT_W / len(header)] * len(header)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEAD_BG),
        ('LINEBELOW', (0, 0), (-1, 0), 0.6, RULE),
        ('GRID', (0, 0), (-1, -1), 0.35, RULE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafbfc')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


# --------------------------------------------------------------------------
# document scaffolding: numbered headings, running footer, contents
# --------------------------------------------------------------------------

class Handbook(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4,
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=18 * mm,
                         title=TITLE, author='Ghaf crown-mapping project',
                         subject=SUBTITLE, **kw)
        frame = Frame(MARGIN, 18 * mm, CONTENT_W,
                      PAGE_H - MARGIN - 18 * mm - 6 * mm, id='body')
        self.addPageTemplates([
            PageTemplate(id='cover', frames=[frame]),
            PageTemplate(id='main', frames=[frame], onPage=self.decorate),
        ])
        self.chapter = ''
        self.chapter_by_page = {}     # from the previous layout pass
        self._pending = {}            # being collected this pass

    def beforeDocument(self):
        # Each pass starts fresh: entries from an earlier pass name pages that
        # have since moved, and a stale one would caption the wrong page.
        self.chapter_by_page = self._pending
        self._pending = {}

    def running_head(self, page):
        """The chapter this page belongs to, from the previous layout pass.

        onPage runs before the page's flowables, so the chapter cannot be
        known on the first pass; multiBuild lays the document out more than
        once, and by the last pass this map is complete.
        """
        known = [n for n in self.chapter_by_page if n <= page]
        return self.chapter_by_page[max(known)] if known else ''

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 4, TITLE)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 4,
                               self.running_head(doc.page))
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, PAGE_H - MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN)
        canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canvas.drawString(MARGIN, 11 * mm, 'Ghaf crown mapping - operating handbook')
        canvas.drawRightString(PAGE_W - MARGIN, 11 * mm, f'page {doc.page}')
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        if style == 'h1':
            text = flowable.getPlainText()
            self.chapter = text
            self._pending[self.page] = text
            self.notify('TOCEntry', (0, text, self.page))
        elif style == 'h2':
            self.notify('TOCEntry', (1, flowable.getPlainText(), self.page))


def build(story):
    doc = Handbook(str(OUT))
    doc.multiBuild(story)


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------

story = []

# ---- cover ---------------------------------------------------------------
story += [
    Spacer(1, 52 * mm),
    Paragraph(TITLE, S['title']),
    Paragraph(SUBTITLE, S['subtitle']),
]
cover = Table([[Paragraph(
    '<i>Prosopis cineraria</i> (Ghaf) crown delineation from area-wide UAV '
    'orthomosaics, using the trained FastViT-MA36 + Mask2Former model.',
    ParagraphStyle('cv', parent=S['cover'], fontSize=11, leading=16))]],
    colWidths=[120 * mm])
cover.setStyle(TableStyle([
    ('LINEABOVE', (0, 0), (-1, 0), 0.8, RULE),
    ('LINEBELOW', (0, 0), (-1, -1), 0.8, RULE),
    ('TOPPADDING', (0, 0), (-1, -1), 12),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
]))
story += [
    Table([[cover]], colWidths=[CONTENT_W],
          style=TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')])),
    Spacer(1, 16 * mm),
    Paragraph(
        'Written for a reader who is comfortable following instructions and '
        'does not write code.<br/>Every step is a command to copy, paste and '
        'run, with what it does, how long it takes,<br/>and what a correct '
        'result looks like.', S['cover']),
    Spacer(1, 22 * mm),
    Paragraph(
        'Repository: github.com/brakuta/Ghaf-Tree-Crown-Mapping-from-UAV-Data'
        '<br/>Accompanies the manuscript "Hybrid Vision-CNN Architecture for '
        'Mapping <i>Prosopis cineraria</i><br/>from Area-wide UAV-based Images"'
        '<br/><br/>First issue, September 2026',
        ParagraphStyle('cv2', parent=S['cover'], fontSize=9, textColor=MUTED)),
    NextPageTemplate('main'),
    PageBreak(),
]

# ---- contents ------------------------------------------------------------
toc = TableOfContents()
toc.levelStyles = [S['toc1'], S['toc2']]
story += [Paragraph('Contents', S['h1']), toc, PageBreak()]

# =========================================================================
H1('1. How to use this handbook', story)
story.append(P("""
This handbook takes you from a computer with nothing installed to crown
polygons you can open in QGIS. It assumes no programming. It does assume you
will read the screen: almost everything that goes wrong announces itself in
the last two or three lines of output, and chapter 12 is a method for turning
those lines into a fix.
"""))

story.append(Paragraph('Conventions', S['h2']))
story.append(P("""
A shaded box is something you type (or paste) and run. Press Enter after it,
and wait for the prompt to come back before typing the next one:
"""))
code(r'python tools\smoke_test.py', story)
story.append(P("""
A box with a green edge is what a correct run prints. Yours will not match
character for character - times and file paths differ - but the shape and the
key numbers should:
"""))
output("""all 6 model(s) built and ran a forward pass""", story, caption='Example')

box("""
**Paste one command at a time.** Some terminals join a multi-line paste into a
single line and run something you did not intend. If a command in this
handbook wraps onto a second line, it is still **one** command: the wrap is
the page being narrow, not a second line to type. Join the pieces with a
single space.
""", story, kind='warn')

story.append(P("""
Paths in this handbook are written as `D:\\ghaf-project\\...`. Substitute
wherever your copy actually lives. Nothing else changes. Commands are written
for the Windows Command Prompt; on macOS or Linux they are identical except
that paths use `/` instead of `\\`.
"""))

story.append(Paragraph('If you are in a hurry', S['h2']))
story.append(P("""
Chapters 5 and 6 install and prove the installation; chapter 7 maps an image.
Those three are the shortest path to a result. Everything else can wait until
you need it.
"""))

story.append(PageBreak())
# =========================================================================
H1('2. What this system does', story)
story.append(P("""
You give it an aerial image. It gives you back the Ghaf tree crowns in that
image, as shapes on a map.
"""))
story.append(P("""
Underneath, a neural network looks at the image in 1024 x 1024 pixel windows
and decides, for every single pixel, whether that pixel is part of a Ghaf
crown or not. The windows overlap by half their width and the answers are
blended where they overlap, so there are no visible seams between windows. The
result is then traced into polygons - one polygon per crown - each carrying
its area in square metres.
"""))
story.append(P("""
The model in use is **FastViT-MA36 + Mask2Former**, which scored **79.32 mIoU**
and **87.22 F1** on the held-out test tiles. Five other models are provided for
comparison; unless you have a reason, use the FastViT one.
"""))

story.append(Paragraph('What it is good at, and what it is not', S['h2']))
table(['It can', 'It cannot'],
      [['Map crowns in an orthomosaic of any size, from a small plot to a '
        'billion-pixel survey',
        'Tell one tree species from another beyond Ghaf and "not Ghaf" - it '
        'was trained on two classes only'],
       ['Return crowns as GIS polygons with areas, ready to count and measure',
        'Work on imagery that is not red-green-blue 8-bit; other band '
        'arrangements need `--bands`'],
       ['Reproduce the published scores exactly, from the weights you were given',
        'Separate two crowns that physically touch and overlap into two '
        'polygons - they may merge into one'],
       ['Be re-trained or fine-tuned on labelled tiles from a new site',
        'Run usefully fast without an NVIDIA GPU, though it will run on a CPU']],
      story, widths=[CONTENT_W * 0.5, CONTENT_W * 0.5])

story.append(Paragraph('The three outputs, in plain terms', S['h2']))
table(['Output', 'What it is', 'What it is for'],
      [['crowns.gpkg', 'A GeoPackage: the crowns as polygons, with an '
        '`area_m2` column',
        'Counting trees, measuring crowns, joining to other GIS data. This is '
        'the one most people want'],
       ['crowns.tif', 'A picture the same size as the input where each pixel '
        'is 1 (crown) or 0 (not)',
        'Overlaying on the imagery, computing canopy cover, differencing '
        'against a labelled mask'],
       ['probability.tif', 'The same shape, but each pixel is the model\'s '
        'confidence from 0.00 to 1.00',
        'Choosing a stricter or looser cut-off after the fact, and seeing '
        'where the model was unsure']],
      story, widths=[CONTENT_W * 0.20, CONTENT_W * 0.40, CONTENT_W * 0.40],
      mono_first=True)

story.append(PageBreak())
# =========================================================================
H1('3. What you were given', story)
story.append(P("""
Three parts, kept separate on purpose. The code is public; the weights and the
imagery are not.
"""))
table(['Part', 'What it is', 'Where it comes from'],
      [['The code', 'Configurations, tools, documentation',
        'Public on GitHub'],
       ['The models', 'Six trained checkpoints, one folder each',
        'Shared with you directly. Not on GitHub'],
       ['The data', 'Labelled tiles, and sample imagery to run on',
        'Shared with you directly. Not on GitHub']],
      story, widths=[CONTENT_W * 0.18, CONTENT_W * 0.47, CONTENT_W * 0.35])

story.append(Paragraph('The folder you received', S['h2']))
output(r"""
D:\ghaf-project\
+-- README.md                          what this is; start here
+-- MANIFEST.json                      every part, its size, and what was checked
+-- code\                              the repository: tools, configs, docs
+-- models\                            2.2 GB
|   +-- fastvit-ma36_mask2former\
|   |   +-- fastvit-ma36_mask2former.py    the complete recipe, self-contained
|   |   +-- best_mIoU_iter_3500.pth        the trained weights
|   |   +-- metadata.json                  fingerprint, size, scores
|   +-- ...                                five more models
+-- init-weights\                      0.9 GB  ImageNet weights, for training offline
+-- data\ghaf\                         15.6 GB
|   +-- training\{images,masks}\           7005 tile pairs
|   +-- validation\{images,masks}\          869 tile pairs
|   +-- testing\ghaf26\{images,masks}\      767 tile pairs
+-- samples\
|   +-- Kalba26_sample.tif                 a small clip: start here, minutes
+-- predictions\testing\               predictions already made for the test split
""", story, caption='The layout assumed throughout this handbook')

story.append(P("""
Tiles are 1024 x 1024 PNG pairs: an image and a mask sharing a filename. In a
mask, `0` is background and `1` is a Ghaf crown.
"""))
story.append(P("""
The test tiles have small companion files beside them (`.pgw`, `.png.aux.xml`,
`.ovr`). These hold the patch of ground each tile covers, and display
pyramids, so predictions made from them open in QGIS already in the right
place. The training and validation tiles have none, which changes nothing
about training or scoring. Do not delete or move these companion files.
"""))

box("""
**MANIFEST.json is the receipt.** It records every part of the bundle, its
size, the number of files, and the exact version of the code it was built
from. If you ever need to prove that a copy is complete, or work out which
version of the code produced a result, it is in there.
""", story)

story.append(Paragraph('What the computer needs', S['h2']))
table(['Item', 'Needed', 'Notes'],
      [['Operating system', 'Windows 10/11, macOS or Linux',
        'This handbook shows Windows commands'],
       ['GPU', 'An NVIDIA GPU, 8 GB or more',
        'The published work used an RTX A5000. Without a GPU everything still '
        'runs with `--device cpu`, perhaps 20-50 times slower'],
       ['Disk for the bundle', 'About 20 GB',
        'Code, models, initialisation weights, tiles, samples'],
       ['Disk for a run', '9 bytes per pixel of the image being mapped',
        'An 8192 x 8192 clip needs 0.6 GB; the full 84 072 x 103 691 mosaic '
        'needs about 79 GB. Freed when the run ends'],
       ['Software', 'Miniconda or Anaconda',
        'Free, from docs.conda.io. Everything else is installed by the '
        'commands in chapter 5']],
      story, widths=[CONTENT_W * 0.20, CONTENT_W * 0.28, CONTENT_W * 0.52])

story.append(PageBreak())
# =========================================================================
H1('4. Words you will meet', story)
story.append(P("""
Skim this once. You do not need to memorise it; come back when a word in a
later chapter is unfamiliar.
"""))
table(['Word', 'What it means here'],
      [['Orthomosaic', 'One large image made by stitching together the '
        'hundreds of photographs taken on a drone flight, geometrically '
        'corrected so that it can be measured like a map'],
       ['Georeferenced', 'The file knows where on Earth it sits. Two layers '
        'that are both georeferenced line up automatically in QGIS'],
       ['CRS', 'Coordinate Reference System - the map projection the '
        'coordinates are in. This project\'s imagery is EPSG:32640 (UTM zone '
        '40 North), which covers the UAE'],
       ['GSD', 'Ground sampling distance: how much ground one pixel covers. '
        'The sample mosaic is about 2.7 cm per pixel'],
       ['Tile', 'A small square cut out of a large image. Training uses '
        '1024 x 1024 tiles'],
       ['Mask', 'An image where the pixel value is a label rather than a '
        'colour. Here, 0 = background, 1 = Ghaf'],
       ['Checkpoint (.pth)', 'A file holding a trained model\'s weights - what '
        'it learned. Roughly 300-800 MB each'],
       ['Config (.py)', 'The complete recipe for a model: its architecture, '
        'the data pipeline, the training schedule'],
       ['Inference', 'Using a trained model to make predictions. The opposite '
        'of training'],
       ['mIoU / F1', 'Accuracy scores between 0 and 100. Higher is better. '
        'They measure how well predicted crowns overlap the hand-drawn ones'],
       ['Threshold', 'The confidence above which a pixel is called a crown. '
        'The default is 0.50'],
       ['Environment (conda)', 'A private, self-contained installation of '
        'Python and its packages, so this project cannot break anything else '
        'on the computer. Ours is named `ghaf`'],
       ['Terminal / Command Prompt', 'The window where you type commands. On '
        'Windows, use the Anaconda Prompt'],
       ['GeoPackage (.gpkg)', 'A standard GIS file holding shapes and their '
        'attributes. Opens in QGIS and ArcGIS'],
       ['Scratch space', 'Temporary disk the program uses while working, '
        'released when it finishes']],
      story, widths=[CONTENT_W * 0.22, CONTENT_W * 0.78])

story.append(PageBreak())
# =========================================================================
H1('5. Installing, once', story)
story.append(P("""
This takes about twenty minutes on a good connection and is done once per
computer. Every command goes in the same window.
"""))

story.append(Paragraph('Step 0 - open the right window', S['h2']))
bullets([
    'Press the Windows key, type **Anaconda Prompt**, and open it.',
    'A black or blue window appears with a line ending in `>`. That line is '
    'the **prompt**; you type after it.',
    'If the prompt starts with `(base)`, conda is installed and working. If '
    'nothing called Anaconda Prompt exists, install Miniconda first from '
    'docs.conda.io, then open a new one.',
], story, numbered=True)

story.append(Paragraph('Step 1 - create the environment and enter it', S['h2']))
code('conda create -n ghaf python=3.9 -y', story)
code('conda activate ghaf', story)
box("""
**The prompt now begins with `(ghaf)`.** That is how you know you are in the
right place. Every command in this handbook assumes it. If you close the
window and open a new one, run `conda activate ghaf` again - this is the
single most common cause of confusing errors later.
""", story)

story.append(Paragraph('Step 2 - install PyTorch', S['h2']))
story.append(P("""
This is the exact version the published models were trained with. It is a
large download, around 2 GB.
"""))
code('python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 '
     '--extra-index-url https://download.pytorch.org/whl/cu113', story)
story.append(P("""
No NVIDIA GPU? Use this instead. Everything still runs, just slowly:
"""))
code('python -m pip install torch==1.12.1 torchvision==0.13.1', story)

story.append(Paragraph('Step 3 - install the OpenMMLab framework', S['h2']))
story.append(P("""
Three commands, in this order. The version numbers matter: they are the ones
the models were built and tested against.
"""))
code('python -m pip install -U openmim', story)
code('python -m mim install mmengine==0.10.7 "mmcv>=2.0.0rc4,<2.2.0"', story)
code('python -m pip install mmsegmentation==1.2.2 mmdet==3.3.0 mmpretrain==1.2.0',
     story)
story.append(P("""
`mmcv` is the slowest of these - it is fetching a large pre-built package
matched to your PyTorch and CUDA versions. Several minutes is normal.
"""))

story.append(Paragraph('Step 4 - install this project', S['h2']))
story.append(P("""
Move into the code folder first. The `/d` is needed when changing to another
drive letter:
"""))
code(r'cd /d D:\ghaf-project\code', story)
code('python -m pip install -r requirements.txt', story)
code('python -m pip install -e ".[test]"', story)

box("""
**Two messages during installation that are not errors.** pip may say that
`opencv-python` requires `numpy>=2`; ignore it, OpenCV works with either.
It may also print warnings about dependency conflicts among packages you are
not using. Only a line beginning `ERROR:` that stops the install matters.
""", story, kind='warn')

story.append(Paragraph('A note about paths with spaces or "&"', S['h2']))
story.append(P("""
If your project folder contains spaces, put the whole path in double quotes.
If it contains an ampersand (`&`), quotes are **required**: without them the
Command Prompt treats `&` as "end of command, start another one", and you get
a confusing "The system cannot find the path specified" twice over.
"""))
code(r'cd /d "Z:\Survey Data\Cineraria_Data & Model\ghaf-project\code"', story)

story.append(PageBreak())
# =========================================================================
H1('6. Proving that it works', story)
story.append(P("""
Three checks, about five minutes in total. Run them before anything else, and
again on any day when something behaves strangely. They confirm that the code,
the weights and the tiles all arrived intact and agree with each other.
"""))

story.append(Paragraph('Check 1 - the code', S['h2']))
story.append(P('No GPU and no data needed, about a minute.'))
code(r'python -m pytest tests\ -q', story)
output("""
.........................................................................
.........................................................................
325 passed, 1 skipped in 74.19s
""", story)
story.append(P("""
Any number of passing tests with no `F` characters and no `failed` in the last
line is a pass. If you see `No module named 'mmengine'`, you are running a
different Python than the one you installed into - see chapter 12, it is the
first entry.
"""))

story.append(Paragraph('Check 2 - the models', S['h2']))
story.append(P("""
This builds all six models from their configurations, loads the weights you
were given, checks every file against its published fingerprint, and runs a
prediction through each. About 90 seconds on a GPU.
"""))
code(r'python tools\smoke_test.py --checkpoints D:\ghaf-project\models', story)
output("""
model                            digest                    weights  prediction
--------------------------------------------------------------------------------
fastvit-ma36_mask2former             ok          all 1,558 matched    0.00% ghaf
poolformer-s36_fpn                   ok            all 436 matched    0.00% ghaf
dpn98_fpn                            ok            all 676 matched    0.00% ghaf
convnext-small_upernet               ok            all 430 matched    0.00% ghaf
resnet-50_mask2former                ok            all 610 matched    0.00% ghaf
efficientnet-b3_fpn                  ok            all 610 matched    0.00% ghaf

all 6 model(s) built and ran a forward pass
""", story)
table(['Column', 'What it proves'],
      [['`digest` = ok', 'The checkpoint file is byte-for-byte the one that '
        'was released. Nothing was corrupted in the copy'],
       ['`all N matched`', 'Every weight in the file found its place in the '
        'model built from the configuration - nothing missing, nothing left '
        'over'],
       ['`+0` in the delta column', 'The number of parameters matches the '
        'published figure exactly'],
       ['`0.00% ghaf`', '**Correct.** This step predicts on a blank synthetic '
        'tile, not on imagery, so finding no trees is what should happen. '
        'Real imagery gives a few percent']],
      story, widths=[CONTENT_W * 0.28, CONTENT_W * 0.72])

box("""
**Every command that loads a model prints screens of `UserWarning` lines** -
about `__floordiv__`, `torch.meshgrid`, binary segmentation and `build_loss`.
They come from inside PyTorch and mmsegmentation, they appear on a perfectly
correct run, and there is nothing to do about them. Read past them to the
table at the end.
""", story)

story.append(Paragraph('Check 3 - the data', S['h2']))
code(r'python tools\check_dataset.py D:\ghaf-project\data\ghaf', story)
output("""
split          images    masks   paired  checked  status
--------------------------------------------------------------
training         7005     7005     7005      200  ok
validation        869      869      869      200  ok
testing           767      767      767      200  ok

8,641 paired tile(s) across 3 split(s)
dataset looks usable
""", story)
table(['Variation', 'When to use it'],
      [[r'check_dataset.py PATH --full', 'Opens every one of the 8641 tiles '
        'instead of a sample of 200. Slow, thorough; worth doing once'],
       [r'check_dataset.py PATH --sample 0', 'Checks only that every image has '
        'a matching mask, without opening any. Seconds, even over a network '
        'drive'],
       [r'check_dataset.py PATH --sample 25', 'Opens 25 tiles per split. A '
        'good compromise after copying the data somewhere new']],
      story, widths=[CONTENT_W * 0.42, CONTENT_W * 0.58], mono_first=True)

story.append(PageBreak())
# =========================================================================
H1('7. Mapping crowns in one image', story)
story.append(P("""
This is the main thing the models are for: give it a UAV orthomosaic, get back
a canopy map. The image can be far larger than the computer's memory - it is
read in overlapping windows and blended, so nothing has to fit at once.
"""))

box("""
**Start with the small clip.** `Kalba26_sample.tif` is a piece cut from the
full survey. It runs in a couple of minutes and produces exactly the same kind
of output, so it proves the whole chain works before you commit to a run of
hours. Only once that has worked is the full mosaic worth starting.
""", story)

story.append(Paragraph('The command', S['h2']))
story.append(P("""
Move into the code folder first, so the rest of the command stays short:
"""))
code(r'cd /d D:\ghaf-project\code', story)
code(r"""python -m ghaf.inference.large_image ..\models\fastvit-ma36_mask2former\fastvit-ma36_mask2former.py ..\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth ..\samples\Kalba26_sample.tif --out-mask ..\output\crowns.tif --out-prob ..\output\probability.tif --out-polygons ..\output\crowns.gpkg --batch-size 4""",
     story)
story.append(P("""
Reading it left to right: the **recipe**, the **weights**, the **image**, then
where to put each of the three outputs. `--batch-size 4` processes four
windows at a time, which is faster on a GPU with memory to spare.
"""))

output("""
INFO Kalba26_sample.tif: 8192 x 8192 px, 225 window(s) of 1024 px (overlap 512), batch 4, 0.6 GB scratch
Kalba26_sample: 100%|##############################| 57/57 [00:40<00:00,  1.42batch/s]
INFO Created 27 records
INFO wrote ..\\output\\crowns.gpkg (27 polygon(s))
INFO canopy: 650055 of 67108864 valid px (0.97%)
""", story)

story.append(P("""
Between 40 seconds and two and a half minutes on one GPU, depending on how
many outputs are written and whether they go to a local disk or a network
drive. The progress bar is the honest guide to how much is left.
"""))

story.append(Paragraph('Is the answer sensible?', S['h2']))
table(['Number', 'Expected', 'If it is far off'],
      [['Canopy percentage', 'About **1%** on this clip. Scattered Ghaf in '
        'desert is a low number by nature',
        '`0.00%` means the run found nothing; above 50% means it found '
        'everything. Both point upstream: usually the wrong checkpoint, or '
        'bands that are not red, green, blue'],
       ['Polygon count', '27 on this clip at the default settings',
        'A wildly larger number usually means single-pixel specks - see '
        '`--min-area` below'],
       ['Crown areas', 'Roughly 2 to 112 m2 for real crowns',
        'Hundreds of square metres suggests crowns merged, or the image is '
        'not at the resolution the model expects']],
      story, widths=[CONTENT_W * 0.20, CONTENT_W * 0.36, CONTENT_W * 0.44])

story.append(Paragraph('Adjustments worth knowing', S['h2']))
table(['Option', 'Effect'],
      [['--threshold 0.6', 'Stricter: fewer, more confident crowns. `0.4` is '
        'more inclusive. Default 0.5'],
       ['--min-area 1', 'Drop crown polygons under 1 m2. Removes the stray '
        'single-pixel specks that any threshold leaves behind, which '
        'otherwise inflate a crown count. On the sample clip this removes 11 '
        'of 27 "crowns" and leaves the 16 real ones'],
       ['--batch-size 4', 'Faster on a GPU with spare memory. Lower it to 1 if '
        'you see `CUDA out of memory`'],
       ['--device cpu', 'Run without a GPU. Much slower, but it works'],
       [r'--scratch-dir E:\scratch', 'Put the temporary working files on a '
        'bigger or faster drive'],
       ['--bands 1 2 3', 'Which bands of the image are red, green and blue. '
        'Only needed for unusual imagery']],
      story, widths=[CONTENT_W * 0.26, CONTENT_W * 0.74], mono_first=True)

story.append(Paragraph('The full survey mosaic', S['h2']))
story.append(P("""
`Kalba26.tif` is 84 072 x 103 691 pixels - 8.7 billion of them. At 9 bytes per
pixel that is about **79 GB of temporary space**, and roughly 33 500 windows
to predict: a run of hours rather than minutes on one GPU. Point
`--scratch-dir` at a drive with room, start it, and leave it. The program
checks the free space before it begins rather than failing part-way through.
"""))

story.append(Paragraph('Cutting your own sample', S['h2']))
story.append(P("""
To try a different part of a mosaic, or to make a quick sample from a new
survey, cut one out first. A minute or two:
"""))
code(r'python tools\make_sample.py ..\samples\Kalba26.tif --output ..\samples\my_sample.tif --size 8192 --origin 30000 40000',
     story)
story.append(P("""
`--size` is the clip in pixels. `--origin` is its top-left corner; leave it out
and the clip is taken from the centre. The tool reports how much of the clip
is real imagery rather than the transparent border around a survey, so a badly
placed window is obvious immediately.
"""))

story.append(PageBreak())
# =========================================================================
H1('8. Mapping a whole folder of images', story)
story.append(P("""
When you have many images rather than one - a season's clips, a set of survey
plots, the frames from one flight - point the batch tool at the folder and
leave it. Each image is windowed exactly as a full mosaic is, so they need not
be the same size and none of them has to fit in memory.
"""))
code(r'cd /d D:\ghaf-project\code', story)
code(r"""python tools\predict_folder.py ..\models\fastvit-ma36_mask2former\fastvit-ma36_mask2former.py ..\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth D:\my-images --out-dir ..\output\folder --batch-size 4 --min-area 1""",
     story)
output(r"""
INFO 1 image(s) in ..\samples -> ..\output\folder
INFO [1/1] Kalba26_sample.tif
INFO Kalba26_sample.tif: 8192 x 8192 px, 225 window(s) of 1024 px (overlap 512), batch 4, 0.6 GB scratch
INFO dropped 11 polygon(s) under 1 m2
INFO wrote ..\output\folder\polygons\Kalba26_sample.gpkg (16 polygon(s))
INFO canopy: 650055 of 67108864 valid px (0.97%)
INFO 1 predicted, 0 skipped, 0 failed; canopy 0.97% of valid px
INFO wrote ..\output\folder\summary.json
""", story)

story.append(Paragraph('What it writes', S['h2']))
output(r"""
folder\
+-- polygons\      one .gpkg of crowns per image, with area_m2      <- the output
+-- masks\         the 0/1 raster, only if you add --save-mask
+-- probability\   the confidence raster, only with --save-probability
+-- summary.json   every image, its canopy share, and any failures
""", story, caption='Output layout')

story.append(P("""
**You get crowns, not rasters.** A GeoPackage of crowns is a few hundred
kilobytes where the mask behind it is hundreds of megabytes, and it is the
crowns you count, measure and drape over the imagery. The mask is still made -
the polygons are traced from it - but it is written to temporary space and
deleted when that image is done. Add `--save-mask` when you want to keep it.
"""))
story.append(P("""
If the input folder has subfolders, the output mirrors them, so two images
with the same name in different folders cannot overwrite each other.
"""))

story.append(Paragraph('Switches for a long batch', S['h2']))
table(['Switch', 'Effect'],
      [['--limit 3', 'Stop after three images. Always do this first on a big '
        'folder: it proves the settings before you commit to hundreds'],
       ['--recursive', 'Include images in subfolders'],
       ['--pattern "*_rgb.tif"', 'Only files whose name matches. Useful when a '
        'folder mixes RGB images with elevation models'],
       ['--skip-existing', 'Carry on where an interrupted run stopped, instead '
        'of redoing finished images'],
       ['--save-mask', 'Also keep the 0/1 raster per image'],
       ['--save-probability', 'Also keep the confidence raster per image'],
       ['--min-area 1', 'Drop crowns under 1 square metre']],
      story, widths=[CONTENT_W * 0.26, CONTENT_W * 0.74], mono_first=True)

box("""
**One bad file does not stop the run.** If an image is unreadable or has an
unexpected number of bands, that image is reported, the batch carries on, and
the failure is listed in `summary.json` at the end. Read `summary.json` after
any long run: it names anything that did not work.
""", story)

story.append(PageBreak())
# =========================================================================
H1('9. Looking at the results', story)
story.append(P("""
Everything the tools write carries the coordinate system of the image it came
from, so it lands in the right place on the map with no manual positioning.
"""))

story.append(Paragraph('Opening the crowns in QGIS', S['h2']))
bullets([
    'Open QGIS. **Layer > Add Layer > Add Vector Layer**, browse to '
    '`crowns.gpkg`, and click **Add**.',
    'Add the imagery underneath it: **Layer > Add Layer > Add Raster Layer**, '
    'and choose the image you mapped. Drag it below the crowns in the Layers '
    'panel.',
    'Right-click the crowns layer > **Properties > Symbology**. Set **Fill '
    'style** to *No brush* and pick a bright outline colour, so you can see '
    'the imagery through each crown.',
    'Right-click > **Open Attribute Table** to see one row per crown, with '
    '`area_m2`. The row count at the top of that window is your tree count.',
    'For totals: **Vector > Analysis Tools > Basic Statistics for Fields**, '
    'choose the crowns layer and the `area_m2` field. It gives the count, sum, '
    'mean, minimum and maximum crown area.',
], story, numbered=True)

story.append(Paragraph('Opening the rasters', S['h2']))
story.append(P("""
`crowns.tif` and `probability.tif` are added the same way, through **Add Raster
Layer**. Two tips that save confusion:
"""))
bullets([
    'The mask looks black at first, because its values are 0 and 1 and QGIS '
    'stretches 0-255 by default. Right-click > **Properties > Symbology**, set '
    'the render type to **Paletted/Unique values**, and click **Classify** - '
    'you then get two colours, one per class.',
    'For the probability raster, use **Singleband pseudocolor** and a colour '
    'ramp from 0 to 1. Anywhere yellow-to-red is where the model was most '
    'confident.',
], story)

story.append(Paragraph('Numbers to sanity-check against', S['h2']))
table(['Quantity', 'Value seen in this project'],
      [['Canopy share, sample clip', 'About 0.97% of valid pixels'],
       ['Canopy share, test split', 'About 3.44% across the 767 test tiles'],
       ['Crown count, sample clip', '27 raw; 16 after `--min-area 1`'],
       ['Crown areas, sample clip', 'About 2.4 to 112 m2'],
       ['Ground sampling distance', 'About 2.68 cm per pixel'],
       ['Total crown area, sample clip', 'About 469 m2, against 467 m2 '
        'computed from the mask pixels - the two agree, as they should']],
      story, widths=[CONTENT_W * 0.36, CONTENT_W * 0.64])

story.append(P("""
Those figures are the yardstick. A result an order of magnitude away from them
is worth investigating before it goes into a report.
"""))

story.append(PageBreak())
# =========================================================================
H1('10. Scoring a model against the labelled tiles', story)
story.append(P("""
This reproduces the published numbers from the weights you were given - the
strongest single check that everything on your machine is correct. About three
minutes on a GPU.
"""))
code(r'cd /d D:\ghaf-project\code', story)
code(r"""python tools\test.py ..\models\fastvit-ma36_mask2former\fastvit-ma36_mask2former.py ..\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth --data-root ..\data\ghaf""",
     story)
story.append(P("""
The last line reports `mIoU`, `mDice` and `mFscore`, with a table above it
giving the two classes separately.
"""))
table(['Model', 'mIoU', 'F1'],
      [['**FastViT-MA36 + Mask2Former**', '**79.32**', '**87.22**'],
       ['PoolFormer-S36 + FPN', '78.65', '86.72'],
       ['DPN-98 + FPN', '78.19', '86.35'],
       ['ConvNeXt-S + UPerNet', '78.02', '86.20'],
       ['ResNet-50 + Mask2Former', '77.69', '85.98'],
       ['EfficientNet-B3 + FPN', '70.77', '80.29']],
      story, widths=[CONTENT_W * 0.56, CONTENT_W * 0.22, CONTENT_W * 0.22])
story.append(P("""
Swap the two paths to score any of the other five. If your number differs from
the table by more than a rounding error, the cause is almost always the data
root pointing somewhere unexpected, or a checkpoint that is not the one it
claims to be - check 2 in chapter 6 settles the second case.
"""))

story.append(Paragraph('Per-tile predictions over a labelled split', S['h2']))
story.append(P("""
Scoring reduces a whole split to a few numbers. If you want the maps behind
those numbers - for figures, or to see *where* the model is wrong rather than
by how much - predict the split tile by tile. A few minutes for the test
split:
"""))
code(r"""python tools\predict_split.py ..\models\fastvit-ma36_mask2former\fastvit-ma36_mask2former.py ..\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth --data-root ..\data\ghaf --split testing --out-dir ..\output\predictions --save-probability""",
     story)
story.append(P("""
One predicted mask per tile, encoded exactly like the ground-truth masks
(`0` background, `1` ghaf), so a prediction and its label can be subtracted
directly to make an error map. `--limit 20` runs a quick partial pass first.
Expect a canopy fraction around **3.4%** over the test split.
"""))

story.append(PageBreak())
# =========================================================================
H1('11. Training and fine-tuning', story)
story.append(P("""
Only worth doing if you are changing something. The six trained models are
already provided, and training the published configuration takes many hours on
one GPU.
"""))

story.append(Paragraph('Training from scratch', S['h2']))
code(r'python tools\train.py configs\ghaf\fastvit-ma36_mask2former.py --data-root ..\data\ghaf',
     story)
story.append(P("""
Checkpoints and logs go to `work_dirs\\<config-name>\\`. The model is scored on
the validation split every 3 500 iterations and the best one is kept as
`best_mIoU_iter_*.pth`. If a run is interrupted - a power cut, a closed window
- continue it, optimiser state and all:
"""))
code(r'python tools\train.py configs\ghaf\fastvit-ma36_mask2former.py --data-root ..\data\ghaf --resume',
     story)
box("""
**Keep `work_dirs\\` when a run finishes.** It holds the loss curves, the
validation history and the exact configuration the run used. Those logs cannot
be reconstructed afterwards, and they are what a reviewer asks for.
""", story)

story.append(Paragraph('Fine-tuning on labels from a new site', S['h2']))
story.append(P("""
Starting from a released checkpoint is much cheaper than training from
scratch, and usually better. First prepare the new tiles in the same layout -
`training/{images,masks}`, `validation/{images,masks}`, 1024 x 1024 PNG pairs,
masks containing only 0 and 1 - and check them:
"""))
code(r'python tools\check_dataset.py D:\ghaf-project\data\new-site --full', story)
story.append(P('Then fine-tune, with a shorter schedule and a smaller learning rate:'))
code(r"""python tools\train.py configs\ghaf\fastvit-ma36_mask2former.py --data-root ..\data\new-site --load-from ..\models\fastvit-ma36_mask2former\best_mIoU_iter_3500.pth --cfg-options train_cfg.max_iters=4000 optim_wrapper.optimizer.lr=1e-5""",
     story)
table(['Flag', 'What it does'],
      [['--load-from', 'Takes the weights and starts a **fresh** schedule. '
        'This is what fine-tuning means'],
       ['--resume', 'Continues an **interrupted** run of your own, keeping the '
        'optimiser state and iteration count. A different thing entirely'],
       ['--cfg-options train_cfg.max_iters=4000', 'A shorter schedule: 4000 '
        'iterations rather than the full run'],
       ['--cfg-options optim_wrapper.optimizer.lr=1e-5', 'A smaller learning '
        'rate, so the model adjusts to the new site without forgetting what '
        'it already knows']],
      story, widths=[CONTENT_W * 0.38, CONTENT_W * 0.62], mono_first=True)
story.append(P("""
Score the result exactly as in chapter 10, pointing `--data-root` at the new
site and the checkpoint at your new `work_dirs\\...\\best_mIoU_iter_*.pth`.
"""))

story.append(PageBreak())
# =========================================================================
H1('12. When something goes wrong', story)
story.append(P("""
Nothing here is dangerous. No command in this handbook deletes your data, and
a failed run can always be started again. The skill worth having is reading
what the computer said.
"""))

story.append(Paragraph('The method: five steps', S['h2']))
bullets([
    '**Read the last three lines first.** Python prints the history of what it '
    'was doing (the "traceback") and then, on the very last line, what '
    'actually went wrong. That last line is the error. Everything above it is '
    'context.',
    '**Find the error type.** The last line looks like '
    '`SomeError: some message`. The part before the colon is the type - '
    '`FileNotFoundError`, `RuntimeError`, `ModuleNotFoundError` - and it tells '
    'you the category of the problem.',
    '**Check the table on the next page.** Most of what happens in practice is '
    'in it, with the fix.',
    '**Ask the three questions** in "Before you search" below. They resolve a '
    'large share of problems without any searching at all.',
    '**Then search**, using the recipe below. Copy the error, not your paths.',
], story, numbered=True)

box("""
**Copy the error text before you do anything else.** Select it in the terminal
window and press Enter (in the Anaconda Prompt this copies), then paste it
into Notepad. If you close the window or run another command the text is
usually gone, and an error you cannot quote is an error nobody can help with.
""", story, kind='warn')

story.append(Paragraph('Before you search: three questions', S['h2']))
table(['Question', 'How to check', 'Why it matters'],
      [['Am I in the `ghaf` environment?', 'The prompt should start with '
        '`(ghaf)`. If not: `conda activate ghaf`',
        'The single most common cause of "module not found". A new terminal '
        'window always starts outside it'],
       ['Am I running the right Python?',
        '`python -c "import sys; print(sys.executable)"` - the path printed '
        'must contain `envs\\ghaf`',
        'A base Anaconda install ahead on the PATH will answer instead, and '
        'has none of this project\'s packages'],
       ['Am I in the right folder?', '`cd` on its own prints where you are. '
        '`dir` lists what is there',
        'Commands here assume the `code` folder. A path typed relative to '
        'somewhere else will not be found']],
      story, widths=[CONTENT_W * 0.26, CONTENT_W * 0.40, CONTENT_W * 0.34])

story.append(PageBreak())
story.append(Paragraph('Errors you may actually meet', S['h2']))
table(['What you see', 'What it means', 'What to do'],
      [['`No module named mmengine` / `pytest` / `ghaf`',
        'A different Python is running than the one you installed into',
        'Run `conda activate ghaf`. Verify with '
        '`python -c "import sys; print(sys.executable)"`. Always start '
        'commands with `python -m`'],
       ['`mmseg ... is not installed in conda environment "X"`',
        'Wrong environment, or the stack was never installed on this machine',
        'The message names the interpreter it asked. `conda activate ghaf`, '
        'then repeat the command'],
       ['`No module named ftfy`',
        'mmsegmentation imports a tokenizer that needs it, although its own '
        'metadata does not say so',
        '`python -m pip install ftfy regex`'],
       ['`RuntimeError: Numpy is not available`',
        'NumPy 2 alongside a PyTorch built for NumPy 1',
        '`python -m pip install "numpy<2"`'],
       ['`opencv-python ... requires numpy>=2` while installing',
        'A note from pip\'s resolver, not an error',
        'Ignore it. OpenCV works with either'],
       ['`CUDA out of memory`',
        'The GPU ran out of room, usually because the batch size is too high '
        'or another program is using the card',
        'Lower `--batch-size` to 2 or 1. Close other GPU programs. Or add '
        '`--device cpu`'],
       ['`not enough scratch space`',
        'The drive holding temporary files is too small for this image',
        'Add `--scratch-dir` pointing at a drive with room. An image needs 9 '
        'bytes per pixel'],
       ['`FileNotFoundError` naming a path',
        'A file or folder in the command does not exist as typed',
        'Check for a typo, a missing `..\\`, or a path with spaces that needs '
        'double quotes'],
       ['`The system cannot find the path specified`, printed twice',
        'The path contains an `&`, which the Command Prompt read as two '
        'separate commands',
        'Put the whole path in double quotes'],
       ['`The process cannot access the file because it is being used by '
        'another process`',
        'QGIS, ArcGIS or another window has the file open',
        'Close the program holding it. If you are deleting a folder, first '
        '`cd` out of it - a terminal sitting inside a folder holds it open'],
       ['`NO TILES` from `check_dataset.py`',
        'The folders exist but hold no `.png` or `.tif` tiles',
        'The row names what it found instead; usually the tiles are one '
        'folder deeper, or in another format'],
       ['`SHA-256 mismatch` from `smoke_test.py`',
        'A checkpoint does not match the one that was released',
        'That copy is damaged. Copy it again from the original'],
       ['`0.00% ghaf` from `smoke_test.py`',
        'Nothing. That step predicts on a blank tile',
        'Expected. See chapter 6'],
       ['`0.00%` canopy on real imagery',
        'The run genuinely found nothing',
        'Usually the wrong checkpoint, or bands that are not red, green, blue. '
        'Check the config and checkpoint paths, then try `--bands`'],
       ['Screens of `UserWarning` about `__floordiv__`, `meshgrid`, '
        '`build_loss`',
        'Deprecation notices from inside PyTorch and mmsegmentation',
        'Nothing. They appear on a correct run'],
       ['`KeyboardInterrupt`',
        'Someone pressed Ctrl+C, or the terminal was closed',
        'Nothing broke. Run the command again; long runs can be resumed with '
        '`--skip-existing` or `--resume`'],
       ['The command did something odd after pasting several lines',
        'The terminal joined them into one line',
        'Paste and run one command at a time']],
      story, widths=[CONTENT_W * 0.30, CONTENT_W * 0.32, CONTENT_W * 0.38])

story.append(PageBreak())
story.append(Paragraph('How to search for an answer', S['h2']))
story.append(P("""
If the error is not in the table, it is almost certainly something another
person has already hit. Searching well is a skill with a recipe.
"""))

story.append(Paragraph('What to paste into the search box', S['h3']))
story.append(P("""
Take the **last line** of the error. Remove anything specific to your
computer - your name, your drive letters, your file names, any long numbers -
because those appear in nobody else's error. Add the name of the library it
came from. So this:
"""))
output(r"""
  File "D:\ghaf-project\code\ghaf\inference\large_image.py", line 322, in predict_large_image
    with rasterio.open(src_path) as src:
rasterio.errors.RasterioIOError: D:\surveys\flight_07\north_block.tif: No such file or directory
""", story, caption='What you saw')
story.append(P('becomes this:'))
output("""
rasterio RasterioIOError No such file or directory
""", story, caption='What to search for')

story.append(Paragraph('Where to search, in order', S['h3']))
table(['Where', 'How', 'Best for'],
      [['A search engine',
        'Paste the cleaned error. Add `mmsegmentation` or `rasterio` or '
        '`pytorch` - whichever library the error came from',
        'Almost everything. Try this first'],
       ['GitHub issues for the library',
        'Go to the library\'s GitHub page, click **Issues**, and search there. '
        'Include closed issues - a closed one usually contains the fix',
        'Errors from inside mmsegmentation, mmcv, mmdet: '
        'github.com/open-mmlab/mmsegmentation'],
       ['Stack Overflow',
        'Search the error text. Read the accepted answer and the comments '
        'under it',
        'General Python, conda and installation problems'],
       ['This project\'s own documents',
        'The `docs\\` folder beside the code: GETTING_STARTED, '
        'AREA_WIDE_INFERENCE, MODEL_ZOO, RELEASE_BUNDLE',
        'Anything about this project specifically rather than the libraries '
        'it uses'],
       ['The tool\'s own help',
        'Add `--help` to any command, for example '
        r'`python tools\predict_folder.py --help`',
        'Remembering what an option is called, and what it does']],
      story, widths=[CONTENT_W * 0.22, CONTENT_W * 0.44, CONTENT_W * 0.34])

story.append(Paragraph('Judging what you find', S['h3']))
bullets([
    '**Check the versions.** An answer written for mmsegmentation 0.x does not '
    'apply to 1.2.2, and one for PyTorch 2.x may not apply to 1.12.1. If the '
    'page does not say which version it is about, treat it with suspicion.',
    '**Prefer answers that explain why.** An answer that only says "run this '
    'command" and cannot say what it fixes is as likely to break something as '
    'to help.',
    '**Never blindly upgrade.** Advice of the form "just upgrade mmcv / torch / '
    'numpy" will usually break this installation: the versions in chapter 5 '
    'were chosen because they work together and match the trained weights. If '
    'you think a version must change, write down what you had first.',
    '**Change one thing at a time**, and re-run the checks in chapter 6 after '
    'each change. Two changes at once and you will not know which one helped.',
], story)

box("""
**The escape hatch.** If an installation gets into a state you cannot explain,
delete the environment and build it again from chapter 5. It costs twenty
minutes and touches none of your data:
`conda deactivate`, then `conda env remove -n ghaf`, then start at Step 1.
""", story)

story.append(Paragraph('Asking a person for help', S['h2']))
story.append(P("""
When you do need to ask someone, send all six of these. An answer usually
comes back in one round instead of five:
"""))
bullets([
    'The **exact command** you ran, copied and pasted, not retyped from memory.',
    'The **complete output**, from the command down to the last line. Not a '
    'screenshot of part of it, and not just the last line.',
    'What you **expected** to happen instead.',
    'The result of `python -c "import sys; print(sys.executable)"`.',
    'The result of `python -m pip list` (or at least the lines for `torch`, '
    '`mmcv`, `mmsegmentation`, `mmdet`, `numpy`).',
    'Whether the checks in chapter 6 pass **now** - and whether they ever did '
    'on this machine.',
], story, numbered=True)

story.append(PageBreak())
# =========================================================================
H1('13. Quick reference', story)
story.append(P("""
Every routine command in one place. All of them assume you have run
`conda activate ghaf` and `cd /d D:\\ghaf-project\\code` in that window, and
that `MODEL` stands for
`..\\models\\fastvit-ma36_mask2former\\fastvit-ma36_mask2former.py` and
`WEIGHTS` for `..\\models\\fastvit-ma36_mask2former\\best_mIoU_iter_3500.pth`.
"""))

story.append(Paragraph('Checks', S['h2']))
code(r"""python -m pytest tests\ -q
python tools\smoke_test.py --checkpoints ..\models
python tools\check_dataset.py ..\data\ghaf""", story)

story.append(Paragraph('Mapping crowns', S['h2']))
code(r"""python -m ghaf.inference.large_image MODEL WEIGHTS image.tif --out-polygons crowns.gpkg --out-mask crowns.tif --batch-size 4 --min-area 1

python tools\predict_folder.py MODEL WEIGHTS D:\my-images --out-dir ..\output\folder --batch-size 4 --min-area 1

python tools\make_sample.py big-mosaic.tif --output sample.tif --size 8192""",
     story)

story.append(Paragraph('Scoring, predicting a split, training', S['h2']))
code(r"""python tools\test.py MODEL WEIGHTS --data-root ..\data\ghaf

python tools\predict_split.py MODEL WEIGHTS --data-root ..\data\ghaf --split testing --out-dir ..\output\predictions

python tools\train.py configs\ghaf\fastvit-ma36_mask2former.py --data-root ..\data\ghaf

python tools\train.py configs\ghaf\fastvit-ma36_mask2former.py --data-root ..\data\new-site --load-from WEIGHTS --cfg-options train_cfg.max_iters=4000 optim_wrapper.optimizer.lr=1e-5""",
     story)

story.append(Paragraph('Getting out of trouble', S['h2']))
code(r"""conda activate ghaf
python -c "import sys; print(sys.executable)"
python tools\predict_folder.py --help
conda env remove -n ghaf""", story)

story.append(Paragraph('Where to read more', S['h2']))
table(['Document', 'Covers'],
      [[r'code\README.md', 'The project, the results, the repository layout'],
       [r'code\docs\GETTING_STARTED.md', 'The same walkthrough as this '
        'handbook, in the code folder'],
       [r'code\docs\MODEL_ZOO.md', 'All six models, per-class scores, training '
        'settings'],
       [r'code\docs\AREA_WIDE_INFERENCE.md', 'How a mosaic is tiled and '
        'blended, and how to tune it'],
       [r'code\docs\RELEASE_BUNDLE.md', 'How the models folder is assembled '
        'and verified'],
       ['MANIFEST.json', 'What is in the bundle, its sizes, and the version of '
        'the code that produced it']],
      story, widths=[CONTENT_W * 0.36, CONTENT_W * 0.64], mono_first=True)

story.append(Spacer(1, 10))
story.append(Paragraph(md(
    'This handbook describes the repository at '
    'github.com/brakuta/Ghaf-Tree-Crown-Mapping-from-UAV-Data. The trained '
    'weights, the UAV imagery and the labelled tiles are not distributed with '
    'the code; they are available from the corresponding author on reasonable '
    'request.'), S['caption']))

build(story)
print('wrote', OUT, OUT.stat().st_size, 'bytes')
