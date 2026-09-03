#!/usr/bin/env python3
"""
Typesetting engine for the project manuals.
=============================================================================
WHY THIS EXISTS RATHER THAN pandoc
  This environment has no LaTeX, no pandoc and no HTML-to-PDF converter, and
  installing a TeX distribution to set forty pages is not a good trade. What
  the manual actually needs is narrow: a table of contents with real page
  numbers, tables that break across pages, code blocks that do not reflow,
  and running heads. ReportLab's Platypus does all four.

WHAT IT IS DELIBERATELY BUILT TO AVOID
  Four failure modes make a generated document look generated, and each one
  is a layout decision rather than a matter of wording:

  1. A page break before every section. It is the single biggest cause of
     half-empty pages. Only a CHAPTER opens a new page here; sections flow.

  2. Headings stranded at the foot of a page. Every heading is glued to the
     block that follows it, so a heading and its first paragraph move
     together or not at all.

  3. Tables and code blocks that break one row before the end. Short blocks
     are kept whole; long tables repeat their header row instead.

  4. Tables running off the right edge of the paper. ReportLab does NOT clip
     or shrink an over-wide table -- it draws it, overflowing the margin and
     off the sheet. Column widths are therefore never trusted as written;
     `fit_widths()` rescales every one of them to the real measure. See the
     comment there, because this is the defect that ruined the first edition.

THE PAGE
  A4 with 30/28 mm side margins: a 431 pt measure, set 8.5/12.2, which is
  about 92 characters. Wide for a book, normal for a technical manual on A4,
  and made readable by the open leading rather than by a narrow column --
  narrowing the column is not available here, because the document is full of
  tables and eighty-character shell commands that have to fit inside it.

THE TYPE
  IBM Plex, bundled in `fonts/` under the SIL Open Font License. Serif for
  running text, Sans for every heading and every table head, Mono for code.
  One superfamily drawn on shared proportions, so the three mix without the
  seams you get from three unrelated faces -- and specifically not the
  DejaVu/FreeFont pair that every Linux box defaults to.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, KeepTogether, CondPageBreak, PageBreak, Flowable,
)
from reportlab.platypus.tableofcontents import TableOfContents

# --------------------------------------------------------------------------
# PAGE GEOMETRY
# Declared before anything else because the table fitter, the code sizer and
# the running head all measure against it. One source of truth: change these
# four numbers and every column width in the document follows.
# --------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN_L, MARGIN_R = 30 * mm, 28 * mm
MARGIN_T, MARGIN_B = 22 * mm, 20 * mm
MEASURE = PAGE_W - MARGIN_L - MARGIN_R          # 430.9 pt
TEXT_H = PAGE_H - MARGIN_T - MARGIN_B

# --------------------------------------------------------------------------
# FONTS
# Bundled rather than taken from the system: this document has to rebuild
# identically on a machine nobody has configured, and the system fonts differ
# between the container, the workstation and whatever laptop the manual is
# regenerated on five years from now.
#
# SemiBold, not Bold, carries the headings. Plex Bold at 18 pt is heavy enough
# to shout; SemiBold holds the hierarchy without it.
# --------------------------------------------------------------------------
_FONT_DIR = Path(__file__).resolve().parent / 'fonts'

_FACES = [
    ('Body',            'IBMPlexSerif-Regular.ttf'),
    ('Body-Bold',       'IBMPlexSerif-SemiBold.ttf'),
    ('Body-Ital',       'IBMPlexSerif-Italic.ttf'),
    ('Body-BoldItal',   'IBMPlexSerif-SemiBoldItalic.ttf'),
    ('Head',            'IBMPlexSans-Regular.ttf'),
    ('Head-Med',        'IBMPlexSans-Medium.ttf'),
    ('Head-Bold',       'IBMPlexSans-SemiBold.ttf'),
    ('Head-Heavy',      'IBMPlexSans-Bold.ttf'),
    ('Head-Ital',       'IBMPlexSans-Italic.ttf'),
    ('Mono',            'IBMPlexMono-Regular.ttf'),
    ('Mono-Bold',       'IBMPlexMono-SemiBold.ttf'),
    ('Mono-Ital',       'IBMPlexMono-Italic.ttf'),
]


def register_fonts():
    for name, fn in _FACES:
        path = _FONT_DIR / fn
        if not path.exists():
            raise SystemExit(
                f'[FATAL] font missing: {path}\n'
                f'        The IBM Plex faces live in docs/manual/fonts/ and are\n'
                f'        part of the repository. If that directory is empty the\n'
                f'        checkout is incomplete.')
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        'Body', normal='Body', bold='Body-Bold',
        italic='Body-Ital', boldItalic='Body-BoldItal')
    pdfmetrics.registerFontFamily(
        'Head', normal='Head', bold='Head-Bold',
        italic='Head-Ital', boldItalic='Head-Bold')
    pdfmetrics.registerFontFamily(
        'Mono', normal='Mono', bold='Mono-Bold',
        italic='Mono-Ital', boldItalic='Mono-Bold')


# --------------------------------------------------------------------------
# COLOUR
# Six values. Every one of them is dark enough, or light enough, to survive a
# monochrome laser printer -- which is how a bench document is actually read.
# The accent is a burnt sienna: it reads as a deliberate second colour on
# screen and as a mid grey on a photocopier, never as a smudge.
# --------------------------------------------------------------------------
INK      = colors.HexColor('#16181c')   # text
SOFT     = colors.HexColor('#5a5f66')   # captions, running heads
RULE     = colors.HexColor('#b9bcc0')   # hairlines between table rows
FAINT    = colors.HexColor('#f3f1ed')   # code ground
ACCENT   = colors.HexColor('#8a3a12')   # chapter numerals, callout rules
ACCENT_D_HEX = '#5c2a1b'                # inline code, as a string for <font>
ACCENT_D = colors.HexColor(ACCENT_D_HEX)

# The body size. Everything else in the stylesheet is expressed against it, so
# the whole document can be scaled from one number without the hierarchy
# drifting apart.
BODY = 8.5
LEAD = 1.435           # leading as a multiple of size


def stylesheet():
    ss = StyleSheet1()
    ss.add(ParagraphStyle(
        'Body', fontName='Body', fontSize=BODY, leading=BODY * LEAD,
        alignment=TA_JUSTIFY, textColor=INK, spaceAfter=BODY * 0.66,
        allowWidows=0, allowOrphans=0,
        # Justified text in a single column, interrupted by long unbreakable
        # monospace tokens, produces lines of four words stretched edge to
        # edge. Hyphenation is what makes justification survive that; without
        # it the measure has to be given up instead.
        hyphenationLang='en_GB', embeddedHyphenation=1,
        # Zero-width spaces are NOT break opportunities in ReportLab (tested),
        # so a long path is atomic and the line before it must stretch to the
        # margin. Allowing the inter-word space to compress by 12% instead of
        # the default 5% lets one more word fit on such a line, which is the
        # only lever available short of giving up justification.
        spaceShrinkage=0.12,
        uriWasteReduce=0.3))
    ss.add(ParagraphStyle(
        'BodyTight', parent=ss['Body'], spaceAfter=2.2))
    ss.add(ParagraphStyle(
        'Lead', parent=ss['Body'], fontSize=BODY + 0.6,
        leading=(BODY + 0.6) * 1.46, textColor=colors.HexColor('#31353b'),
        spaceAfter=9))
    ss.add(ParagraphStyle(
        'Bullet', parent=ss['Body'], leftIndent=12, bulletIndent=2,
        spaceAfter=3, alignment=TA_LEFT))

    # -- headings ----------------------------------------------------------
    ss.add(ParagraphStyle(
        'ChapterNum', fontName='Head', fontSize=27, leading=27,
        textColor=ACCENT, spaceAfter=0, alignment=TA_LEFT))
    ss.add(ParagraphStyle(
        'ChapterKick', fontName='Head-Bold', fontSize=6.6, leading=8,
        textColor=SOFT, spaceAfter=2, alignment=TA_LEFT))
    ss.add(ParagraphStyle(
        'Chapter', fontName='Head-Bold', fontSize=17.5, leading=20.5,
        textColor=INK, spaceAfter=0, spaceBefore=0, alignment=TA_LEFT))
    ss.add(ParagraphStyle(
        'Section', fontName='Head-Bold', fontSize=10.6, leading=12.8,
        textColor=INK, spaceBefore=13, spaceAfter=4.2, alignment=TA_LEFT))
    ss.add(ParagraphStyle(
        'Sub', fontName='Head-Med', fontSize=9.0, leading=11.2,
        textColor=INK, spaceBefore=9, spaceAfter=2.8, alignment=TA_LEFT))

    # -- monospace and tables ---------------------------------------------
    ss.add(ParagraphStyle(
        'Code', fontName='Mono', fontSize=7.6, leading=10.2,
        textColor=INK, alignment=TA_LEFT,
        leftIndent=8, rightIndent=4, spaceBefore=1, spaceAfter=1))
    ss.add(ParagraphStyle(
        'TableCell', fontName='Body', fontSize=TABLE_BASE, leading=TABLE_BASE * 1.34,
        textColor=INK, alignment=TA_LEFT))
    ss.add(ParagraphStyle(
        'TableHead', fontName='Head-Bold', fontSize=TABLE_BASE - 0.7,
        leading=(TABLE_BASE - 0.7) * 1.3, textColor=INK, alignment=TA_LEFT))
    ss.add(ParagraphStyle(
        'Caption', fontName='Head-Ital', fontSize=7.4, leading=9.8,
        textColor=SOFT, spaceBefore=2.5, spaceAfter=8))

    # -- asides ------------------------------------------------------------
    ss.add(ParagraphStyle(
        'Callout', parent=ss['Body'], fontSize=BODY - 0.25,
        leading=(BODY - 0.25) * 1.4, leftIndent=9, rightIndent=6,
        spaceAfter=3.5, spaceBefore=0))
    ss.add(ParagraphStyle(
        'CalloutLabel', fontName='Head-Bold', fontSize=6.8, leading=8.6,
        textColor=ACCENT, spaceAfter=3, leftIndent=9))

    # -- contents ----------------------------------------------------------
    ss.add(ParagraphStyle(
        'TOC0', fontName='Head-Bold', fontSize=9, leading=15.5,
        textColor=INK, spaceBefore=6))
    ss.add(ParagraphStyle(
        'TOC1', fontName='Body', fontSize=8.5, leading=12.4,
        textColor=colors.HexColor('#2c3037'), leftIndent=15))

    # -- title page --------------------------------------------------------
    ss.add(ParagraphStyle(
        'TitleKick', fontName='Head-Bold', fontSize=7.4, leading=9,
        textColor=ACCENT, alignment=TA_LEFT, spaceAfter=10))
    ss.add(ParagraphStyle(
        'TitleBig', fontName='Head-Bold', fontSize=27, leading=31,
        textColor=INK, alignment=TA_LEFT, spaceAfter=5))
    ss.add(ParagraphStyle(
        'TitleSub', fontName='Head', fontSize=12, leading=16,
        textColor=colors.HexColor('#43484f'), alignment=TA_LEFT,
        spaceAfter=16))
    ss.add(ParagraphStyle(
        'TitleMeta', fontName='Body', fontSize=8.4, leading=12.5,
        textColor=colors.HexColor('#43484f'), alignment=TA_LEFT))
    return ss


# The size table cells were set at in the first edition. Call sites in the
# content modules pass `size=8.4`, `size=9` and so on, all chosen against that
# 9 pt baseline; rather than touch forty-four of them, `table()` rescales what
# it is given onto the current baseline. Change TABLE_BASE and every table in
# the document moves together.
TABLE_BASE = 7.8
_TABLE_LEGACY_BASE = 9.0

SS = None  # populated by build()
QA_FILL = []  # (page number, fraction of frame filled), for check_layout.py


# --------------------------------------------------------------------------
# INLINE MARKUP
# A deliberately small dialect: **bold**, *italic*, `code`. Anything richer
# would be a second markdown implementation, and this document does not need
# one.
# --------------------------------------------------------------------------
def esc(text: str) -> str:
    # `&nbsp;` is resolved to the real character first. Escaping runs after,
    # so an ampersand written by the author still becomes `&amp;` -- but the
    # entity the author meant as a non-breaking space is not mangled into
    # visible text, which is exactly what happened on the first build.
    text = text.replace('&nbsp;', ' ')
    return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def inline(text: str, mono: float | None = None) -> str:
    """Convert the small markup dialect to ReportLab's inline tags.

    `mono` is the size for `code spans`, which has to follow the surrounding
    type: a fixed value looks correct in body text and oversized in an 7.8 pt
    table cell. Plex Mono runs slightly larger on the eye than Plex Serif at
    the same size, so it is set at 0.94 of its host.

    Order matters: code spans are extracted FIRST and restored last, so that
    a path containing an asterisk (rare but real -- glob patterns appear all
    over this project's documentation) is not read as emphasis.
    """
    size = mono if mono else BODY * 0.94
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return f'\x00{len(spans) - 1}\x00'

    text = re.sub(r'`([^`]+)`', stash, text)
    text = esc(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<![\w*])\*([^*\n]+?)\*(?![\w*])', r'<i>\1</i>', text)

    def restore(m):
        c = esc(spans[int(m.group(1))])
        return (f'<font face="Mono" size="{size:.2f}" '
                f'color="{ACCENT_D_HEX}">{c}</font>')

    return re.sub(r'\x00(\d+)\x00', restore, text)


# --------------------------------------------------------------------------
# COLUMN FITTING
# --------------------------------------------------------------------------
def fit_widths(widths, avail=None, min_free=42.0):
    """Rescale a column-width list so it sums to EXACTLY the measure.

    This function exists because of a real defect in the first edition. A
    ReportLab `Table` given `colWidths=[276, 175, None]` inside a 431 pt frame
    does not complain, does not shrink and does not clip: it lays the third
    column out at its natural width and draws the whole table straight off the
    edge of the paper. Several tables lost their last column that way.

    The rule here is that the numbers a call site passes are PROPORTIONS, not
    promises. Fixed columns keep their ratios to one another; `None` columns
    share whatever is left, with a floor of `min_free` so a greedy fixed
    column cannot squeeze one down to nothing. Whatever comes in, the sum that
    goes out is the measure.
    """
    if not widths:
        return None
    avail = MEASURE if avail is None else avail
    n_free = sum(1 for w in widths if not w)
    total_fixed = float(sum(w for w in widths if w))

    if n_free == 0:
        if total_fixed <= 0:
            return None
        return [w * avail / total_fixed for w in widths]

    room = avail - n_free * min_free
    if total_fixed > room:
        # The fixed columns want more than the page has. Shrink them
        # proportionally rather than letting the free column disappear.
        scale = max(room, 1.0) / total_fixed
        widths = [(w * scale if w else None) for w in widths]
        total_fixed = sum(w for w in widths if w)
    each = (avail - total_fixed) / n_free
    return [(w if w else each) for w in widths]


# --------------------------------------------------------------------------
# FLOWABLES
# --------------------------------------------------------------------------
class HRule(Flowable):
    """A hairline. Used sparingly -- under chapter titles and nowhere else."""

    def __init__(self, width=None, thickness=0.6, colour=RULE, space=4):
        Flowable.__init__(self)
        self._w, self._t, self._c, self._space = width, thickness, colour, space

    def wrap(self, aw, ah):
        self._aw = self._w or aw
        return (self._aw, self._t + self._space)

    def draw(self):
        self.canv.setStrokeColor(self._c)
        self.canv.setLineWidth(self._t)
        self.canv.line(0, self._space, self._aw, self._space)


def para(text, style='Body'):
    return Paragraph(inline(text), SS[style])


def bullets(items, style='Bullet'):
    return [Paragraph(inline(i), SS[style], bulletText='•') for i in items]


def numbered(items):
    out = []
    for n, i in enumerate(items, 1):
        out.append(Paragraph(inline(i), SS['Bullet'], bulletText=f'{n}.'))
    return out


# Reported by code() when a line will not fit at the minimum size, and printed
# by build.py. A silently shrunk 5 pt line is worse than a loud complaint.
CODE_OVERLONG = []
CODE_MAX = 7.6
CODE_MIN = 6.7


def code(text, keep=True):
    """A monospace block on a tinted ground with an accent rule at the left.

    Lines are NOT wrapped: a wrapped command is a command that cannot be
    copied. The block is set at the largest size at which its longest line
    still fits the measure, computed from the real font metrics rather than
    from a guess at characters-per-inch, and floored at CODE_MIN -- below that
    the type stops being legible and the honest fix is to break the line in
    the source, which is what CODE_OVERLONG exists to demand.
    """
    lines = text.strip('\n').split('\n')
    inner = MEASURE - 8 - 7 - 5          # leftIndent, left padding, right padding
    widest = max((pdfmetrics.stringWidth(l, 'Mono', 10) for l in lines),
                 default=1.0) / 10.0     # width of the longest line at 1 pt
    size = CODE_MAX if widest <= 0 else min(CODE_MAX, inner / widest)
    if size < CODE_MIN:
        longest = max(lines, key=len)
        CODE_OVERLONG.append((round(size, 2), len(longest), longest[:78]))
        size = CODE_MIN
    st = ParagraphStyle('c', parent=SS['Code'], fontSize=size,
                        leading=size * 1.36)
    rows = [[Paragraph(esc(l).replace(' ', '&nbsp;') or '&nbsp;', st)]
            for l in lines]
    t = Table(rows, colWidths=[MEASURE])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), FAINT),
        ('LINEBEFORE', (0, 0), (0, -1), 1.6, colors.HexColor('#c8bfb2')),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 0.9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.9),
        ('TOPPADDING', (0, 0), (-1, 0), 4.5),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 4.5),
    ]))
    block = [Spacer(1, 3.5), t, Spacer(1, 6)]
    # Only glue short blocks; a 40-line block that must stay whole would
    # push a mostly-empty page ahead of itself, which is the exact defect
    # this engine exists to avoid.
    return [KeepTogether(block)] if (keep and len(lines) <= 16) else block


def table(rows, widths=None, head=True, align=None, size=None, caption=None):
    """A table with horizontal rules only, fitted to the measure.

    Vertical rules and full boxes make a dense reference table harder to
    scan, not easier: the eye already tracks columns by alignment. The head is
    set in Sans against a Serif body, which separates it more cleanly than a
    tint or a heavier rule would.

    `size` is quoted on the original 9 pt baseline -- see TABLE_BASE.
    """
    pts = TABLE_BASE if not size else size * TABLE_BASE / _TABLE_LEGACY_BASE
    cs = ParagraphStyle('tc', parent=SS['TableCell'], fontSize=pts,
                        leading=pts * 1.34)
    hs = ParagraphStyle('th', parent=SS['TableHead'], fontSize=pts - 0.6,
                        leading=(pts - 0.6) * 1.3)
    data = []
    for r, row in enumerate(rows):
        is_head = head and r == 0
        st, mono = (hs, pts * 0.9) if is_head else (cs, pts * 0.94)
        data.append([Paragraph(inline(str(c), mono=mono), st) for c in row])

    t = Table(data, colWidths=fit_widths(widths or [None] * len(data[0])),
              repeatRows=1 if head else 0)
    cmds = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 3.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.2),
    ]
    if head:
        cmds += [
            ('LINEABOVE', (0, 0), (-1, 0), 0.9, INK),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, INK),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4.4),
            ('TOPPADDING', (0, 0), (-1, 0), 3.6),
            ('LINEBELOW', (0, -1), (-1, -1), 0.9, INK),
        ]
        rng = range(1, len(data) - 1)
    else:
        cmds.append(('LINEABOVE', (0, 0), (-1, 0), 0.9, INK))
        cmds.append(('LINEBELOW', (0, -1), (-1, -1), 0.9, INK))
        rng = range(len(data) - 1)
    for r in rng:
        cmds.append(('LINEBELOW', (0, r), (-1, r), 0.25, RULE))
    for spec in (align or []):
        cmds.append(spec)
    t.setStyle(TableStyle(cmds))
    out = [Spacer(1, 3), t]
    if caption:
        out.append(Paragraph(inline(caption, mono=7.0), SS['Caption']))
    else:
        out.append(Spacer(1, 8))
    return out


def callout(label, text_lines, kind='warning'):
    """A boxed aside with a coloured left rule.

    Reserved for things that fail SILENTLY. A note that merely repeats the
    surrounding prose in a box teaches the reader to skip boxes, and then the
    one that mattered gets skipped too.
    """
    warn = kind == 'warning'
    accent = ACCENT if warn else colors.HexColor('#28527a')
    ground = colors.HexColor('#faf6f1') if warn else colors.HexColor('#f4f7fa')
    inner = [Paragraph(inline(label.upper(), mono=7.0), ParagraphStyle(
        'cl', parent=SS['CalloutLabel'], textColor=accent))]
    for t in text_lines:
        inner.append(Paragraph(inline(t, mono=(BODY - 0.25) * 0.94),
                               SS['Callout']))
    t = Table([[inner]], colWidths=[MEASURE])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ground),
        ('LINEBEFORE', (0, 0), (0, -1), 2.4, accent),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5),
    ]))
    return [Spacer(1, 4), KeepTogether([t]), Spacer(1, 8)]


# --------------------------------------------------------------------------
# HEADINGS -- each one registers itself with the table of contents and is
# glued to whatever follows it.
# --------------------------------------------------------------------------
_counter = {'ch': 0, 'sec': 0, 'sub': 0}


class _Anchored(Paragraph):
    """A heading that reports itself to the TOC on draw."""

    def __init__(self, text, style, level, key, toc_text=None):
        Paragraph.__init__(self, text, style)
        self._level, self._key = level, key
        # The printed heading and the contents entry differ on purpose: the
        # page already carries the chapter numeral beside the title, so
        # repeating the number in the title would be noise -- but the contents
        # page has no such context and needs it.
        self._plain = re.sub(r'<[^>]+>', '', toc_text or text)

    def draw(self):
        # The heading draws itself and plants a named destination. It does NOT
        # notify the table of contents from here: `notify` lives on the
        # document template, not on the canvas, and the canonical place to
        # raise it is afterFlowable() below -- which also guarantees the page
        # number is the settled one rather than one mid-split.
        Paragraph.draw(self)
        self.canv.bookmarkPage(self._key)
        self.canv.addOutlineEntry(self._plain, self._key, self._level, 0)


class _TocMark(Flowable):
    """A zero-height marker that reports a chapter to the contents.

    ReportLab calls `afterFlowable` only for flowables at the TOP LEVEL of the
    story. The chapter opener puts its title inside a two-column table, so the
    heading itself is nested and never reaches afterFlowable -- which silently
    cost the contents page all twelve of its chapter lines, and the running
    heads their chapter names. This marker sits beside the opener at the top
    level and carries the notification instead.
    """

    def __init__(self, level, text, key):
        Flowable.__init__(self)
        self._level, self._plain, self._key = level, text, key
        self.width = self.height = 0

    def wrap(self, aw, ah):
        return (0, 0)

    def draw(self):
        pass


def chapter(title, blurb=None):
    """Opens a new page. The ONLY thing in this engine that does.

    The opener is a two-column rule: the numeral hangs in its own narrow
    column at the left, the title and its standfirst run in the wide one. It
    gives the chapter openings a shape that repeats down the document and
    makes the numeral findable when thumbing through printed pages, which a
    small "CHAPTER 3" line above the title does not.
    """
    _counter['ch'] += 1
    _counter['sec'] = 0
    n = _counter['ch']
    key = f'ch{n}'
    # Wide enough for two digits at 27 pt plus the 12 pt gutter before the
    # rule. At 38 pt "12" wrapped and the opener read as a stacked 1 over 2.
    num_w = 52.0
    right = [Paragraph('CHAPTER', SS['ChapterKick']),
             _Anchored(inline(title), SS['Chapter'], 0, key,
                       toc_text=f'{n}  {title}')]
    head = Table([[Paragraph(str(n), SS['ChapterNum']), right]],
                 colWidths=[num_w, MEASURE - num_w])
    head.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (0, 0), 1.5),
        ('TOPPADDING', (1, 0), (1, 0), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LINEAFTER', (0, 0), (0, 0), 0.7, colors.HexColor('#d8cfc4')),
        ('RIGHTPADDING', (0, 0), (0, 0), 12),
        ('LEFTPADDING', (1, 0), (1, 0), 12),
    ]))
    # A chapter opens a new page ONLY when too little room is left for a
    # useful start. An unconditional PageBreak here is what leaves a chapter's
    # final page 7% full -- the single most common reason a generated document
    # looks padded. 200 points is roughly the opener, the standfirst and four
    # lines of the first section.
    out = [CondPageBreak(200), Spacer(1, 14), head,
           _TocMark(0, f'{n}  {title}', key), Spacer(1, 6),
           HRule(thickness=0.9, colour=INK, space=1)]
    if blurb:
        out.append(Spacer(1, 7))
        out.append(Paragraph(inline(blurb, mono=BODY * 0.94), SS['Lead']))
    return out


def section(title):
    _counter['sec'] += 1
    _counter['sub'] = 0
    n = f"{_counter['ch']}.{_counter['sec']}"
    key = f'sec{_counter["ch"]}_{_counter["sec"]}'
    return _Anchored(inline(f'{n}  {title}'), SS['Section'], 1, key)


def sub(title):
    _counter['sub'] += 1
    return Paragraph(inline(title), SS['Sub'])


def glue(*blocks):
    """Keep a heading with the block that follows it.

    Applied to every heading in the content file. Without it ReportLab will
    happily leave a section title alone at the foot of a page.
    """
    flat = []
    for b in blocks:
        if b is None:          # `glue(section(...), None)` -- a heading whose
            continue           # following block is supplied separately
        flat.extend(b if isinstance(b, list) else [b])
    return KeepTogether(flat)


# --------------------------------------------------------------------------
# PAGE FURNITURE
# --------------------------------------------------------------------------
class Manual(BaseDocTemplate):
    """One frame, one page template, furniture suppressed on the title page.

    The first edition registered a separate 'cover' template and never issued
    a NextPageTemplate, so the document stayed on it for all thirty-eight
    pages and the running heads and folios never appeared at all. A single
    template that checks the page number cannot fail that way.
    """

    def __init__(self, path, title, running, cover_pages=1, **kw):
        self.running_title = running
        self.doc_title = title
        self.cover_pages = cover_pages
        BaseDocTemplate.__init__(
            self, path, pagesize=A4,
            leftMargin=MARGIN_L, rightMargin=MARGIN_R,
            topMargin=MARGIN_T, bottomMargin=MARGIN_B,
            title=title, author='Mohamed Barakat Gibril',
            subject=running, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id='body',
                      leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id='body', frames=[frame], onPage=self._furniture)])
        self._chapter_now = ''

    @staticmethod
    def _tracked(canv, x, y, text, space=0.9, right=False):
        """Letterspaced small type. Tracking is what makes 6.5 pt sans read as
        a deliberate running head rather than as a caption that got lost."""
        # Tracking goes through drawString's charSpace keyword: the Canvas has
        # no setCharSpace -- that lives on the text object, not here.
        if right:
            canv.drawRightString(x, y, text, charSpace=space)
        else:
            canv.drawString(x, y, text, charSpace=space)

    def _furniture(self, canv, doc):
        if canv.getPageNumber() <= self.cover_pages:
            return
        canv.saveState()
        y = PAGE_H - 14 * mm
        canv.setFont('Head-Bold', 6.4)
        canv.setFillColor(SOFT)
        self._tracked(canv, MARGIN_L, y, self.running_title.upper())
        if self._chapter_now:
            canv.setFont('Head', 6.9)
            canv.setFillColor(colors.HexColor('#7b8087'))
            canv.drawRightString(PAGE_W - MARGIN_R, y, self._chapter_now)
        canv.setStrokeColor(colors.HexColor('#d5d7da'))
        canv.setLineWidth(0.5)
        canv.line(MARGIN_L, y - 4, PAGE_W - MARGIN_R, y - 4)

        # Folio: outer edge of the foot, with a short accent rule beside it.
        n = str(canv.getPageNumber())
        canv.setFont('Head-Bold', 8)
        canv.setFillColor(colors.HexColor('#3a3f45'))
        canv.drawRightString(PAGE_W - MARGIN_R, MARGIN_B - 11.5, n)
        w = canv.stringWidth(n, 'Head-Bold', 8)
        canv.setStrokeColor(ACCENT)
        canv.setLineWidth(1.1)
        canv.line(PAGE_W - MARGIN_R - w - 12, MARGIN_B - 8.6,
                  PAGE_W - MARGIN_R - w - 4, MARGIN_B - 8.6)
        canv.restoreState()

    def beforeDocument(self):
        # multiBuild runs the story twice and the template instance survives
        # between passes. Without this the second pass opens with the LAST
        # chapter of the first still in the running head, so the contents page
        # is headed "13 Appendix -- command index".
        self._chapter_now = ''

    def handle_pageEnd(self):
        # Record how much of the text frame this page actually used. Text
        # extraction would measure the same thing less reliably (a table of
        # short cells extracts as very little text but fills the page), and
        # this needs no third-party reader. QA_FILL is consumed by
        # check_layout.py.
        try:
            f = self.frame
            used = (f._y1 + f.height) - f._y
            if QA_FILL and QA_FILL[-1][0] >= self.page:
                QA_FILL.clear()      # a new multiBuild pass has started
            QA_FILL.append((self.page, used / f.height, self._chapter_now))
        except Exception:
            pass
        BaseDocTemplate.handle_pageEnd(self)

    def afterFlowable(self, flowable):
        if not isinstance(flowable, (_Anchored, _TocMark)):
            return
        self.notify('TOCEntry',
                    (flowable._level, flowable._plain,
                     self.page, flowable._key))
        if flowable._level == 0:
            self._chapter_now = flowable._plain


def toc():
    t = TableOfContents()
    t.levelStyles = [SS['TOC0'], SS['TOC1']]
    t.dotsMinLevel = 0
    return t


def build(path, title, subtitle, meta_lines, story, running=None):
    """Two-pass build so the table of contents carries real page numbers."""
    global SS
    doc = Manual(str(path), title, running or title)
    doc.multiBuild(story)
    return path
