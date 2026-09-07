#!/usr/bin/env python3
"""Assemble the technical manual.

    python docs/manual/build.py [output.pdf]

Front matter is built here; the chapters live in content_a.py and
content_b.py. The build is two-pass, so the contents page carries real page
numbers rather than the placeholders of a single pass.

Any code line that will not fit the measure at the minimum size is reported as
[WARN] before the document is written. A warning is not cosmetic: it means a
command in the manual has been set smaller than it should be, and the fix is
to break the line in the source rather than to let the type shrink.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import typeset as T  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    KeepTogether,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
)

TITLE = 'Ghaf tree-crown mapping from UAV imagery'
SUBTITLE = 'Technical manual'
RUNNING = 'Ghaf crown mapping · technical manual'


def front_matter():
    """Title page, then the contents.

    The signpost matters more than the rest of it. A reader who needs one map
    made should not have to decide for themselves which of sixteen chapters
    they can skip, and a manual that does not say so is read from the front
    until the reader gives up.
    """
    from order import ch, chapters
    from typeset import HRule, para

    s = [
        Spacer(1, 62),
        Paragraph('TECHNICAL MANUAL', T.SS['TitleKick']),
        Paragraph(TITLE, T.SS['TitleBig']),
        Paragraph('Delineating <i>Prosopis cineraria</i> crowns in '
                  'area-wide UAV orthomosaics', T.SS['TitleSub']),
        HRule(thickness=1.1, colour=T.INK, space=4),
        Spacer(1, 14),
        Paragraph(
            'FastViT-MA36 with a Mask2Former decode head, and five '
            'alternatives.<br/>Six trained models, 8&nbsp;641 labelled tiles, '
            'and an inference path that maps a whole survey.',
            T.SS['TitleMeta']),
        Spacer(1, 190),
        Paragraph(
            f'Repository <b>{_fact_repo()}</b><br/>'
            f'Documented at commit <b>{_fact_commit()}</b><br/>'
            'Values are traceable to <b>docs/handover/FACTS.yml</b>, which '
            'wins wherever it and this manual could disagree.<br/>'
            'First issue, September 2026',
            T.SS['TitleMeta']),
        PageBreak(),
    ]

    s += [
        Spacer(1, 6),
        Paragraph('Contents', T.SS['Chapter']),
        HRule(thickness=0.9, colour=T.INK, space=3),
        Spacer(1, 8),
        T.toc(),
        Spacer(1, 16),
        Paragraph('You can stop reading here', T.SS['Section']),
        para(
            f'{chapters("installing", "verifying", "orthomosaic").capitalize()}'
            ' are the whole job if what you need is a canopy map from an '
            'orthomosaic: install the environment, prove it, run the model. '
            f'{ch("errors").capitalize()} is the error catalogue and the only '
            'other chapter worth reading before something goes wrong. '
            'Everything between exists for the reader who has to change '
            'something.'),
        para(
            'The chapters run in the order the work is done: the data, then '
            'what the configuration will do with it, then training, scoring '
            'and prediction. Read '
            f'{ch("repository")} when you have to find a file, and '
            f'{chapters("data", "configuration")} before building a '
            'dataset of your own — the mask encoding is the part that fails '
            f'silently, and {ch("configuration")} says which settings can be '
            'changed without invalidating the published scores. Skip '
            f'{ch("training")} and {ch("adapting")} unless you are training: '
            'six trained models are supplied, and reproducing one of them '
            f'costs a day of GPU time. {ch("handover").capitalize()} is for '
            'whoever has to hand the system on again.'),
    ]
    # No PageBreak here on purpose. Chapter 1 carries CondPageBreak(200): it
    # opens a new page only if too little room is left, and otherwise starts
    # under the signpost. An unconditional break left this page 19.7% full.
    return s


def _facts():
    """Read the two scalars the title page quotes out of FACTS.yml.

    Pulling in PyYAML for a repository name and a commit would add a
    dependency to a document that otherwise needs only ReportLab. The two
    values are matched by their keys, so a reordering of the file does not
    silently change what the title page claims.
    """
    path = HERE.parent / 'handover' / 'FACTS.yml'
    out = {'repository': 'NOT ESTABLISHED', 'commit': 'NOT ESTABLISHED'}
    if not path.exists():
        return out
    text = path.read_text(encoding='utf-8')
    m = re.search(r'^\s*repository:\s*(\S+)', text, re.M)
    if m:
        out['repository'] = m.group(1)
    m = re.search(r'commit_of_record:\s*\n\s*value:\s*(\S+)', text)
    if m:
        out['commit'] = m.group(1)
    return out


def _fact_repo():
    return _facts()['repository']


def _fact_commit():
    return _facts()['commit']


def glue_headings(story):
    """Keep every section heading with the block that follows it.

    The content modules append a heading and its first block as separate
    statements, which is readable but leaves the heading unattached: the
    engine's `glue(section(...), None)` idiom returns the heading alone, and
    ReportLab will then happily set it as the last line on a page. Section 6.4
    of the first build landed exactly there. Pairing them here rather than at
    every call site means a heading cannot be stranded by an edit that adds a
    paragraph above it.

    A heading is paired with a paragraph unconditionally, and with a table or
    a code block only when it is short enough that carrying it over cannot
    leave a half-empty page behind — the same trade the engine makes for code
    blocks. Section 9.1 of the reordered manual is where the code-block case
    was found: a heading whose next block is a command, and nothing else,
    was set as the last line on the page.
    """
    def heading_of(flowable):
        if isinstance(flowable, T._Anchored):
            return flowable
        if isinstance(flowable, KeepTogether):
            parts = getattr(flowable, '_content', [])
            if len(parts) == 1 and isinstance(parts[0], T._Anchored):
                return parts[0]
        return None

    out, i = [], 0
    while i < len(story):
        head = heading_of(story[i])
        rest = story[i + 1:i + 4]
        if head is not None and rest:
            if isinstance(rest[0], Paragraph):
                out.append(KeepTogether([head, rest[0]]))
                i += 2
                continue
            if (len(rest) >= 2 and isinstance(rest[0], Spacer)
                    and isinstance(rest[1], Table)
                    and len(rest[1]._cellvalues) <= 6):
                out.append(KeepTogether([head, rest[0], rest[1]]))
                i += 3
                continue
            if _short_code_block(rest[0]):
                # Flattened rather than nested: a KeepTogether inside a
                # KeepTogether measures as taller than its contents in
                # ReportLab, and nesting them broke six pages.
                out.append(KeepTogether([head] + list(rest[0]._content)))
                i += 2
                continue
        out.append(story[i])
        i += 1
    return out


def _short_code_block(flowable, limit=6):
    """Is this a code block short enough to carry over with a heading?

    ``code()`` returns one ``KeepTogether`` holding a spacer, a one-column
    table of lines, and a spacer. Anything longer than ``limit`` lines is
    left alone: dragging it to the next page would leave more white space
    behind than a stranded heading costs.
    """
    if not isinstance(flowable, KeepTogether):
        return False
    parts = getattr(flowable, '_content', [])
    # [Spacer, Table, Spacer] is a code block. A callout is one Table on its
    # own inside a KeepTogether, and is far too tall to drag after a heading:
    # gluing those as well took the manual from 27 pages to 33, nine of them
    # under 70 per cent full.
    if len(parts) != 3 or not isinstance(parts[1], Table):
        return False
    return (len(parts[1]._cellvalues) <= limit
            and len(parts[1]._argW) == 1)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        HERE / 'Ghaf-Crown-Mapping-Technical-Manual.pdf')

    T.register_fonts()
    T.SS = T.stylesheet()

    story = front_matter()

    import content_a
    story += content_a.story()
    try:
        import content_b
        story += content_b.story()
    except ImportError:
        print('[note] content_b.py not present; building chapters 1 onward '
              'from content_a.py only')

    # Printed before the build, because code() runs while the story is being
    # assembled, not while it is being laid out.
    for size, n, line in T.CODE_OVERLONG:
        print(f'[WARN] code line of {n} chars needs {size} pt: {line}')

    doc = T.Manual(str(out), title=TITLE, running=RUNNING)
    doc.multiBuild(glue_headings(story))

    pages = len(T.QA_FILL)
    print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {pages} pages)')
    return out


if __name__ == '__main__':
    main()
