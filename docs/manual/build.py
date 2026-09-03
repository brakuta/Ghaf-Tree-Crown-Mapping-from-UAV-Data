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

import typeset as T                                             # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, Spacer     # noqa: E402

TITLE = 'Ghaf tree-crown mapping from UAV imagery'
SUBTITLE = 'Technical manual'
RUNNING = 'Ghaf crown mapping · technical manual'


def front_matter():
    """Title page, then the contents.

    The signpost matters more than the rest of it. A reader who needs one map
    made should not have to decide for themselves which of thirteen chapters
    they can skip, and a manual that does not say so will be read from the
    front until the reader gives up.
    """
    from typeset import para, HRule

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
            'Chapters 3 and 5 are the whole job if what you need is a canopy '
            'map from an orthomosaic: install the environment, prove it, run '
            'the model. Chapter 12 is the error catalogue, and it is the only '
            'other chapter worth reading before something goes wrong. '
            'Everything between them exists for the reader who has to change '
            'something — retrain on new labels, adapt to a second site, or '
            'work out why a number moved.'),
        para(
            'Read chapter 2 if you have to find a file. Read chapter 4 before '
            'you build a dataset of your own; the two-class mask encoding is '
            'the part that goes wrong silently. Skip 9 and 10 entirely unless '
            'you are training, since six trained models are supplied and '
            'training one takes a day of GPU time to reproduce what is '
            'already in the bundle.'),
        PageBreak(),
    ]
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
    doc.multiBuild(story)

    pages = len(T.QA_FILL)
    print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {pages} pages)')
    return out


if __name__ == '__main__':
    main()
