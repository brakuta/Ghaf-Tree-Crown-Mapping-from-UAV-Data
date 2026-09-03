#!/usr/bin/env python3
"""Layout QA. Reports any page that is under 70 per cent full.

    python docs/manual/check_layout.py [output.pdf]

Rebuilds the manual and reads `typeset.QA_FILL`, which the document template
records at the end of every page from the frame's own cursor. A half-empty
page in a generated document is nearly always a page break someone inserted to
tidy a section; this is the check that catches it before a reader does.

Exit status is 1 if any page falls below the threshold, so the check can gate
a commit.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import typeset as T          # noqa: E402
import build                 # noqa: E402

THRESHOLD = 0.70


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        HERE / 'Ghaf-Crown-Mapping-Technical-Manual.pdf')
    sys.argv = [sys.argv[0], str(out)]
    build.main()

    fills = list(T.QA_FILL)
    if not fills:
        print('[FAIL] no page fills recorded; the template did not run')
        return 1

    # The last page of a document is short by nature, and the title page is
    # deliberately open. Neither is a defect, so both are reported and
    # neither is counted against the threshold.
    last = fills[-1][0]
    offenders = [(n, f, ch) for n, f, ch in fills
                 if f < THRESHOLD and n not in (1, last)]

    for n, f, ch in fills:
        mark = '  ' if f >= THRESHOLD or n in (1, last) else '<-'
        print(f'{mark} page {n:>3}  {f * 100:5.1f}%  {ch}')

    mean = sum(f for _, f, _ in fills) / len(fills)
    body = [f for n, f, _ in fills if n not in (1, last)]
    print(f'\n{len(fills)} pages, mean fill {mean * 100:.1f}%, '
          f'body mean {sum(body) / len(body) * 100:.1f}%')

    if offenders:
        print(f'[FAIL] {len(offenders)} page(s) under '
              f'{THRESHOLD * 100:.0f}% full: '
              f'{", ".join(str(n) for n, _, _ in offenders)}')
        return 1
    print(f'[OK] no page under {THRESHOLD * 100:.0f}% full '
          f'(title page and last page excepted)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
