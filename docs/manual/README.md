# The manuals

Two PDFs, built from source in this folder.

| File | Pages | For |
|---|---|---|
| `Ghaf-Crown-Mapping-Technical-Manual.pdf` | 22 | the complete reference — what the system does, the repository, install, verify, data, inference, evaluation, training, adaptation, handover, errors |
| `Ghaf-Crown-Mapping-Quick-Reference.pdf` | 2 | the sheet that sits next to the machine |

## Rebuilding

```bash
pip install reportlab pyphen pymupdf
python docs/manual/build.py       # the manual
python docs/manual/quickref.py    # the quick reference
python docs/manual/check_layout.py  # layout QA; exits 1 on a page under 70% full
```

## The files

| File | What it is |
|---|---|
| `typeset.py` | the typesetting engine — styles, flowables, table of contents, running heads |
| `content_a.py` | chapters 1–5 |
| `content_b.py` | chapters 6–14 |
| `build.py` | front matter, and assembles the manual |
| `quickref.py` | the quick reference, content and build in one file |
| `check_layout.py` | layout QA — reports any page under 70% full |
| `fonts/` | IBM Plex Serif, Sans and Mono, bundled under the SIL Open Font License (`fonts/LICENSE-IBM-Plex.txt`) |

The fonts were taken from the official `@ibm/plex` 6.4.0 package, which ships
woff2 only; each face was decompressed to TTF with fontTools. Same outlines,
same metrics.

## The layout rules that matter

Four things make a generated document look generated, and each is a layout
decision rather than a matter of wording:

1. **A page break before every section.** The single biggest cause of
   half-empty pages. Only a chapter opens a new page here, and even then only
   when fewer than 200 points remain. The first build of this manual added one
   `PageBreak()` in the front matter and left a page 19.7% full; removing it
   was the fix.
2. **Headings stranded at the foot of a page.** Every heading is glued to the
   block that follows it. `glue_headings()` in `build.py` pairs them at
   assembly time, because a heading appended as its own statement is not
   attached to anything.
3. **Blocks that break one row before the end.** Short code blocks are kept
   whole; long tables repeat their header row instead.
4. **Tables running off the edge of the paper.** ReportLab neither clips nor
   shrinks an over-wide table: it draws it, straight off the sheet. Column
   widths in the content files are *proportions*, not promises —
   `fit_widths()` rescales every row to sum to exactly the measure.

## Verification

Every build is checked five ways. The first four are automatic:

| Check | Last result |
|---|---|
| `build.py` prints no `[WARN]` | no over-long code lines |
| `check_layout.py` | 22 pages, zero under 70% full, body mean 95.6% |
| spans crossing the right margin | 0, in both documents |
| extracted text scanned for `&nbsp`, `**`, NUL | 0 of each |

The fifth is looking at the rendered pages, and it is not optional. Two
defects in the first build passed all four automated checks and were obvious
on sight: a missing glyph (`▸` is not in IBM Plex, so it set as nothing), and
section 6.4 left as the last line on a page.

One thing the checks report and nobody has fixed: ReportLab declares
`Helvetica` in every page's font resources whether or not anything prints in
it. No content stream selects it. It is a resource declaration, not text.

## Where the content comes from

`docs/handover/FACTS.yml` is the authority. Every entry carries its
provenance: read from a file in the repository, measured by a command that was
run, derived arithmetically from other entries, or `NOT_ESTABLISHED`. Ten
values are marked never established — the full-mosaic run time, training
times, the validation scores of five of the six models — and the manual says
so rather than supplying a plausible figure.
