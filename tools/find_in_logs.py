"""Search a work_dirs tree for a literal string in its text logs.

Training logs record every validation score, so locating a reported figure is
faster and cheaper than re-reading checkpoints: the logs are small text files
and the match is exact.

Usage
-----
    python tools/find_in_logs.py ROOT PATTERN [PATTERN ...] [--ext .json .log]

Several patterns can be given at once, which is the usual case when checking
a table of reported figures against the runs that produced them.
"""

import argparse
import pathlib
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("patterns", nargs="+",
                        help="literal text to look for, e.g. 80.14 79.32")
    parser.add_argument("--ext", nargs="+", default=[".json", ".log", ".txt", ".py"],
                        help="file extensions to search (default: %(default)s)")
    parser.add_argument("--max-per-file", type=int, default=1, metavar="N",
                        help="matching lines to show per file (default: %(default)s; "
                             "0 shows every match)")
    args = parser.parse_args(argv)

    exts = {e.lower() for e in args.ext}

    # Read each file once and test every pattern against it, rather than
    # walking the tree once per pattern.
    found = {p: 0 for p in args.patterns}
    scanned = 0
    for path in args.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in args.patterns:
            if pattern not in text:
                continue
            found[pattern] += 1
            print(f"[{pattern}] {path}")
            shown = 0
            for n, line in enumerate(text.splitlines(), 1):
                if pattern not in line:
                    continue
                print(f"    {n}: {line.strip()[:400]}")
                shown += 1
                if args.max_per_file and shown >= args.max_per_file:
                    break

    print(f"\nsearched {scanned} file(s):", file=sys.stderr)
    for pattern, count in found.items():
        print(f"  {pattern!r}: {count} file(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
