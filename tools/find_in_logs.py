"""Search a work_dirs tree for a literal string in its text logs.

Training logs record every validation score, so locating a reported figure is
faster and cheaper than re-reading checkpoints: the logs are small text files
and the match is exact.

Usage
-----
    python tools/find_in_logs.py ROOT PATTERN [--ext .json .log]
"""

import argparse
import pathlib
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("pattern", help="literal text to look for, e.g. 80.14")
    parser.add_argument("--ext", nargs="+", default=[".json", ".log", ".txt", ".py"],
                        help="file extensions to search (default: %(default)s)")
    parser.add_argument("--max-per-file", type=int, default=1, metavar="N",
                        help="matching lines to show per file (default: %(default)s; "
                             "0 shows every match)")
    args = parser.parse_args(argv)

    exts = {e.lower() for e in args.ext}
    hits = scanned = 0
    for path in args.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if args.pattern in text:
            hits += 1
            print(path)
            shown = 0
            for n, line in enumerate(text.splitlines(), 1):
                if args.pattern not in line:
                    continue
                print(f"    {n}: {line.strip()[:400]}")
                shown += 1
                if args.max_per_file and shown >= args.max_per_file:
                    break

    print(f"\n{hits} file(s) contain {args.pattern!r}, out of {scanned} searched.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
