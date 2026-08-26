"""Compare two directories of label masks by filename and by content.

A benchmark is only a comparison if every model was scored against the same
ground truth. Where different runs point at differently-named label
directories, this answers whether those are the same data under two names or
genuinely different annotations - and if different, by how much.

Usage
-----
    python tools/compare_dirs.py DIR_A DIR_B [--sample N]

Reports the file counts, which names are unique to each side, and among the
names they share, how many are byte-identical. For those that differ it also
reports how many pixels changed label, when Pillow and NumPy are available.
"""

import argparse
import hashlib
import pathlib
import sys


def file_digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def index(directory):
    """Map filename -> path for every file directly inside a directory."""
    return {p.name: p for p in sorted(directory.iterdir()) if p.is_file()}


def pixel_difference(path_a, path_b):
    """Fraction of pixels whose label differs, or None if it can't be read."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        a = np.array(Image.open(path_a))
        b = np.array(Image.open(path_b))
    except Exception:
        return None
    if a.shape != b.shape:
        return 1.0
    return float((a != b).sum()) / a.size if a.size else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir_a", type=pathlib.Path)
    parser.add_argument("dir_b", type=pathlib.Path)
    parser.add_argument("--sample", type=int, default=25, metavar="N",
                        help="how many differing files to measure pixel-wise "
                             "(default: %(default)s; 0 disables)")
    args = parser.parse_args(argv)

    for d in (args.dir_a, args.dir_b):
        if not d.is_dir():
            print(f"Not a directory: {d}", file=sys.stderr)
            return 1

    a, b = index(args.dir_a), index(args.dir_b)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    shared = sorted(set(a) & set(b))

    print(f"A  {args.dir_a}")
    print(f"   {len(a)} file(s)")
    print(f"B  {args.dir_b}")
    print(f"   {len(b)} file(s)")
    print()
    print(f"shared filenames : {len(shared)}")
    print(f"only in A        : {len(only_a)}")
    print(f"only in B        : {len(only_b)}")
    for name in only_a[:5]:
        print(f"    A only: {name}")
    for name in only_b[:5]:
        print(f"    B only: {name}")

    if not shared:
        print("\nNo filenames in common - these are unrelated sets.")
        return 0

    identical, differing = [], []
    for name in shared:
        if file_digest(a[name]) == file_digest(b[name]):
            identical.append(name)
        else:
            differing.append(name)

    print()
    print(f"of the {len(shared)} shared names:")
    print(f"    byte-identical : {len(identical)}")
    print(f"    differing      : {len(differing)}")

    if not differing:
        print("\nVERDICT: the two directories hold the same label data.")
        print("The differing directory names are cosmetic.")
        return 0

    print("\nVERDICT: these are DIFFERENT annotations, not a renaming.")

    if args.sample:
        measured = []
        for name in differing[: args.sample]:
            frac = pixel_difference(a[name], b[name])
            if frac is not None:
                measured.append((name, frac))
        if measured:
            avg = sum(f for _, f in measured) / len(measured)
            worst = max(measured, key=lambda kv: kv[1])
            print(f"\nmeasured {len(measured)} of the differing files:")
            print(f"    mean pixels relabelled : {avg * 100:.3f}%")
            print(f"    worst file             : {worst[0]} at {worst[1] * 100:.3f}%")
        else:
            print("\n(install pillow and numpy for a pixel-level measurement)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
