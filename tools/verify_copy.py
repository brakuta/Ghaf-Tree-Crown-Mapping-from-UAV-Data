"""Prove a copy is complete and uncorrupted, file by file.

Copying tens of gigabytes off a machine you are about to lose access to is
only worth doing if you can show it arrived intact. This compares two trees by
relative path and SHA-256, and names every file that is missing, extra, or
different. A clean run is evidence the copy can be trusted; anything else
names exactly what to re-copy while the source is still reachable.

Usage
-----
    python tools/verify_copy.py SOURCE DEST [--manifest copy-manifest.json]
                               [--quick] [--ignore PATTERN ...]

``--quick`` compares size only, which is fast enough to run over a whole disk
and catches truncated transfers, but will not catch silent corruption.
"""

import argparse
import fnmatch
import hashlib
import json
import pathlib
import sys


def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def index(root, ignore):
    """Relative path -> (size, absolute path) for every file under root."""
    out = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in ignore):
            continue
        try:
            out[rel] = (path.stat().st_size, path)
        except OSError:
            pass
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("dest", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path("copy-manifest.json"))
    parser.add_argument("--quick", action="store_true",
                        help="compare sizes only; catches truncation, not corruption")
    parser.add_argument("--ignore", nargs="*", default=["*.tmp", "Thumbs.db",
                                                        ".DS_Store"],
                        metavar="PATTERN")
    args = parser.parse_args(argv)

    for d in (args.source, args.dest):
        if not d.is_dir():
            print(f"Not a directory: {d}", file=sys.stderr)
            return 2

    src = index(args.source, args.ignore)
    dst = index(args.dest, args.ignore)

    missing = sorted(set(src) - set(dst))
    extra = sorted(set(dst) - set(src))
    shared = sorted(set(src) & set(dst))

    size_mismatch, hash_mismatch, verified = [], [], []
    for rel in shared:
        s_size, s_path = src[rel]
        d_size, d_path = dst[rel]
        if s_size != d_size:
            size_mismatch.append((rel, s_size, d_size))
            continue
        if args.quick:
            verified.append(rel)
            continue
        try:
            s_hash, d_hash = digest(s_path), digest(d_path)
        except OSError as exc:
            hash_mismatch.append((rel, f"unreadable: {exc}"))
            continue
        if s_hash == d_hash:
            verified.append(rel)
        else:
            hash_mismatch.append((rel, "content differs"))

    total = sum(s for s, _ in src.values())
    print(f"source : {args.source}   {len(src)} files, {total / 2**30:.2f} GB")
    print(f"dest   : {args.dest}   {len(dst)} files")
    print()
    print(f"  verified {'identical' if not args.quick else 'same size'} : {len(verified)}")
    print(f"  MISSING from dest                : {len(missing)}")
    print(f"  size mismatch                    : {len(size_mismatch)}")
    print(f"  content mismatch                 : {len(hash_mismatch)}")
    print(f"  present in dest only             : {len(extra)}")

    for rel in missing[:20]:
        print(f"    MISSING  {rel}")
    if len(missing) > 20:
        print(f"    ... and {len(missing) - 20} more")
    for rel, a, b in size_mismatch[:20]:
        print(f"    SIZE     {rel}  ({a} vs {b})")
    for rel, why in hash_mismatch[:20]:
        print(f"    CONTENT  {rel}  ({why})")

    args.manifest.write_text(json.dumps({
        "source": str(args.source), "dest": str(args.dest),
        "mode": "size-only" if args.quick else "sha256",
        "counts": {"source_files": len(src), "dest_files": len(dst),
                   "verified": len(verified), "missing": len(missing),
                   "size_mismatch": len(size_mismatch),
                   "content_mismatch": len(hash_mismatch), "extra": len(extra)},
        "missing": missing,
        "size_mismatch": [{"path": r, "source": a, "dest": b}
                          for r, a, b in size_mismatch],
        "content_mismatch": [{"path": r, "reason": w} for r, w in hash_mismatch],
        "extra": extra,
    }, indent=2), encoding="utf-8")
    print(f"\nManifest written to {args.manifest}")

    bad = missing or size_mismatch or hash_mismatch
    if bad:
        print("\nCOPY IS NOT COMPLETE - re-copy the files listed above "
              "while the source is still reachable.", file=sys.stderr)
        return 1
    print("\nCOPY VERIFIED: every source file is present in the destination "
          f"and {'byte-identical' if not args.quick else 'the same size'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
