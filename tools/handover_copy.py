"""Copy the checkpoints a manifest names, and prove each arrived intact.

Reads the JSON written by handover_survey.py, copies every checkpoint that
still exists, and verifies each by SHA-256 as it goes. Nothing is typed by
hand, so a path cannot be mistyped and a file cannot be silently missed.

The digests are written alongside the copies. They are what lets whoever
receives this archive prove, later and independently, that the weights they
hold are the ones that produced the published numbers.

Usage
-----
    python tools/handover_copy.py handover-manifest.json DEST [--all]

By default only checkpoints used by an evaluation on a given date are copied
when --only-runs is supplied; otherwise every existing checkpoint in the
manifest is copied.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys


def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("dest", type=pathlib.Path)
    parser.add_argument("--only-runs", nargs="*", metavar="PREFIX",
                        help="copy only checkpoints used by runs whose name "
                             "starts with one of these, e.g. 20250318")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [c for c in data.get("checkpoints", []) if c.get("exists")]
    skipped = [c for c in data.get("checkpoints", []) if not c.get("exists")]

    if args.only_runs:
        entries = [c for c in entries
                   if any(run.startswith(p)
                          for run in c.get("used_by", [])
                          for p in args.only_runs)]

    if not entries:
        print("No existing checkpoints selected.", file=sys.stderr)
        return 1

    total = sum(c["size_bytes"] or 0 for c in entries)
    print(f"{len(entries)} checkpoint(s), {total / 2**30:.2f} GB -> {args.dest}\n")

    receipts, failures = [], []
    for c in entries:
        src = pathlib.Path(c["path"])
        # Keep the run directory so two files of the same name cannot collide.
        out_dir = args.dest / src.parent.name
        out = out_dir / src.name
        label = f"{src.parent.name}/{src.name}"

        if args.dry_run:
            print(f"  would copy  {label}  ({(c['size_bytes'] or 0)/2**20:.1f} MB)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  copying {label} ...", end="", flush=True)
        try:
            shutil.copy2(src, out)
            src_hash = digest(src)
            dst_hash = digest(out)
        except OSError as exc:
            print(f" FAILED ({exc})")
            failures.append({"checkpoint": label, "error": str(exc)})
            continue

        ok = src_hash == dst_hash
        print(" verified" if ok else " HASH MISMATCH")
        if not ok:
            failures.append({"checkpoint": label, "error": "sha256 mismatch"})
        receipts.append({
            "source": str(src), "copied_to": str(out), "sha256": src_hash,
            "size_bytes": src.stat().st_size, "verified": ok,
            "used_by": c.get("used_by", []),
        })

    if args.dry_run:
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "CHECKPOINTS.json").write_text(
        json.dumps({"checkpoints": receipts, "not_copied": skipped,
                    "failures": failures}, indent=2), encoding="utf-8")

    lines = ["# Checkpoints", "",
             "SHA-256 digests of the weights behind the reported results.",
             "Verify a copy with `certutil -hashfile <file> SHA256` on Windows",
             "or `sha256sum <file>` elsewhere.", ""]
    for r in receipts:
        lines += [f"## {pathlib.Path(r['copied_to']).parent.name}/"
                  f"{pathlib.Path(r['copied_to']).name}",
                  f"- sha256: `{r['sha256']}`",
                  f"- size: {r['size_bytes']:,} bytes",
                  f"- original path: `{r['source']}`",
                  f"- used by evaluation run(s): {', '.join(r['used_by']) or 'n/a'}",
                  ""]
    (args.dest / "CHECKPOINTS.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{len(receipts)} copied and verified. "
          f"Digests written to {args.dest / 'CHECKPOINTS.md'}")
    if skipped:
        print(f"{len(skipped)} checkpoint(s) in the manifest no longer exist "
              f"and were skipped.", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} FAILED - re-copy these while the source is "
              f"still reachable:", file=sys.stderr)
        for f in failures:
            print(f"  {f['checkpoint']}: {f['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
