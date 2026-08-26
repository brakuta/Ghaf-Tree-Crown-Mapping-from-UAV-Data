"""Collect the log and config material for specific runs, and nothing else.

A work_dirs tree accumulates every experiment a machine ever ran. For a
handover, only the runs behind the published table matter: each reported
figure's evaluation, and the training run whose checkpoint that evaluation
loaded. This derives both from the survey manifest rather than being told
them, so no directory is missed and none is included by accident.

Weights are excluded - they are handled by handover_copy.py, which verifies
each one by digest.

Usage
-----
    python tools/collect_runs.py handover-manifest.json DEST [--only-runs PREFIX ...]

    --only-runs 20250318     just the evaluation session behind the paper
"""

import argparse
import json
import pathlib
import shutil
import sys

WEIGHTS = {".pth", ".pt", ".ckpt", ".pyth", ".safetensors"}


def tree_size(root):
    total = files = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
                files += 1
            except OSError:
                pass
    return total, files


def copy_tree(src, dest):
    """Copy a run directory without its weights. Returns (bytes, files)."""
    total = files = 0
    for path in src.rglob("*"):
        if not path.is_file() or path.suffix.lower() in WEIGHTS:
            continue
        target = dest / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, target)
            total += target.stat().st_size
            files += 1
        except OSError as exc:
            print(f"    could not copy {path}: {exc}", file=sys.stderr)
    return total, files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("dest", type=pathlib.Path)
    parser.add_argument("--only-runs", nargs="*", metavar="PREFIX",
                        help="restrict to evaluations whose run directory name "
                             "starts with one of these")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    work_dirs = pathlib.Path(data["work_dirs"]).resolve()
    evaluations = data.get("evaluations", [])

    selected = []
    for record in evaluations:
        run = pathlib.Path(record["log"]).parent
        if args.only_runs and not any(run.name.startswith(p) for p in args.only_runs):
            continue
        selected.append(record)

    if not selected:
        print("No evaluations matched.", file=sys.stderr)
        return 1

    # Each evaluation needs its own directory, plus the training run whose
    # checkpoint it loaded. Deduplicate: several evaluations share a source.
    wanted = {}
    for record in selected:
        eval_dir = pathlib.Path(record["log"]).parent
        wanted.setdefault(eval_dir, set()).add("evaluation")
        ck = record.get("load_from_resolved")
        if ck:
            train_dir = pathlib.Path(ck).parent
            if train_dir.is_dir():
                wanted.setdefault(train_dir, set()).add("training")

    print(f"{len(selected)} evaluation(s) selected, "
          f"needing {len(wanted)} directory(ies):\n")

    grand_bytes = grand_files = 0
    for src in sorted(wanted):
        roles = "+".join(sorted(wanted[src]))
        try:
            rel = src.resolve().relative_to(work_dirs)
        except ValueError:
            rel = pathlib.Path(src.name)
        size, count = tree_size(src)
        print(f"  [{roles:<20}] {rel}")
        print(f"{'':25}{size / 2**20:.1f} MB total on disk, {count} files")

        if args.dry_run:
            continue
        copied_bytes, copied_files = copy_tree(src, args.dest / rel)
        grand_bytes += copied_bytes
        grand_files += copied_files
        print(f"{'':25}copied {copied_files} file(s), "
              f"{copied_bytes / 2**20:.1f} MB (weights excluded)")

    if args.dry_run:
        print("\nDry run: nothing copied.")
        return 0

    print(f"\nCollected {grand_files} file(s), {grand_bytes / 2**20:.1f} MB "
          f"into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
