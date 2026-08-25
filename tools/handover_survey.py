"""Inventory everything a set of reported results depends on, before access is lost.

Reads a work_dirs tree and, for every run that performed an evaluation, records
which checkpoint it loaded, which data it read, and what it scored. The result
is a manifest naming the exact files that must be preserved for the reported
numbers to remain reproducible - and flagging any that are already missing.

Run this while the training machine is still reachable. It only reads.

Usage
-----
    python tools/handover_survey.py WORK_DIRS [-o handover-manifest.json]
                                    [--extra PATH ...] [--no-sizes]

``--extra`` adds directories to size-check alongside the ones discovered in the
configs, for anything the configs do not mention (source trees, notes).
"""

import argparse
import json
import os
import pathlib
import re
import sys

# The summary line mmengine writes when an evaluation loop finishes.
TEST_LINE = re.compile(r"Iter\((test|val)\)\s*\[\s*(\d+)\s*/\s*(\d+)\]\s+(.*mIoU:.*)")
METRIC = re.compile(r"(\w+):\s*([0-9.]+)")
ASSIGN = re.compile(r"^(load_from|work_dir)\s*=\s*(.+)$", re.M)


def dir_size(path):
    """Total bytes and file count under a path. Returns (None, None) if unreadable."""
    total = files = 0
    try:
        for root, _dirs, names in os.walk(path, onerror=lambda e: None):
            for name in names:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                    files += 1
                except OSError:
                    pass
    except OSError:
        return None, None
    return total, files


def config_beside(log_path):
    """The dumped config mmengine writes next to a log, if it is there."""
    candidate = log_path.parent / "vis_data" / "config.py"
    return candidate if candidate.is_file() else None


def read_config(path):
    """Execute a dumped config and return the fields we care about."""
    namespace = {}
    exec(path.read_text(encoding="utf-8", errors="replace"), namespace)  # noqa: S102

    def loader(name):
        block = (namespace.get(name) or {}).get("dataset") or {}
        prefix = block.get("data_prefix") or {}
        return {
            "data_root": block.get("data_root"),
            "img_path": prefix.get("img_path"),
            "seg_map_path": prefix.get("seg_map_path"),
        }

    model = namespace.get("model") or {}
    backbone = model.get("backbone") or {}
    return {
        "backbone": backbone.get("type"),
        "backbone_arch": backbone.get("arch") or backbone.get("depth"),
        "decode_head": (model.get("decode_head") or {}).get("type"),
        "load_from": namespace.get("load_from"),
        "train": loader("train_dataloader"),
        "val": loader("val_dataloader"),
        "test": loader("test_dataloader"),
    }


def scrape_log(text):
    """Fall back to reading the config mmengine echoes into the log itself."""
    found = {"load_from": None, "work_dir": None}
    for key, value in ASSIGN.findall(text):
        found[key] = value.strip().strip("'\"")
    return found


def evaluations(root):
    """Yield one record per completed evaluation found under root."""
    for log_path in sorted(root.rglob("*.log")):
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        finished = [m for m in TEST_LINE.finditer(text)
                    if m.group(2) == m.group(3)]  # final line only: [N/N]
        if not finished:
            continue
        last = finished[-1]
        metrics = {k: float(v) for k, v in METRIC.findall(last.group(4))
                   if k not in ("data_time", "time")}

        record = {
            "log": str(log_path),
            "loop": last.group(1),
            "images": int(last.group(3)),
            "metrics": metrics,
        }

        cfg_path = config_beside(log_path)
        if cfg_path:
            try:
                record.update(read_config(cfg_path))
                record["config"] = str(cfg_path)
            except Exception as exc:
                record["config_error"] = f"{type(exc).__name__}: {exc}"
        if not record.get("load_from"):
            record.update({k: v for k, v in scrape_log(text).items() if v})
        yield record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("work_dirs", type=pathlib.Path)
    parser.add_argument("-o", "--out", type=pathlib.Path,
                        default=pathlib.Path("handover-manifest.json"))
    parser.add_argument("--extra", nargs="*", default=[], metavar="PATH",
                        help="further directories to size-check")
    parser.add_argument("--no-sizes", action="store_true",
                        help="skip directory size measurement (much faster)")
    args = parser.parse_args(argv)

    if not args.work_dirs.is_dir():
        print(f"Not a directory: {args.work_dirs}", file=sys.stderr)
        return 1

    records = list(evaluations(args.work_dirs))
    if not records:
        print(f"No completed evaluations found under {args.work_dirs}", file=sys.stderr)
        return 1

    print(f"=== {len(records)} completed evaluation(s) ===\n")
    print(f"{'images':>7}  {'mIoU':>7}  {'mFscore':>8}  backbone / run")
    print("-" * 96)
    for r in sorted(records, key=lambda r: r["log"]):
        m = r["metrics"]
        bb = r.get("backbone") or "?"
        arch = r.get("backbone_arch")
        label = f"{bb}{f'({arch})' if arch else ''}"
        run = pathlib.Path(r["log"]).parent.name
        print(f"{r['images']:>7}  {m.get('mIoU', 0):>7.2f}  "
              f"{m.get('mFscore', 0):>8.2f}  {label:<30} {run}")

    # Everything the evaluations depend on.
    checkpoints, data_roots = {}, {}
    for r in records:
        ck = r.get("load_from")
        if ck and ck not in (None, "None"):
            checkpoints.setdefault(ck, []).append(pathlib.Path(r["log"]).parent.name)
        for split in ("train", "val", "test"):
            block = r.get(split) or {}
            if block.get("data_root"):
                key = (block["data_root"], block.get("seg_map_path"))
                data_roots.setdefault(key, set()).add(split)

    print(f"\n=== checkpoints these evaluations loaded ({len(checkpoints)}) ===")
    manifest_ck = []
    for path, runs in sorted(checkpoints.items()):
        p = pathlib.Path(path)
        exists = p.is_file()
        size = p.stat().st_size if exists else None
        mark = f"{size / 2**20:9.1f} MB" if exists else "  MISSING   "
        print(f"  {mark}  {path}")
        print(f"{'':14}used by: {', '.join(sorted(set(runs)))}")
        manifest_ck.append({"path": path, "exists": exists, "size_bytes": size,
                            "used_by": sorted(set(runs))})

    print(f"\n=== data directories these evaluations read ({len(data_roots)}) ===")
    manifest_data = []
    for (root, seg), splits in sorted(data_roots.items(), key=lambda kv: str(kv[0])):
        full = os.path.join(root, seg) if seg else root
        if args.no_sizes:
            size, count, note = None, None, "(not measured)"
        else:
            size, count = dir_size(full)
            note = (f"{size / 2**30:.2f} GB, {count} files" if size is not None
                    else "UNREADABLE or missing")
        print(f"  [{','.join(sorted(splits)):<14}] {full}")
        print(f"{'':18}{note}")
        manifest_data.append({"data_root": root, "seg_map_path": seg,
                              "splits": sorted(splits), "resolved": full,
                              "size_bytes": size, "file_count": count})

    manifest_extra = []
    if args.extra:
        print(f"\n=== additional paths ({len(args.extra)}) ===")
        for path in args.extra:
            size, count = (None, None) if args.no_sizes else dir_size(path)
            note = (f"{size / 2**30:.2f} GB, {count} files" if size is not None
                    else "(not measured)" if args.no_sizes else "UNREADABLE or missing")
            print(f"  {path}\n{'':18}{note}")
            manifest_extra.append({"path": path, "size_bytes": size,
                                   "file_count": count})

    args.out.write_text(json.dumps({
        "work_dirs": str(args.work_dirs),
        "evaluations": records,
        "checkpoints": manifest_ck,
        "data": manifest_data,
        "extra": manifest_extra,
    }, indent=2, default=str), encoding="utf-8")

    missing = [c for c in manifest_ck if not c["exists"]]
    total = sum(c["size_bytes"] or 0 for c in manifest_ck)
    print(f"\nManifest written to {args.out}")
    human = (f"{total / 2**30:.2f} GB" if total >= 2**30
             else f"{total / 2**20:.1f} MB")
    print(f"Checkpoints to preserve: {len(manifest_ck)}, {human} total")
    if missing:
        print(f"\n{len(missing)} referenced checkpoint(s) NO LONGER EXIST:",
              file=sys.stderr)
        for c in missing:
            print(f"  {c['path']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
