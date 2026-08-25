"""Extract provenance from MMSegmentation/mmengine checkpoints without moving them.

An mmengine ``.pth`` stores the resolved config, the iteration reached, the
dataset metadata and the full parameter tensors alongside the weights. This
script reads all of that and writes a small text report per checkpoint, so the
provenance of a run can be recovered and shared even when the checkpoint itself
is far too large to commit.

Usage
-----
    python tools/inspect_checkpoints.py <root> [-o OUTDIR]

``<root>`` is searched recursively for ``*.pth``. Reports land in OUTDIR
(default ``checkpoint-reports/``), one directory per checkpoint:

    recovered_config.py   the config the run actually executed
    summary.json          iteration, parameter counts, dataset metadata
    param_shapes.txt      every tensor name and shape in the state dict
    metrics.json          logged metric history, when the checkpoint carries it

Run it inside the environment that has PyTorch installed (the training env),
not necessarily the GIS one.
"""

import argparse
import json
import pathlib
import sys
import traceback
from collections import OrderedDict


def _stub_class(module, name):
    """Build a placeholder standing in for a class that cannot be imported."""

    def __init__(self, *args, **kwargs):
        self._stub_args = args
        self._stub_kwargs = kwargs

    def __setstate__(self, state):
        # Pickle restores most objects by instantiating then replaying state.
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.data = state

    def __repr__(self):
        return f"<unresolved {module}.{name}>"

    return type(
        f"Stub_{name}",
        (),
        {"__init__": __init__, "__setstate__": __setstate__, "__repr__": __repr__},
    )


def _tolerant_pickle_module():
    """A pickle shim that substitutes stubs for unimportable classes.

    Checkpoints reference classes from whatever wrote them - mmengine's
    HistoryBuffer, numpy dtypes, project-local modules. Those imports fail when
    the script runs outside the training environment, and the failure aborts a
    load that would otherwise have yielded a perfectly good config and state
    dict. Substituting a stub keeps the recoverable parts recoverable.
    """
    import pickle
    import types

    class TolerantUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _stub_class(module, name)

    shim = types.ModuleType("tolerant_pickle")
    shim.Unpickler = TolerantUnpickler
    shim.load = pickle.load
    shim.loads = pickle.loads
    shim.Pickler = pickle.Pickler
    shim.dump = pickle.dump
    shim.dumps = pickle.dumps
    shim.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    shim.DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL
    shim.UnpicklingError = pickle.UnpicklingError
    return shim


def load_checkpoint(path):
    """Load a checkpoint, preferring full metadata but degrading gracefully.

    Three attempts, most informative first:

    1. ``weights_only=False`` - everything, when every referenced class imports.
    2. a tolerant unpickler - everything, with stubs for classes that do not.
    3. ``weights_only=True`` - tensors only, the last resort.

    Returns the payload and a label naming which path succeeded, so the report
    records how much of the metadata is trustworthy.
    """
    import torch

    errors = []

    try:
        return torch.load(path, map_location="cpu", weights_only=False), "full"
    except Exception as exc:
        errors.append(f"weights_only=False: {exc}")

    try:
        return (
            torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                pickle_module=_tolerant_pickle_module(),
            ),
            "full-with-stubs",
        )
    except Exception as exc:
        errors.append(f"tolerant unpickler: {exc}")

    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights-only"
    except Exception as exc:
        errors.append(f"weights_only=True: {exc}")

    raise RuntimeError(
        "could not load checkpoint; attempts:\n  " + "\n  ".join(errors)
    )


def tensor_entries(state_dict):
    for key, value in state_dict.items():
        if hasattr(value, "numel") and hasattr(value, "shape"):
            yield key, value


def parameter_counts(state_dict):
    """Total parameters, plus a breakdown by top-level module prefix."""
    total = 0
    by_prefix = {}
    for key, value in tensor_entries(state_dict):
        n = int(value.numel())
        total += n
        prefix = key.split(".", 1)[0]
        by_prefix[prefix] = by_prefix.get(prefix, 0) + n
    return total, OrderedDict(sorted(by_prefix.items(), key=lambda kv: -kv[1]))


def stage_widths(state_dict):
    """Channel widths of the backbone stages, in depth order.

    Backbone variants of the same family differ mainly in their per-stage
    widths and depths, so this is usually enough to name a variant outright.
    """
    widths = []
    seen = set()
    for key, value in tensor_entries(state_dict):
        if not key.startswith("backbone."):
            continue
        if value.ndim != 4:  # convolution kernels only
            continue
        out_channels = int(value.shape[0])
        if out_channels not in seen:
            seen.add(out_channels)
            widths.append((key, out_channels))
    return widths


def extract_metrics(checkpoint):
    """Pull the logged scalar history out of the message hub, if present."""
    hub = checkpoint.get("message_hub")
    if not isinstance(hub, dict):
        return None
    out = {}
    for section in ("log_scalars", "runtime_info"):
        block = hub.get(section)
        if not isinstance(block, dict):
            continue
        rendered = {}
        for name, value in block.items():
            # HistoryBuffer exposes .data; anything else is taken as-is.
            data = getattr(value, "data", value)
            try:
                json.dumps(data)
                rendered[name] = data
            except (TypeError, ValueError):
                rendered[name] = repr(data)[:2000]
        out[section] = rendered
    return out or None


def report(path, outdir):
    checkpoint, mode = load_checkpoint(path)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"unexpected checkpoint payload: {type(checkpoint)!r}")

    meta = checkpoint.get("meta") or {}
    state_dict = checkpoint.get("state_dict", checkpoint)

    outdir.mkdir(parents=True, exist_ok=True)

    cfg = meta.get("cfg")
    if cfg:
        (outdir / "recovered_config.py").write_text(str(cfg), encoding="utf-8")

    total, by_prefix = parameter_counts(state_dict)

    summary = {
        "checkpoint": str(path),
        "load_mode": mode,
        "size_bytes": path.stat().st_size,
        "iter": meta.get("iter"),
        "epoch": meta.get("epoch"),
        "experiment_name": meta.get("experiment_name"),
        "seed": meta.get("seed"),
        "time": meta.get("time"),
        "mmengine_version": meta.get("mmengine_version"),
        "dataset_meta": meta.get("dataset_meta"),
        "config_recovered": bool(cfg),
        "total_parameters": total,
        "parameters_by_module": by_prefix,
        "backbone_stage_widths": stage_widths(state_dict),
        "top_level_keys": sorted(k for k in checkpoint if k != "state_dict"),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    with (outdir / "param_shapes.txt").open("w", encoding="utf-8") as fh:
        for key, value in tensor_entries(state_dict):
            fh.write(f"{key}\t{tuple(value.shape)}\n")

    metrics = extract_metrics(checkpoint)
    if metrics:
        (outdir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=pathlib.Path,
                        help="directory searched recursively for checkpoints")
    parser.add_argument("-o", "--outdir", type=pathlib.Path,
                        default=pathlib.Path("checkpoint-reports"),
                        help="where to write the reports")
    parser.add_argument("--glob", default="*.pth",
                        help="filename pattern to match (default: %(default)s). "
                             "A work_dirs tree holds a periodic checkpoint every "
                             "few thousand iterations; 'best_*.pth' skips those.")
    parser.add_argument("--max-per-dir", type=int, default=0, metavar="N",
                        help="keep only the N most recently modified matches in "
                             "each directory (default: no limit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be read, with sizes, and stop")
    args = parser.parse_args(argv)

    checkpoints = sorted(args.root.rglob(args.glob))
    if not checkpoints:
        print(f"No files matching {args.glob!r} under {args.root}", file=sys.stderr)
        return 1

    if args.max_per_dir:
        by_dir = {}
        for path in checkpoints:
            by_dir.setdefault(path.parent, []).append(path)
        kept = []
        for group in by_dir.values():
            group.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            kept.extend(group[: args.max_per_dir])
        checkpoints = sorted(kept)

    total_bytes = sum(p.stat().st_size for p in checkpoints)

    if args.dry_run:
        for path in checkpoints:
            print(f"  {path.stat().st_size / 2**20:9.1f} MB  {path}")
        print(f"\n{len(checkpoints)} file(s), "
              f"{total_bytes / 2**30:.2f} GB to read. Re-run without --dry-run.")
        return 0

    print(f"Found {len(checkpoints)} checkpoint(s) under {args.root} "
          f"({total_bytes / 2**30:.2f} GB to read)\n")
    failures = 0
    for path in checkpoints:
        # Mirror the layout under root so sibling runs stay distinguishable.
        rel = path.relative_to(args.root)
        outdir = args.outdir / rel.parent / rel.stem
        try:
            summary = report(path, outdir)
        except Exception:
            failures += 1
            print(f"FAILED  {rel}")
            traceback.print_exc(limit=3)
            continue
        print(
            f"  {str(rel):<58} iter={summary['iter']}  "
            f"params={summary['total_parameters']:,}  "
            f"cfg={'yes' if summary['config_recovered'] else 'NO'}"
        )

    print(f"\nReports written to {args.outdir}/")
    if failures:
        print(f"{failures} checkpoint(s) failed to load.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
