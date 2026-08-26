"""Rename copied checkpoint folders after the models they actually contain.

The working directories on the training machine were named after whichever
config was copied to create them, not after the model that ended up inside.
A folder called mask2former_swin-t_...fast_all_data holds FastViT; one called
dual_path holds CoAtNet. Handing those names to anyone else guarantees the
wrong model is cited.

This reads the survey manifest to find what each checkpoint's config actually
declared, renames each folder accordingly, and records the original path so
the trail back to the training machine is never lost.

Usage
-----
    python tools/label_checkpoints.py handover-manifest.json CHECKPOINT_DIR [--dry-run]
"""

import argparse
import json
import pathlib
import sys

# Class names in this repository that do not describe the architecture they
# build. Each is justified by the source, not by inference.
ALIASES = {
    # fastvit.py defines class fastvit_small with layers [6,6,18,6] and
    # embed_dims [76,152,304,608]; the same file names that exact combination
    # fastvit_ma36 in a commented-out definition directly above it.
    "fastvit_small": "fastvit-ma36",
}

HEADS = {
    "Mask2FormerHead": "mask2former",
    "UPerHead": "upernet",
    "FPNHead": "fpn",
    "SegformerHead": "segformer",
    "DepthwiseSeparableASPPHead": "deeplabv3plus",
}


def model_label(backbone, arch):
    """A folder name that says what the weights are."""
    if not backbone:
        return "unknown-backbone"
    if backbone in ALIASES:
        return ALIASES[backbone]
    name = backbone.split(".")[-1].replace("_timm", "").replace("_", "-").lower()
    if arch not in (None, "", "None"):
        name = f"{name}-{str(arch).lower()}"
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("checkpoint_dir", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    receipt_path = args.checkpoint_dir / "CHECKPOINTS.json"
    if not receipt_path.is_file():
        print(f"No CHECKPOINTS.json in {args.checkpoint_dir}. Run "
              f"handover_copy.py first.", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    # What each checkpoint's config declared, keyed by the checkpoint it loaded.
    declared = {}
    for record in manifest.get("evaluations", []):
        resolved = record.get("load_from_resolved")
        if not resolved:
            continue
        declared.setdefault(resolved, {
            "backbone": record.get("backbone"),
            "arch": record.get("backbone_arch"),
            "head": record.get("decode_head"),
        })

    renamed, skipped = [], []
    for entry in receipt.get("checkpoints", []):
        source = entry["source"]
        current = pathlib.Path(entry["copied_to"])
        info = declared.get(source)
        if not info:
            skipped.append((current.parent.name, "no config found in manifest"))
            continue

        label = model_label(info["backbone"], info["arch"])
        head = HEADS.get(info["head"], (info["head"] or "").lower() or "head")
        new_name = f"{label}_{head}"

        old_dir = current.parent
        new_dir = old_dir.parent / new_name
        arch_note = f", arch={info['arch']}" if info["arch"] else ""
        print(f"  {old_dir.name}")
        print(f"    -> {new_name}    "
              f"({info['backbone']}{arch_note}, {info['head']})")

        entry["original_work_dir"] = pathlib.Path(source).parent.name
        entry["model"] = new_name

        if args.dry_run or old_dir.name == new_name:
            continue
        if new_dir.exists():
            skipped.append((old_dir.name, f"{new_name} already exists"))
            continue
        old_dir.rename(new_dir)
        entry["copied_to"] = str(new_dir / current.name)
        renamed.append((old_dir.name, new_name))

    if args.dry_run:
        print("\nDry run: nothing renamed.")
        return 0

    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    lines = ["# Checkpoints", "",
             "Folders are named after the model each checkpoint actually contains,",
             "which is not always what the original working directory was called.",
             "The original path is recorded for every entry.", "",
             "Verify a copy with `certutil -hashfile <file> SHA256` on Windows",
             "or `sha256sum <file>` elsewhere.", ""]
    for entry in receipt.get("checkpoints", []):
        path = pathlib.Path(entry["copied_to"])
        lines += [f"## {entry.get('model', path.parent.name)}/{path.name}",
                  f"- sha256: `{entry['sha256']}`",
                  f"- size: {entry['size_bytes']:,} bytes",
                  f"- original working directory: `{entry.get('original_work_dir', '?')}`",
                  f"- original path: `{entry['source']}`",
                  f"- used by evaluation run(s): "
                  f"{', '.join(entry.get('used_by', [])) or 'n/a'}",
                  ""]
    (args.checkpoint_dir / "CHECKPOINTS.md").write_text("\n".join(lines),
                                                        encoding="utf-8")

    print(f"\nRenamed {len(renamed)} folder(s); receipts updated.")
    for name, why in skipped:
        print(f"  skipped {name}: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
