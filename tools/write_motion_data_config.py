#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


def _yaml_quote(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _iter_pkl_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(".pkl"):
                files.append(Path(dirpath) / name)
    files.sort()
    return files


def _load_excludes_from_jump_config(jump_config_yaml: Path) -> Set[str]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required for --exclude-from-jump-config") from e

    with open(jump_config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return set()

    datasets = cfg.get("datasets", {})
    if not isinstance(datasets, dict):
        return set()

    out: Set[str] = set()
    for _, v in datasets.items():
        if not isinstance(v, dict):
            continue
        for key in ("root_pos_jump", "root_rot_jump", "dof_jump"):
            items = v.get(key, [])
            if isinstance(items, list):
                out.update(str(x) for x in items if isinstance(x, str))
    return out


def _resolve_scan_roots(dataset_root: Path, subdirs: Optional[List[str]]) -> List[Tuple[str, Path]]:
    if subdirs:
        roots: List[Tuple[str, Path]] = []
        for s in subdirs:
            p = (dataset_root / s).resolve()
            roots.append((s, p))
        return roots

    roots = []
    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            roots.append((child.name, child.resolve()))
    return roots


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a legged_gym motion_data_configs YAML from a folder of *_stageii.pkl motions."
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Humanoid_WBC_Dataset_GMR_30fps_GMR"),
        help="Dataset root folder containing one or more subfolders of .pkl motions.",
    )
    ap.add_argument(
        "--subdirs",
        nargs="*",
        default=None,
        help="Optional list of subdirectories (relative to dataset root) to include; default includes all immediate subdirs.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output YAML path, e.g. legged_gym/motion_data_configs/humanoid_wbc_30fps.yaml",
    )
    ap.add_argument("--weight", type=float, default=1.0, help="Sampling weight for each motion entry.")
    ap.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional description to write for each entry; default uses the top-level subdir name.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, only write up to this many motions (after shuffling if --shuffle).",
    )
    ap.add_argument("--shuffle", action="store_true", help="Shuffle motion list before applying --max-files.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle.")
    ap.add_argument(
        "--exclude-from-jump-config",
        type=Path,
        default=None,
        help="Optional path to a jumpy-sample config.yaml; excludes any listed paths (relative to dataset root).",
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    excludes: Set[str] = set()
    if args.exclude_from_jump_config is not None:
        excludes = _load_excludes_from_jump_config(args.exclude_from_jump_config)

    scan_roots = _resolve_scan_roots(dataset_root, args.subdirs)
    motion_entries: List[Tuple[str, str]] = []
    for tag, root in scan_roots:
        for pkl_path in _iter_pkl_files(root):
            # NOTE: some datasets may contain symlinks to files outside dataset_root.
            # Use relpath instead of Path.relative_to to support those cases.
            rel = os.path.relpath(str(pkl_path), str(dataset_root)).replace(os.sep, "/")
            if rel in excludes:
                continue
            motion_entries.append((rel, tag))

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(motion_entries)

    if args.max_files and int(args.max_files) > 0:
        motion_entries = motion_entries[: int(args.max_files)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"root_path: {_yaml_quote(str(dataset_root))}\n")
        f.write("motions:\n")
        for rel, tag in motion_entries:
            desc = args.description or tag
            f.write(f"- file: {_yaml_quote(rel)}\n")
            f.write(f"  weight: {float(args.weight)}\n")
            f.write(f"  description: {_yaml_quote(desc)}\n")

    print(f"Wrote: {args.out} ({len(motion_entries)} motions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
