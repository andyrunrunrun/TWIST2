#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _iter_motion_pkls(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(".pkl"):
                files.append(Path(dirpath) / name)
    files.sort()
    return files


def _maybe_tqdm(it: Iterable[Path], total: int):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, total=total, desc="Converting to .npz")
    except Exception:
        return it


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x
    return x.astype(np.float32, copy=False)


def convert_one(pkl_path: Path, npz_path: Path, cast_float32: bool, compress: bool) -> None:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise TypeError(f"Unexpected pickle type: {type(d)}")

    fps = float(d["fps"])
    root_pos = np.asarray(d["root_pos"])
    root_rot = np.asarray(d["root_rot"])
    dof_pos = np.asarray(d["dof_pos"])
    local_body_pos = np.asarray(d["local_body_pos"])
    link_body_list = d["link_body_list"]
    if not isinstance(link_body_list, list) or not all(isinstance(x, str) for x in link_body_list):
        raise TypeError("link_body_list must be a list[str]")
    link_body_arr = np.asarray(link_body_list, dtype=str)

    if cast_float32:
        root_pos = _as_float32(root_pos)
        root_rot = _as_float32(root_rot)
        dof_pos = _as_float32(dof_pos)
        local_body_pos = _as_float32(local_body_pos)

    _ensure_parent(npz_path)
    save = np.savez_compressed if compress else np.savez
    save(
        npz_path,
        fps=np.asarray(fps, dtype=np.float32),
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
        local_body_pos=local_body_pos,
        link_body_list=link_body_arr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Convert StageII motion *.pkl to *.npz.\n"
            "This is useful when the pkls were generated with NumPy 2.x and crash or fail to load under the "
            "IsaacGym (py38 / NumPy 1.x) environment.\n"
            "Run this script in a NumPy 2.x environment (e.g. py311) that can load the original pkls."
        )
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder containing motion pkls (may include symlinked subdirs).",
    )
    ap.add_argument(
        "--subdirs",
        nargs="*",
        default=None,
        help="Optional list of subdirectories (relative to dataset root) to convert; default converts everything under dataset root.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help=(
            "If set, write *.npz under this root mirroring the relative path under --dataset-root. "
            "If omitted, writes next to each input pkl (same folder, just .npz extension)."
        ),
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files.")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed (slower, smaller).")
    ap.add_argument("--no-float32", action="store_true", help="Do not cast arrays to float32.")
    ap.add_argument("--max-files", type=int, default=0, help="If >0, only convert the first N files (sorted).")
    args = ap.parse_args()

    dataset_root = Path(os.path.abspath(os.path.expanduser(str(args.dataset_root))))

    scan_roots: List[Tuple[str, Path]] = []
    if args.subdirs:
        for s in args.subdirs:
            scan_roots.append((s, dataset_root / s))
    else:
        scan_roots.append((".", dataset_root))

    all_files: List[Path] = []
    for _, root in scan_roots:
        if root.exists():
            all_files.extend(_iter_motion_pkls(root))
    all_files.sort()

    if args.max_files and int(args.max_files) > 0:
        all_files = all_files[: int(args.max_files)]

    cast_float32 = not args.no_float32

    converted = 0
    skipped = 0
    failed = 0

    it = _maybe_tqdm(all_files, total=len(all_files))
    for pkl_path in it:
        try:
            if args.out_root is None:
                npz_path = pkl_path.with_suffix(".npz")
            else:
                rel = os.path.relpath(str(pkl_path), str(dataset_root))
                npz_path = (args.out_root / rel).with_suffix(".npz")

            if npz_path.exists() and not args.overwrite:
                skipped += 1
                continue

            convert_one(pkl_path, npz_path, cast_float32=cast_float32, compress=args.compress)
            converted += 1
        except Exception as e:
            failed += 1
            # Keep going; large datasets may contain a few bad files.
            print(f"[WARN] Failed: {pkl_path} -> {e}")

    print(f"Done. converted={converted} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

