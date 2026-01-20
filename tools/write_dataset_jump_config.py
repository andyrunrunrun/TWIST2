#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List


def _yaml_quote(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _relpath_str(path: str, root: Path) -> str:
    try:
        return os.path.relpath(path, str(root))
    except Exception:
        return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/weijin/source/Datasets/Humanoid_WBC_Dataset_GMR_30fps_GMR"),
        help="Where to write config.yaml",
    )
    ap.add_argument(
        "--scan-root",
        type=Path,
        default=Path("outputs/jumpy_scan"),
        help="Folder containing <dataset_name>/all_sorted.csv from tools/find_jumpy_samples.py",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["AMASS_g1_GMR8_PHC_missing_30fps", "PHUMA_filtered", "TWIST2_dataset"],
    )
    ap.add_argument("--out", type=Path, default=None, help="Output yaml path (default: <dataset-root>/config.yaml)")
    ap.add_argument("--root-step-gt-m", type=float, default=0.5)
    ap.add_argument("--root-dang-gt-deg", type=float, default=45.0)
    ap.add_argument("--dof-step-gt-rad", type=float, default=1.0)
    args = ap.parse_args()

    dataset_root = args.dataset_root
    out_path = args.out or (dataset_root / "config.yaml")

    thresholds = {
        "root_step_gt_m": float(args.root_step_gt_m),
        "root_dang_gt_deg": float(args.root_dang_gt_deg),
        "dof_step_gt_rad": float(args.dof_step_gt_rad),
    }

    per_dataset: Dict[str, Dict[str, List[str] | int | Dict[str, int] | str]] = {}

    for name in args.datasets:
        csv_path = args.scan_root / name / "all_sorted.csv"
        if not csv_path.exists():
            per_dataset[name] = {"error": f"missing scan csv: {csv_path}"}
            continue

        root_pos_jump: List[str] = []
        root_rot_jump: List[str] = []
        dof_jump: List[str] = []
        total = 0

        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                total += 1
                rel = _relpath_str(row["path"], dataset_root)
                if int(row["n_root_step_gt"]) > 0:
                    root_pos_jump.append(rel)
                if int(row["n_root_dang_gt"]) > 0:
                    root_rot_jump.append(rel)
                if int(row["n_dof_step_gt"]) > 0:
                    dof_jump.append(rel)

        per_dataset[name] = {
            "total_files": total,
            "counts": {
                "root_pos_jump_files": len(root_pos_jump),
                "root_rot_jump_files": len(root_rot_jump),
                "dof_jump_files": len(dof_jump),
            },
            "root_pos_jump": sorted(root_pos_jump),
            "root_rot_jump": sorted(root_rot_jump),
            "dof_jump": sorted(dof_jump),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("version: 1\n")
        f.write(f"generated_at: {_yaml_quote(time.strftime('%Y-%m-%d %H:%M:%S'))}\n")
        f.write("thresholds:\n")
        for k, v in thresholds.items():
            f.write(f"  {k}: {v}\n")
        f.write("notes:\n")
        f.write("  definition:\n")
        f.write(
            f"    root_pos_jump: {_yaml_quote('exists frame t where ||root_pos[t]-root_pos[t-1]|| > root_step_gt_m')}\n"
        )
        f.write(
            f"    root_rot_jump: {_yaml_quote('exists frame t where angle(root_rot[t-1]^-1 * root_rot[t]) > root_dang_gt_deg')}\n"
        )
        f.write(
            f"    dof_jump: {_yaml_quote('exists frame t where max_j |dof_pos[t,j]-dof_pos[t-1,j]| > dof_step_gt_rad')}\n"
        )
        f.write(f"  paths: {_yaml_quote('All paths are relative to Humanoid_WBC_Dataset_GMR_30fps_GMR root.')}\n")
        f.write(f"  scan_root: {_yaml_quote(str(Path(args.scan_root).resolve()))}\n")
        f.write("datasets:\n")

        for name in args.datasets:
            ds = per_dataset.get(name, {})
            f.write(f"  {name}:\n")
            if "error" in ds:
                f.write(f"    error: {_yaml_quote(str(ds['error']))}\n")
                continue
            f.write(f"    total_files: {ds['total_files']}\n")
            f.write("    counts:\n")
            counts = ds["counts"]  # type: ignore[assignment]
            for ck, cv in counts.items():  # type: ignore[union-attr]
                f.write(f"      {ck}: {int(cv)}\n")

            def emit_list(key: str, items: List[str]):
                f.write(f"    {key}:\n")
                if not items:
                    f.write("      []\n")
                    return
                for it in items:
                    f.write(f"      - {_yaml_quote(it)}\n")

            emit_list("root_pos_jump", ds["root_pos_jump"])  # type: ignore[arg-type]
            emit_list("root_rot_jump", ds["root_rot_jump"])  # type: ignore[arg-type]
            emit_list("dof_jump", ds["dof_jump"])  # type: ignore[arg-type]

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

