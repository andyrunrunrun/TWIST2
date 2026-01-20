#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _iter_pkl_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(".pkl"):
                files.append(Path(dirpath) / name)
    files.sort()
    return files


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_quat_xyzw(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return q / n


def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=False)
    return np.concatenate([-out[..., :3], out[..., 3:4]], axis=-1)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def _quat_delta_angle_rad(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray) -> np.ndarray:
    q0 = _normalize_quat_xyzw(q0_xyzw)
    q1 = _normalize_quat_xyzw(q1_xyzw)
    dq = _quat_mul_xyzw(q1, _quat_conjugate_xyzw(q0))
    dq = np.where(dq[..., 3:4] < 0.0, -dq, dq)  # shortest path
    v_norm = np.linalg.norm(dq[..., :3], axis=-1)
    w = np.clip(dq[..., 3], -1.0, 1.0)
    return 2.0 * np.arctan2(v_norm, w)


@dataclass
class Metrics:
    path: str
    fps: float
    frames: int
    seconds: float

    root_step_max_m: float
    root_speed_max_mps: float
    root_dang_max_rad: float
    root_ang_speed_max_rps: float
    dof_step_max_rad: float
    dof_vel_max_rps: float

    n_root_step_gt: int
    n_root_dang_gt: int
    n_dof_step_gt: int

    first_root_step_gt_frame: int
    first_root_dang_gt_frame: int
    first_dof_step_gt_frame: int

    @property
    def score(self) -> float:
        # A simple composite score for sorting (higher = worse)
        return (
            10.0 * self.root_step_max_m
            + 2.0 * (self.root_dang_max_rad * 180.0 / math.pi) / 45.0
            + 0.5 * self.dof_step_max_rad
            + 0.01 * (self.n_root_step_gt + self.n_root_dang_gt + self.n_dof_step_gt)
        )


def _first_idx(mask: np.ndarray) -> int:
    if not np.any(mask):
        return -1
    return int(np.argmax(mask)) + 1  # delta index -> frame index


def compute_metrics(
    path: Path,
    root_step_gt_m: float,
    root_dang_gt_deg: float,
    dof_step_gt_rad: float,
    dof_vel_clip_rps: float,
) -> Optional[Metrics]:
    try:
        d = _load_pickle(path)
        if not isinstance(d, dict):
            return None
        fps = float(d["fps"])
        root_pos = np.asarray(d["root_pos"], dtype=np.float64)
        root_rot = np.asarray(d["root_rot"], dtype=np.float64)
        dof_pos = np.asarray(d["dof_pos"], dtype=np.float64)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            return None
        if root_rot.ndim != 2 or root_rot.shape[1] != 4:
            return None
        if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
            return None

        T = int(min(root_pos.shape[0], root_rot.shape[0], dof_pos.shape[0]))
        if T < 2:
            return None
        root_pos = np.nan_to_num(root_pos[:T], copy=False)
        root_rot = np.nan_to_num(root_rot[:T], copy=False)
        dof_pos = np.nan_to_num(dof_pos[:T], copy=False)

        dt = 1.0 / fps
        seconds = dt * (T - 1)

        dpos = root_pos[1:] - root_pos[:-1]
        root_step = np.linalg.norm(dpos, axis=1)
        root_speed = root_step / dt

        dang = _quat_delta_angle_rad(root_rot[:-1], root_rot[1:])
        root_ang_speed = dang / dt

        ddof = dof_pos[1:] - dof_pos[:-1]
        dof_step = np.max(np.abs(ddof), axis=1)
        dof_vel = np.clip(np.max(np.abs(ddof / dt), axis=1), 0.0, dof_vel_clip_rps)

        root_dang_gt_rad = float(root_dang_gt_deg) * math.pi / 180.0

        mask_rs = root_step > float(root_step_gt_m)
        mask_ra = dang > root_dang_gt_rad
        mask_ds = dof_step > float(dof_step_gt_rad)

        return Metrics(
            path=str(path),
            fps=fps,
            frames=T,
            seconds=seconds,
            root_step_max_m=float(np.max(root_step)),
            root_speed_max_mps=float(np.max(root_speed)),
            root_dang_max_rad=float(np.max(dang)),
            root_ang_speed_max_rps=float(np.max(root_ang_speed)),
            dof_step_max_rad=float(np.max(dof_step)),
            dof_vel_max_rps=float(np.max(dof_vel)),
            n_root_step_gt=int(np.sum(mask_rs)),
            n_root_dang_gt=int(np.sum(mask_ra)),
            n_dof_step_gt=int(np.sum(mask_ds)),
            first_root_step_gt_frame=_first_idx(mask_rs),
            first_root_dang_gt_frame=_first_idx(mask_ra),
            first_dof_step_gt_frame=_first_idx(mask_ds),
        )
    except Exception:
        return None


def _write_csv(path: Path, rows: List[Metrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "score",
                "path",
                "fps",
                "frames",
                "seconds",
                "root_step_max_m",
                "root_speed_max_mps",
                "root_dang_max_deg",
                "root_ang_speed_max_rps",
                "dof_step_max_rad",
                "dof_vel_max_rps",
                "n_root_step_gt",
                "n_root_dang_gt",
                "n_dof_step_gt",
                "first_root_step_gt_frame",
                "first_root_dang_gt_frame",
                "first_dof_step_gt_frame",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    f"{r.score:.6f}",
                    r.path,
                    f"{r.fps:.6f}",
                    r.frames,
                    f"{r.seconds:.6f}",
                    f"{r.root_step_max_m:.6f}",
                    f"{r.root_speed_max_mps:.6f}",
                    f"{r.root_dang_max_rad * 180.0 / math.pi:.6f}",
                    f"{r.root_ang_speed_max_rps:.6f}",
                    f"{r.dof_step_max_rad:.6f}",
                    f"{r.dof_vel_max_rps:.6f}",
                    r.n_root_step_gt,
                    r.n_root_dang_gt,
                    r.n_dof_step_gt,
                    r.first_root_step_gt_frame,
                    r.first_root_dang_gt_frame,
                    r.first_dof_step_gt_frame,
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path, help="Directory to scan (e.g. .../TWIST2_dataset)")
    ap.add_argument("--out", type=Path, default=Path("outputs/jumpy_scan"))
    ap.add_argument("--limit", type=int, default=200, help="How many top rows to keep in summary")
    ap.add_argument("--root-step-gt-m", type=float, default=0.5, help="Per-frame root position jump threshold (meters)")
    ap.add_argument("--root-dang-gt-deg", type=float, default=45.0, help="Per-frame root rotation jump threshold (degrees)")
    ap.add_argument("--dof-step-gt-rad", type=float, default=1.0, help="Per-frame max(|Δdof|) threshold (rad)")
    ap.add_argument("--dof-vel-clip-rps", type=float, default=500.0, help="Clip for dof velocity max to avoid inf")
    args = ap.parse_args()

    files = _iter_pkl_files(args.dataset_dir)
    rows: List[Metrics] = []
    bad: List[Metrics] = []
    for p in files:
        m = compute_metrics(
            p,
            root_step_gt_m=args.root_step_gt_m,
            root_dang_gt_deg=args.root_dang_gt_deg,
            dof_step_gt_rad=args.dof_step_gt_rad,
            dof_vel_clip_rps=args.dof_vel_clip_rps,
        )
        if m is None:
            continue
        rows.append(m)
        if (m.n_root_step_gt + m.n_root_dang_gt + m.n_dof_step_gt) > 0:
            bad.append(m)

    out_dir = args.out / args.dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write full list sorted by composite score
    rows_sorted = sorted(rows, key=lambda r: r.score, reverse=True)
    _write_csv(out_dir / "all_sorted.csv", rows_sorted)

    # Write "bad" only
    bad_sorted = sorted(bad, key=lambda r: (r.n_root_step_gt + r.n_root_dang_gt + r.n_dof_step_gt, r.score), reverse=True)
    _write_csv(out_dir / "bad_sorted.csv", bad_sorted)

    # Print quick summary (top N by different criteria)
    def top_by(key, n):
        return sorted(rows, key=key, reverse=True)[:n]

    topn = min(int(args.limit), 2000)
    prints = {
        "top_root_step_max": top_by(lambda r: r.root_step_max_m, topn),
        "top_root_dang_max": top_by(lambda r: r.root_dang_max_rad, topn),
        "top_dof_step_max": top_by(lambda r: r.dof_step_max_rad, topn),
        "top_bad_by_counts": sorted(bad, key=lambda r: (r.n_root_step_gt + r.n_root_dang_gt + r.n_dof_step_gt), reverse=True)[:topn],
        "top_by_score": rows_sorted[:topn],
    }

    for name, arr in prints.items():
        _write_csv(out_dir / f"{name}.csv", arr)

    print(f"Scanned {len(files)} files, parsed {len(rows)}, bad {len(bad)}")
    print(f"Wrote CSVs to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

