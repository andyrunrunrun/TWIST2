#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _parse_g1_hinge_joints_from_mjcf(xml_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_names: List[str] = []
    joint_mins: List[float] = []
    joint_maxs: List[float] = []

    for joint in root.iter("joint"):
        # Skip implicit freejoint; in MJCF it is <freejoint>, not <joint>.
        jtype = joint.attrib.get("type", "hinge")
        if jtype != "hinge":
            continue
        name = joint.attrib.get("name")
        if not name:
            continue
        rng = joint.attrib.get("range")
        if not rng:
            # If not present, treat as unbounded.
            jmin, jmax = -math.pi, math.pi
        else:
            parts = rng.split()
            if len(parts) != 2:
                continue
            jmin, jmax = float(parts[0]), float(parts[1])

        joint_names.append(name)
        joint_mins.append(jmin)
        joint_maxs.append(jmax)

    if len(joint_names) != 29:
        raise ValueError(f"Expected 29 hinge joints from {xml_path}, got {len(joint_names)}")

    return joint_names, np.asarray(joint_mins, dtype=np.float64), np.asarray(joint_maxs, dtype=np.float64)


def _normalize_quat_xyzw(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected (...,4) quaternion, got {q.shape}")
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return q / n


def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True)
    out[..., :3] *= -1.0
    return out


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
    # Ensure shortest path: negate if w < 0 (same rotation).
    dq = np.where(dq[..., 3:4] < 0.0, -dq, dq)
    v_norm = np.linalg.norm(dq[..., :3], axis=-1)
    w = np.clip(dq[..., 3], -1.0, 1.0)
    return 2.0 * np.arctan2(v_norm, w)


@dataclass
class Tops:
    k: int
    root_step: List[Tuple[float, str, int]]
    root_ang: List[Tuple[float, str, int]]
    dof_step: List[Tuple[float, str, int, int]]  # value, path, frame_idx, dof_idx

    @staticmethod
    def create(k: int) -> "Tops":
        return Tops(k=k, root_step=[], root_ang=[], dof_step=[])

    def _push(self, arr: List, item: Tuple):
        arr.append(item)
        arr.sort(key=lambda x: x[0], reverse=True)
        if len(arr) > self.k:
            del arr[self.k :]

    def add_root_step(self, value: float, path: str, frame_idx: int) -> None:
        self._push(self.root_step, (float(value), path, int(frame_idx)))

    def add_root_ang(self, value: float, path: str, frame_idx: int) -> None:
        self._push(self.root_ang, (float(value), path, int(frame_idx)))

    def add_dof_step(self, value: float, path: str, frame_idx: int, dof_idx: int) -> None:
        self._push(self.dof_step, (float(value), path, int(frame_idx), int(dof_idx)))

    def merge(self, other: "Tops") -> None:
        for x in other.root_step:
            self._push(self.root_step, x)
        for x in other.root_ang:
            self._push(self.root_ang, x)
        for x in other.dof_step:
            self._push(self.dof_step, x)


@dataclass
class Acc:
    # counts
    files_ok: int
    files_bad: int
    frames: int
    vel_frames: int  # T-1 summed
    seconds: float

    # per-sequence (file) frame/duration stats
    seq_frames_min: int
    seq_frames_max: int
    seq_frames_sum: int
    seq_seconds_min: float
    seq_seconds_max: float
    seq_seconds_sum: float
    seq_frames_hist: np.ndarray
    seq_seconds_hist: np.ndarray
    seq_frames_list: List[int]
    seq_seconds_list: List[float]
    fps_counts: Dict[str, int]

    # fps
    fps_min: float
    fps_max: float
    fps_sum: float

    # root_pos (3)
    root_pos_min: np.ndarray
    root_pos_max: np.ndarray
    root_pos_sum: np.ndarray
    root_pos_sumsq: np.ndarray

    # dof_pos (29)
    dof_pos_min: np.ndarray
    dof_pos_max: np.ndarray
    dof_pos_sum: np.ndarray
    dof_pos_sumsq: np.ndarray

    # dof_vel (29)
    dof_vel_min: np.ndarray
    dof_vel_max: np.ndarray
    dof_vel_sum: np.ndarray
    dof_vel_sumsq: np.ndarray

    # root speed / ang speed (scalar)
    root_speed_min: float
    root_speed_max: float
    root_speed_sum: float
    root_speed_sumsq: float
    ang_speed_min: float
    ang_speed_max: float
    ang_speed_sum: float
    ang_speed_sumsq: float

    # hist
    dof_pos_hist: np.ndarray  # (29, nbins+2)
    dof_vel_hist: np.ndarray  # (29, nbins+2)
    root_speed_hist: np.ndarray  # (nbins,)
    ang_speed_hist: np.ndarray  # (nbins,)
    root_dang_hist: np.ndarray  # (nbins,)

    # sanity / missing
    nan_root_pos: int
    nan_root_rot: int
    nan_dof_pos: int

    # jump counters (fixed thresholds)
    jumps_root_step_gt_05m: int
    jumps_root_ang_gt_45deg: int
    jumps_dof_step_gt_1rad: int

    tops: Tops

    @staticmethod
    def create(
        nbins_dof: int,
        nbins_vel: int,
        nbins_root_speed: int,
        nbins_ang_speed: int,
        nbins_root_dang: int,
        seq_frames_hist_bins: int,
        seq_seconds_hist_bins: int,
        topk: int,
    ) -> "Acc":
        inf = float("inf")
        return Acc(
            files_ok=0,
            files_bad=0,
            frames=0,
            vel_frames=0,
            seconds=0.0,
            seq_frames_min=2**31 - 1,
            seq_frames_max=0,
            seq_frames_sum=0,
            seq_seconds_min=inf,
            seq_seconds_max=-inf,
            seq_seconds_sum=0.0,
            seq_frames_hist=np.zeros((seq_frames_hist_bins,), dtype=np.int64),
            seq_seconds_hist=np.zeros((seq_seconds_hist_bins,), dtype=np.int64),
            seq_frames_list=[],
            seq_seconds_list=[],
            fps_counts={},
            fps_min=inf,
            fps_max=-inf,
            fps_sum=0.0,
            root_pos_min=np.full((3,), inf, dtype=np.float64),
            root_pos_max=np.full((3,), -inf, dtype=np.float64),
            root_pos_sum=np.zeros((3,), dtype=np.float64),
            root_pos_sumsq=np.zeros((3,), dtype=np.float64),
            dof_pos_min=np.full((29,), inf, dtype=np.float64),
            dof_pos_max=np.full((29,), -inf, dtype=np.float64),
            dof_pos_sum=np.zeros((29,), dtype=np.float64),
            dof_pos_sumsq=np.zeros((29,), dtype=np.float64),
            dof_vel_min=np.full((29,), inf, dtype=np.float64),
            dof_vel_max=np.full((29,), -inf, dtype=np.float64),
            dof_vel_sum=np.zeros((29,), dtype=np.float64),
            dof_vel_sumsq=np.zeros((29,), dtype=np.float64),
            root_speed_min=inf,
            root_speed_max=-inf,
            root_speed_sum=0.0,
            root_speed_sumsq=0.0,
            ang_speed_min=inf,
            ang_speed_max=-inf,
            ang_speed_sum=0.0,
            ang_speed_sumsq=0.0,
            dof_pos_hist=np.zeros((29, nbins_dof + 2), dtype=np.int64),
            dof_vel_hist=np.zeros((29, nbins_vel + 2), dtype=np.int64),
            root_speed_hist=np.zeros((nbins_root_speed,), dtype=np.int64),
            ang_speed_hist=np.zeros((nbins_ang_speed,), dtype=np.int64),
            root_dang_hist=np.zeros((nbins_root_dang,), dtype=np.int64),
            nan_root_pos=0,
            nan_root_rot=0,
            nan_dof_pos=0,
            jumps_root_step_gt_05m=0,
            jumps_root_ang_gt_45deg=0,
            jumps_dof_step_gt_1rad=0,
            tops=Tops.create(topk),
        )

    def merge(self, other: "Acc") -> None:
        self.files_ok += other.files_ok
        self.files_bad += other.files_bad
        self.frames += other.frames
        self.vel_frames += other.vel_frames
        self.seconds += other.seconds

        self.seq_frames_min = min(self.seq_frames_min, other.seq_frames_min)
        self.seq_frames_max = max(self.seq_frames_max, other.seq_frames_max)
        self.seq_frames_sum += other.seq_frames_sum
        self.seq_seconds_min = min(self.seq_seconds_min, other.seq_seconds_min)
        self.seq_seconds_max = max(self.seq_seconds_max, other.seq_seconds_max)
        self.seq_seconds_sum += other.seq_seconds_sum
        self.seq_frames_hist += other.seq_frames_hist
        self.seq_seconds_hist += other.seq_seconds_hist
        self.seq_frames_list.extend(other.seq_frames_list)
        self.seq_seconds_list.extend(other.seq_seconds_list)
        for k, v in other.fps_counts.items():
            self.fps_counts[k] = self.fps_counts.get(k, 0) + int(v)

        self.fps_min = min(self.fps_min, other.fps_min)
        self.fps_max = max(self.fps_max, other.fps_max)
        self.fps_sum += other.fps_sum

        self.root_pos_min = np.minimum(self.root_pos_min, other.root_pos_min)
        self.root_pos_max = np.maximum(self.root_pos_max, other.root_pos_max)
        self.root_pos_sum += other.root_pos_sum
        self.root_pos_sumsq += other.root_pos_sumsq

        self.dof_pos_min = np.minimum(self.dof_pos_min, other.dof_pos_min)
        self.dof_pos_max = np.maximum(self.dof_pos_max, other.dof_pos_max)
        self.dof_pos_sum += other.dof_pos_sum
        self.dof_pos_sumsq += other.dof_pos_sumsq

        self.dof_vel_min = np.minimum(self.dof_vel_min, other.dof_vel_min)
        self.dof_vel_max = np.maximum(self.dof_vel_max, other.dof_vel_max)
        self.dof_vel_sum += other.dof_vel_sum
        self.dof_vel_sumsq += other.dof_vel_sumsq

        self.root_speed_min = min(self.root_speed_min, other.root_speed_min)
        self.root_speed_max = max(self.root_speed_max, other.root_speed_max)
        self.root_speed_sum += other.root_speed_sum
        self.root_speed_sumsq += other.root_speed_sumsq

        self.ang_speed_min = min(self.ang_speed_min, other.ang_speed_min)
        self.ang_speed_max = max(self.ang_speed_max, other.ang_speed_max)
        self.ang_speed_sum += other.ang_speed_sum
        self.ang_speed_sumsq += other.ang_speed_sumsq

        self.dof_pos_hist += other.dof_pos_hist
        self.dof_vel_hist += other.dof_vel_hist
        self.root_speed_hist += other.root_speed_hist
        self.ang_speed_hist += other.ang_speed_hist
        self.root_dang_hist += other.root_dang_hist

        self.nan_root_pos += other.nan_root_pos
        self.nan_root_rot += other.nan_root_rot
        self.nan_dof_pos += other.nan_dof_pos

        self.jumps_root_step_gt_05m += other.jumps_root_step_gt_05m
        self.jumps_root_ang_gt_45deg += other.jumps_root_ang_gt_45deg
        self.jumps_dof_step_gt_1rad += other.jumps_dof_step_gt_1rad

        self.tops.merge(other.tops)


def _hist_dof_pos(
    dof_pos: np.ndarray, jmin: np.ndarray, jmax: np.ndarray, nbins: int
) -> np.ndarray:
    # counts shape (29, nbins+2): [underflow, 1..nbins, overflow]
    counts = np.zeros((29, nbins + 2), dtype=np.int64)
    # Vectorized binning per joint
    rng = np.maximum(jmax - jmin, 1e-9)
    scaled = (dof_pos - jmin[None, :]) / rng[None, :]
    idx = np.floor(scaled * nbins).astype(np.int64) + 1
    idx = np.where(dof_pos < jmin[None, :], 0, idx)
    idx = np.where(dof_pos > jmax[None, :], nbins + 1, idx)
    idx = np.clip(idx, 0, nbins + 1)
    # accumulate counts
    for j in range(29):
        counts[j] = np.bincount(idx[:, j], minlength=nbins + 2)
    return counts


def _hist_dof_vel(dof_vel: np.ndarray, vmax: float, nbins: int) -> np.ndarray:
    counts = np.zeros((29, nbins + 2), dtype=np.int64)
    vmin = -float(vmax)
    vmax = float(vmax)
    rng = vmax - vmin
    scaled = (dof_vel - vmin) / rng
    idx = np.floor(scaled * nbins).astype(np.int64) + 1
    idx = np.where(dof_vel < vmin, 0, idx)
    idx = np.where(dof_vel > vmax, nbins + 1, idx)
    idx = np.clip(idx, 0, nbins + 1)
    for j in range(29):
        counts[j] = np.bincount(idx[:, j], minlength=nbins + 2)
    return counts


def _hist_1d(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(x, bins=edges)[0].astype(np.int64, copy=False)


def _process_chunk(
    paths: Sequence[str],
    jmin: np.ndarray,
    jmax: np.ndarray,
    nbins_dof: int,
    nbins_vel: int,
    dof_vel_vmax: float,
    root_speed_edges: np.ndarray,
    ang_speed_edges: np.ndarray,
    root_dang_edges: np.ndarray,
    seq_frames_edges: np.ndarray,
    seq_seconds_edges: np.ndarray,
    topk: int,
) -> Acc:
    acc = Acc.create(
        nbins_dof=nbins_dof,
        nbins_vel=nbins_vel,
        nbins_root_speed=len(root_speed_edges) - 1,
        nbins_ang_speed=len(ang_speed_edges) - 1,
        nbins_root_dang=len(root_dang_edges) - 1,
        seq_frames_hist_bins=len(seq_frames_edges) - 1,
        seq_seconds_hist_bins=len(seq_seconds_edges) - 1,
        topk=topk,
    )

    for p in paths:
        path = Path(p)
        try:
            d = _load_pickle(path)
            if not isinstance(d, dict):
                raise TypeError("not a dict")
            fps = float(d["fps"])
            root_pos = np.asarray(d["root_pos"])
            root_rot = np.asarray(d["root_rot"])
            dof_pos = np.asarray(d["dof_pos"])

            if root_pos.ndim != 2 or root_pos.shape[1] != 3:
                raise ValueError(f"bad root_pos shape {root_pos.shape}")
            if root_rot.ndim != 2 or root_rot.shape[1] != 4:
                raise ValueError(f"bad root_rot shape {root_rot.shape}")
            if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
                raise ValueError(f"bad dof_pos shape {dof_pos.shape}")

            T = int(min(root_pos.shape[0], root_rot.shape[0], dof_pos.shape[0]))
            if T < 2:
                raise ValueError("T < 2")
            root_pos = root_pos[:T]
            root_rot = root_rot[:T]
            dof_pos = dof_pos[:T]

            dt = 1.0 / fps

            acc.files_ok += 1
            acc.frames += T
            acc.vel_frames += T - 1
            seq_seconds = dt * (T - 1)
            acc.seconds += seq_seconds

            # per-sequence distributions
            acc.seq_frames_min = min(acc.seq_frames_min, T)
            acc.seq_frames_max = max(acc.seq_frames_max, T)
            acc.seq_frames_sum += T
            acc.seq_seconds_min = min(acc.seq_seconds_min, seq_seconds)
            acc.seq_seconds_max = max(acc.seq_seconds_max, seq_seconds)
            acc.seq_seconds_sum += seq_seconds

            fi = int(np.searchsorted(seq_frames_edges, T, side="right") - 1)
            fi = max(0, min(fi, acc.seq_frames_hist.shape[0] - 1))
            acc.seq_frames_hist[fi] += 1

            si = int(np.searchsorted(seq_seconds_edges, seq_seconds, side="right") - 1)
            si = max(0, min(si, acc.seq_seconds_hist.shape[0] - 1))
            acc.seq_seconds_hist[si] += 1

            acc.seq_frames_list.append(int(T))
            acc.seq_seconds_list.append(float(seq_seconds))

            k = f"{fps:.3f}"
            acc.fps_counts[k] = acc.fps_counts.get(k, 0) + 1

            acc.fps_min = min(acc.fps_min, fps)
            acc.fps_max = max(acc.fps_max, fps)
            acc.fps_sum += fps

            # NaNs
            acc.nan_root_pos += int(np.isnan(root_pos).sum())
            acc.nan_root_rot += int(np.isnan(root_rot).sum())
            acc.nan_dof_pos += int(np.isnan(dof_pos).sum())

            # Root pos stats
            rp = np.nan_to_num(root_pos, copy=False)
            acc.root_pos_min = np.minimum(acc.root_pos_min, np.min(rp, axis=0))
            acc.root_pos_max = np.maximum(acc.root_pos_max, np.max(rp, axis=0))
            acc.root_pos_sum += np.sum(rp, axis=0)
            acc.root_pos_sumsq += np.sum(rp * rp, axis=0)

            # dof stats
            dp = np.nan_to_num(dof_pos, copy=False)
            acc.dof_pos_min = np.minimum(acc.dof_pos_min, np.min(dp, axis=0))
            acc.dof_pos_max = np.maximum(acc.dof_pos_max, np.max(dp, axis=0))
            acc.dof_pos_sum += np.sum(dp, axis=0)
            acc.dof_pos_sumsq += np.sum(dp * dp, axis=0)
            acc.dof_pos_hist += _hist_dof_pos(dp, jmin, jmax, nbins_dof)

            # Deltas
            dpos = rp[1:] - rp[:-1]
            step = np.linalg.norm(dpos, axis=1)
            speed = step / dt

            q0 = np.nan_to_num(root_rot[:-1], copy=False)
            q1 = np.nan_to_num(root_rot[1:], copy=False)
            dang = _quat_delta_angle_rad(q0, q1)
            ang_speed = dang / dt

            d_dof = dp[1:] - dp[:-1]
            dof_vel = d_dof / dt

            # velocity stats (dof)
            acc.dof_vel_min = np.minimum(acc.dof_vel_min, np.min(dof_vel, axis=0))
            acc.dof_vel_max = np.maximum(acc.dof_vel_max, np.max(dof_vel, axis=0))
            acc.dof_vel_sum += np.sum(dof_vel, axis=0)
            acc.dof_vel_sumsq += np.sum(dof_vel * dof_vel, axis=0)
            acc.dof_vel_hist += _hist_dof_vel(dof_vel, dof_vel_vmax, nbins_vel)

            # root speed stats
            acc.root_speed_min = min(acc.root_speed_min, float(np.min(speed)))
            acc.root_speed_max = max(acc.root_speed_max, float(np.max(speed)))
            acc.root_speed_sum += float(np.sum(speed))
            acc.root_speed_sumsq += float(np.sum(speed * speed))
            acc.root_speed_hist += _hist_1d(speed, root_speed_edges)

            # angular speed stats
            acc.ang_speed_min = min(acc.ang_speed_min, float(np.min(ang_speed)))
            acc.ang_speed_max = max(acc.ang_speed_max, float(np.max(ang_speed)))
            acc.ang_speed_sum += float(np.sum(ang_speed))
            acc.ang_speed_sumsq += float(np.sum(ang_speed * ang_speed))
            acc.ang_speed_hist += _hist_1d(ang_speed, ang_speed_edges)
            acc.root_dang_hist += _hist_1d(dang, root_dang_edges)

            # fixed-threshold jump counters
            acc.jumps_root_step_gt_05m += int(np.sum(step > 0.5))
            acc.jumps_root_ang_gt_45deg += int(np.sum(dang > (math.pi / 4.0)))
            acc.jumps_dof_step_gt_1rad += int(np.sum(np.abs(d_dof) > 1.0))

            # top outliers (by per-file maxima)
            i_step = int(np.argmax(step))
            acc.tops.add_root_step(float(step[i_step]), str(path), i_step + 1)

            i_ang = int(np.argmax(dang))
            acc.tops.add_root_ang(float(dang[i_ang]), str(path), i_ang + 1)

            abs_d = np.abs(d_dof)
            flat = int(np.argmax(abs_d))
            fi, ji = divmod(flat, 29)
            acc.tops.add_dof_step(float(abs_d[fi, ji]), str(path), fi + 1, ji)
        except Exception:
            acc.files_bad += 1
            continue

    return acc


def _chunked(seq: Sequence[str], chunk_size: int) -> List[List[str]]:
    return [list(seq[i : i + chunk_size]) for i in range(0, len(seq), chunk_size)]


def _quantiles_from_hist(edges: np.ndarray, counts: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    # counts are per-bin over [edges[i], edges[i+1])
    total = int(np.sum(counts))
    if total <= 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}

    cdf = np.cumsum(counts, dtype=np.int64)
    out: Dict[str, float] = {}
    for q in qs:
        target = int(math.ceil(q * total))
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(max(idx, 0), len(counts) - 1)
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        out[f"p{int(q*100)}"] = (lo + hi) * 0.5
    return out


def _dof_edges(jmin: np.ndarray, jmax: np.ndarray, nbins: int) -> np.ndarray:
    # (29, nbins+1) edges within the limit; hist has +2 bins for under/over
    edges = np.zeros((29, nbins + 1), dtype=np.float64)
    for j in range(29):
        edges[j] = np.linspace(float(jmin[j]), float(jmax[j]), nbins + 1)
    return edges


def _dof_vel_edges(vmax: float, nbins: int) -> np.ndarray:
    return np.linspace(-float(vmax), float(vmax), nbins + 1, dtype=np.float64)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _finalize_report(
    name: str,
    paths: List[Path],
    acc: Acc,
    joint_names: List[str],
    jmin: np.ndarray,
    jmax: np.ndarray,
    nbins_dof: int,
    nbins_vel: int,
    dof_vel_vmax: float,
    root_speed_edges: np.ndarray,
    ang_speed_edges: np.ndarray,
    root_dang_edges: np.ndarray,
    seq_frames_edges: np.ndarray,
    seq_seconds_edges: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fps_mean = acc.fps_sum / max(acc.files_ok, 1)
    seq_frames_mean = acc.seq_frames_sum / max(acc.files_ok, 1)
    seq_seconds_mean = acc.seq_seconds_sum / max(acc.files_ok, 1)
    root_pos_mean = acc.root_pos_sum / max(acc.frames, 1)
    dof_pos_mean = acc.dof_pos_sum / max(acc.frames, 1)
    dof_vel_mean = acc.dof_vel_sum / max(acc.vel_frames, 1)

    root_pos_var = acc.root_pos_sumsq / max(acc.frames, 1) - root_pos_mean**2
    dof_pos_var = acc.dof_pos_sumsq / max(acc.frames, 1) - dof_pos_mean**2
    dof_vel_var = acc.dof_vel_sumsq / max(acc.vel_frames, 1) - dof_vel_mean**2

    root_speed_mean = acc.root_speed_sum / max(acc.vel_frames, 1)
    root_speed_var = acc.root_speed_sumsq / max(acc.vel_frames, 1) - root_speed_mean**2
    ang_speed_mean = acc.ang_speed_sum / max(acc.vel_frames, 1)
    ang_speed_var = acc.ang_speed_sumsq / max(acc.vel_frames, 1) - ang_speed_mean**2

    # Quantiles
    qs = (0.01, 0.05, 0.5, 0.95, 0.99)
    root_speed_q = _quantiles_from_hist(root_speed_edges, acc.root_speed_hist, qs)
    ang_speed_q = _quantiles_from_hist(ang_speed_edges, acc.ang_speed_hist, qs)
    root_dang_q = _quantiles_from_hist(root_dang_edges, acc.root_dang_hist, qs)
    # Exact sequence quantiles (list is only per-file, so stays small enough even for 100k files)
    if acc.seq_frames_list:
        qv = np.quantile(np.asarray(acc.seq_frames_list, dtype=np.float64), qs).tolist()
        seq_frames_q = {f"p{int(q*100)}": float(v) for q, v in zip(qs, qv)}
    else:
        seq_frames_q = _quantiles_from_hist(seq_frames_edges, acc.seq_frames_hist, qs)
    if acc.seq_seconds_list:
        qv = np.quantile(np.asarray(acc.seq_seconds_list, dtype=np.float64), qs).tolist()
        seq_seconds_q = {f"p{int(q*100)}": float(v) for q, v in zip(qs, qv)}
    else:
        seq_seconds_q = _quantiles_from_hist(seq_seconds_edges, acc.seq_seconds_hist, qs)

    # dof quantiles from per-joint hist (handle under/over bins by clipping into edge range)
    dof_edges = _dof_edges(jmin, jmax, nbins_dof)
    dof_qs: List[Dict[str, float]] = []
    for j in range(29):
        counts = acc.dof_pos_hist[j]
        counts_in = counts[1:-1]
        edges = dof_edges[j]
        dof_qs.append(_quantiles_from_hist(edges, counts_in, qs))

    vel_edges = _dof_vel_edges(dof_vel_vmax, nbins_vel)
    dof_vel_qs: List[Dict[str, float]] = []
    for j in range(29):
        counts = acc.dof_vel_hist[j]
        counts_in = counts[1:-1]
        dof_vel_qs.append(_quantiles_from_hist(vel_edges, counts_in, qs))

    # Write artifacts
    np.savez_compressed(
        out_dir / "hists.npz",
        dof_pos_hist=acc.dof_pos_hist,
        dof_vel_hist=acc.dof_vel_hist,
        root_speed_hist=acc.root_speed_hist,
        ang_speed_hist=acc.ang_speed_hist,
        root_dang_hist=acc.root_dang_hist,
        dof_pos_edges=dof_edges,
        dof_vel_edges=vel_edges,
        root_speed_edges=root_speed_edges,
        ang_speed_edges=ang_speed_edges,
        root_dang_edges=root_dang_edges,
        joint_names=np.asarray(joint_names, dtype=object),
    )

    # Per-joint CSVs
    rows_pos: List[List[Any]] = []
    rows_vel: List[List[Any]] = []
    for j, jn in enumerate(joint_names):
        rows_pos.append(
            [
                j,
                jn,
                float(acc.dof_pos_min[j]),
                float(acc.dof_pos_max[j]),
                float(dof_pos_mean[j]),
                float(math.sqrt(max(float(dof_pos_var[j]), 0.0))),
                float(jmin[j]),
                float(jmax[j]),
                dof_qs[j]["p1"],
                dof_qs[j]["p5"],
                dof_qs[j]["p50"],
                dof_qs[j]["p95"],
                dof_qs[j]["p99"],
                int(acc.dof_pos_hist[j, 0]),
                int(acc.dof_pos_hist[j, -1]),
            ]
        )
        rows_vel.append(
            [
                j,
                jn,
                float(acc.dof_vel_min[j]),
                float(acc.dof_vel_max[j]),
                float(dof_vel_mean[j]),
                float(math.sqrt(max(float(dof_vel_var[j]), 0.0))),
                dof_vel_qs[j]["p1"],
                dof_vel_qs[j]["p5"],
                dof_vel_qs[j]["p50"],
                dof_vel_qs[j]["p95"],
                dof_vel_qs[j]["p99"],
                int(acc.dof_vel_hist[j, 0]),
                int(acc.dof_vel_hist[j, -1]),
            ]
        )

    _write_csv(
        out_dir / "dof_pos_stats.csv",
        [
            "dof_idx",
            "joint_name",
            "min",
            "max",
            "mean",
            "std",
            "limit_min",
            "limit_max",
            "p1",
            "p5",
            "p50",
            "p95",
            "p99",
            "underflow_frames",
            "overflow_frames",
        ],
        rows_pos,
    )
    _write_csv(
        out_dir / "dof_vel_stats.csv",
        [
            "dof_idx",
            "joint_name",
            "min_rad_per_s",
            "max_rad_per_s",
            "mean_rad_per_s",
            "std_rad_per_s",
            "p1",
            "p5",
            "p50",
            "p95",
            "p99",
            "underflow_frames",
            "overflow_frames",
        ],
        rows_vel,
    )

    report: Dict[str, Any] = {
        "name": name,
        "paths_count": len(paths),
        "files_ok": acc.files_ok,
        "files_bad": acc.files_bad,
        "frames_total": acc.frames,
        "vel_frames_total": acc.vel_frames,
        "seconds_total": acc.seconds,
        "seq_frames": {
            "min": int(acc.seq_frames_min if acc.files_ok else 0),
            "max": int(acc.seq_frames_max if acc.files_ok else 0),
            "mean": float(seq_frames_mean),
            **seq_frames_q,
        },
        "seq_seconds": {
            "min": float(acc.seq_seconds_min if acc.files_ok else 0.0),
            "max": float(acc.seq_seconds_max if acc.files_ok else 0.0),
            "mean": float(seq_seconds_mean),
            **seq_seconds_q,
        },
        "fps_counts": dict(sorted(acc.fps_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "fps": {"min": acc.fps_min, "max": acc.fps_max, "mean": fps_mean},
        "root_pos": {
            "min_xyz": acc.root_pos_min.tolist(),
            "max_xyz": acc.root_pos_max.tolist(),
            "mean_xyz": root_pos_mean.tolist(),
            "std_xyz": np.sqrt(np.maximum(root_pos_var, 0.0)).tolist(),
        },
        "root_speed": {
            "min_m_per_s": acc.root_speed_min,
            "max_m_per_s": acc.root_speed_max,
            "mean_m_per_s": root_speed_mean,
            "std_m_per_s": float(math.sqrt(max(root_speed_var, 0.0))),
            **root_speed_q,
        },
        "root_delta_angle": {
            "p1_rad": root_dang_q["p1"],
            "p50_rad": root_dang_q["p50"],
            "p99_rad": root_dang_q["p99"],
        },
        "root_ang_speed": {
            "min_rad_per_s": acc.ang_speed_min,
            "max_rad_per_s": acc.ang_speed_max,
            "mean_rad_per_s": ang_speed_mean,
            "std_rad_per_s": float(math.sqrt(max(ang_speed_var, 0.0))),
            **ang_speed_q,
        },
        "jumps": {
            "root_step_gt_0p5m_frames": acc.jumps_root_step_gt_05m,
            "root_delta_angle_gt_45deg_frames": acc.jumps_root_ang_gt_45deg,
            "dof_step_gt_1rad_frames": acc.jumps_dof_step_gt_1rad,
        },
        "nan_counts": {
            "root_pos": acc.nan_root_pos,
            "root_rot": acc.nan_root_rot,
            "dof_pos": acc.nan_dof_pos,
        },
        "tops": {
            "root_step_m": acc.tops.root_step,
            "root_delta_angle_rad": acc.tops.root_ang,
            "dof_step_rad": acc.tops.dof_step,
        },
        "artifacts": {
            "hists_npz": str((out_dir / "hists.npz").resolve()),
            "dof_pos_stats_csv": str((out_dir / "dof_pos_stats.csv").resolve()),
            "dof_vel_stats_csv": str((out_dir / "dof_vel_stats.csv").resolve()),
        },
    }

    _write_json(out_dir / "report.json", report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("/home/weijin/source/Datasets/Humanoid_WBC_Dataset_GMR_30fps_GMR"),
        help="Dataset root (contains 3 dirs)",
    )
    ap.add_argument(
        "--mjcf",
        type=Path,
        default=Path("assets/g1/g1_mocap_29dof.xml"),
        help="MJCF used to name/limit 29 DoFs",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/dataset_stats_stageii"))
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 2, 2))
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--nbins-dof", type=int, default=200)
    ap.add_argument("--nbins-vel", type=int, default=200)
    ap.add_argument("--dof-vel-vmax", type=float, default=20.0, help="rad/s bin range for dof velocities")
    args = ap.parse_args()

    joint_names, jmin, jmax = _parse_g1_hinge_joints_from_mjcf(args.mjcf)

    base = args.base
    datasets = {
        "AMASS_g1_GMR8_PHC_missing_30fps": base / "AMASS_g1_GMR8_PHC_missing_30fps",
        "PHUMA_filtered": base / "PHUMA_filtered",
        "TWIST2_dataset": base / "TWIST2_dataset",
    }

    # Histogram edges
    # Root speed: [0, 1e-3..50] + inf
    root_speed_edges = np.concatenate(
        [np.array([0.0, 1e-6], dtype=np.float64), np.logspace(-3, math.log10(50.0), 120), np.array([np.inf])]
    )
    root_speed_edges = np.unique(root_speed_edges)

    # Angular speed: [0, 1e-3..200] + inf
    ang_speed_edges = np.concatenate(
        [np.array([0.0, 1e-6], dtype=np.float64), np.logspace(-3, math.log10(200.0), 120), np.array([np.inf])]
    )
    ang_speed_edges = np.unique(ang_speed_edges)

    # Delta angle per frame: [0..pi] + inf
    root_dang_edges = np.concatenate([np.linspace(0.0, math.pi, 181, dtype=np.float64), np.array([np.inf])])

    # Sequence frames / durations (coarse, for quantiles)
    seq_frames_edges = np.asarray(
        [0, 30, 60, 90, 120, 150, 180, 240, 300, 360, 450, 600, 900, 1200, 1800, 2400, 3600, 4800, 7200, 10000, 20000, np.inf],
        dtype=np.float64,
    )
    seq_seconds_edges = np.asarray(
        [0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, 1200, 3600, np.inf],
        dtype=np.float64,
    )

    out_root = args.out / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    overall = Acc.create(
        nbins_dof=args.nbins_dof,
        nbins_vel=args.nbins_vel,
        nbins_root_speed=len(root_speed_edges) - 1,
        nbins_ang_speed=len(ang_speed_edges) - 1,
        nbins_root_dang=len(root_dang_edges) - 1,
        seq_frames_hist_bins=len(seq_frames_edges) - 1,
        seq_seconds_hist_bins=len(seq_seconds_edges) - 1,
        topk=args.topk,
    )

    reports: Dict[str, Any] = {"base": str(base), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for ds_name, ds_path in datasets.items():
        if not ds_path.exists():
            reports[ds_name] = {"error": f"missing: {ds_path}"}
            continue

        t0 = time.time()
        files = _iter_pkl_files(ds_path)
        paths = [str(p) for p in files]
        chunks = _chunked(paths, args.chunk_size)

        acc = Acc.create(
            nbins_dof=args.nbins_dof,
            nbins_vel=args.nbins_vel,
            nbins_root_speed=len(root_speed_edges) - 1,
            nbins_ang_speed=len(ang_speed_edges) - 1,
            nbins_root_dang=len(root_dang_edges) - 1,
            seq_frames_hist_bins=len(seq_frames_edges) - 1,
            seq_seconds_hist_bins=len(seq_seconds_edges) - 1,
            topk=args.topk,
        )

        if args.workers <= 1:
            for ch in chunks:
                acc.merge(
                    _process_chunk(
                        ch,
                        jmin=jmin,
                        jmax=jmax,
                        nbins_dof=args.nbins_dof,
                        nbins_vel=args.nbins_vel,
                        dof_vel_vmax=args.dof_vel_vmax,
                        root_speed_edges=root_speed_edges,
                        ang_speed_edges=ang_speed_edges,
                        root_dang_edges=root_dang_edges,
                        seq_frames_edges=seq_frames_edges,
                        seq_seconds_edges=seq_seconds_edges,
                        topk=args.topk,
                    )
                )
        else:
            import multiprocessing as mp

            start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
            with mp.get_context(start_method).Pool(processes=args.workers) as pool:
                fn = partial(
                    _process_chunk,
                    jmin=jmin,
                    jmax=jmax,
                    nbins_dof=args.nbins_dof,
                    nbins_vel=args.nbins_vel,
                    dof_vel_vmax=args.dof_vel_vmax,
                    root_speed_edges=root_speed_edges,
                    ang_speed_edges=ang_speed_edges,
                    root_dang_edges=root_dang_edges,
                    seq_frames_edges=seq_frames_edges,
                    seq_seconds_edges=seq_seconds_edges,
                    topk=args.topk,
                )
                it = pool.imap_unordered(
                    fn,
                    chunks,
                    chunksize=1,
                )
                for part in it:
                    acc.merge(part)

        elapsed = time.time() - t0
        out_dir = out_root / ds_name
        rep = _finalize_report(
            name=ds_name,
            paths=files,
            acc=acc,
            joint_names=joint_names,
            jmin=jmin,
            jmax=jmax,
            nbins_dof=args.nbins_dof,
            nbins_vel=args.nbins_vel,
            dof_vel_vmax=args.dof_vel_vmax,
            root_speed_edges=root_speed_edges,
            ang_speed_edges=ang_speed_edges,
            root_dang_edges=root_dang_edges,
            out_dir=out_dir,
            seq_frames_edges=seq_frames_edges,
            seq_seconds_edges=seq_seconds_edges,
        )
        rep["elapsed_s"] = elapsed
        reports[ds_name] = rep

        overall.merge(acc)

    reports["overall"] = _finalize_report(
        name="overall",
        paths=[],
        acc=overall,
        joint_names=joint_names,
        jmin=jmin,
        jmax=jmax,
        nbins_dof=args.nbins_dof,
        nbins_vel=args.nbins_vel,
        dof_vel_vmax=args.dof_vel_vmax,
        root_speed_edges=root_speed_edges,
        ang_speed_edges=ang_speed_edges,
        root_dang_edges=root_dang_edges,
        out_dir=out_root / "overall",
        seq_frames_edges=seq_frames_edges,
        seq_seconds_edges=seq_seconds_edges,
    )
    _write_json(out_root / "index.json", reports)
    print(f"Wrote reports to: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
