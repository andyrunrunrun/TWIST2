#!/usr/bin/env python3
"""
MuJoCo single-process (policy + sim) evaluator for TWIST2 motion configs.

Reads a MotionLib-style YAML (root_path + motions[].file), runs an ONNX policy in MuJoCo driven by
per-frame mimic targets, and writes per-motion metrics to a CSV.

This is inspired by HY-Humanoid/evaluation/exec_vs_gmr but adapted to TWIST2's training motion YAMLs.

python tools/mujoco_exec_eval.py --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --out_csv ./outputs/twist2_exec_metrics.csv --policy_path assets/ckpts/twist2_1017_20k.onnx \
    --xml_path assets/g1/g1_sim2sim_29dof.xml --disable_termination --body_set joint_bodies29  --workers 16

legged_gym/logs/g1_stu_future/0116_student_ddp/model_42500.onnx

python tools/mujoco_exec_eval.py --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --out_csv ./outputs/twist2_exec_metrics_teacher_deltalocal.csv --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher_deltalocal/model_85000.pt \
    --xml_path assets/g1/g1_sim2sim_29dof.xml --disable_termination --body_set joint_bodies29  --workers 128
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import multiprocessing as mp

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

try:
    import mujoco
except Exception as e:  # pragma: no cover
    mujoco = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Make local editable packages importable without requiring `pip install -e`.
# - `pose/pose/...` is used for quaternion + slerp utilities.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_POSE_ROOT = _REPO_ROOT / "pose"
if _POSE_ROOT.exists() and str(_POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(_POSE_ROOT))
_RSL_RL_ROOT = _REPO_ROOT / "rsl_rl"
if _RSL_RL_ROOT.exists() and str(_RSL_RL_ROOT) not in sys.path:
    # Needed for unpickling `.pt` checkpoints that reference `rsl_rl.*`.
    sys.path.insert(0, str(_RSL_RL_ROOT))
_LEGGED_GYM_ROOT = _REPO_ROOT / "legged_gym"
if _LEGGED_GYM_ROOT.exists() and str(_LEGGED_GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGGED_GYM_ROOT))

from pose.utils.torch_utils import euler_from_quaternion as _euler_from_quat_xyzw
from pose.utils.torch_utils import quat_to_exp_map as _quat_to_exp_map_xyzw
from pose.utils.torch_utils import slerp as _slerp_xyzw
from pose.utils.isaacgym_torch_utils import quat_conjugate as _quat_conj_xyzw
from pose.utils.isaacgym_torch_utils import quat_mul as _quat_mul_xyzw
from pose.utils.isaacgym_torch_utils import quat_rotate_inverse as _quat_rotate_inverse_xyzw


DEFAULT_KEYPOINT_BODIES = (
    # torso / waist
    "waist_yaw_link",
    "torso_link",
    # arms
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_rubber_hand",
    "right_rubber_hand",
    # legs
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _finite_difference_np(x: np.ndarray, dt: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    T = int(x.shape[0])
    if T <= 1:
        return np.zeros_like(x, dtype=np.float32)
    if T == 2:
        v = (x[1:2] - x[0:1]) / float(dt)
        out = np.concatenate([v, v], axis=0)
        return out.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * float(dt))
    out[0] = (x[1] - x[0]) / float(dt)
    out[-1] = (x[-1] - x[-2]) / float(dt)
    return out.astype(np.float32, copy=False)


def _quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, float(eps))
    return (q / n).astype(np.float32, copy=False)


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out.astype(np.float32, copy=False)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1).astype(np.float32, copy=False)


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize_wxyz(q)
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., 1:].astype(np.float64)
    qw = q[..., 0:1].astype(np.float64)
    t = 2.0 * np.cross(qv, v, axis=-1)
    out = v + qw * t + np.cross(qv, t, axis=-1)
    return out.astype(np.float32, copy=False)


def _quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_apply_wxyz(_quat_conjugate_wxyz(q), v)


def _quat_from_euler_zyx_wxyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    roll = np.asarray(roll, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.stack([w, x, y, z], axis=-1)
    return _quat_normalize_wxyz(q)


def _euler_from_quat_wxyz(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = _quat_normalize_wxyz(q).astype(np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll.astype(np.float32, copy=False), pitch.astype(np.float32, copy=False), yaw.astype(np.float32, copy=False)


def quat_angle_error_deg_wxyz(q_exec: np.ndarray, q_tgt: np.ndarray) -> np.ndarray:
    q_exec = _quat_normalize_wxyz(q_exec).astype(np.float64)
    q_tgt = _quat_normalize_wxyz(q_tgt).astype(np.float64)
    q_err = _quat_mul_wxyz(q_exec, _quat_conjugate_wxyz(q_tgt))
    q_err = _quat_normalize_wxyz(q_err).astype(np.float64)
    w = np.clip(np.abs(q_err[..., 0]), 0.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return (ang * 180.0 / math.pi).astype(np.float32, copy=False)


def reconstruct_qpos_from_mimic_target(
    mimic_target: np.ndarray,
    *,
    dt: float,
    init_xy: tuple[float, float],
    init_yaw: float,
) -> np.ndarray:
    """
    Reconstructs a (T,36) MuJoCo qpos sequence (root_pos + root_quat_wxyz + dof_pos)
    from TWIST2's mimic target (T,35):
      [vx_local, vy_local, z, roll, pitch, yaw_rate, dof_pos(29)].
    """
    m = np.asarray(mimic_target, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 35:
        raise ValueError(f"mimic_target must be (T,35), got {m.shape}")
    if m.shape[0] < 2:
        raise ValueError("mimic_target must have at least 2 frames")

    vx_local = m[:, 0]
    vy_local = m[:, 1]
    z = m[:, 2]
    roll = m[:, 3]
    pitch = m[:, 4]
    yaw_rate = m[:, 5]
    dof = m[:, 6:]

    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be >0, got {dt}")

    yaw = np.empty((m.shape[0],), dtype=np.float64)
    yaw[0] = float(init_yaw)
    for i in range(m.shape[0] - 1):
        yaw[i + 1] = yaw[i] + float(yaw_rate[i]) * dt

    root_quat = _quat_from_euler_zyx_wxyz(roll=roll, pitch=pitch, yaw=yaw).astype(np.float64)

    v_world_z = np.gradient(z, dt, axis=0)

    ex_world = _quat_apply_wxyz(root_quat, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    ey_world = _quat_apply_wxyz(root_quat, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    ez_world = _quat_apply_wxyz(root_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    r_zx = ex_world[:, 2].astype(np.float64)
    r_zy = ey_world[:, 2].astype(np.float64)
    r_zz = ez_world[:, 2].astype(np.float64)

    denom = np.where(np.abs(r_zz) < 1e-6, 1.0, r_zz)
    vz_local = (v_world_z - r_zx * vx_local - r_zy * vy_local) / denom
    vz_local = np.where(np.abs(r_zz) < 1e-6, 0.0, vz_local)

    v_local = np.stack([vx_local, vy_local, vz_local], axis=1)
    v_world = _quat_apply_wxyz(root_quat, v_local).astype(np.float64)

    x0, y0 = float(init_xy[0]), float(init_xy[1])
    x = np.empty((m.shape[0],), dtype=np.float64)
    y = np.empty((m.shape[0],), dtype=np.float64)
    x[0] = x0
    y[0] = y0
    if m.shape[0] >= 2:
        x[1:] = x0 + np.cumsum(v_world[:-1, 0]) * dt
        y[1:] = y0 + np.cumsum(v_world[:-1, 1]) * dt

    root_pos = np.stack([x, y, z], axis=1)
    qpos = np.concatenate([root_pos, root_quat, dof], axis=1)
    if qpos.shape[1] != 36:
        raise RuntimeError(f"qpos reconstructed wrong shape: {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("qpos reconstructed contains NaN/Inf")
    return qpos.astype(np.float32, copy=False)


def _crop_indices_for_eval(
    T: int,
    *,
    publish_hz: float,
    idle_s: float,
    tail_s: float,
    transition_s: float,
    future_step: int,
    loop: bool,
) -> tuple[int, int]:
    hz = float(publish_hz)
    if hz <= 0:
        raise ValueError(f"publish_hz must be >0, got {hz}")
    if T <= 0:
        return 0, 0

    n_idle = int(round(float(idle_s) * hz))
    n_tail = int(round(float(tail_s) * hz))

    transition_s = float(transition_s)
    if transition_s <= 0:
        n_ramp_in_extra = 0
        n_ramp_out_extra = 0
    else:
        n = int(round(transition_s * hz)) + 1
        n = max(n, int(future_step) + 2, 2)
        n_ramp_in_extra = n - 1
        n_ramp_out_extra = 0 if bool(loop) else (n - 1)

    start = int(n_idle + n_ramp_in_extra)
    end = int(T - (n_tail + n_ramp_out_extra))
    start = max(0, min(start, T))
    end = max(start, min(end, T))
    return start, end


def _apply_start_end_transition(
    mimic: np.ndarray,
    *,
    idle: np.ndarray,
    publish_hz: float,
    future_step: int,
    loop: bool,
    transition_s: float,
) -> np.ndarray:
    mimic = np.asarray(mimic, dtype=np.float32)
    idle = np.asarray(idle, dtype=np.float32).reshape(1, -1)
    if mimic.ndim != 2 or mimic.shape[1] != 35:
        raise ValueError(f"mimic must be (T,35), got {mimic.shape}")
    if idle.shape[1] != 35:
        raise ValueError(f"idle must be len=35, got {idle.shape}")
    if float(transition_s) <= 0:
        return mimic

    hz = float(publish_hz)
    n = int(round(float(transition_s) * hz)) + 1
    n = max(n, int(future_step) + 2, 2)
    if int(mimic.shape[0]) < 2:
        return mimic

    # ramp in: idle -> first motion frame
    ramp_in = np.linspace(0.0, 1.0, num=n, dtype=np.float32)[:, None]
    start = mimic[0:1]
    ramp_in_seq = (1.0 - ramp_in) * idle + ramp_in * start

    out = mimic.copy()
    out[:n] = ramp_in_seq[: min(n, out.shape[0])]

    if not bool(loop):
        # ramp out: last motion frame -> idle
        ramp_out = np.linspace(0.0, 1.0, num=n, dtype=np.float32)[:, None]
        end = out[-1:]
        ramp_out_seq = (1.0 - ramp_out) * end + ramp_out * idle
        out[-n:] = ramp_out_seq[-min(n, out.shape[0]) :]
    return out


def _prepend_append_idle(
    mimic: np.ndarray,
    *,
    idle: np.ndarray,
    publish_hz: float,
    idle_s: float,
    tail_s: float,
) -> np.ndarray:
    mimic = np.asarray(mimic, dtype=np.float32)
    idle = np.asarray(idle, dtype=np.float32).reshape(1, -1)
    hz = float(publish_hz)
    n_idle = int(round(float(idle_s) * hz))
    n_tail = int(round(float(tail_s) * hz))
    pre = np.repeat(idle, max(0, n_idle), axis=0)
    post = np.repeat(idle, max(0, n_tail), axis=0)
    return np.concatenate([pre, mimic, post], axis=0).astype(np.float32, copy=False)


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw)
    return np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)


def _detect_quat_order_xyzw_or_wxyz(root_rot: np.ndarray, *, max_frames: int = 200) -> str:
    if torch is None:
        raise RuntimeError("torch is required for quat_order=auto detection")
    rr = np.asarray(root_rot, dtype=np.float64)
    if rr.ndim != 2 or rr.shape[1] != 4:
        raise ValueError(f"root_rot must be (T,4), got {rr.shape}")
    if rr.shape[0] < 2:
        return "xyzw"

    sample = rr[: min(int(max_frames), int(rr.shape[0]))]
    q_xyzw = torch.from_numpy(sample.astype(np.float32))
    q_xyzw = q_xyzw / torch.norm(q_xyzw, dim=-1, keepdim=True).clamp(min=1e-9)
    e1 = _euler_from_quat_xyzw(q_xyzw)
    score_xyzw = float(torch.mean(torch.abs(e1[:, 0])) + torch.mean(torch.abs(e1[:, 1])))

    q_wxyz = torch.from_numpy(_quat_xyzw_to_wxyz(sample).astype(np.float32))
    q_as_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    q_as_xyzw = q_as_xyzw / torch.norm(q_as_xyzw, dim=-1, keepdim=True).clamp(min=1e-9)
    e2 = _euler_from_quat_xyzw(q_as_xyzw)
    score_wxyz = float(torch.mean(torch.abs(e2[:, 0])) + torch.mean(torch.abs(e2[:, 1])))

    return "xyzw" if score_xyzw <= score_wxyz else "wxyz"


@dataclass(frozen=True)
class Motion:
    fps: float
    root_pos: np.ndarray  # (T,3)
    root_rot_xyzw: np.ndarray  # (T,4)
    dof_pos: np.ndarray  # (T,29)
    link_body_list: tuple[str, ...] | None = None
    local_body_pos: np.ndarray | None = None  # (T,B,3)


def load_motion_pkl_or_npz(path: str | Path, *, quat_order: str = "auto") -> Motion:
    if torch is None:
        raise RuntimeError("torch is required (pose utils dependency)")
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            obj = {
                "fps": float(z["fps"]),
                "root_pos": z["root_pos"],
                "root_rot": z["root_rot"],
                "dof_pos": z["dof_pos"],
            }
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")

    fps = float(obj["fps"])
    root_pos = np.asarray(obj["root_pos"], dtype=np.float32)
    root_rot = np.asarray(obj["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(obj["dof_pos"], dtype=np.float32)
    link_body_list: tuple[str, ...] | None = None
    local_body_pos: np.ndarray | None = None
    if "link_body_list" in obj and "local_body_pos" in obj:
        try:
            link_body_list = tuple(str(x) for x in obj["link_body_list"])
            local_body_pos = np.asarray(obj["local_body_pos"], dtype=np.float32)
        except Exception:
            link_body_list = None
            local_body_pos = None

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (T,3), got {root_pos.shape} in {path}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must be (T,4), got {root_rot.shape} in {path}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"dof_pos must be (T,29), got {dof_pos.shape} in {path}")
    if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError(f"length mismatch in {path}: root_pos={root_pos.shape} root_rot={root_rot.shape} dof_pos={dof_pos.shape}")
    if root_pos.shape[0] < 2:
        raise ValueError(f"T must be >=2, got T={root_pos.shape[0]} in {path}")

    quat_order = str(quat_order).strip().lower()
    if quat_order == "auto":
        quat_order = _detect_quat_order_xyzw_or_wxyz(root_rot)
    if quat_order not in {"xyzw", "wxyz"}:
        raise ValueError(f"quat_order must be 'auto'|'xyzw'|'wxyz', got {quat_order!r}")
    if quat_order == "wxyz":
        root_rot_xyzw = root_rot[:, [1, 2, 3, 0]]
    else:
        root_rot_xyzw = root_rot

    return Motion(
        fps=float(fps),
        root_pos=root_pos,
        root_rot_xyzw=root_rot_xyzw,
        dof_pos=dof_pos,
        link_body_list=link_body_list,
        local_body_pos=local_body_pos,
    )


def _resample_xyzw_sequence(
    x: torch.Tensor,
    q_xyzw: torch.Tensor,
    y: torch.Tensor,
    *,
    src_fps: float,
    publish_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resamples (x,q,y) from src_fps to publish_hz with linear interp for x,y and slerp for q.
    x: (T,3), q: (T,4 xyzw), y: (T,29)
    Returns tensors sampled at times t=k/publish_hz, k=0..K-1, where t < length.
    """
    if x.shape[0] != q_xyzw.shape[0] or x.shape[0] != y.shape[0]:
        raise ValueError(f"length mismatch: x={x.shape} q={q_xyzw.shape} y={y.shape}")
    if x.shape[0] < 2:
        raise ValueError("need at least 2 frames to resample")

    src_dt = 1.0 / float(src_fps)
    length = src_dt * float(int(x.shape[0]) - 1)
    pub_dt = 1.0 / float(publish_hz)
    if length <= 0:
        raise ValueError(f"invalid length={length}")

    # time grid: t in [0, length) with fixed step pub_dt
    n = int(math.floor(length / pub_dt))
    n = max(n, 2)
    times = torch.arange(n, dtype=torch.float32) * float(pub_dt)
    times = torch.clamp(times, 0.0, float(length) - 1e-6)

    phase = times / float(length)
    phase = torch.clamp(phase, 0.0, 1.0)
    t_float = phase * float(int(x.shape[0]) - 1)
    idx0 = torch.floor(t_float).to(torch.int64)
    idx1 = torch.clamp(idx0 + 1, max=int(x.shape[0]) - 1)
    blend = (t_float - idx0.to(torch.float32)).to(torch.float32)

    x0 = x[idx0]
    x1 = x[idx1]
    y0 = y[idx0]
    y1 = y[idx1]
    q0 = q_xyzw[idx0]
    q1 = q_xyzw[idx1]

    blend_x = blend.unsqueeze(-1)
    x_s = (1.0 - blend_x) * x0 + blend_x * x1
    y_s = (1.0 - blend_x) * y0 + blend_x * y1
    q_s = _slerp_xyzw(q0, q1, blend)
    q_s = q_s / torch.norm(q_s, dim=-1, keepdim=True).clamp(min=1e-9)
    return x_s, q_s, y_s


def build_mimic_target_from_motion(motion: Motion, *, publish_hz: float) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required to build mimic targets")
    root_pos = torch.from_numpy(motion.root_pos)
    root_rot = torch.from_numpy(motion.root_rot_xyzw)
    dof_pos = torch.from_numpy(motion.dof_pos)

    root_rot = root_rot / torch.norm(root_rot, dim=-1, keepdim=True).clamp(min=1e-9)

    root_pos_s, root_rot_s, dof_pos_s = _resample_xyzw_sequence(
        root_pos, root_rot, dof_pos, src_fps=float(motion.fps), publish_hz=float(publish_hz)
    )

    dt = 1.0 / float(publish_hz)
    # Root velocity (world) from resampled positions
    root_vel = torch.from_numpy(_finite_difference_np(root_pos_s.numpy(), dt))
    # Angular velocity (world) using SO(3) derivative on resampled quats
    q_prev = root_rot_s[:-2]
    q_next = root_rot_s[2:]
    q_rel = _quat_mul_xyzw(q_next, _quat_conj_xyzw(q_prev))
    omega_mid = _quat_to_exp_map_xyzw(q_rel) / (2.0 * float(dt))
    q_start_rel = _quat_mul_xyzw(root_rot_s[1], _quat_conj_xyzw(root_rot_s[0]))
    omega_start = _quat_to_exp_map_xyzw(q_start_rel) / float(dt)
    q_end_rel = _quat_mul_xyzw(root_rot_s[-1], _quat_conj_xyzw(root_rot_s[-2]))
    omega_end = _quat_to_exp_map_xyzw(q_end_rel) / float(dt)
    root_ang_vel = torch.cat([omega_start.unsqueeze(0), omega_mid, omega_end.unsqueeze(0)], dim=0)

    euler = _euler_from_quat_xyzw(root_rot_s)
    roll = euler[:, 0:1]
    pitch = euler[:, 1:2]

    root_vel_local = _quat_rotate_inverse_xyzw(root_rot_s, root_vel)
    root_ang_vel_local = _quat_rotate_inverse_xyzw(root_rot_s, root_ang_vel)

    mimic = torch.cat(
        [
            root_vel_local[:, 0:2],
            root_pos_s[:, 2:3],
            roll,
            pitch,
            root_ang_vel_local[:, 2:3],
            dof_pos_s,
        ],
        dim=-1,
    )
    if mimic.shape[1] != 35:
        raise RuntimeError(f"mimic has wrong dim: {mimic.shape}")
    return mimic.detach().cpu().numpy().astype(np.float32, copy=False)


def _parse_index_spec(spec: str, n: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(n))
    indices: list[int] = []
    parts = [p.strip() for p in spec.replace(" ", "").split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            if a_s == "" or b_s == "":
                raise ValueError(f"Invalid range token '{part}' in motion_ids='{spec}'")
            a = int(a_s)
            b = int(b_s)
            if a < 0:
                a = n + a
            if b < 0:
                b = n + b
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if i < 0 or i >= n:
                    raise IndexError(f"motion_ids index {i} out of range [0, {n-1}]")
                indices.append(i)
        else:
            i = int(part)
            if i < 0:
                i = n + i
            if i < 0 or i >= n:
                raise IndexError(f"motion_ids index {i} out of range [0, {n-1}]")
            indices.append(i)
    seen: set[int] = set()
    out: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


@dataclass(frozen=True)
class MotionEntry:
    idx: int
    file_abs: Path
    file_rel: str
    weight: float


def iter_motion_config_files(
    motion_yaml: str | Path,
    *,
    motion_ids: str = "",
    max_motions: int = 0,
    shuffle: bool = False,
    shuffle_seed: int = 0,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> Iterator[MotionEntry]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read motion YAML configs")
    motion_yaml = Path(motion_yaml).expanduser().resolve()
    with open(motion_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict yaml in {motion_yaml}, got {type(cfg)}")
    root_path = Path(str(cfg["root_path"])).expanduser()
    motion_list = list(cfg["motions"])
    if not motion_list:
        return

    if motion_ids.strip():
        indices = _parse_index_spec(motion_ids, len(motion_list))
        motion_list = [motion_list[i] for i in indices]
    elif bool(shuffle):
        rng = np.random.RandomState(int(shuffle_seed))
        order = rng.permutation(len(motion_list)).tolist()
        motion_list = [motion_list[i] for i in order]

    if int(max_motions) > 0:
        motion_list = motion_list[: int(max_motions)]

    shard_idx = int(shard_idx)
    num_shards = int(num_shards)
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >0, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"invalid shard_idx={shard_idx} for num_shards={num_shards}")

    for i, entry in enumerate(motion_list):
        rel = str(entry["file"])
        weight = float(entry.get("weight", 1.0))
        abs_path = (root_path / rel).expanduser().resolve()
        if abs_path.suffix == ".pkl":
            npz_path = abs_path.with_suffix(".npz")
            if npz_path.exists():
                abs_path = npz_path

        # deterministic sharding by relpath
        h = int.from_bytes(hashlib.md5(rel.encode("utf-8")).digest()[:8], "little", signed=False)
        if (h % num_shards) != shard_idx:
            continue
        yield MotionEntry(idx=int(i), file_abs=abs_path, file_rel=rel, weight=weight)


class OnnxPolicy:
    def __init__(self, policy_path: str | Path, *, device: str = "cpu") -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
        policy_path = Path(policy_path).expanduser().resolve()
        if not policy_path.exists():
            raise FileNotFoundError(policy_path)
        device = str(device)
        providers: list[str] = []
        available = ort.get_available_providers()
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(policy_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_index = 0
        self.expected_obs_dim: int | None = None
        try:
            shape = self.session.get_inputs()[0].shape
            if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
                self.expected_obs_dim = int(shape[1])
        except Exception:
            self.expected_obs_dim = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if self.expected_obs_dim is not None and int(obs.shape[1]) != int(self.expected_obs_dim):
            raise ValueError(f"ONNX policy expected obs_dim={self.expected_obs_dim}, got {obs.shape}")
        out = self.session.run(None, {self.input_name: obs})[self.output_index]
        return np.asarray(out, dtype=np.float32)


def _infer_actor_critic_mimic_init(sd: dict[str, Any]) -> dict[str, Any]:
    """Infer ActorCriticMimic constructor args from a saved state_dict."""
    if torch is None:
        raise RuntimeError("torch is required for .pt policies")
    if "std" not in sd:
        raise ValueError("Not an RSL-RL actor_critic checkpoint (missing 'std')")

    std = sd["std"]
    num_actions = int(std.numel()) if hasattr(std, "numel") else int(len(std))

    w_me = sd.get("actor.motion_encoder.encoder.0.weight", None)
    if w_me is None or getattr(w_me, "ndim", 0) != 2:
        raise ValueError("Unsupported .pt policy (missing actor.motion_encoder.encoder.0.weight)")
    num_single_motion_obs = int(w_me.shape[1])

    conv0 = sd.get("actor.motion_encoder.conv_layers.0.weight", None)
    if conv0 is None or getattr(conv0, "ndim", 0) != 3:
        tsteps = 1
    else:
        ks = int(conv0.shape[-1])
        if ks == 8:
            tsteps = 50
        elif ks == 6:
            tsteps = 20
        elif ks == 4:
            tsteps = 10
        else:
            tsteps = 1

    w_lat = sd.get("actor.motion_encoder.linear_output.weight", None)
    if w_lat is None or getattr(w_lat, "ndim", 0) != 2:
        raise ValueError("Unsupported .pt policy (missing actor.motion_encoder.linear_output.weight)")
    motion_latent_dim = int(w_lat.shape[0])

    num_motion_observations = int(num_single_motion_obs * tsteps)

    if "actor.actor_backbone.0.weight" not in sd or "critic.0.weight" not in sd:
        raise ValueError("Unsupported .pt policy (missing actor/critic backbone weights)")

    actor_in = int(sd["actor.actor_backbone.0.weight"].shape[1])
    num_observations = int(actor_in + num_motion_observations - num_single_motion_obs - motion_latent_dim)

    critic_in = int(sd["critic.0.weight"].shape[1])
    num_critic_observations = int(critic_in + num_motion_observations - num_single_motion_obs - motion_latent_dim)

    import re

    actor_linears: list[tuple[int, Any]] = []
    for k, v in sd.items():
        m = re.match(r"^actor\.actor_backbone\.(\d+)\.weight$", str(k))
        if not m:
            continue
        if getattr(v, "ndim", 0) != 2:
            continue
        actor_linears.append((int(m.group(1)), v))
    actor_linears.sort(key=lambda x: x[0])
    if not actor_linears:
        raise ValueError("Failed to infer actor backbone linears from checkpoint")
    if int(actor_linears[-1][1].shape[0]) != num_actions:
        raise ValueError("Actor output layer not found in checkpoint (unexpected shapes)")
    actor_hidden_dims = [int(v.shape[0]) for _, v in actor_linears[:-1]]
    if not actor_hidden_dims:
        raise ValueError("Failed to infer actor_hidden_dims from checkpoint")

    critic_linears: list[tuple[int, Any]] = []
    for k, v in sd.items():
        m = re.match(r"^critic\.(\d+)\.weight$", str(k))
        if not m:
            continue
        if getattr(v, "ndim", 0) != 2:
            continue
        critic_linears.append((int(m.group(1)), v))
    critic_linears.sort(key=lambda x: x[0])
    if not critic_linears:
        raise ValueError("Failed to infer critic backbone linears from checkpoint")
    if int(critic_linears[-1][1].shape[0]) != 1:
        raise ValueError("Critic output layer not found in checkpoint (unexpected shapes)")
    critic_hidden_dims = [int(v.shape[0]) for _, v in critic_linears[:-1]]
    if not critic_hidden_dims:
        raise ValueError("Failed to infer critic_hidden_dims from checkpoint")

    layer_norm = any(
        k.startswith("actor.actor_backbone.") and k.endswith(".weight") and getattr(v, "ndim", 0) == 1 for k, v in sd.items()
    )

    return {
        "num_observations": int(num_observations),
        "num_critic_observations": int(num_critic_observations),
        "num_motion_observations": int(num_motion_observations),
        "num_motion_steps": int(tsteps),
        "num_actions": int(num_actions),
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "motion_latent_dim": int(motion_latent_dim),
        "layer_norm": bool(layer_norm),
    }


class TorchPolicy:
    """Loads an RSL-RL `.pt` checkpoint (ActorCriticMimic) for inference."""

    def __init__(self, policy_path: str | Path, *, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("torch is required for .pt policy inference but is not installed.")
        policy_path = Path(policy_path).expanduser().resolve()
        if not policy_path.exists():
            raise FileNotFoundError(policy_path)

        self.device = str(device)
        self.torch_device = torch.device(self.device if (self.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

        # NOTE: local training checkpoints include Normalizer objects; we load with weights_only=False.
        ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Unsupported .pt policy format: expected dict with model_state_dict, got {type(ckpt)}")
        sd = ckpt["model_state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("Unsupported .pt policy: model_state_dict is not a dict")

        # Detect ActorCriticMimic by motion_encoder keys.
        if not any(str(k).startswith("actor.motion_encoder.") for k in sd.keys()):
            raise ValueError(
                "Unsupported .pt policy (not ActorCriticMimic). "
                "If this is a different policy, export to ONNX or extend TorchPolicy."
            )

        init = _infer_actor_critic_mimic_init(sd)
        self.expected_obs_dim = int(init["num_observations"])
        self.num_motion_steps = int(init["num_motion_steps"])
        self.num_single_motion_obs = int(init["num_motion_observations"] // init["num_motion_steps"])

        from rsl_rl.modules.actor_critic_mimic import ActorCriticMimic

        self.model = ActorCriticMimic(
            num_observations=init["num_observations"],
            num_critic_observations=init["num_critic_observations"],
            num_motion_observations=init["num_motion_observations"],
            num_motion_steps=init["num_motion_steps"],
            num_actions=init["num_actions"],
            actor_hidden_dims=init["actor_hidden_dims"],
            critic_hidden_dims=init["critic_hidden_dims"],
            motion_latent_dim=init["motion_latent_dim"],
            activation="elu",
            init_noise_std=1.0,
            fix_action_std=False,
            layer_norm=bool(init["layer_norm"]),
        )
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.torch_device)
        self.model.eval()

        self.normalizer = ckpt.get("normalizer", None)
        if self.normalizer is not None:
            try:
                self.normalizer.to(self.torch_device)
            except Exception:
                pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if int(obs.shape[1]) != int(self.expected_obs_dim):
            raise ValueError(f"Torch policy expected obs_dim={self.expected_obs_dim}, got {obs.shape}")
        x = torch.from_numpy(obs).to(self.torch_device)
        if self.normalizer is not None:
            x = self.normalizer.normalize(x)
        with torch.inference_mode():
            # Some repo variants have `act_inference(..., eval=True)` calling Actor with an unsupported kwarg.
            # `act_inference()` already returns the deterministic mean action.
            act = self.model.act_inference(x)
        return act.detach().to("cpu").numpy().astype(np.float32, copy=False)


def _default_pt_to_onnx_cache_dir() -> Path:
    mmdd = time.strftime("%m%d", time.localtime())
    return Path(f"/tmp/codex-{mmdd}-pt-to-onnx")


def export_actor_critic_mimic_ckpt_to_onnx(
    ckpt_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    opset: int = 11,
) -> Path:
    """
    Export an RSL-RL ActorCriticMimic `.pt/.pth` checkpoint into an ONNX policy (action mean).

    The exported graph includes the checkpoint Normalizer (if present), so ONNX inference matches
    TorchPolicy behavior.
    """
    if torch is None:
        raise RuntimeError("torch is required to export `.pt` checkpoints to ONNX")
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if ckpt_path.suffix.lower() not in (".pt", ".pth"):
        raise ValueError(f"Expected a .pt/.pth checkpoint, got {ckpt_path}")

    out_dir = _default_pt_to_onnx_cache_dir() if out_dir is None else Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    st = ckpt_path.stat()
    cache_key = f"{ckpt_path}|{st.st_mtime_ns}|{st.st_size}|opset={int(opset)}"
    h = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    out_path = out_dir / f"{ckpt_path.stem}-{h}.onnx"
    if out_path.exists():
        return out_path

    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    fd: int | None = None
    try:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # Another process is exporting; wait for completion.
            t0 = time.time()
            while time.time() - t0 < 120.0:
                if out_path.exists():
                    return out_path
                time.sleep(0.1)
            raise TimeoutError(f"Timed out waiting for ONNX export lock: {lock_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Unsupported .pt policy format: expected dict with model_state_dict, got {type(ckpt)}")
        sd = ckpt["model_state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("Unsupported .pt policy: model_state_dict is not a dict")

        init = _infer_actor_critic_mimic_init(sd)
        expected_obs_dim = int(init["num_observations"])

        from rsl_rl.modules.actor_critic_mimic import ActorCriticMimic

        model = ActorCriticMimic(
            num_observations=init["num_observations"],
            num_critic_observations=init["num_critic_observations"],
            num_motion_observations=init["num_motion_observations"],
            num_motion_steps=init["num_motion_steps"],
            num_actions=init["num_actions"],
            actor_hidden_dims=init["actor_hidden_dims"],
            critic_hidden_dims=init["critic_hidden_dims"],
            motion_latent_dim=init["motion_latent_dim"],
            activation="elu",
            init_noise_std=1.0,
            fix_action_std=False,
            layer_norm=bool(init["layer_norm"]),
        )
        model.load_state_dict(sd, strict=True)
        model.eval()

        normalizer = ckpt.get("normalizer", None)
        if normalizer is not None:
            try:
                normalizer.eval()
            except Exception:
                pass

        class _ExportWrapper(torch.nn.Module):
            def __init__(self, m: torch.nn.Module, n: Any | None) -> None:
                super().__init__()
                self.m = m
                self.n = n

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                if self.n is not None:
                    obs = self.n.normalize(obs)
                return self.m.actor(obs)

        wrapper = _ExportWrapper(model, normalizer)
        wrapper.eval()

        dummy = torch.zeros((1, expected_obs_dim), dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            dummy,
            str(tmp_path),
            export_params=True,
            opset_version=int(opset),
            do_constant_folding=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        )
        os.replace(str(tmp_path), str(out_path))
        return out_path
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def load_policy(policy_path: str | Path, *, device: str = "cpu") -> tuple[Any, str]:
    policy_path = Path(policy_path).expanduser().resolve()
    suf = policy_path.suffix.lower()
    if suf == ".onnx":
        return OnnxPolicy(policy_path, device=str(device)), "onnx"
    if suf in (".pt", ".pth"):
        return TorchPolicy(policy_path, device=str(device)), "torch"
    raise ValueError(f"Unsupported policy file extension: {policy_path} (expected .onnx or .pt)")


class _EmaSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._initialized = False
        self._value: np.ndarray | None = None

    def reset(self) -> None:
        self._initialized = False
        self._value = None

    def smooth(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if not self._initialized:
            self._initialized = True
            self._value = x.copy()
            return self._value
        assert self._value is not None
        self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value


@dataclass(frozen=True)
class Twist2SimConfig:
    xml_path: Path
    policy_path: Path
    device: str = "cpu"
    policy_frequency: float = 100.0
    sim_dt: float = 0.001
    smooth_body: float = 0.0
    obs_mode: str = "auto"  # auto|student_future|teacher_priv_mimic


class Twist2SimRunner:
    """
    Headless TWIST2 simulation runner (no Redis, no viewer).
    Reproduces key settings from deploy_real/server_low_level_g1_sim.py.
    """

    def __init__(self, cfg: Twist2SimConfig) -> None:
        if mujoco is None:
            raise ImportError("mujoco is required but not installed.")
        self.cfg = cfg
        self.policy, self.policy_backend = load_policy(cfg.policy_path, device=str(cfg.device))

        self.model = mujoco.MjModel.from_xml_path(str(Path(cfg.xml_path).expanduser().resolve()))
        self.model.opt.timestep = float(cfg.sim_dt)
        self.data = mujoco.MjData(self.model)

        self.num_actions = 29
        self.sim_dt = float(cfg.sim_dt)
        self.policy_frequency = float(cfg.policy_frequency)
        self.sim_decimation = int(round(1.0 / (self.policy_frequency * self.sim_dt)))
        if self.sim_decimation <= 0:
            raise ValueError(f"Invalid sim_decimation={self.sim_decimation} (policy_frequency={self.policy_frequency}, dt={self.sim_dt})")

        # --- G1 constants (match deploy_real/server_low_level_g1_sim.py) ---
        self.default_dof_pos = np.array(
            [
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.mujoco_default_qpos = np.concatenate(
            [
                np.array([0.0, 0.0, 0.793], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array(
                    [
                        -0.2,
                        0.0,
                        0.0,
                        0.4,
                        -0.2,
                        0.0,
                        -0.2,
                        0.0,
                        0.0,
                        0.4,
                        -0.2,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.2,
                        0.0,
                        1.2,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.2,
                        0.0,
                        1.2,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        self.stiffness = np.array(
            [
                100,
                100,
                100,
                150,
                40,
                40,
                100,
                100,
                100,
                150,
                40,
                40,
                150,
                150,
                150,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
            ],
            dtype=np.float32,
        )
        self.damping = np.array(
            [
                2,
                2,
                2,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                2,
                2,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                0.2,
                0.2,
                0.2,
                5,
                5,
                5,
                5,
                0.2,
                0.2,
                0.2,
            ],
            dtype=np.float32,
        )
        self.torque_limits = np.array(
            [
                100,
                100,
                100,
                150,
                40,
                40,
                100,
                100,
                100,
                150,
                40,
                40,
                150,
                150,
                150,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
            ],
            dtype=np.float32,
        )
        self.action_scale = np.array(
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            dtype=np.float32,
        )

        self.ankle_idx = [4, 5, 10, 11]

        self.n_mimic_obs = 35
        self.n_proprio = 3 + 2 + 3 * self.num_actions
        self.n_obs_single = self.n_mimic_obs + self.n_proprio
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs

        self._last_action = np.zeros((self.num_actions,), dtype=np.float32)
        self._history = np.zeros((self.history_len, self.n_obs_single), dtype=np.float32)
        self._body_smoother = _EmaSmoother(alpha=float(cfg.smooth_body)) if float(cfg.smooth_body) > 0.0 else None

        self._pelvis_body_id = None
        try:
            self._pelvis_body_id = int(self.model.body("pelvis").id)
        except Exception:
            self._pelvis_body_id = None

        # Observation mode selection.
        self.obs_mode = str(getattr(cfg, "obs_mode", "auto") or "auto")
        if self.obs_mode == "auto":
            if hasattr(self.policy, "expected_obs_dim") and getattr(self.policy, "expected_obs_dim") is not None:
                exp = int(getattr(self.policy, "expected_obs_dim"))
                if exp == int(self.total_obs_size):
                    self.obs_mode = "student_future"
                elif exp == 1734:
                    self.obs_mode = "teacher_priv_mimic"
                else:
                    raise ValueError(
                        f"Unsupported policy obs_dim={exp}. "
                        f"Supported: student_future({self.total_obs_size}) or teacher_priv_mimic(1734)."
                    )
            else:
                self.obs_mode = "student_future"

        # Teacher (priv mimic) constants (G1).
        self._teacher_key_bodies = (
            "left_rubber_hand",
            "right_rubber_hand",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_knee_link",
            "right_knee_link",
            "left_elbow_link",
            "right_elbow_link",
            "head_mocap",
        )
        self._teacher_key_bodies_resolved: tuple[str, ...] = self._teacher_key_bodies
        self._teacher_tar_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        self._teacher_key_body_ids_sim: np.ndarray | None = None
        self._teacher_feet_body_ids_sim: np.ndarray | None = None
        if self.obs_mode == "teacher_priv_mimic":
            if hasattr(self.policy, "expected_obs_dim") and getattr(self.policy, "expected_obs_dim") is not None:
                exp = int(getattr(self.policy, "expected_obs_dim"))
                if exp != 1734:
                    raise ValueError(f"teacher_priv_mimic requires policy obs_dim=1734, got {exp}")
            fallbacks = {"head_mocap": ("imu_in_torso", "torso_link")}
            resolved_names: list[str] = []
            sim_ids: list[int] = []
            for name in self._teacher_key_bodies:
                candidates = (name,) + tuple(fallbacks.get(name, ()))
                found = None
                for cand in candidates:
                    try:
                        bid = int(self.model.body(cand).id)
                        found = (cand, bid)
                        break
                    except Exception:
                        continue
                if found is None:
                    raise ValueError(f"MuJoCo XML missing required body for teacher obs: {name} (tried {candidates})")
                resolved_names.append(found[0])
                sim_ids.append(found[1])
            self._teacher_key_bodies_resolved = tuple(resolved_names)
            self._teacher_key_body_ids_sim = np.asarray(sim_ids, dtype=np.int32)
            feet = []
            for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
                try:
                    feet.append(int(self.model.body(name).id))
                except Exception as e:
                    raise ValueError(f"MuJoCo XML missing foot body for contact: {name}") from e
            self._teacher_feet_body_ids_sim = np.asarray(feet, dtype=np.int32)

    def reset(self, *, sim_seed: int = 0) -> None:
        if mujoco is None:
            raise RuntimeError("mujoco not available")
        mujoco.mj_resetData(self.model, self.data)
        if int(sim_seed) != 0:
            np.random.seed(int(sim_seed))
        self.data.qpos[:] = self.mujoco_default_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_action[:] = 0.0
        self._history[:] = 0.0
        if self._body_smoother is not None:
            self._body_smoother.reset()

    def _build_obs(self, action_mimic: np.ndarray) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        dof_pos = qpos[7 : 7 + self.num_actions].astype(np.float32, copy=False)
        dof_vel = qvel[6 : 6 + self.num_actions].astype(np.float32, copy=False)
        ang_vel = qvel[3:6].astype(np.float32, copy=False)
        quat_wxyz = qpos[3:7].astype(np.float32, copy=False)
        roll, pitch, _yaw = _euler_from_quat_wxyz(quat_wxyz.reshape(1, 4))
        rpy_rp = np.array([roll[0], pitch[0]], dtype=np.float32)

        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[self.ankle_idx] = 0.0
        obs_proprio = np.concatenate(
            [
                ang_vel * 0.25,
                rpy_rp,
                (dof_pos - self.default_dof_pos),
                obs_body_dof_vel * 0.05,
                self._last_action,
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if obs_proprio.shape[0] != self.n_proprio:
            raise RuntimeError(f"obs_proprio wrong shape: {obs_proprio.shape}")

        action_mimic = np.asarray(action_mimic, dtype=np.float32).reshape(-1)
        if action_mimic.shape[0] != self.n_mimic_obs:
            raise ValueError(f"action_mimic must be len=35, got {action_mimic.shape}")
        if self._body_smoother is not None:
            action_mimic = self._body_smoother.smooth(action_mimic)

        obs_full = np.concatenate([action_mimic, obs_proprio], axis=0).astype(np.float32, copy=False)
        if obs_full.shape[0] != self.n_obs_single:
            raise RuntimeError(f"obs_full wrong shape: {obs_full.shape}")

        obs_hist = self._history.reshape(-1).copy()
        # Roll history and append current
        self._history[:-1] = self._history[1:]
        self._history[-1] = obs_full

        # Current-frame future obs (matches deploy_real/server_low_level_g1_sim.py behavior)
        future_obs = action_mimic.copy()
        obs_buf = np.concatenate([obs_full, obs_hist, future_obs], axis=0)
        if obs_buf.shape[0] != self.total_obs_size:
            raise RuntimeError(f"obs_buf wrong size: {obs_buf.shape[0]} (expected {self.total_obs_size})")
        return obs_buf

    def _prepare_teacher_ref(
        self,
        motion: Motion,
        *,
        publish_hz: float,
        future_step: int,
        idle_s: float,
        tail_s: float,
        transition_s: float,
        loop: bool,
        idle_root_pos_z: float = 0.8,
    ) -> dict[str, np.ndarray]:
        if torch is None:
            raise RuntimeError("torch is required for teacher_priv_mimic observation building")
        if motion.local_body_pos is None or motion.link_body_list is None:
            raise ValueError("teacher_priv_mimic requires motion files with local_body_pos + link_body_list (use .pkl stageii)")

        root_pos = torch.from_numpy(motion.root_pos)
        root_rot = torch.from_numpy(motion.root_rot_xyzw)
        dof_pos = torch.from_numpy(motion.dof_pos)
        root_rot = root_rot / torch.norm(root_rot, dim=-1, keepdim=True).clamp(min=1e-9)

        root_pos_s, root_rot_s, dof_pos_s = _resample_xyzw_sequence(
            root_pos, root_rot, dof_pos, src_fps=float(motion.fps), publish_hz=float(publish_hz)
        )

        # Resample local_body_pos linearly using the same time grid.
        src_dt = 1.0 / float(motion.fps)
        length = src_dt * float(int(root_pos.shape[0]) - 1)
        pub_dt = 1.0 / float(publish_hz)
        n = int(math.floor(length / pub_dt))
        n = max(n, 2)
        times = torch.arange(n, dtype=torch.float32) * float(pub_dt)
        times = torch.clamp(times, 0.0, float(length) - 1e-6)
        phase = torch.clamp(times / float(length), 0.0, 1.0)
        t_float = phase * float(int(root_pos.shape[0]) - 1)
        idx0 = torch.floor(t_float).to(torch.int64)
        idx1 = torch.clamp(idx0 + 1, max=int(root_pos.shape[0]) - 1)
        blend = (t_float - idx0.to(torch.float32)).to(torch.float32).unsqueeze(-1).unsqueeze(-1)

        lb = torch.from_numpy(np.asarray(motion.local_body_pos, dtype=np.float32))  # (T,B,3)
        lb0 = lb[idx0]
        lb1 = lb[idx1]
        local_body_pos_s = ((1.0 - blend) * lb0 + blend * lb1).detach().cpu().numpy().astype(np.float32, copy=False)

        # Apply the same idle/transition logic as mimic_target generation so teacher ref aligns in time/length.
        hz = float(publish_hz)
        n_idle = int(round(float(idle_s) * hz))
        n_tail = int(round(float(tail_s) * hz))

        transition_s = float(transition_s)
        if transition_s <= 0:
            n_ramp = 0
        else:
            n_ramp = int(round(transition_s * hz)) + 1
            n_ramp = max(n_ramp, int(future_step) + 2, 2)

        # Base arrays (motion-only, already resampled to publish_hz).
        root_pos_np_m = root_pos_s.detach().cpu().numpy().astype(np.float32, copy=False)
        root_rot_xyzw_m = root_rot_s.detach().cpu().numpy().astype(np.float32, copy=False)
        dof_pos_np_m = dof_pos_s.detach().cpu().numpy().astype(np.float32, copy=False)

        # Idle frame (matches MotionEvaluator idle defaults).
        idle_pos = np.array([0.0, 0.0, float(idle_root_pos_z)], dtype=np.float32)
        idle_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        idle_dof = self.default_dof_pos.astype(np.float32, copy=False)
        idle_local_body = local_body_pos_s[0].astype(np.float32, copy=False)

        # Ramp in/out on the motion segment.
        if n_ramp > 0 and root_pos_np_m.shape[0] >= 2:
            ramp = np.linspace(0.0, 1.0, num=n_ramp, dtype=np.float32)[:, None]
            nn = min(n_ramp, int(root_pos_np_m.shape[0]))
            root_pos_np_m[:nn] = (1.0 - ramp[:nn]) * idle_pos[None, :] + ramp[:nn] * root_pos_np_m[:1]
            dof_pos_np_m[:nn] = (1.0 - ramp[:nn]) * idle_dof[None, :] + ramp[:nn] * dof_pos_np_m[:1]
            local_body_pos_s[:nn] = (1.0 - ramp[:nn, None]) * idle_local_body[None, :, :] + ramp[:nn, None] * local_body_pos_s[:1]

            # slerp quats in xyzw
            q0 = torch.from_numpy(np.repeat(idle_quat_xyzw[None, :], nn, axis=0))
            q1 = torch.from_numpy(root_rot_xyzw_m[:1].repeat(nn, axis=0))
            blend = torch.from_numpy(np.linspace(0.0, 1.0, num=nn, dtype=np.float32))
            q_s = _slerp_xyzw(q0, q1, blend)
            q_s = q_s / torch.norm(q_s, dim=-1, keepdim=True).clamp(min=1e-9)
            root_rot_xyzw_m[:nn] = q_s.numpy().astype(np.float32, copy=False)

            if not bool(loop):
                nn = min(n_ramp, int(root_pos_np_m.shape[0]))
                ramp2 = np.linspace(0.0, 1.0, num=nn, dtype=np.float32)[:, None]
                root_pos_np_m[-nn:] = (1.0 - ramp2) * root_pos_np_m[-1:] + ramp2 * idle_pos[None, :]
                dof_pos_np_m[-nn:] = (1.0 - ramp2) * dof_pos_np_m[-1:] + ramp2 * idle_dof[None, :]
                local_body_pos_s[-nn:] = (1.0 - ramp2[:, None]) * local_body_pos_s[-1:] + ramp2[:, None] * idle_local_body[None, :, :]

                q0 = torch.from_numpy(root_rot_xyzw_m[-1:].repeat(nn, axis=0))
                q1 = torch.from_numpy(np.repeat(idle_quat_xyzw[None, :], nn, axis=0))
                blend = torch.from_numpy(np.linspace(0.0, 1.0, num=nn, dtype=np.float32))
                q_s = _slerp_xyzw(q0, q1, blend)
                q_s = q_s / torch.norm(q_s, dim=-1, keepdim=True).clamp(min=1e-9)
                root_rot_xyzw_m[-nn:] = q_s.numpy().astype(np.float32, copy=False)

        # Prepend/append idle segments.
        pre_pos = np.repeat(idle_pos[None, :], max(0, n_idle), axis=0)
        pre_rot = np.repeat(idle_quat_xyzw[None, :], max(0, n_idle), axis=0)
        pre_dof = np.repeat(idle_dof[None, :], max(0, n_idle), axis=0)
        pre_lb = np.repeat(idle_local_body[None, :, :], max(0, n_idle), axis=0)
        post_pos = np.repeat(idle_pos[None, :], max(0, n_tail), axis=0)
        post_rot = np.repeat(idle_quat_xyzw[None, :], max(0, n_tail), axis=0)
        post_dof = np.repeat(idle_dof[None, :], max(0, n_tail), axis=0)
        post_lb = np.repeat(idle_local_body[None, :, :], max(0, n_tail), axis=0)

        root_pos_np = np.concatenate([pre_pos, root_pos_np_m, post_pos], axis=0).astype(np.float32, copy=False)
        root_rot_xyzw = np.concatenate([pre_rot, root_rot_xyzw_m, post_rot], axis=0).astype(np.float32, copy=False)
        dof_pos_np = np.concatenate([pre_dof, dof_pos_np_m, post_dof], axis=0).astype(np.float32, copy=False)
        local_body_pos_full = np.concatenate([pre_lb, local_body_pos_s, post_lb], axis=0).astype(np.float32, copy=False)

        root_rot_wxyz = _quat_xyzw_to_wxyz(root_rot_xyzw).astype(np.float32, copy=False)

        dt = 1.0 / float(publish_hz)
        root_vel_world = _finite_difference_np(root_pos_np, dt).astype(np.float32, copy=False)

        q_full = torch.from_numpy(root_rot_xyzw)
        q_full = q_full / torch.norm(q_full, dim=-1, keepdim=True).clamp(min=1e-9)
        q_prev = q_full[:-2]
        q_next = q_full[2:]
        q_rel = _quat_mul_xyzw(q_next, _quat_conj_xyzw(q_prev))
        omega_mid = _quat_to_exp_map_xyzw(q_rel) / (2.0 * float(dt))
        q_start_rel = _quat_mul_xyzw(q_full[1], _quat_conj_xyzw(q_full[0]))
        omega_start = _quat_to_exp_map_xyzw(q_start_rel) / float(dt)
        q_end_rel = _quat_mul_xyzw(q_full[-1], _quat_conj_xyzw(q_full[-2]))
        omega_end = _quat_to_exp_map_xyzw(q_end_rel) / float(dt)
        root_ang_vel_world = torch.cat([omega_start.unsqueeze(0), omega_mid, omega_end.unsqueeze(0)], dim=0).detach().cpu().numpy().astype(np.float32, copy=False)

        root_pos_delta_local = np.zeros_like(root_pos_np, dtype=np.float32)
        root_pos_delta_local[1:] = root_pos_np[1:] - root_pos_np[:-1]
        root_pos_delta_local[1:] = _quat_rotate_inverse_wxyz(root_rot_wxyz[:-1], root_pos_delta_local[1:]).astype(np.float32, copy=False)

        root_rot_delta_local = np.zeros_like(root_pos_np, dtype=np.float32)
        if root_rot_wxyz.shape[0] >= 2:
            q_prev_w = root_rot_wxyz[:-1]
            q_next_w = root_rot_wxyz[1:]
            q_rel_w = _quat_mul_wxyz(q_next_w, _quat_conjugate_wxyz(q_prev_w))
            dr, dp, dy = _euler_from_quat_wxyz(q_rel_w)
            de = np.stack([dr, dp, dy], axis=-1).astype(np.float32, copy=False)
            root_rot_delta_local[1:] = _quat_rotate_inverse_wxyz(q_prev_w, de).astype(np.float32, copy=False)

        link = list(motion.link_body_list)
        idxs = []
        for name in self._teacher_key_bodies_resolved:
            if name not in link:
                raise ValueError(f"Motion file missing key body '{name}' in link_body_list")
            idxs.append(int(link.index(name)))
        idxs_np = np.asarray(idxs, dtype=np.int32)
        key_body_pos_local = local_body_pos_full[:, idxs_np, :].reshape(local_body_pos_full.shape[0], -1).astype(np.float32, copy=False)

        return {
            "loop": np.array(bool(loop), dtype=np.bool_),
            "root_pos": root_pos_np,
            "root_rot_wxyz": root_rot_wxyz,
            "root_vel_world": root_vel_world,
            "root_ang_vel_world": root_ang_vel_world,
            "root_pos_delta_local": root_pos_delta_local,
            "root_rot_delta_local": root_rot_delta_local,
            "dof_pos": dof_pos_np,
            "key_body_pos_local": key_body_pos_local,
        }

    def _build_obs_teacher_priv_mimic(self, ref: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self._teacher_key_body_ids_sim is None or self._teacher_feet_body_ids_sim is None:
            raise RuntimeError("Teacher obs not initialized")
        qpos = self.data.qpos
        qvel = self.data.qvel
        root_pos = qpos[0:3].astype(np.float32, copy=False)
        quat_wxyz = qpos[3:7].astype(np.float32, copy=False)

        T = int(ref["root_pos"].shape[0])
        steps = self._teacher_tar_steps
        if hasattr(self.policy, "num_motion_steps") and len(steps) != int(getattr(self.policy, "num_motion_steps")):
            raise ValueError(f"teacher tar_steps length {len(steps)} != policy num_motion_steps {int(getattr(self.policy, 'num_motion_steps'))}")

        rows: list[np.ndarray] = []
        for s in steps:
            idx = int(t + s)
            if idx >= T:
                idx = (idx % T) if bool(ref["loop"].item()) else (T - 1)

            tar_root_pos = ref["root_pos"][idx]
            tar_quat = ref["root_rot_wxyz"][idx]
            tar_roll, tar_pitch, tar_yaw = _euler_from_quat_wxyz(tar_quat.reshape(1, 4))
            tar_rpy = np.array([float(tar_roll[0]), float(tar_pitch[0]), float(tar_yaw[0])], dtype=np.float32)

            tar_root_vel_local = _quat_rotate_inverse_wxyz(tar_quat.reshape(1, 4), ref["root_vel_world"][idx].reshape(1, 3)).reshape(-1).astype(np.float32, copy=False)
            tar_root_ang_vel_local = _quat_rotate_inverse_wxyz(tar_quat.reshape(1, 4), ref["root_ang_vel_world"][idx].reshape(1, 3)).reshape(-1).astype(np.float32, copy=False)

            row = np.concatenate(
                [
                    tar_root_pos.astype(np.float32, copy=False),
                    (tar_root_pos - root_pos).astype(np.float32, copy=False),
                    tar_rpy,
                    tar_root_vel_local,
                    tar_root_ang_vel_local,
                    ref["root_pos_delta_local"][idx].astype(np.float32, copy=False),
                    ref["root_rot_delta_local"][idx].astype(np.float32, copy=False),
                    ref["dof_pos"][idx].astype(np.float32, copy=False),
                    ref["key_body_pos_local"][idx].astype(np.float32, copy=False),
                ],
                axis=0,
            )
            rows.append(row)
        motion_obs = np.concatenate(rows, axis=0).astype(np.float32, copy=False)

        # proprio (matches g1_mimic_distill.py, but without reindex/noise)
        dof_pos = qpos[7 : 7 + self.num_actions].astype(np.float32, copy=False)
        dof_vel = qvel[6 : 6 + self.num_actions].astype(np.float32, copy=False)
        ang_vel = qvel[3:6].astype(np.float32, copy=False)
        roll, pitch, _yaw = _euler_from_quat_wxyz(quat_wxyz.reshape(1, 4))
        imu_rp = np.array([float(roll[0]), float(pitch[0])], dtype=np.float32)

        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[self.ankle_idx] = 0.0
        proprio = np.concatenate(
            [
                ang_vel * 0.25,
                imu_rp,
                (dof_pos - self.default_dof_pos),
                obs_dof_vel * 0.05,
                self._last_action,
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        # priv_info (matches g1_mimic_distill.py; domain-rand fields set to nominal)
        lin_vel_world = qvel[0:3].astype(np.float32, copy=False)
        base_lin_vel = _quat_rotate_inverse_wxyz(quat_wxyz.reshape(1, 4), lin_vel_world.reshape(1, 3)).reshape(-1).astype(np.float32, copy=False)

        body_pos = np.asarray(self.data.xpos[self._teacher_key_body_ids_sim], dtype=np.float32)  # (9,3)
        rel = body_pos - root_pos[None, :]
        rel_local = _quat_rotate_inverse_wxyz(quat_wxyz.reshape(1, 4), rel).reshape(-1).astype(np.float32, copy=False)

        cfrc = np.asarray(self.data.cfrc_ext[self._teacher_feet_body_ids_sim], dtype=np.float32)  # (2,6)
        foot_contact = (cfrc[:, 2] > 5.0).astype(np.float32, copy=False)

        mass_params = np.zeros((4,), dtype=np.float32)
        friction = np.ones((1,), dtype=np.float32)
        motor_strength = np.zeros((2 * self.num_actions,), dtype=np.float32)

        priv_info = np.concatenate(
            [
                base_lin_vel,
                root_pos.astype(np.float32, copy=False),
                quat_wxyz.astype(np.float32, copy=False),
                rel_local,
                foot_contact,
                mass_params,
                friction,
                motor_strength,
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        return np.concatenate([motion_obs, proprio, priv_info], axis=0).astype(np.float32, copy=False)

    def _policy_to_pd_target(self, obs_buf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.policy(obs_buf).reshape(-1)
        if raw.shape[0] != self.num_actions:
            raise RuntimeError(f"policy output wrong shape: {raw.shape}")
        self._last_action = raw.astype(np.float32, copy=True)
        raw = np.clip(raw, -10.0, 10.0).astype(np.float32, copy=False)
        pd_target = raw * self.action_scale + self.default_dof_pos
        return raw, pd_target.astype(np.float32, copy=False)

    def run(
        self,
        mimic_target: np.ndarray,
        *,
        motion: Motion | None = None,
        loop: bool = False,
        future_step: int = 1,
        idle_s: float = 0.5,
        tail_s: float = 0.5,
        transition_s: float = 0.4,
        sim_seed: int = 0,
        z_min: float = 0.55,
        angle_max_deg: float = 60.0,
        disable_termination: bool = True,
    ) -> dict[str, Any]:
        if mujoco is None:
            raise RuntimeError("mujoco not available")
        mimic_target = np.asarray(mimic_target, dtype=np.float32)
        if mimic_target.ndim != 2 or mimic_target.shape[1] != 35:
            raise ValueError(f"mimic_target must be (T,35), got {mimic_target.shape}")
        if mimic_target.shape[0] < 2:
            raise ValueError("mimic_target must have at least 2 frames")
        if not np.all(np.isfinite(mimic_target)):
            raise ValueError("mimic_target contains NaN/Inf")

        self.reset(sim_seed=int(sim_seed))

        T = int(mimic_target.shape[0])
        qpos = np.empty((T, int(self.model.nq)), dtype=np.float32)
        qvel = np.empty((T, int(self.model.nv)), dtype=np.float32)
        torque = np.empty((T, self.num_actions), dtype=np.float32)
        pelvis_z = np.empty((T,), dtype=np.float32)
        roll = np.empty((T,), dtype=np.float32)
        pitch = np.empty((T,), dtype=np.float32)

        terminated = False
        fail_detected = False
        fail_reason = ""
        fail_step = -1

        def _record(i: int) -> None:
            qpos[i] = self.data.qpos.astype(np.float32, copy=False)
            qvel[i] = self.data.qvel.astype(np.float32, copy=False)
            torque[i] = self.data.ctrl.astype(np.float32, copy=False)
            if self._pelvis_body_id is not None:
                pelvis_z[i] = float(self.data.xpos[self._pelvis_body_id][2])
            else:
                pelvis_z[i] = float(self.data.qpos[2])
            r, p, _y = _euler_from_quat_wxyz(self.data.qpos[3:7].reshape(1, 4))
            roll[i] = float(r[0])
            pitch[i] = float(p[0])

        _record(0)
        angle_max = float(angle_max_deg) * math.pi / 180.0
        z_min = float(z_min)

        teacher_ref: dict[str, np.ndarray] | None = None
        if self.obs_mode == "teacher_priv_mimic":
            if motion is None:
                raise ValueError("teacher_priv_mimic requires passing `motion` into Twist2SimRunner.run()")
            teacher_ref = self._prepare_teacher_ref(
                motion,
                publish_hz=float(self.policy_frequency),
                future_step=int(future_step),
                idle_s=float(idle_s),
                tail_s=float(tail_s),
                transition_s=float(transition_s),
                loop=bool(loop),
            )

        for t in range(T - 1):
            if self.obs_mode == "teacher_priv_mimic":
                assert teacher_ref is not None
                obs_buf = self._build_obs_teacher_priv_mimic(teacher_ref, t)
            else:
                obs_buf = self._build_obs(mimic_target[t])
            _raw, pd_target = self._policy_to_pd_target(obs_buf)

            for _ in range(self.sim_decimation):
                dof_pos = self.data.qpos[7 : 7 + self.num_actions]
                dof_vel = self.data.qvel[6 : 6 + self.num_actions]
                tau = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
                tau = np.clip(tau, -self.torque_limits, self.torque_limits)
                self.data.ctrl[:] = tau
                mujoco.mj_step(self.model, self.data)

            _record(t + 1)

            if not np.all(np.isfinite(qpos[t + 1])) or not np.all(np.isfinite(qvel[t + 1])) or not np.all(np.isfinite(torque[t + 1])):
                terminated = True
                fail_detected = True
                fail_reason = "nan_or_inf"
                fail_step = t + 1
                break

            # Detect failure conditions. If disable_termination=True, do not stop the rollout;
            # only record the first failure time/reason.
            reason = ""
            if pelvis_z[t + 1] < z_min:
                reason = "fell_pelvis_z"
            elif abs(roll[t + 1]) > angle_max or abs(pitch[t + 1]) > angle_max:
                reason = "fell_angle"

            if reason:
                if not fail_detected:
                    fail_detected = True
                    fail_reason = reason
                    fail_step = t + 1
                if not bool(disable_termination):
                    terminated = True
                    break

        if terminated and fail_step >= 0:
            qpos = qpos[: fail_step + 1]
            qvel = qvel[: fail_step + 1]
            torque = torque[: fail_step + 1]
            mimic_target = mimic_target[: fail_step + 1]
            pelvis_z = pelvis_z[: fail_step + 1]
            roll = roll[: fail_step + 1]
            pitch = pitch[: fail_step + 1]

        return {
            "qpos": qpos,
            "qvel": qvel,
            "torque": torque,
            "mimic_target": mimic_target,
            "terminated": bool(terminated),
            "fail_detected": bool(fail_detected),
            "fail_reason": str(fail_reason),
            "fail_step": int(fail_step),
            "pelvis_z": pelvis_z,
            "roll": roll,
            "pitch": pitch,
        }


@dataclass(frozen=True)
class BodySet:
    name: str
    body_names: tuple[str, ...]
    body_ids: np.ndarray


def _build_body_sets(model) -> dict[str, BodySet]:
    kp_names = tuple(DEFAULT_KEYPOINT_BODIES)
    kp_ids = np.array([int(model.body(n).id) for n in kp_names], dtype=np.int32)

    hinge_joint_idxs: list[int] = []
    for j in range(int(model.njnt)):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        hinge_joint_idxs.append(int(j))
    hinge_joint_idxs.sort(key=lambda j: int(model.jnt_qposadr[j]))

    joint_body_names: list[str] = []
    joint_body_ids: list[int] = []
    for j in hinge_joint_idxs:
        body_id = int(model.jnt_bodyid[j])
        # Prefer name lookup for debug, but it's fine if missing.
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        joint_body_names.append(str(body_name))
        joint_body_ids.append(body_id)

    jb_names = tuple(joint_body_names)
    jb_ids = np.asarray(joint_body_ids, dtype=np.int32)
    return {
        "keypoints14": BodySet(name="keypoints14", body_names=kp_names, body_ids=kp_ids),
        f"joint_bodies{len(jb_names)}": BodySet(name=f"joint_bodies{len(jb_names)}", body_names=jb_names, body_ids=jb_ids),
    }


class _Kinematics:
    def __init__(self, *, xml_path: Path, body_set: str) -> None:
        if mujoco is None:
            raise RuntimeError("mujoco not available")
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).expanduser().resolve()))
        self.data = mujoco.MjData(self.model)
        self.pelvis_body_id = int(self.model.body("pelvis").id)
        body_sets = _build_body_sets(self.model)
        if str(body_set) not in body_sets:
            raise ValueError(f"unknown body_set={body_set!r}, choices={sorted(body_sets.keys())}")
        self.body_set = body_sets[str(body_set)]

    def bodies_rel_pelvis(self, qpos_wxyz: np.ndarray) -> np.ndarray:
        qpos_wxyz = np.asarray(qpos_wxyz, dtype=np.float32).reshape(-1)
        if qpos_wxyz.shape[0] != int(self.model.nq):
            raise ValueError(f"qpos must have len={int(self.model.nq)}, got {qpos_wxyz.shape}")
        self.data.qpos[:] = qpos_wxyz
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        pelvis_pos = np.asarray(self.data.xpos[self.pelvis_body_id], dtype=np.float32)
        body_pos = np.asarray(self.data.xpos[self.body_set.body_ids], dtype=np.float32)
        delta = body_pos - pelvis_pos[None, :]
        pelvis_quat = qpos_wxyz[3:7]
        rel = _quat_rotate_inverse_wxyz(pelvis_quat.reshape(1, 4), delta)
        return np.asarray(rel, dtype=np.float32)


@dataclass(frozen=True)
class EvalResult:
    status: str  # ok|too_short|error
    motion_relpath: str
    motion_idx: int
    fps_src: float
    T_src: int
    policy_hz: float
    T_mimic_full: int
    T_exec: int
    terminated: bool
    fail_detected: bool
    fail_reason: str
    fail_step: int
    fail_time_s: float
    crop_start_full: int
    crop_end_full: int
    crop_start_used: int
    crop_end_used: int
    core_expected_len: int
    core_used_len: int
    core_coverage: float
    core_progress_to_fail: float
    root_pos_mean_l2_m: float
    root_pos_mean_l1_m: float
    root_rot_mean_deg: float
    joint_dof_mean_l1: float
    joint_vel_mean_l1: float
    fk_rel_mean_l2_m: float
    error: str

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "motion_relpath": self.motion_relpath,
            "motion_idx": int(self.motion_idx),
            "fps_src": float(self.fps_src),
            "T_src": int(self.T_src),
            "policy_hz": float(self.policy_hz),
            "T_mimic_full": int(self.T_mimic_full),
            "T_exec": int(self.T_exec),
            "terminated": bool(self.terminated),
            "fail_detected": bool(self.fail_detected),
            "fail_reason": str(self.fail_reason),
            "fail_step": int(self.fail_step),
            "fail_time_s": float(self.fail_time_s),
            "crop_start_full": int(self.crop_start_full),
            "crop_end_full": int(self.crop_end_full),
            "crop_start_used": int(self.crop_start_used),
            "crop_end_used": int(self.crop_end_used),
            "core_expected_len": int(self.core_expected_len),
            "core_used_len": int(self.core_used_len),
            "core_coverage": float(self.core_coverage),
            "core_progress_to_fail": float(self.core_progress_to_fail),
            "root_pos_mean_l2_m": float(self.root_pos_mean_l2_m),
            "root_pos_mean_l1_m": float(self.root_pos_mean_l1_m),
            "root_rot_mean_deg": float(self.root_rot_mean_deg),
            "joint_dof_mean_l1": float(self.joint_dof_mean_l1),
            "joint_vel_mean_l1": float(self.joint_vel_mean_l1),
            "fk_rel_mean_l2_m": float(self.fk_rel_mean_l2_m),
            "error": str(self.error),
        }


class MotionEvaluator:
    def __init__(self, *, sim: Twist2SimRunner, xml_path: Path, body_set: str) -> None:
        self.sim = sim
        self.kin = _Kinematics(xml_path=xml_path, body_set=str(body_set))

    def prepare_mimic_target_from_motion(
        self,
        motion: Motion,
        *,
        future_step: int,
        idle_s: float,
        tail_s: float,
        transition_s: float,
        loop: bool,
        idle_root_pos_z: float = 0.8,
    ) -> np.ndarray:
        publish_hz = float(self.sim.policy_frequency)
        mimic = build_mimic_target_from_motion(motion, publish_hz=publish_hz)

        # Idle mimic matches deploy_real/data_utils/params.py defaults (G1).
        idle = np.concatenate(
            [
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([float(idle_root_pos_z)], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                self.sim.default_dof_pos.astype(np.float32, copy=False),
            ],
            axis=0,
        )
        mimic = _apply_start_end_transition(
            mimic,
            idle=idle,
            publish_hz=publish_hz,
            future_step=int(future_step),
            loop=bool(loop),
            transition_s=float(transition_s),
        )
        mimic = _prepend_append_idle(
            mimic,
            idle=idle,
            publish_hz=publish_hz,
            idle_s=float(idle_s),
            tail_s=float(tail_s),
        )
        return np.asarray(mimic, dtype=np.float32)

    def run_and_eval(
        self,
        mimic_target: np.ndarray,
        *,
        motion: Motion,
        motion_relpath: str,
        motion_idx: int,
        fps_src: float,
        T_src: int,
        future_step: int,
        idle_s: float,
        tail_s: float,
        transition_s: float,
        loop: bool,
        sim_seed: int,
        z_min: float,
        angle_max_deg: float,
        disable_termination: bool,
        fk_stride: int,
    ) -> EvalResult:
        publish_hz = float(self.sim.policy_frequency)
        mimic_target = np.asarray(mimic_target, dtype=np.float32)
        T_full = int(mimic_target.shape[0])
        start_full, end_full = _crop_indices_for_eval(
            T_full,
            publish_hz=publish_hz,
            idle_s=float(idle_s),
            tail_s=float(tail_s),
            transition_s=float(transition_s),
            future_step=int(future_step),
            loop=bool(loop),
        )
        core_expected_len = int(end_full - start_full)

        try:
            sim_out = self.sim.run(
                mimic_target,
                motion=motion,
                loop=bool(loop),
                future_step=int(future_step),
                idle_s=float(idle_s),
                tail_s=float(tail_s),
                transition_s=float(transition_s),
                sim_seed=int(sim_seed),
                z_min=float(z_min),
                angle_max_deg=float(angle_max_deg),
                disable_termination=bool(disable_termination),
            )
            qpos_exec = np.asarray(sim_out["qpos"], dtype=np.float32)
            qvel_exec = np.asarray(sim_out["qvel"], dtype=np.float32)
            terminated = bool(sim_out.get("terminated", False))
            fail_detected = bool(sim_out.get("fail_detected", False))
            fail_reason = str(sim_out.get("fail_reason", ""))
            fail_step = int(sim_out.get("fail_step", -1))

            T_exec = int(qpos_exec.shape[0])
            mimic_used = mimic_target[:T_exec]

            # Align target reconstruction to the executed initial (x,y,yaw) so translation/yaw offsets don't dominate.
            _r0, _p0, yaw0 = _euler_from_quat_wxyz(qpos_exec[0, 3:7].reshape(1, 4))
            qpos_tgt = reconstruct_qpos_from_mimic_target(
                mimic_used,
                dt=1.0 / float(publish_hz),
                init_xy=(float(qpos_exec[0, 0]), float(qpos_exec[0, 1])),
                init_yaw=float(yaw0[0]),
            )

            start_used = int(min(start_full, T_exec))
            end_used = int(min(end_full, T_exec))
            core_used_len = int(max(0, end_used - start_used))
            core_coverage = float(core_used_len / core_expected_len) if core_expected_len > 0 else float("nan")

            if fail_step >= 0:
                fail_time_s = float(fail_step / publish_hz)
            else:
                fail_time_s = float("nan")

            if core_expected_len > 0:
                if fail_step < 0:
                    core_progress_to_fail = 1.0
                else:
                    clamped = int(min(max(fail_step, start_full), end_full))
                    core_progress_to_fail = float((clamped - start_full) / core_expected_len)
            else:
                core_progress_to_fail = float("nan")

            if core_used_len < 2:
                return EvalResult(
                    status="too_short",
                    motion_relpath=str(motion_relpath),
                    motion_idx=int(motion_idx),
                    fps_src=float(fps_src),
                    T_src=int(T_src),
                    policy_hz=float(publish_hz),
                    T_mimic_full=int(T_full),
                    T_exec=int(T_exec),
                    terminated=bool(terminated),
                    fail_detected=bool(fail_detected),
                    fail_reason=str(fail_reason),
                    fail_step=int(fail_step),
                    fail_time_s=float(fail_time_s),
                    crop_start_full=int(start_full),
                    crop_end_full=int(end_full),
                    crop_start_used=int(start_used),
                    crop_end_used=int(end_used),
                    core_expected_len=int(core_expected_len),
                    core_used_len=int(core_used_len),
                    core_coverage=float(core_coverage),
                    core_progress_to_fail=float(core_progress_to_fail),
                    root_pos_mean_l2_m=float("nan"),
                    root_pos_mean_l1_m=float("nan"),
                    root_rot_mean_deg=float("nan"),
                    joint_dof_mean_l1=float("nan"),
                    joint_vel_mean_l1=float("nan"),
                    fk_rel_mean_l2_m=float("nan"),
                    error="core segment too short",
                )

            qpos_exec_core = qpos_exec[start_used:end_used]
            qpos_tgt_core = qpos_tgt[start_used:end_used]
            qvel_exec_core = qvel_exec[start_used:end_used]

            dp = (qpos_exec_core[:, 0:3].astype(np.float64) - qpos_tgt_core[:, 0:3].astype(np.float64))
            dp_l2 = np.linalg.norm(dp, axis=1)
            root_pos_mean_l2_m = float(np.mean(dp_l2))
            root_pos_mean_l1_m = float(np.mean(np.mean(np.abs(dp), axis=1)))

            rot_err_deg = quat_angle_error_deg_wxyz(qpos_exec_core[:, 3:7], qpos_tgt_core[:, 3:7]).astype(np.float64)
            root_rot_mean_deg = float(np.mean(rot_err_deg))

            dof_exec = qpos_exec_core[:, 7 : 7 + 29].astype(np.float64)
            dof_tgt = qpos_tgt_core[:, 7 : 7 + 29].astype(np.float64)
            joint_dof_mean_l1 = float(np.mean(np.mean(np.abs(dof_exec - dof_tgt), axis=1)))

            dt = 1.0 / float(publish_hz)
            dof_tgt_vel = _finite_difference_np(dof_tgt, dt).astype(np.float64)
            dof_exec_vel = qvel_exec_core[:, 6 : 6 + 29].astype(np.float64)
            joint_vel_mean_l1 = float(np.mean(np.mean(np.abs(dof_exec_vel - dof_tgt_vel), axis=1)))

            fk_stride = int(max(1, fk_stride))
            rel_err_per_frame: list[float] = []
            for t in range(0, int(qpos_exec_core.shape[0]), fk_stride):
                rel_e = self.kin.bodies_rel_pelvis(qpos_exec_core[t])
                rel_t = self.kin.bodies_rel_pelvis(qpos_tgt_core[t])
                l2 = np.linalg.norm((rel_e - rel_t).astype(np.float64), axis=1)
                rel_err_per_frame.append(float(np.mean(l2)))
            fk_rel_mean_l2_m = float(np.mean(np.asarray(rel_err_per_frame, dtype=np.float64))) if rel_err_per_frame else float("nan")

            return EvalResult(
                status="ok",
                motion_relpath=str(motion_relpath),
                motion_idx=int(motion_idx),
                fps_src=float(fps_src),
                T_src=int(T_src),
                policy_hz=float(publish_hz),
                T_mimic_full=int(T_full),
                T_exec=int(T_exec),
                terminated=bool(terminated),
                fail_detected=bool(fail_detected),
                fail_reason=str(fail_reason),
                fail_step=int(fail_step),
                fail_time_s=float(fail_time_s),
                crop_start_full=int(start_full),
                crop_end_full=int(end_full),
                crop_start_used=int(start_used),
                crop_end_used=int(end_used),
                core_expected_len=int(core_expected_len),
                core_used_len=int(core_used_len),
                core_coverage=float(core_coverage),
                core_progress_to_fail=float(core_progress_to_fail),
                root_pos_mean_l2_m=float(root_pos_mean_l2_m),
                root_pos_mean_l1_m=float(root_pos_mean_l1_m),
                root_rot_mean_deg=float(root_rot_mean_deg),
                joint_dof_mean_l1=float(joint_dof_mean_l1),
                joint_vel_mean_l1=float(joint_vel_mean_l1),
                fk_rel_mean_l2_m=float(fk_rel_mean_l2_m),
                error="",
            )
        except Exception as e:
            return EvalResult(
                status="error",
                motion_relpath=str(motion_relpath),
                motion_idx=int(motion_idx),
                fps_src=float(fps_src),
                T_src=int(T_src),
                policy_hz=float(publish_hz),
                T_mimic_full=int(T_full),
                T_exec=0,
                terminated=False,
                fail_detected=False,
                fail_reason="",
                fail_step=-1,
                fail_time_s=float("nan"),
                crop_start_full=int(start_full),
                crop_end_full=int(end_full),
                crop_start_used=0,
                crop_end_used=0,
                core_expected_len=int(core_expected_len),
                core_used_len=0,
                core_coverage=0.0,
                core_progress_to_fail=float("nan"),
                root_pos_mean_l2_m=float("nan"),
                root_pos_mean_l1_m=float("nan"),
                root_rot_mean_deg=float("nan"),
                joint_dof_mean_l1=float("nan"),
                joint_vel_mean_l1=float("nan"),
                fk_rel_mean_l2_m=float("nan"),
                error=f"{type(e).__name__}: {e}",
            )


_WORKER_SIM: Twist2SimRunner | None = None
_WORKER_EVAL: MotionEvaluator | None = None
_WORKER_ARGS: dict[str, Any] | None = None


def _worker_init(cfg: dict[str, Any]) -> None:
    global _WORKER_SIM, _WORKER_EVAL, _WORKER_ARGS
    _WORKER_ARGS = cfg
    _WORKER_SIM = Twist2SimRunner(
        Twist2SimConfig(
            xml_path=Path(cfg["xml_path"]).expanduser().resolve(),
            policy_path=Path(cfg["policy_path"]).expanduser().resolve(),
            device=str(cfg["device"]),
            policy_frequency=float(cfg["policy_frequency"]),
            sim_dt=float(cfg["sim_dt"]),
            smooth_body=float(cfg["smooth_body"]),
            obs_mode=str(cfg.get("obs_mode", "auto")),
        )
    )
    _WORKER_EVAL = MotionEvaluator(sim=_WORKER_SIM, xml_path=Path(cfg["xml_path"]), body_set=str(cfg["body_set"]))


def _worker_eval_entry(entry: MotionEntry) -> dict[str, Any]:
    if _WORKER_EVAL is None or _WORKER_ARGS is None:
        raise RuntimeError("Worker not initialized")
    cfg = _WORKER_ARGS

    try:
        motion = load_motion_pkl_or_npz(Path(entry.file_abs), quat_order=str(cfg["quat_order"]))
        mimic = _WORKER_EVAL.prepare_mimic_target_from_motion(
            motion,
            future_step=int(cfg["future_step"]),
            idle_s=float(cfg["idle_s"]),
            tail_s=float(cfg["tail_s"]),
            transition_s=float(cfg["transition_s"]),
            loop=bool(cfg["loop"]),
        )
        res = _WORKER_EVAL.run_and_eval(
            mimic,
            motion=motion,
            motion_relpath=str(entry.file_rel),
            motion_idx=int(entry.idx),
            fps_src=float(motion.fps),
            T_src=int(motion.root_pos.shape[0]),
            future_step=int(cfg["future_step"]),
            idle_s=float(cfg["idle_s"]),
            tail_s=float(cfg["tail_s"]),
            transition_s=float(cfg["transition_s"]),
            loop=bool(cfg["loop"]),
            sim_seed=int(cfg["sim_seed"]),
            z_min=float(cfg["z_min"]),
            angle_max_deg=float(cfg["angle_max_deg"]),
            disable_termination=bool(cfg["disable_termination"]),
            fk_stride=int(cfg["fk_stride"]),
        )
    except Exception as e:
        res = EvalResult(
            status="error",
            motion_relpath=str(entry.file_rel),
            motion_idx=int(entry.idx),
            fps_src=float("nan"),
            T_src=-1,
            policy_hz=float(cfg["policy_frequency"]),
            T_mimic_full=0,
            T_exec=0,
            terminated=False,
            fail_detected=False,
            fail_reason="",
            fail_step=-1,
            fail_time_s=float("nan"),
            crop_start_full=0,
            crop_end_full=0,
            crop_start_used=0,
            crop_end_used=0,
            core_expected_len=0,
            core_used_len=0,
            core_coverage=0.0,
            core_progress_to_fail=float("nan"),
            root_pos_mean_l2_m=float("nan"),
            root_pos_mean_l1_m=float("nan"),
            root_rot_mean_deg=float("nan"),
            joint_dof_mean_l1=float("nan"),
            joint_vel_mean_l1=float("nan"),
            fk_rel_mean_l2_m=float("nan"),
            error=f"{type(e).__name__}: {e}",
        )

    return res.to_flat_dict()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a TWIST2 policy in MuJoCo for all motions in a training motion YAML; output per-motion CSV metrics. "
        "Supports .onnx directly; for .pt/.pth checkpoints the script exports a cached ONNX first."
    )
    ap.add_argument("--motion_yaml", type=str, required=True, help="Motion config YAML (root_path + motions[].file)")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")

    ap.add_argument("--policy_path", type=str, default="assets/ckpts/twist2_1017_20k.onnx")
    ap.add_argument(
        "--onnx_cache_dir",
        type=str,
        default="",
        help="Where to cache exported ONNX when --policy_path is a .pt/.pth (default: /tmp/codex-<mmdd>-pt-to-onnx/)",
    )
    ap.add_argument("--xml_path", type=str, default="assets/g1/g1_sim2sim_29dof.xml")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | cuda:<id>")
    ap.add_argument("--policy_frequency", type=float, default=100.0, choices=[50.0, 100.0])
    ap.add_argument("--smooth_body", type=float, default=0.0)
    ap.add_argument(
        "--obs_mode",
        type=str,
        default="auto",
        choices=["auto", "student_future", "teacher_priv_mimic"],
        help="Observation builder for the policy; auto selects based on policy type/obs_dim.",
    )
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes for CPU evaluation")
    ap.add_argument(
        "--mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="multiprocessing start method; 'spawn' is safest with MuJoCo/onnxruntime",
    )

    ap.add_argument("--quat_order", type=str, default="auto", choices=["auto", "xyzw", "wxyz"], help="Quaternion order in motion files (root_rot)")
    ap.add_argument("--motion_ids", type=str, default="", help="Subset of YAML motions by indices (e.g. '0,3,10-20')")
    ap.add_argument("--max_motions", type=int, default=0, help="Process at most N motions (0 = no limit)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle motion order before applying max_motions (ignored if motion_ids given)")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)

    ap.add_argument("--future_step", type=int, default=1)
    ap.add_argument("--idle_s", type=float, default=0.5)
    ap.add_argument("--tail_s", type=float, default=0.5)
    ap.add_argument("--transition_s", type=float, default=0.4)
    ap.add_argument("--loop", action="store_true")

    ap.add_argument("--sim_seed", type=int, default=0)
    ap.add_argument("--z_min", type=float, default=0.55)
    ap.add_argument("--angle_max_deg", type=float, default=60.0)
    ap.add_argument(
        "--disable_termination",
        action="store_true",
        help="Disable early termination on pelvis_z/angle (still records first failure time); always stops on NaN/Inf.",
    )

    ap.add_argument("--body_set", type=str, default="joint_bodies29", choices=["keypoints14", "joint_bodies29"])
    ap.add_argument("--fk_stride", type=int, default=1)
    args = ap.parse_args()

    if torch is None:
        raise RuntimeError("torch is required (pose utils dependency)")
    if mujoco is None:
        raise RuntimeError("mujoco is required")

    policy_path = Path(args.policy_path).expanduser().resolve()
    if policy_path.suffix.lower() in (".pt", ".pth"):
        cache_dir = _default_pt_to_onnx_cache_dir() if not str(args.onnx_cache_dir).strip() else Path(args.onnx_cache_dir).expanduser().resolve()
        exported = export_actor_critic_mimic_ckpt_to_onnx(policy_path, out_dir=cache_dir)
        print(f"[info] exported ckpt -> onnx: {policy_path} -> {exported}", file=sys.stderr)
        args.policy_path = str(exported)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if bool(args.append) else "w"

    fieldnames = [
        "motion_idx",
        "motion_relpath",
        "status",
        "fps_src",
        "T_src",
        "policy_hz",
        "T_mimic_full",
        "T_exec",
        "terminated",
        "fail_detected",
        "fail_reason",
        "fail_step",
        "fail_time_s",
        "core_expected_len",
        "core_used_len",
        "core_coverage",
        "core_progress_to_fail",
        "root_pos_mean_l2_m",
        "root_pos_mean_l1_m",
        "root_rot_mean_deg",
        "joint_dof_mean_l1",
        "joint_vel_mean_l1",
        "fk_rel_mean_l2_m",
        "error",
    ]

    write_header = (mode == "w") or (not out_csv.exists()) or (out_csv.stat().st_size == 0)
    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        n_total = 0
        n_ok = 0
        n_err = 0

        entries = list(
            iter_motion_config_files(
                args.motion_yaml,
                motion_ids=str(args.motion_ids),
                max_motions=int(args.max_motions),
                shuffle=bool(args.shuffle),
                shuffle_seed=int(args.shuffle_seed),
                shard_idx=int(args.shard_idx),
                num_shards=int(args.num_shards),
            )
        )

        workers = int(args.workers)
        if workers < 1:
            raise ValueError(f"--workers must be >= 1, got {workers}")
        if str(args.device).startswith("cuda") and workers > 1:
            print("[warn] --device is cuda*; forcing --workers=1 to avoid multi-process GPU contention.", file=sys.stderr)
            workers = 1

        worker_cfg: dict[str, Any] = {
            "xml_path": str(Path(args.xml_path).expanduser().resolve()),
            "policy_path": str(Path(args.policy_path).expanduser().resolve()),
            "device": str(args.device),
            "policy_frequency": float(args.policy_frequency),
            "sim_dt": 0.001,
            "smooth_body": float(args.smooth_body),
            "obs_mode": str(args.obs_mode),
            "body_set": str(args.body_set),
            "quat_order": str(args.quat_order),
            "future_step": int(args.future_step),
            "idle_s": float(args.idle_s),
            "tail_s": float(args.tail_s),
            "transition_s": float(args.transition_s),
            "loop": bool(args.loop),
            "sim_seed": int(args.sim_seed),
            "z_min": float(args.z_min),
            "angle_max_deg": float(args.angle_max_deg),
            "disable_termination": bool(args.disable_termination),
            "fk_stride": int(args.fk_stride),
        }

        if workers == 1:
            sim = Twist2SimRunner(
                Twist2SimConfig(
                    xml_path=Path(args.xml_path).expanduser().resolve(),
                    policy_path=Path(args.policy_path).expanduser().resolve(),
                    device=str(args.device),
                    policy_frequency=float(args.policy_frequency),
                    sim_dt=0.001,
                    smooth_body=float(args.smooth_body),
                    obs_mode=str(args.obs_mode),
                )
            )
            evaluator = MotionEvaluator(sim=sim, xml_path=Path(args.xml_path), body_set=str(args.body_set))
            for entry in entries:
                n_total += 1
                try:
                    motion = load_motion_pkl_or_npz(entry.file_abs, quat_order=str(args.quat_order))
                    mimic = evaluator.prepare_mimic_target_from_motion(
                        motion,
                        future_step=int(args.future_step),
                        idle_s=float(args.idle_s),
                        tail_s=float(args.tail_s),
                        transition_s=float(args.transition_s),
                        loop=bool(args.loop),
                    )
                    res = evaluator.run_and_eval(
                        mimic,
                        motion=motion,
                        motion_relpath=str(entry.file_rel),
                        motion_idx=int(entry.idx),
                        fps_src=float(motion.fps),
                        T_src=int(motion.root_pos.shape[0]),
                        future_step=int(args.future_step),
                        idle_s=float(args.idle_s),
                        tail_s=float(args.tail_s),
                        transition_s=float(args.transition_s),
                        loop=bool(args.loop),
                        sim_seed=int(args.sim_seed),
                        z_min=float(args.z_min),
                        angle_max_deg=float(args.angle_max_deg),
                        disable_termination=bool(args.disable_termination),
                        fk_stride=int(args.fk_stride),
                    )
                except Exception as e:
                    res = EvalResult(
                        status="error",
                        motion_relpath=str(entry.file_rel),
                        motion_idx=int(entry.idx),
                        fps_src=float("nan"),
                        T_src=-1,
                        policy_hz=float(args.policy_frequency),
                        T_mimic_full=0,
                        T_exec=0,
                        terminated=False,
                        fail_detected=False,
                        fail_reason="",
                        fail_step=-1,
                        fail_time_s=float("nan"),
                        crop_start_full=0,
                        crop_end_full=0,
                        crop_start_used=0,
                        crop_end_used=0,
                        core_expected_len=0,
                        core_used_len=0,
                        core_coverage=0.0,
                        core_progress_to_fail=float("nan"),
                        root_pos_mean_l2_m=float("nan"),
                        root_pos_mean_l1_m=float("nan"),
                        root_rot_mean_deg=float("nan"),
                        joint_dof_mean_l1=float("nan"),
                        joint_vel_mean_l1=float("nan"),
                        fk_rel_mean_l2_m=float("nan"),
                        error=f"{type(e).__name__}: {e}",
                    )
                row = {k: res.to_flat_dict().get(k) for k in fieldnames}
                w.writerow(row)

                if res.status == "ok":
                    n_ok += 1
                else:
                    n_err += 1

                if (n_total % 20) == 0:
                    print(f"[{_now()}] processed={n_total} ok={n_ok} err={n_err}", flush=True)
        else:
            ctx = mp.get_context(str(args.mp_start_method))
            with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(worker_cfg,)) as pool:
                for row_dict in pool.imap_unordered(_worker_eval_entry, entries, chunksize=1):
                    n_total += 1
                    status = str(row_dict.get("status", "error"))
                    if status == "ok":
                        n_ok += 1
                    else:
                        n_err += 1
                    w.writerow({k: row_dict.get(k) for k in fieldnames})
                    if (n_total % 20) == 0:
                        print(f"[{_now()}] processed={n_total} ok={n_ok} err={n_err}", flush=True)

    print(f"[done] out_csv={out_csv} processed={n_total} ok={n_ok} err={n_err}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
