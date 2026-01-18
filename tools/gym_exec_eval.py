#!/usr/bin/env python3
"""
IsaacGym/legged_gym evaluator for TWIST2 policies over MotionLib-style YAML configs.

This is a "fully gym version":
- NO MuJoCo simulation (unlike tools/mujoco_exec_eval*.py)
- Uses IsaacGym-based legged_gym envs (same code path as legged_gym/legged_gym/scripts/play.py)

It evaluates a subset of motions from a YAML (root_path + motions[].file) by forcing motion_id
via env.reset_idx(..., motion_ids=...) and stepping until the env resets. Per-motion results are
written to CSV, including termination reason and basic tracking error metrics from env.get_episode_log().

Policy sources:
- --policy_type=runner (default): loads an RSL-RL checkpoint via task_registry.make_alg_runner()
- --policy_type=onnx: runs a standalone ONNX policy on gym observations via onnxruntime


python tools/gym_exec_eval.py --task g1_priv_mimic --proj_name g1_priv_mimic --resumeid 0106_teacher --checkpoint -1 \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml --out_csv outputs/wbc_0117_230k_teacher0106_cpu_w64.csv \
    --device cpu --headless --num_envs 1 --episode_length_s 300 --log_stride 200 --runner_backend onnx --workers 64 --mp_start_method spawn

# Fastest single-GPU large-scale eval (resume after Ctrl-C with --append):
# python tools/gym_exec_eval.py --queue_eval --task g1_priv_mimic --proj_name g1_priv_mimic --resumeid 0106_teacher --checkpoint -1 \
#     --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml --out_csv outputs/wbc_0117_230k_teacher0106_gpu4096.csv --append \
#     --device cuda:0 --headless --num_envs 4096 --episode_length_s 300 --queue_metrics fast
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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
class SelectedMotion:
    original_idx: int
    relpath: str
    weight: float


def _select_motions_from_yaml(
    motion_yaml: Path,
    *,
    motion_ids: str,
    max_motions: int,
    shuffle: bool,
    shuffle_seed: int,
    shard_idx: int,
    num_shards: int,
) -> tuple[str, list[SelectedMotion]]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read motion YAML configs")
    motion_yaml = Path(motion_yaml).expanduser().resolve()
    with open(motion_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict yaml in {motion_yaml}, got {type(cfg)}")
    root_path = str(cfg["root_path"])
    motion_list = list(cfg["motions"])
    if not motion_list:
        return root_path, []

    if str(motion_ids).strip():
        indices = _parse_index_spec(str(motion_ids), len(motion_list))
        kept = [(i, motion_list[i]) for i in indices]
    else:
        kept = list(enumerate(motion_list))
        if bool(shuffle):
            rng = np.random.RandomState(int(shuffle_seed))
            order = rng.permutation(len(kept)).tolist()
            kept = [kept[i] for i in order]

    if int(max_motions) > 0:
        kept = kept[: int(max_motions)]

    shard_idx = int(shard_idx)
    num_shards = int(num_shards)
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >0, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"invalid shard_idx={shard_idx} for num_shards={num_shards}")

    selected: list[SelectedMotion] = []
    for orig_i, entry in kept:
        rel = str(entry["file"])
        weight = float(entry.get("weight", 1.0))
        h = int.from_bytes(hashlib.md5(rel.encode("utf-8")).digest()[:8], "little", signed=False)
        if (h % num_shards) != shard_idx:
            continue
        selected.append(SelectedMotion(original_idx=int(orig_i), relpath=rel, weight=float(weight)))
    return root_path, selected


def _write_subset_yaml(*, root_path: str, motions: list[SelectedMotion], out_path: Path) -> None:
    if yaml is None:
        raise RuntimeError("pyyaml is required to write motion YAML configs")
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"root_path": str(root_path), "motions": [{"file": m.relpath, "weight": float(m.weight)} for m in motions]}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _safe_float(val: str) -> Optional[float]:
    s = str(val or "").strip()
    if not s:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _stats_from_array(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _summarize_csv(csv_path: Path, *, out_json: Path) -> dict[str, Any]:
    csv_path = Path(csv_path).expanduser().resolve()
    out_json = Path(out_json).expanduser().resolve()
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    values_all: dict[str, list[float]] = {}
    values_ok: dict[str, list[float]] = {}
    exclude = {"motion_idx_original", "motion_id_loaded", "motion_relpath", "status", "done_reason", "error"}

    total = 0
    ok = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            total += 1
            status = str(row.get("status", "") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "ok":
                ok += 1
            reason = str(row.get("done_reason", "") or "")
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            for k, v in row.items():
                if k in exclude:
                    continue
                val = _safe_float(str(v))
                if val is None:
                    continue
                values_all.setdefault(k, []).append(val)
                if status == "ok":
                    values_ok.setdefault(k, []).append(val)

    summary: dict[str, Any] = {
        "generated_at": _now(),
        "csv": str(csv_path),
        "total_rows": int(total),
        "ok_rows": int(ok),
        "ok_rate": float(ok / total) if total > 0 else 0.0,
        "status_counts": status_counts,
        "done_reason_counts": reason_counts,
        "metrics_all": {},
        "metrics_ok": {},
    }

    for name, vals in sorted(values_all.items()):
        summary["metrics_all"][name] = _stats_from_array(np.asarray(vals, dtype=np.float64))
    for name, vals in sorted(values_ok.items()):
        summary["metrics_ok"][name] = _stats_from_array(np.asarray(vals, dtype=np.float64))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def _summary_out_path(out_csv: Path, summary_json: str) -> Path:
    if str(summary_json).strip():
        return Path(summary_json).expanduser().resolve()
    return Path(out_csv).with_suffix(Path(out_csv).suffix + ".summary.json")


def _maybe_write_summary(out_csv: Path, args: argparse.Namespace) -> None:
    if bool(getattr(args, "no_summary", False)):
        return
    out_json = _summary_out_path(Path(out_csv), str(getattr(args, "summary_json", "")))
    _summarize_csv(Path(out_csv), out_json=out_json)
    print(f"[done] summary_json={out_json}", flush=True)


class _MetricsAccumulator:
    def __init__(self, env, *, log_stride: int, active_mask=None) -> None:
        import torch
        from isaacgym.torch_utils import quat_conjugate, quat_from_euler_xyz, quat_mul
        from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos
        from legged_gym.envs.base.legged_robot import euler_from_quaternion

        self.env = env
        self.log_stride = max(1, int(log_stride))
        self.active_mask = active_mask
        self.num_envs = int(env.num_envs)
        self._torch = torch
        self._quat_conjugate = quat_conjugate
        self._quat_mul = quat_mul
        self._quat_from_euler_xyz = quat_from_euler_xyz
        self._euler_from_quaternion = euler_from_quaternion
        self._convert_to_local_root_body_pos = convert_to_local_root_body_pos
        self.has_keybody = hasattr(env, "_key_body_ids") and hasattr(env, "_ref_body_pos") and hasattr(env, "rigid_body_states")

        self.sum_root_pos_l2 = torch.zeros((self.num_envs,), device=env.device)
        self.sum_root_pos_l1 = torch.zeros((self.num_envs,), device=env.device)
        self.sum_root_rot_rad = torch.zeros((self.num_envs,), device=env.device)
        self.sum_dof_pos_l2 = torch.zeros((self.num_envs,), device=env.device)
        self.sum_dof_pos_l1 = torch.zeros((self.num_envs,), device=env.device)
        self.sum_dof_vel_l2 = torch.zeros((self.num_envs,), device=env.device)
        self.sum_dof_vel_l1 = torch.zeros((self.num_envs,), device=env.device)
        self.sum_keybody_l1 = torch.zeros((self.num_envs,), device=env.device)
        self.sum_keybody_l2 = torch.zeros((self.num_envs,), device=env.device)
        self.count = torch.zeros((self.num_envs,), device=env.device, dtype=torch.long)
        self._install()

    def _quat_angle_diff_rad(self, q, q_ref):
        dq = self._quat_mul(q_ref, self._quat_conjugate(q))
        dq_xyz = dq[:, 0:3]
        dq_w = self._torch.abs(dq[:, 3])
        return 2.0 * self._torch.atan2(self._torch.norm(dq_xyz, dim=-1), dq_w.clamp(min=1e-8))

    def _accumulate(self) -> None:
        env = self.env
        torch = self._torch
        mask = (env.episode_length_buf % int(self.log_stride)) == 0
        if self.active_mask is not None:
            mask = mask & self.active_mask
        if not bool(torch.any(mask)):
            return

        root_pos_diff = env._ref_root_pos - env.root_states[:, 0:3]
        root_pos_l2 = torch.norm(root_pos_diff, dim=-1)
        root_pos_l1 = torch.mean(torch.abs(root_pos_diff), dim=-1)
        root_rot_rad = self._quat_angle_diff_rad(env.root_states[:, 3:7], env._ref_root_rot)

        dof_pos_diff = env._ref_dof_pos - env.dof_pos
        dof_pos_l2 = torch.norm(dof_pos_diff, dim=-1)
        dof_pos_l1 = torch.mean(torch.abs(dof_pos_diff), dim=-1)

        dof_vel_diff = env._ref_dof_vel - env.dof_vel
        dof_vel_l2 = torch.norm(dof_vel_diff, dim=-1)
        dof_vel_l1 = torch.mean(torch.abs(dof_vel_diff), dim=-1)

        if self.has_keybody:
            key_body_pos = env.rigid_body_states[:, env._key_body_ids, 0:3]
            key_body_pos = key_body_pos - env.root_states[:, 0:3].unsqueeze(1)
            tar_key_body_pos = env._ref_body_pos[:, env._key_body_ids, :]
            tar_key_body_pos = tar_key_body_pos - env._ref_root_pos.unsqueeze(1)
            if not getattr(env, "global_obs", False):
                base_yaw = env.yaw
                base_yaw_quat = self._quat_from_euler_xyz(0 * base_yaw, 0 * base_yaw, base_yaw)
                key_body_pos = self._convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)
                _, _, ref_yaw = self._euler_from_quaternion(env._ref_root_rot)
                ref_yaw_quat = self._quat_from_euler_xyz(0 * ref_yaw, 0 * ref_yaw, ref_yaw)
                tar_key_body_pos = self._convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)
            key_diff = key_body_pos - tar_key_body_pos
            key_l1 = torch.mean(torch.abs(key_diff), dim=-1).mean(dim=-1)
            key_l2 = torch.norm(key_diff, dim=-1).mean(dim=-1)
        else:
            key_l1 = torch.zeros_like(root_pos_l2)
            key_l2 = torch.zeros_like(root_pos_l2)

        self.sum_root_pos_l2[mask] += root_pos_l2[mask]
        self.sum_root_pos_l1[mask] += root_pos_l1[mask]
        self.sum_root_rot_rad[mask] += root_rot_rad[mask]
        self.sum_dof_pos_l2[mask] += dof_pos_l2[mask]
        self.sum_dof_pos_l1[mask] += dof_pos_l1[mask]
        self.sum_dof_vel_l2[mask] += dof_vel_l2[mask]
        self.sum_dof_vel_l1[mask] += dof_vel_l1[mask]
        self.sum_keybody_l1[mask] += key_l1[mask]
        self.sum_keybody_l2[mask] += key_l2[mask]
        self.count[mask] += 1

    def _install(self) -> None:
        orig = self.env._post_physics_step_callback

        def _wrapped():
            orig()
            self._accumulate()

        self.env._post_physics_step_callback = _wrapped

    def reset(self, env_ids) -> None:
        if env_ids is None:
            return
        if hasattr(env_ids, "numel") and int(env_ids.numel()) == 0:
            return
        idx = env_ids
        self.sum_root_pos_l2[idx] = 0.0
        self.sum_root_pos_l1[idx] = 0.0
        self.sum_root_rot_rad[idx] = 0.0
        self.sum_dof_pos_l2[idx] = 0.0
        self.sum_dof_pos_l1[idx] = 0.0
        self.sum_dof_vel_l2[idx] = 0.0
        self.sum_dof_vel_l1[idx] = 0.0
        self.sum_keybody_l1[idx] = 0.0
        self.sum_keybody_l2[idx] = 0.0
        self.count[idx] = 0

    def means(self, env_ids) -> dict[str, Any]:
        torch = self._torch
        if env_ids is None or (hasattr(env_ids, "numel") and int(env_ids.numel()) == 0):
            return {}
        idx = env_ids
        count = self.count[idx].to(dtype=torch.float32)
        denom = torch.where(count > 0, count, torch.ones_like(count))

        def _mean(t):
            out = t[idx] / denom
            return torch.where(count > 0, out, torch.full_like(out, float("nan")))

        return {
            "root_pos_l2": _mean(self.sum_root_pos_l2),
            "root_pos_l1": _mean(self.sum_root_pos_l1),
            "root_rot_rad": _mean(self.sum_root_rot_rad),
            "dof_pos_l2": _mean(self.sum_dof_pos_l2),
            "dof_pos_l1": _mean(self.sum_dof_pos_l1),
            "dof_vel_l2": _mean(self.sum_dof_vel_l2),
            "dof_vel_l1": _mean(self.sum_dof_vel_l1),
            "keybody_l1": _mean(self.sum_keybody_l1) if self.has_keybody else torch.full_like(count, float("nan")),
            "keybody_l2": _mean(self.sum_keybody_l2) if self.has_keybody else torch.full_like(count, float("nan")),
            "count": count,
        }


def _install_reset_reason_trace(env) -> None:
    orig = env.check_termination
    env._exec_eval_last_reason = [""] * int(env.num_envs)  # type: ignore[attr-defined]
    env._exec_eval_last_time_s = [float("nan")] * int(env.num_envs)  # type: ignore[attr-defined]
    env._exec_eval_last_step = [-1] * int(env.num_envs)  # type: ignore[attr-defined]
    env._exec_eval_last_motion_id = [-1] * int(env.num_envs)  # type: ignore[attr-defined]

    def _wrapped_check_termination():
        orig()
        try:
            import torch
        except Exception:  # pragma: no cover
            return

        reset_buf = getattr(env, "reset_buf", None)
        if reset_buf is None or (not torch.any(reset_buf)):
            return

        contact_force_termination = torch.any(
            torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1.0, dim=1
        )
        root_height_diff = torch.abs(env.root_states[:, 2] - env._ref_root_pos[:, 2])
        height_cutoff = root_height_diff > float(env.cfg.rewards.root_height_diff_threshold)
        roll_cut = torch.abs(env.roll) > float(env.cfg.rewards.termination_roll)
        pitch_cut = torch.abs(env.pitch) > float(env.cfg.rewards.termination_pitch)
        motion_end = env.episode_length_buf * env.dt >= env._motion_lib.get_motion_length(env._motion_ids)
        vel_too_large = torch.norm(env.root_states[:, 7:10], dim=-1) > 5.0

        reset_ids = torch.nonzero(reset_buf, as_tuple=False).flatten()
        for env_i in reset_ids.tolist():
            reason = "unknown"
            if bool(contact_force_termination[env_i].item()):
                reason = "contact_force"
            elif bool(height_cutoff[env_i].item()):
                reason = "height_cutoff"
            elif bool(roll_cut[env_i].item()):
                reason = "roll_limit"
            elif bool(pitch_cut[env_i].item()):
                reason = "pitch_limit"
            elif bool(motion_end[env_i].item()):
                reason = "motion_end"
            elif bool(vel_too_large[env_i].item()):
                reason = "vel_too_large"

            env._exec_eval_last_reason[env_i] = str(reason)  # type: ignore[attr-defined]
            env._exec_eval_last_time_s[env_i] = float(env.episode_length_buf[env_i].item() * env.dt)  # type: ignore[attr-defined]
            try:
                env._exec_eval_last_step[env_i] = int(env.episode_length_buf[env_i].item())  # type: ignore[attr-defined]
            except Exception:
                env._exec_eval_last_step[env_i] = -1  # type: ignore[attr-defined]
            try:
                env._exec_eval_last_motion_id[env_i] = int(env._motion_ids[env_i].item())  # type: ignore[attr-defined]
            except Exception:
                env._exec_eval_last_motion_id[env_i] = -1  # type: ignore[attr-defined]

    env.check_termination = _wrapped_check_termination


def _refresh_obs(env):
    from isaacgym.torch_utils import quat_rotate_inverse
    from legged_gym.envs.base.legged_robot import euler_from_quaternion

    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    try:
        env.gym.refresh_dof_state_tensor(env.sim)
    except Exception:
        pass

    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    env.compute_observations()
    return env.get_observations()


def _call_policy(policy, obs, *, hist_encoding: bool = True):
    try:
        return policy(obs, hist_encoding=bool(hist_encoding))
    except TypeError:
        return policy(obs)


def _make_onnx_policy(onnx_path: str, *, device: str):
    try:
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"onnxruntime is required for --policy_type onnx: {type(e).__name__}: {e}") from e

    device = str(device).strip().lower()
    device_id = 0
    use_cuda = device.startswith("cuda")
    if device.startswith("cuda:"):
        try:
            device_id = int(device.split(":", 1)[1])
        except Exception:
            device_id = 0

    providers = []
    available = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": int(device_id)}))
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(Path(onnx_path).expanduser().resolve()), providers=providers)
    input0 = session.get_inputs()[0]
    input_name = input0.name
    expected_obs_dim = None
    try:
        shape = input0.shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            expected_obs_dim = int(shape[1])
    except Exception:
        expected_obs_dim = None

    def _infer(obs_torch):
        import torch

        obs = obs_torch.detach()
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_np = obs.to("cpu").numpy().astype("float32", copy=False)
        if expected_obs_dim is not None and int(obs_np.shape[1]) != int(expected_obs_dim):
            raise ValueError(f"ONNX expected obs_dim={expected_obs_dim}, got {tuple(obs_np.shape)}")
        outs = session.run(None, {input_name: obs_np})
        act = np.asarray(outs[0], dtype=np.float32)
        if act.ndim == 1:
            act = act[None, :]
        return torch.from_numpy(act).to(device=obs_torch.device)

    return _infer


def _default_ckpt_to_onnx_cache_dir() -> Path:
    mmdd = time.strftime("%m%d", time.localtime())
    return Path(f"/tmp/codex-{mmdd}-gym-pt-to-onnx")

def _default_tmp_dir() -> Path:
    mmdd = time.strftime("%m%d", time.localtime())
    return Path(f"/tmp/codex-{mmdd}-gym-exec-eval")


def _resolve_ckpt_path(*, proj_name: str, resumeid: str, checkpoint: int) -> Path:
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.gym_utils.helpers import get_load_path

    if not str(resumeid).strip():
        raise ValueError("--resumeid is required to resolve the checkpoint path for ONNX export")
    root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / str(proj_name) / str(resumeid)
    ckpt = Path(get_load_path(str(root), checkpoint=int(checkpoint))).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    return ckpt


def _acquire_lock(lock_path: Path, *, timeout_s: float = 600.0, poll_s: float = 0.2) -> int:
    """
    Cross-process lock using an exclusive lock file.
    Returns an fd that must be closed by _release_lock().
    """
    lock_path = Path(lock_path).expanduser().resolve()
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            try:
                os.write(fd, str(os.getpid()).encode("utf-8"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            if (time.time() - t0) > float(timeout_s):
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(float(poll_s))


def _release_lock(fd: int, lock_path: Path) -> None:
    try:
        os.close(fd)
    finally:
        try:
            Path(lock_path).unlink(missing_ok=True)
        except Exception:
            pass


def _export_actor_critic_to_onnx(
    *,
    actor_critic,
    normalizer,
    out_path: Path,
    opset: int,
    include_normalizer: bool,
    obs_dim: int,
) -> None:
    import torch

    ac = copy.deepcopy(actor_critic).to("cpu").eval()
    n = None
    if include_normalizer and (normalizer is not None):
        try:
            n = copy.deepcopy(normalizer).to("cpu").eval()
        except Exception:
            n = None

    class _Wrapper(torch.nn.Module):
        def __init__(self, ac0, n0):
            super().__init__()
            self.ac = ac0
            self.n = n0

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = obs
            if self.n is not None:
                x = self.n.normalize(x)
            # act_inference signature varies; always request eval=True (deterministic mean).
            try:
                return self.ac.act_inference(x, eval=True, hist_encoding=True)
            except TypeError:
                try:
                    return self.ac.act_inference(x, eval=True)
                except TypeError:
                    return self.ac.act_inference(x)

    wrapper = _Wrapper(ac, n).eval()

    dummy = torch.zeros((1, int(obs_dim)), dtype=torch.float32)
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid race conditions when multiple workers export the same ckpt simultaneously.
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    fd = _acquire_lock(lock_path)
    try:
        if out_path.exists():
            return
        torch.onnx.export(
            wrapper,
            dummy,
            str(tmp_path),
            opset_version=int(opset),
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        )
        os.replace(str(tmp_path), str(out_path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        _release_lock(fd, lock_path)


def _merge_csvs(*, inputs: list[Path], out_csv: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for p in inputs:
        with open(p, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(r.fieldnames or [])
            for row in r:
                rows.append(dict(row))
    if fieldnames is None:
        raise RuntimeError("No input CSVs to merge")

    def _k(d: dict[str, str]):
        try:
            return (int(d.get("motion_idx_original", "0")), d.get("motion_relpath", ""))
        except Exception:
            return (0, d.get("motion_relpath", ""))

    rows.sort(key=_k)
    out_csv = Path(out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _load_done_motion_ids_loaded(*, csv_path: Path, motions: list[SelectedMotion]) -> set[int]:
    """Load already-evaluated motion ids (0..len(motions)-1) from an existing CSV."""
    csv_path = Path(csv_path).expanduser().resolve()
    num_motions = int(len(motions))
    done: set[int] = set()
    orig_to_loaded: dict[int, int] = {int(m.original_idx): int(i) for i, m in enumerate(motions)}
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                mid_s = str(row.get("motion_id_loaded", "") or "").strip()
                if mid_s:
                    try:
                        mid = int(mid_s)
                    except Exception:
                        try:
                            mid = int(float(mid_s))
                        except Exception:
                            mid = None
                    if mid is not None and 0 <= int(mid) < num_motions:
                        done.add(int(mid))
                        continue

                orig_s = str(row.get("motion_idx_original", "") or "").strip()
                if not orig_s:
                    continue
                try:
                    orig = int(orig_s)
                except Exception:
                    continue
                mid2 = orig_to_loaded.get(int(orig))
                if mid2 is not None:
                    done.add(int(mid2))
    except FileNotFoundError:
        return set()
    except Exception as e:
        print(f"[warn] Failed to read resume CSV {csv_path}: {type(e).__name__}: {e}", flush=True)
    return done


def _worker_entry(args_dict: dict[str, Any], shard_idx: int, num_shards: int, out_csv: str) -> str:
    args = argparse.Namespace(**args_dict)
    args.shard_idx = int(shard_idx)
    args.num_shards = int(num_shards)
    args.out_csv = str(out_csv)
    args.append = False
    args.skip_motions = 0
    _run_eval_single_process(args)
    return str(out_csv)


def _run_eval_single_process(args: argparse.Namespace) -> None:
    if yaml is None:
        raise RuntimeError("pyyaml is required")

    # IsaacGym should be imported before torch.
    import isaacgym  # noqa: F401
    from isaacgym import gymapi  # noqa: F401
    import torch  # noqa: F401

    import legged_gym.envs  # noqa: F401
    from legged_gym.gym_utils import task_registry
    from legged_gym.gym_utils.helpers import get_args as get_legged_gym_args

    gym_args = get_legged_gym_args()
    # Keep our args as the source of truth (avoid relying on sys.argv parsing behavior).
    if str(getattr(args, "proj_name", "")).strip():
        gym_args.proj_name = str(args.proj_name)
    if str(getattr(args, "resumeid", "")).strip():
        gym_args.resumeid = str(args.resumeid)
    # In this repo, `--no_rand` also rewrites cfg_train.runner.policy_class_name to non-mimic defaults.
    # We control eval determinism via env_cfg overrides below.
    try:
        gym_args.no_rand = False
    except Exception:
        pass

    tmp_dir = _default_tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    root_path, motions = _select_motions_from_yaml(
        Path(args.motion_yaml),
        motion_ids=str(args.motion_ids),
        max_motions=int(args.max_motions),
        shuffle=bool(args.shuffle),
        shuffle_seed=int(args.shuffle_seed),
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
    )
    if not motions:
        raise RuntimeError("No motions selected; check --motion_ids/--max_motions/--shard_idx/--num_shards.")

    subset_yaml = tmp_dir / f"subset_{Path(args.motion_yaml).stem}_n{len(motions)}_sh{args.shard_idx}of{args.num_shards}.yaml"
    _write_subset_yaml(root_path=root_path, motions=motions, out_path=subset_yaml)

    env_cfg, train_cfg = task_registry.get_cfgs(name=str(args.task))
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.env.debug_viz = False
    env_cfg.env.episode_length_s = float(args.episode_length_s)
    env_cfg.env.record_video = False
    env_cfg.env.rand_reset = False

    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "domain_rand"):
        try:
            env_cfg.domain_rand.domain_rand_general = False
        except Exception:
            pass

    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False
        env_cfg.motion.motion_file = str(subset_yaml)
        env_cfg.motion.max_motions = -1
        env_cfg.motion.motion_ids = ""
        env_cfg.motion.shuffle_motions = False

    train_cfg.runner.resume = True
    env, _ = task_registry.make_env(name=str(args.task), args=gym_args, env_cfg=env_cfg)
    _install_reset_reason_trace(env)

    policy_type = str(args.policy_type).strip().lower()
    normalizer = None
    infer_actions: Callable[[Any], Any]
    if policy_type == "runner":
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            log_root="default", env=env, name=str(args.task), args=gym_args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        if getattr(env_cfg.env, "normalize_obs", False):
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
            except Exception:
                normalizer = None

        runner_backend = str(args.runner_backend).strip().lower()
        if runner_backend not in {"onnx", "torch"}:
            raise ValueError(f"Unsupported --runner_backend={args.runner_backend!r}")

        if runner_backend == "torch":
            def infer_actions(obs_in):
                return _call_policy(policy, obs_in, hist_encoding=True)
        else:
            ckpt_path = _resolve_ckpt_path(
                proj_name=str(args.proj_name),
                resumeid=str(args.resumeid),
                checkpoint=int(args.checkpoint),
            )

            cache_dir = _default_ckpt_to_onnx_cache_dir() if not str(args.onnx_cache_dir).strip() else Path(args.onnx_cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            st = ckpt_path.stat()
            include_norm_pref = bool(args.onnx_include_normalizer) or (normalizer is not None)

            def _cache_path(norm_flag: bool) -> Path:
                key = f"{ckpt_path}:{st.st_size}:{getattr(st, 'st_mtime_ns', int(st.st_mtime*1e9))}:{int(args.onnx_opset)}:norm{int(norm_flag)}"
                h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
                return cache_dir / f"{ckpt_path.stem}-{h}.onnx"

            out_onnx = _cache_path(include_norm_pref)

            def _use_onnx(path: Path, *, baked_normalizer: bool) -> None:
                nonlocal normalizer, infer_actions
                if baked_normalizer:
                    normalizer = None
                infer_onnx = _make_onnx_policy(str(path), device=str(args.device))

                def _infer(obs_in):
                    return infer_onnx(obs_in)

                infer_actions = _infer

            if out_onnx.exists():
                _use_onnx(out_onnx, baked_normalizer=include_norm_pref)
            else:
                try:
                    _export_actor_critic_to_onnx(
                        actor_critic=ppo_runner.alg.actor_critic,
                        normalizer=normalizer,
                        out_path=out_onnx,
                        opset=int(args.onnx_opset),
                        include_normalizer=include_norm_pref,
                        obs_dim=int(env.num_obs),
                    )
                except Exception as e:
                    if include_norm_pref and (normalizer is not None):
                        out_onnx2 = _cache_path(False)
                        try:
                            _export_actor_critic_to_onnx(
                                actor_critic=ppo_runner.alg.actor_critic,
                                normalizer=None,
                                out_path=out_onnx2,
                                opset=int(args.onnx_opset),
                                include_normalizer=False,
                                obs_dim=int(env.num_obs),
                            )
                        except Exception as e2:
                            print(f"[warn] ONNX export failed ({type(e2).__name__}: {e2}); falling back to torch policy.", flush=True)

                            def infer_actions(obs_in):
                                return _call_policy(policy, obs_in, hist_encoding=True)
                        else:
                            _use_onnx(out_onnx2, baked_normalizer=False)
                    else:
                        print(f"[warn] ONNX export failed ({type(e).__name__}: {e}); falling back to torch policy.", flush=True)

                        def infer_actions(obs_in):
                            return _call_policy(policy, obs_in, hist_encoding=True)
                else:
                    _use_onnx(out_onnx, baked_normalizer=include_norm_pref)

    elif policy_type == "onnx":
        if not str(args.onnx_path).strip():
            raise ValueError("--onnx_path is required when --policy_type=onnx")
        infer_onnx = _make_onnx_policy(str(args.onnx_path), device=str(args.device))

        def infer_actions(obs_in):
            return infer_onnx(obs_in)
    else:
        raise ValueError(f"Unsupported --policy_type={args.policy_type!r}")

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if bool(args.append) else "w"
    write_header = (mode == "w") or (not out_csv.exists()) or (out_csv.stat().st_size == 0)

    fieldnames = [
        "motion_idx_original",
        "motion_id_loaded",
        "motion_relpath",
        "status",
        "done_reason",
        "done_time_s",
        "motion_len_s",
        "progress",
        "wall_time_s",
        "steps_exec",
        "err_root_pos_l2_mean",
        "err_root_rot_deg_mean",
        "err_dof_pos_l2_mean",
        "err_dof_vel_l2_mean",
        "err_keybody_pos_l1_mean",
        "root_pos_mean_l2_m",
        "root_pos_mean_l1_m",
        "root_rot_mean_deg",
        "joint_dof_mean_l1",
        "joint_vel_mean_l1",
        "fk_rel_mean_l2_m",
        "error",
    ]

    import torch
    log_stride = max(1, int(args.log_stride))
    metrics = _MetricsAccumulator(env, log_stride=log_stride)
    print(f"[{_now()}] selected_motions={len(motions)} env_dt={float(env.dt):.4f}s log_stride={log_stride}", flush=True)

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for local_id, m in enumerate(motions):
            if int(args.skip_motions) > 0 and int(local_id) < int(args.skip_motions):
                continue
            row: dict[str, Any] = {
                "motion_idx_original": int(m.original_idx),
                "motion_id_loaded": int(local_id),
                "motion_relpath": str(m.relpath),
            }
            try:
                t_wall0 = time.perf_counter()
                env_ids = torch.arange(env.num_envs, device=env.device)
                motion_id_tensor = torch.full((env.num_envs,), int(local_id), device=env.device, dtype=torch.long)
                env.reset_idx(env_ids, motion_ids=motion_id_tensor)
                obs = _refresh_obs(env)
                metrics.reset(env_ids)

                motion_len_s = float(env._motion_lib.get_motion_length(motion_id_tensor[:1]).item())
                max_steps = max(1, int(math.ceil(motion_len_s / float(env.dt))))
                steps_exec = 0
                done_reason = ""
                done_time_s = float("nan")

                for step in range(max_steps):
                    steps_exec = step + 1
                    if normalizer is not None:
                        obs_in = normalizer.normalize(obs.detach())
                    else:
                        obs_in = obs.detach()
                    actions = infer_actions(obs_in)
                    step_out = env.step(actions.detach())
                    obs = step_out[0]
                    dones = step_out[3]

                    if bool(dones[0].item()):
                        done_reason = str(getattr(env, "_exec_eval_last_reason", [""])[0])  # type: ignore[attr-defined]
                        done_time_s = float(getattr(env, "_exec_eval_last_time_s", [float("nan")])[0])  # type: ignore[attr-defined]
                        break

                if not done_reason:
                    done_reason = "unknown"
                progress = float(done_time_s / motion_len_s) if (motion_len_s > 0 and math.isfinite(done_time_s)) else float("nan")
                status = "ok" if done_reason == "motion_end" else "fail"
                wall_time_s = float(time.perf_counter() - t_wall0)
                m = metrics.means(torch.tensor([0], device=env.device))
                root_pos_l2_mean = float(m.get("root_pos_l2")[0].item()) if m else float("nan")
                root_pos_l1_mean = float(m.get("root_pos_l1")[0].item()) if m else float("nan")
                root_rot_deg_mean = float(m.get("root_rot_rad")[0].item()) * 180.0 / math.pi if m else float("nan")
                dof_pos_l2_mean = float(m.get("dof_pos_l2")[0].item()) if m else float("nan")
                dof_pos_l1_mean = float(m.get("dof_pos_l1")[0].item()) if m else float("nan")
                dof_vel_l2_mean = float(m.get("dof_vel_l2")[0].item()) if m else float("nan")
                dof_vel_l1_mean = float(m.get("dof_vel_l1")[0].item()) if m else float("nan")
                keybody_l1_mean = float(m.get("keybody_l1")[0].item()) if m else float("nan")
                keybody_l2_mean = float(m.get("keybody_l2")[0].item()) if m else float("nan")

                row.update(
                    {
                        "status": status,
                        "done_reason": done_reason,
                        "done_time_s": float(done_time_s),
                        "motion_len_s": float(motion_len_s),
                        "progress": float(progress),
                        "wall_time_s": float(wall_time_s),
                        "steps_exec": int(steps_exec),
                        "err_root_pos_l2_mean": float(root_pos_l2_mean),
                        "err_root_rot_deg_mean": float(root_rot_deg_mean),
                        "err_dof_pos_l2_mean": float(dof_pos_l2_mean),
                        "err_dof_vel_l2_mean": float(dof_vel_l2_mean),
                        "err_keybody_pos_l1_mean": float(keybody_l1_mean),
                        "root_pos_mean_l2_m": float(root_pos_l2_mean),
                        "root_pos_mean_l1_m": float(root_pos_l1_mean),
                        "root_rot_mean_deg": float(root_rot_deg_mean),
                        "joint_dof_mean_l1": float(dof_pos_l1_mean),
                        "joint_vel_mean_l1": float(dof_vel_l1_mean),
                        "fk_rel_mean_l2_m": float(keybody_l2_mean),
                        "error": "",
                    }
                )
            except Exception as e:
                row.update(
                    {
                        "status": "error",
                        "done_reason": "",
                        "done_time_s": float("nan"),
                        "motion_len_s": float("nan"),
                        "progress": float("nan"),
                        "wall_time_s": float("nan"),
                        "steps_exec": int(0),
                        "err_root_pos_l2_mean": float("nan"),
                        "err_root_rot_deg_mean": float("nan"),
                        "err_dof_pos_l2_mean": float("nan"),
                        "err_dof_vel_l2_mean": float("nan"),
                        "err_keybody_pos_l1_mean": float("nan"),
                        "root_pos_mean_l2_m": float("nan"),
                        "root_pos_mean_l1_m": float("nan"),
                        "root_rot_mean_deg": float("nan"),
                        "joint_dof_mean_l1": float("nan"),
                        "joint_vel_mean_l1": float("nan"),
                        "fk_rel_mean_l2_m": float("nan"),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

            w.writerow({k: row.get(k) for k in fieldnames})
            if ((local_id + 1) % 10) == 0:
                print(f"[{_now()}] processed={local_id+1}/{len(motions)}", flush=True)

    print(f"[done] out_csv={out_csv}")
    _maybe_write_summary(out_csv, args)


def _run_eval_queue_mode(args: argparse.Namespace) -> None:
    """
    High-throughput evaluator for large motion sets:
    - Runs a single IsaacGym process with many envs (e.g. --num_envs 4096 on one GPU)
    - Assigns a different motion_id per env
    - On episode end, logs per-motion result and immediately assigns the next motion to that env

    This mirrors training-style throughput (vectorized rollouts) while still producing per-motion CSV rows.
    """
    if yaml is None:
        raise RuntimeError("pyyaml is required")

    # IsaacGym should be imported before torch.
    import isaacgym  # noqa: F401
    from isaacgym import gymapi  # noqa: F401
    import torch

    import legged_gym.envs  # noqa: F401
    from legged_gym.gym_utils import task_registry
    from legged_gym.gym_utils.helpers import get_args as get_legged_gym_args

    if int(args.workers) != 1:
        raise ValueError("--queue_eval runs in a single process; do not combine with --workers")
    if int(args.skip_motions) > 0:
        raise ValueError("--skip_motions is not supported with --queue_eval (use --motion_ids/--max_motions/--shard_idx).")

    gym_args = get_legged_gym_args()
    # Keep our args as the source of truth (avoid relying on sys.argv parsing behavior).
    if str(getattr(args, "proj_name", "")).strip():
        gym_args.proj_name = str(args.proj_name)
    if str(getattr(args, "resumeid", "")).strip():
        gym_args.resumeid = str(args.resumeid)
    try:
        gym_args.no_rand = False
    except Exception:
        pass

    tmp_dir = _default_tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    root_path, motions = _select_motions_from_yaml(
        Path(args.motion_yaml),
        motion_ids=str(args.motion_ids),
        max_motions=int(args.max_motions),
        shuffle=bool(args.shuffle),
        shuffle_seed=int(args.shuffle_seed),
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
    )
    if not motions:
        raise RuntimeError("No motions selected; check --motion_ids/--max_motions/--shard_idx/--num_shards.")

    # Avoid writing a huge subset YAML when selection is identity (full evaluation in YAML order).
    need_subset = (
        bool(str(args.motion_ids).strip())
        or bool(args.shuffle)
        or (int(args.max_motions) > 0)
        or (int(args.shard_idx) != 0)
        or (int(args.num_shards) != 1)
    )
    if need_subset:
        subset_yaml = tmp_dir / f"subset_{Path(args.motion_yaml).stem}_n{len(motions)}_sh{args.shard_idx}of{args.num_shards}.yaml"
        _write_subset_yaml(root_path=root_path, motions=motions, out_path=subset_yaml)
        motion_file_for_env = subset_yaml
    else:
        motion_file_for_env = Path(args.motion_yaml).expanduser().resolve()

    # Output CSV (single file, streaming writes). With --append, resume from existing rows.
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if bool(args.append) else "w"
    if mode == "w" and out_csv.exists():
        out_csv.unlink()

    done_loaded: set[int] = set()
    if mode == "a" and out_csv.exists() and out_csv.stat().st_size > 0:
        done_loaded = _load_done_motion_ids_loaded(csv_path=out_csv, motions=motions)
        if done_loaded:
            print(f"[resume] {len(done_loaded)}/{len(motions)} motions already in {out_csv}", flush=True)

    if len(done_loaded) >= len(motions):
        print(f"[done] already complete: {out_csv}", flush=True)
        _maybe_write_summary(out_csv, args)
        return

    env_cfg, train_cfg = task_registry.get_cfgs(name=str(args.task))
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.env.debug_viz = False
    env_cfg.env.episode_length_s = float(args.episode_length_s)
    env_cfg.env.record_video = False
    env_cfg.env.rand_reset = False

    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "domain_rand"):
        try:
            env_cfg.domain_rand.domain_rand_general = False
        except Exception:
            pass

    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False
        env_cfg.motion.motion_file = str(motion_file_for_env)
        env_cfg.motion.max_motions = -1
        env_cfg.motion.motion_ids = ""
        env_cfg.motion.shuffle_motions = False

    train_cfg.runner.resume = True
    env, _ = task_registry.make_env(name=str(args.task), args=gym_args, env_cfg=env_cfg)

    policy_type = str(args.policy_type).strip().lower()
    if policy_type != "runner":
        raise ValueError("--queue_eval currently supports only --policy_type runner (use torch policy like training)")

    # For GPU throughput, prefer torch inference (avoids torch->numpy copies).
    if str(args.runner_backend).strip().lower() != "torch":
        print("[warn] --queue_eval on GPU is fastest with --runner_backend torch; forcing torch backend.", flush=True)

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root="default", env=env, name=str(args.task), args=gym_args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    normalizer = None
    if getattr(env_cfg.env, "normalize_obs", False):
        try:
            normalizer = ppo_runner.get_normalizer(device=env.device)
        except Exception:
            normalizer = None

    # Precompute motion lengths on device for fast batch indexing.
    num_motions = int(len(motions))
    motion_ids_all = torch.arange(num_motions, device=env.device, dtype=torch.long)
    motion_len_all = env._motion_lib.get_motion_length(motion_ids_all).detach()

    fieldnames = [
        "motion_idx_original",
        "motion_id_loaded",
        "motion_relpath",
        "status",
        "done_reason",
        "done_time_s",
        "motion_len_s",
        "progress",
        "wall_time_s",
        "steps_exec",
        "err_root_pos_l2_mean",
        "err_root_rot_deg_mean",
        "err_dof_pos_l2_mean",
        "err_dof_vel_l2_mean",
        "err_keybody_pos_l1_mean",
        "root_pos_mean_l2_m",
        "root_pos_mean_l1_m",
        "root_rot_mean_deg",
        "joint_dof_mean_l1",
        "joint_vel_mean_l1",
        "fk_rel_mean_l2_m",
        "error",
    ]

    metrics_mode = str(getattr(args, "queue_metrics", "fast") or "fast").strip().lower()
    if metrics_mode not in {"fast", "final", "mean"}:
        raise ValueError("--queue_metrics must be 'fast', 'final', or 'mean'")
    log_stride = max(1, int(getattr(args, "log_stride", 1) or 1))
    flush_rows = int(getattr(args, "queue_flush_rows", 2048) or 2048)
    log_every = int(getattr(args, "queue_log_every", 10000) or 10000)

    num_envs = int(env.num_envs)
    next_mid = 0
    active_mask = None
    metrics = None
    if metrics_mode == "mean":
        active_mask = torch.zeros((num_envs,), device=env.device, dtype=torch.bool)
        metrics = _MetricsAccumulator(env, log_stride=log_stride, active_mask=active_mask)

    # Track per-env assignment start wall time (for wall_time_s per motion).
    env_start_wall = [float("nan")] * num_envs
    env_active = [False] * num_envs

    # Results buffering.
    buf: list[dict[str, Any]] = []
    processed_total = int(len(done_loaded))
    processed_new = 0
    processed_at_start = int(processed_total)
    t0 = time.perf_counter()

    def _pop_next_motion_id() -> Optional[int]:
        nonlocal next_mid
        while next_mid < num_motions and (int(next_mid) in done_loaded):
            next_mid += 1
        if next_mid >= num_motions:
            return None
        mid = int(next_mid)
        next_mid += 1
        return mid

    # Patch reset_idx so auto-resets (motion_ids=None) pull from our queue instead of random sampling.
    orig_reset_idx = env.reset_idx

    def _reset_idx_queue(env_ids, motion_ids=None):
        nonlocal processed_total, processed_new
        # If user explicitly passes motion_ids, respect it (used for initial assignment).
        if motion_ids is not None:
            return orig_reset_idx(env_ids, motion_ids=motion_ids)

        if len(env_ids) == 0:
            return

        # Log completions for these envs before they get new assignments.
        ids_list = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        now = time.perf_counter()

        # Batch-read required tensors for these envs to CPU to avoid per-env CUDA syncs.
        env_ids_t = env_ids
        motion_ids_curr_t = env._motion_ids[env_ids_t].detach()
        steps_exec_t = env.episode_length_buf[env_ids_t].detach()
        done_time_s_t = steps_exec_t.to(dtype=torch.float32) * float(env.dt)
        motion_len_s_t = motion_len_all[motion_ids_curr_t]
        motion_end_t = done_time_s_t >= (motion_len_s_t - float(env.dt) * 0.5)

        # Detailed termination reason classification (subset-only, runs only at resets).
        bsz = int(motion_ids_curr_t.shape[0])
        reason_code_t = torch.zeros((bsz,), device=env.device, dtype=torch.int64)

        # Priority order matches _install_reset_reason_trace().
        try:
            term_idx = env.termination_contact_indices
            cf = env.contact_forces[env_ids_t][:, term_idx, :]
            contact_force = torch.any(torch.norm(cf, dim=-1) > 1.0, dim=1)
        except Exception:
            contact_force = torch.zeros((bsz,), device=env.device, dtype=torch.bool)
        reason_code_t = torch.where(contact_force, torch.ones_like(reason_code_t), reason_code_t)

        try:
            root_height_diff = torch.abs(env.root_states[env_ids_t, 2] - env._ref_root_pos[env_ids_t, 2])
            height_cutoff = root_height_diff > float(env.cfg.rewards.root_height_diff_threshold)
        except Exception:
            height_cutoff = torch.zeros((bsz,), device=env.device, dtype=torch.bool)
        reason_code_t = torch.where((reason_code_t == 0) & height_cutoff, torch.full_like(reason_code_t, 2), reason_code_t)

        try:
            roll_cut = torch.abs(env.roll[env_ids_t]) > float(env.cfg.rewards.termination_roll)
        except Exception:
            roll_cut = torch.zeros((bsz,), device=env.device, dtype=torch.bool)
        reason_code_t = torch.where((reason_code_t == 0) & roll_cut, torch.full_like(reason_code_t, 3), reason_code_t)

        try:
            pitch_cut = torch.abs(env.pitch[env_ids_t]) > float(env.cfg.rewards.termination_pitch)
        except Exception:
            pitch_cut = torch.zeros((bsz,), device=env.device, dtype=torch.bool)
        reason_code_t = torch.where((reason_code_t == 0) & pitch_cut, torch.full_like(reason_code_t, 4), reason_code_t)

        reason_code_t = torch.where((reason_code_t == 0) & motion_end_t, torch.full_like(reason_code_t, 5), reason_code_t)

        try:
            vel_too_large = torch.norm(env.root_states[env_ids_t, 7:10], dim=-1) > 5.0
        except Exception:
            vel_too_large = torch.zeros((bsz,), device=env.device, dtype=torch.bool)
        reason_code_t = torch.where((reason_code_t == 0) & vel_too_large, torch.full_like(reason_code_t, 6), reason_code_t)

        motion_ids_curr = motion_ids_curr_t.to("cpu").numpy()
        steps_exec_cpu = steps_exec_t.to("cpu").numpy()
        done_time_s_cpu = done_time_s_t.to("cpu").numpy()
        motion_len_s_cpu = motion_len_s_t.to("cpu").numpy()
        reason_code_cpu = reason_code_t.to("cpu").numpy()

        mean_root_pos_l2 = mean_root_pos_l1 = mean_root_rot_deg = None
        mean_dof_pos_l2 = mean_dof_pos_l1 = mean_dof_vel_l2 = mean_dof_vel_l1 = None
        mean_keybody_l1 = mean_keybody_l2 = None
        if metrics_mode == "mean" and metrics is not None:
            m = metrics.means(env_ids_t)
            mean_root_pos_l2 = m["root_pos_l2"].to("cpu").numpy()
            mean_root_pos_l1 = m["root_pos_l1"].to("cpu").numpy()
            mean_root_rot_deg = (m["root_rot_rad"] * (180.0 / math.pi)).to("cpu").numpy()
            mean_dof_pos_l2 = m["dof_pos_l2"].to("cpu").numpy()
            mean_dof_pos_l1 = m["dof_pos_l1"].to("cpu").numpy()
            mean_dof_vel_l2 = m["dof_vel_l2"].to("cpu").numpy()
            mean_dof_vel_l1 = m["dof_vel_l1"].to("cpu").numpy()
            mean_keybody_l1 = m["keybody_l1"].to("cpu").numpy()
            mean_keybody_l2 = m["keybody_l2"].to("cpu").numpy()

        for j, env_i in enumerate(ids_list):
            env_i = int(env_i)
            if not env_active[env_i]:
                continue
            mid = int(motion_ids_curr[j])
            if mid in done_loaded:
                env_active[env_i] = False
                env_start_wall[env_i] = float("nan")
                continue
            done_loaded.add(mid)
            done_time_s = float(done_time_s_cpu[j])
            motion_len_s = float(motion_len_s_cpu[j])
            steps_exec = int(steps_exec_cpu[j])
            rc = int(reason_code_cpu[j])
            reason = {
                0: "unknown",
                1: "contact_force",
                2: "height_cutoff",
                3: "roll_limit",
                4: "pitch_limit",
                5: "motion_end",
                6: "vel_too_large",
            }.get(rc, "unknown")
            status = "ok" if reason == "motion_end" else "fail"
            progress = float(done_time_s / motion_len_s) if (motion_len_s > 0 and math.isfinite(done_time_s)) else float("nan")
            wall_time_s = float(now - float(env_start_wall[env_i])) if math.isfinite(env_start_wall[env_i]) else float("nan")

            err_root_pos = float("nan")
            err_root_rot_deg = float("nan")
            err_dof_pos = float("nan")
            err_dof_vel = float("nan")
            err_keybody = float("nan")
            root_pos_l1 = float("nan")
            dof_pos_l1 = float("nan")
            dof_vel_l1 = float("nan")
            keybody_l2 = float("nan")
            if metrics_mode == "mean" and mean_root_pos_l2 is not None:
                err_root_pos = float(mean_root_pos_l2[j])
                err_root_rot_deg = float(mean_root_rot_deg[j])
                err_dof_pos = float(mean_dof_pos_l2[j])
                err_dof_vel = float(mean_dof_vel_l2[j])
                err_keybody = float(mean_keybody_l1[j])
                root_pos_l1 = float(mean_root_pos_l1[j])
                dof_pos_l1 = float(mean_dof_pos_l1[j])
                dof_vel_l1 = float(mean_dof_vel_l1[j])
                keybody_l2 = float(mean_keybody_l2[j])
            elif metrics_mode == "final":
                try:
                    log = env.get_episode_log(env_ids=env_i)
                    if "err_root_pos_l2" in log:
                        err_root_pos = float(log["err_root_pos_l2"])
                    if "err_root_rot_rad" in log:
                        err_root_rot_deg = float(log["err_root_rot_rad"]) * 180.0 / math.pi
                    if "err_dof_pos_l2" in log:
                        err_dof_pos = float(log["err_dof_pos_l2"])
                    if "err_dof_vel_l2" in log:
                        err_dof_vel = float(log["err_dof_vel_l2"])
                    if "err_keybody_pos_l1" in log:
                        err_keybody = float(log["err_keybody_pos_l1"])
                except Exception:
                    pass

            buf.append(
                {
                    "motion_idx_original": int(motions[mid].original_idx),
                    "motion_id_loaded": int(mid),
                    "motion_relpath": str(motions[mid].relpath),
                    "status": status,
                    "done_reason": reason,
                    "done_time_s": float(done_time_s),
                    "motion_len_s": float(motion_len_s),
                    "progress": float(progress),
                    "wall_time_s": float(wall_time_s),
                    "steps_exec": int(steps_exec),
                    "err_root_pos_l2_mean": float(err_root_pos),
                    "err_root_rot_deg_mean": float(err_root_rot_deg),
                    "err_dof_pos_l2_mean": float(err_dof_pos),
                    "err_dof_vel_l2_mean": float(err_dof_vel),
                    "err_keybody_pos_l1_mean": float(err_keybody),
                    "root_pos_mean_l2_m": float(err_root_pos) if metrics_mode == "mean" else float("nan"),
                    "root_pos_mean_l1_m": float(root_pos_l1),
                    "root_rot_mean_deg": float(err_root_rot_deg) if metrics_mode == "mean" else float("nan"),
                    "joint_dof_mean_l1": float(dof_pos_l1),
                    "joint_vel_mean_l1": float(dof_vel_l1),
                    "fk_rel_mean_l2_m": float(keybody_l2),
                    "error": "",
                }
            )
            processed_total += 1
            processed_new += 1
            env_active[env_i] = False
            env_start_wall[env_i] = float("nan")

        if metrics is not None:
            metrics.reset(env_ids_t)

        # Assign next motions for these envs.
        next_ids = []
        next_active = []
        for env_i in ids_list:
            mid_next = _pop_next_motion_id()
            if mid_next is not None:
                next_ids.append(mid_next)
                next_active.append(True)
                env_active[int(env_i)] = True
                env_start_wall[int(env_i)] = float(now)
            else:
                # No more work: keep env alive with a dummy motion_id=0, but mark inactive so we ignore its resets.
                next_ids.append(0)
                next_active.append(False)
                env_active[int(env_i)] = False
                env_start_wall[int(env_i)] = float("nan")

        if active_mask is not None:
            active_mask[env_ids_t] = torch.tensor(next_active, device=env.device, dtype=torch.bool)

        motion_ids_t = torch.tensor(next_ids, device=env.device, dtype=torch.long)
        return orig_reset_idx(env_ids, motion_ids=motion_ids_t)

    env.reset_idx = _reset_idx_queue  # type: ignore[assignment]

    # Initial assignment to all envs.
    if num_motions == 0:
        raise RuntimeError("No motions selected")
    remaining = int(num_motions - processed_total)
    if remaining < num_envs:
        print(
            f"[info] remaining_motions={remaining} < num_envs={num_envs}; only first {remaining} envs will be active.",
            flush=True,
        )

    init_ids = []
    init_active = []
    for env_i in range(num_envs):
        mid0 = _pop_next_motion_id()
        if mid0 is not None:
            init_ids.append(mid0)
            init_active.append(True)
            env_active[env_i] = True
            env_start_wall[env_i] = float(t0)
        else:
            init_ids.append(0)
            init_active.append(False)
            env_active[env_i] = False
            env_start_wall[env_i] = float("nan")

    env_ids_all = torch.arange(num_envs, device=env.device)
    env.reset_idx(env_ids_all, motion_ids=torch.tensor(init_ids, device=env.device, dtype=torch.long))
    if active_mask is not None:
        active_mask[:] = torch.tensor(init_active, device=env.device, dtype=torch.bool)
    if metrics is not None:
        metrics.reset(env_ids_all)
    obs = _refresh_obs(env)

    write_header = (mode == "w") or (not out_csv.exists()) or (out_csv.stat().st_size == 0)
    with open(out_csv, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        # Main rollout loop.
        while processed_total < num_motions:
            if normalizer is not None:
                obs_in = normalizer.normalize(obs.detach())
            else:
                obs_in = obs.detach()
            actions = _call_policy(policy, obs_in, hist_encoding=True)
            step_out = env.step(actions.detach())
            obs = step_out[0]

            if len(buf) >= flush_rows:
                for row in buf:
                    w.writerow({k: row.get(k) for k in fieldnames})
                buf.clear()

            if (log_every > 0) and (processed_new > 0) and ((processed_new % log_every) == 0):
                elapsed = time.perf_counter() - t0
                rate = float(processed_new / max(1e-9, elapsed))
                print(
                    f"[{_now()}] processed={processed_total}/{num_motions} (+{processed_new} since resume, {rate:.1f} motions/s)",
                    flush=True,
                )

        # Flush tail.
        for row in buf:
            w.writerow({k: row.get(k) for k in fieldnames})

    elapsed = time.perf_counter() - t0
    print(
        f"[done] out_csv={out_csv} processed={processed_total}/{num_motions} (+{processed_total-processed_at_start}) elapsed_s={elapsed:.1f}",
        flush=True,
    )
    _maybe_write_summary(out_csv, args)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate legged_gym (IsaacGym) policies on a subset of motions; output per-motion CSV.")
    ap.add_argument("--motion_yaml", type=str, required=True, help="Motion config YAML (root_path + motions[].file)")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    ap.add_argument("--summary_json", type=str, default="", help="Summary report path (default: <out_csv>.summary.json)")
    ap.add_argument("--no_summary", action="store_true", help="Disable summary report generation")

    ap.add_argument("--motion_ids", type=str, default="", help="Subset of YAML motions by indices (e.g. '0,3,10-20')")
    ap.add_argument("--max_motions", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--skip_motions", type=int, default=0, help="Skip the first K selected motions (useful with --append to resume)")

    ap.add_argument("--task", type=str, default="g1_priv_mimic")
    ap.add_argument("--proj_name", type=str, default="g1_priv_mimic")
    ap.add_argument("--exptid", type=str, default="", help="Alias of --resumeid (matches play.py)")
    ap.add_argument("--resumeid", type=str, default="")
    ap.add_argument("--checkpoint", type=int, default=-1)
    ap.add_argument("--policy_type", type=str, default="runner", choices=["runner", "onnx"])
    ap.add_argument("--onnx_path", type=str, default="")
    ap.add_argument(
        "--runner_backend",
        type=str,
        default="onnx",
        choices=["onnx", "torch"],
        help="When --policy_type=runner: export ckpt to ONNX and run onnxruntime (onnx) or run torch directly (torch).",
    )
    ap.add_argument("--onnx_cache_dir", type=str, default="", help="Cache dir for exported ONNX (default: /tmp/codex-<mmdd>-gym-pt-to-onnx/)")
    ap.add_argument("--onnx_opset", type=int, default=11)
    ap.add_argument("--onnx_include_normalizer", action="store_true", help="Bake observation normalizer into exported ONNX (default: auto)")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--no_rand", action="store_true")
    ap.add_argument("--episode_length_s", type=float, default=120.0)
    ap.add_argument("--log_stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes (IsaacGym per-process sharding)")
    ap.add_argument("--mp_start_method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--no_merge", action="store_true", help="Do not merge worker CSVs; keep per-worker outputs in /tmp")
    ap.add_argument(
        "--queue_eval",
        action="store_true",
        help="High-throughput mode: run many envs in one process and automatically assign new motions on reset (best for GPU).",
    )
    ap.add_argument(
        "--queue_metrics",
        type=str,
        default="fast",
        choices=["fast", "final", "mean"],
        help="In --queue_eval: 'fast' skips metrics; 'final' logs final-step errors; 'mean' logs full-episode means (uses --log_stride).",
    )
    ap.add_argument("--queue_flush_rows", type=int, default=4096, help="In --queue_eval: CSV flush batch size")
    ap.add_argument("--queue_log_every", type=int, default=20000, help="In --queue_eval: print progress every N motions (0 disables)")
    args, _unknown = ap.parse_known_args()
    if (not str(args.resumeid).strip()) and str(getattr(args, "exptid", "")).strip():
        args.resumeid = str(args.exptid)

    if bool(args.queue_eval):
        _run_eval_queue_mode(args)
        return

    if int(args.workers) <= 1:
        _run_eval_single_process(args)
        return

    if bool(args.append) or int(args.skip_motions) > 0:
        raise ValueError("--append/--skip_motions are not supported with --workers > 1 (use output sharding instead).")
    if int(args.num_shards) != 1 or int(args.shard_idx) != 0:
        raise ValueError("With --workers > 1, do not set --num_shards/--shard_idx (they are controlled by workers).")

    if str(args.device).startswith("cuda") and int(args.workers) > 1:
        print("[warn] --workers > 1 on a single GPU can be unstable in IsaacGym; prefer one process per GPU.", flush=True)

    import multiprocessing as mp

    ctx = mp.get_context(str(args.mp_start_method))
    tmp_dir = _default_tmp_dir()
    # Avoid collisions between different runs on the same day.
    run_key = hashlib.sha1(repr(sorted(vars(args).items())).encode("utf-8")).hexdigest()[:10]
    run_dir = tmp_dir / f"mp_{run_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    args_dict = dict(vars(args))
    args_dict["append"] = False
    args_dict["skip_motions"] = 0
    args_dict["workers"] = 1

    jobs: list[tuple[dict[str, Any], int, int, str]] = []
    worker_csvs: list[Path] = []
    for wi in range(int(args.workers)):
        p = run_dir / f"worker_{wi}_of_{int(args.workers)}.csv"
        worker_csvs.append(p)
        jobs.append((args_dict, wi, int(args.workers), str(p)))

    with ctx.Pool(processes=int(args.workers)) as pool:
        pool.starmap(_worker_entry, jobs)

    if bool(args.no_merge):
        print(f"[done] worker_csvs={[str(p) for p in worker_csvs]}", flush=True)
        return

    out_csv = Path(args.out_csv).expanduser().resolve()
    _merge_csvs(inputs=worker_csvs, out_csv=out_csv)
    print(f"[done] out_csv={out_csv} (merged {len(worker_csvs)} workers)", flush=True)
    _maybe_write_summary(out_csv, args)


if __name__ == "__main__":
    main()
