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
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


def _install_reset_reason_trace(env) -> None:
    orig = env.check_termination
    env._exec_eval_last_reason = [""] * int(env.num_envs)  # type: ignore[attr-defined]
    env._exec_eval_last_time_s = [float("nan")] * int(env.num_envs)  # type: ignore[attr-defined]

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

    device = str(device)
    device_id = 0
    if device.startswith("cuda:"):
        try:
            device_id = int(device.split(":", 1)[1])
        except Exception:
            device_id = 0

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate legged_gym (IsaacGym) policies on a subset of motions; output per-motion CSV.")
    ap.add_argument("--motion_yaml", type=str, required=True, help="Motion config YAML (root_path + motions[].file)")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")

    ap.add_argument("--motion_ids", type=str, default="", help="Subset of YAML motions by indices (e.g. '0,3,10-20')")
    ap.add_argument("--max_motions", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--skip_motions", type=int, default=0, help="Skip the first K selected motions (useful with --append to resume)")

    ap.add_argument("--task", type=str, default="g1_priv_mimic")
    ap.add_argument("--proj_name", type=str, default="g1_priv_mimic")
    ap.add_argument("--resumeid", type=str, default="")
    ap.add_argument("--checkpoint", type=int, default=-1)
    ap.add_argument("--policy_type", type=str, default="runner", choices=["runner", "onnx"])
    ap.add_argument("--onnx_path", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--no_rand", action="store_true")
    ap.add_argument("--episode_length_s", type=float, default=120.0)
    ap.add_argument("--log_stride", type=int, default=1)
    args, _unknown = ap.parse_known_args()

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
    # In this repo, `--no_rand` also rewrites cfg_train.runner.policy_class_name to non-mimic defaults.
    # We control eval determinism via env_cfg overrides below.
    try:
        gym_args.no_rand = False
    except Exception:
        pass

    mmdd = time.strftime("%m%d", time.localtime())
    tmp_dir = Path(f"/tmp/codex-{mmdd}-gym-exec-eval")
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

        def infer_actions(obs_in):
            return _call_policy(policy, obs_in, hist_encoding=True)

    else:
        if not str(args.onnx_path).strip():
            raise ValueError("--onnx_path is required when --policy_type=onnx")
        infer_onnx = _make_onnx_policy(str(args.onnx_path), device=str(args.device))

        def infer_actions(obs_in):
            return infer_onnx(obs_in)

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
        "err_root_pos_l2_mean",
        "err_root_rot_deg_mean",
        "err_dof_pos_l2_mean",
        "err_dof_vel_l2_mean",
        "err_keybody_pos_l1_mean",
        "error",
    ]

    log_stride = max(1, int(args.log_stride))
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
                env_ids = torch.arange(env.num_envs, device=env.device)
                motion_id_tensor = torch.full((env.num_envs,), int(local_id), device=env.device, dtype=torch.long)
                env.reset_idx(env_ids, motion_ids=motion_id_tensor)
                obs = _refresh_obs(env)

                motion_len_s = float(env._motion_lib.get_motion_length(motion_id_tensor[:1]).item())
                max_steps = max(1, int(math.ceil(motion_len_s / float(env.dt))))

                n_log = 0
                sum_root_pos = 0.0
                sum_root_rot_deg = 0.0
                sum_dof_pos = 0.0
                sum_dof_vel = 0.0
                sum_keybody = 0.0
                done_reason = ""
                done_time_s = float("nan")

                for step in range(max_steps):
                    if normalizer is not None:
                        obs_in = normalizer.normalize(obs.detach())
                    else:
                        obs_in = obs.detach()
                    actions = infer_actions(obs_in)
                    step_out = env.step(actions.detach())
                    obs = step_out[0]
                    dones = step_out[3]

                    if (step % log_stride) == 0:
                        log = env.get_episode_log(env_ids=0)
                        if "err_root_pos_l2" in log:
                            sum_root_pos += float(log["err_root_pos_l2"])
                        if "err_root_rot_rad" in log:
                            sum_root_rot_deg += float(log["err_root_rot_rad"]) * 180.0 / math.pi
                        if "err_dof_pos_l2" in log:
                            sum_dof_pos += float(log["err_dof_pos_l2"])
                        if "err_dof_vel_l2" in log:
                            sum_dof_vel += float(log["err_dof_vel_l2"])
                        if "err_keybody_pos_l1" in log:
                            sum_keybody += float(log["err_keybody_pos_l1"])
                        n_log += 1

                    if bool(dones[0].item()):
                        done_reason = str(getattr(env, "_exec_eval_last_reason", [""])[0])  # type: ignore[attr-defined]
                        done_time_s = float(getattr(env, "_exec_eval_last_time_s", [float("nan")])[0])  # type: ignore[attr-defined]
                        break

                if not done_reason:
                    done_reason = "unknown"
                progress = float(done_time_s / motion_len_s) if (motion_len_s > 0 and math.isfinite(done_time_s)) else float("nan")
                status = "ok" if done_reason == "motion_end" else "fail"

                row.update(
                    {
                        "status": status,
                        "done_reason": done_reason,
                        "done_time_s": float(done_time_s),
                        "motion_len_s": float(motion_len_s),
                        "progress": float(progress),
                        "err_root_pos_l2_mean": float(sum_root_pos / max(1, n_log)),
                        "err_root_rot_deg_mean": float(sum_root_rot_deg / max(1, n_log)),
                        "err_dof_pos_l2_mean": float(sum_dof_pos / max(1, n_log)),
                        "err_dof_vel_l2_mean": float(sum_dof_vel / max(1, n_log)),
                        "err_keybody_pos_l1_mean": float(sum_keybody / max(1, n_log)),
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
                        "err_root_pos_l2_mean": float("nan"),
                        "err_root_rot_deg_mean": float("nan"),
                        "err_dof_pos_l2_mean": float("nan"),
                        "err_dof_vel_l2_mean": float("nan"),
                        "err_keybody_pos_l1_mean": float("nan"),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

            w.writerow({k: row.get(k) for k in fieldnames})
            if ((local_id + 1) % 10) == 0:
                print(f"[{_now()}] processed={local_id+1}/{len(motions)}", flush=True)

    print(f"[done] out_csv={out_csv}")


if __name__ == "__main__":
    main()

