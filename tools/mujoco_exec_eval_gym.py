#!/usr/bin/env python3
"""
Gym/Gymnasium-based TWIST2 sim2sim evaluator (policy + MuJoCo) for motion YAML configs.

This is a "gym version" of `tools/mujoco_exec_eval.py`: instead of rolling out the policy via a
custom for-loop, it wraps the same MuJoCo + observation logic into a Gym-compatible environment
(`Twist2MujocoGymEnv`) and drives rollouts via `env.reset()` / `env.step()`.

It keeps the same CLI contract and CSV metrics as `tools/mujoco_exec_eval.py`, including:
- reading MotionLib-style YAMLs (root_path + motions[].file)
- loading .onnx policies (and exporting .pt/.pth ActorCriticMimic checkpoints to cached ONNX)
- per-motion metrics (root pos/rot error, joint error, FK relative error, failure time)

Example:
  python tools/mujoco_exec_eval_gym.py --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \\
      --out_csv ./outputs/twist2_exec_metrics_gym.csv --policy_path assets/ckpts/twist2_1017_20k.onnx \\
      --xml_path assets/g1/g1_sim2sim_29dof.xml --disable_termination --body_set joint_bodies29 --workers 16
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import multiprocessing as mp

# Import the original implementation as a module (tools/ isn't a package by default).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import mujoco_exec_eval as base

try:
    import gymnasium as gym

    _GYMNASIUM = True
except Exception:  # pragma: no cover
    import gym  # type: ignore[no-redef]

    _GYMNASIUM = False

try:
    from gymnasium import spaces  # type: ignore[assignment]
except Exception:  # pragma: no cover
    from gym import spaces  # type: ignore[no-redef]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class Twist2MujocoGymEnv(gym.Env):
    """
    Gym wrapper around `base.Twist2SimRunner` that:
    - stores a per-episode mimic target sequence (T,35)
    - returns policy observations matching TWIST2 sim2sim deployment
    - applies PD control for a single policy step per `env.step(action)`

    Notes:
    - This env is intended for headless rollouts (no rendering).
    - It supports both obs modes used by TWIST2 evaluation:
      `student_future` and `teacher_priv_mimic` (auto supported via sim runner).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: base.Twist2SimConfig) -> None:
        super().__init__()
        self.sim = base.Twist2SimRunner(cfg)

        self._t: int = 0
        self._mimic: np.ndarray | None = None
        self._motion: base.Motion | None = None
        self._teacher_ref: dict[str, np.ndarray] | None = None

        self._terminated: bool = False
        self._fail_detected: bool = False
        self._fail_reason: str = ""
        self._fail_step: int = -1

        self._z_min: float = 0.55
        self._angle_max_rad: float = float(60.0) * math.pi / 180.0
        self._disable_termination: bool = True

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(int(self.sim.num_actions),),
            dtype=np.float32,
        )
        obs_dim = int(self.sim.total_obs_size) if self.sim.obs_mode != "teacher_priv_mimic" else 1734
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _pelvis_z(self) -> float:
        if getattr(self.sim, "_pelvis_body_id", None) is not None:
            return float(self.sim.data.xpos[self.sim._pelvis_body_id][2])
        return float(self.sim.data.qpos[2])

    def _roll_pitch(self) -> tuple[float, float]:
        r, p, _y = base._euler_from_quat_wxyz(self.sim.data.qpos[3:7].reshape(1, 4))
        return float(r[0]), float(p[0])

    def _build_obs_at(self, t: int) -> np.ndarray:
        assert self._mimic is not None
        if self.sim.obs_mode == "teacher_priv_mimic":
            if self._teacher_ref is None:
                raise RuntimeError("teacher_priv_mimic requires teacher_ref (did you pass motion to reset?)")
            return self.sim._build_obs_teacher_priv_mimic(self._teacher_ref, int(t))
        return self.sim._build_obs(self._mimic[int(t)])

    def _maybe_mark_failure(self, *, frame_idx: int) -> None:
        if not np.all(np.isfinite(self.sim.data.qpos)) or not np.all(np.isfinite(self.sim.data.qvel)) or not np.all(np.isfinite(self.sim.data.ctrl)):
            self._terminated = True
            if not self._fail_detected:
                self._fail_detected = True
                self._fail_reason = "nan_or_inf"
                self._fail_step = int(frame_idx)
            return

        reason = ""
        if self._pelvis_z() < float(self._z_min):
            reason = "fell_pelvis_z"
        else:
            roll, pitch = self._roll_pitch()
            if abs(roll) > float(self._angle_max_rad) or abs(pitch) > float(self._angle_max_rad):
                reason = "fell_angle"

        if reason:
            if not self._fail_detected:
                self._fail_detected = True
                self._fail_reason = str(reason)
                self._fail_step = int(frame_idx)
            if not bool(self._disable_termination):
                self._terminated = True

    def _info(self) -> dict[str, Any]:
        return {
            "t": int(self._t),
            "qpos": np.asarray(self.sim.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(self.sim.data.qvel, dtype=np.float32).copy(),
            "torque": np.asarray(self.sim.data.ctrl, dtype=np.float32).copy(),
            "pelvis_z": float(self._pelvis_z()),
            "roll": float(self._roll_pitch()[0]),
            "pitch": float(self._roll_pitch()[1]),
            "terminated": bool(self._terminated),
            "fail_detected": bool(self._fail_detected),
            "fail_reason": str(self._fail_reason),
            "fail_step": int(self._fail_step),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        del seed
        options = {} if options is None else dict(options)

        mimic_target = np.asarray(options["mimic_target"], dtype=np.float32)
        if mimic_target.ndim != 2 or mimic_target.shape[1] != 35:
            raise ValueError(f"mimic_target must be (T,35), got {mimic_target.shape}")
        if mimic_target.shape[0] < 2:
            raise ValueError("mimic_target must have at least 2 frames")
        if not np.all(np.isfinite(mimic_target)):
            raise ValueError("mimic_target contains NaN/Inf")

        self._mimic = mimic_target
        self._motion = options.get("motion", None)
        self._t = 0

        self._terminated = False
        self._fail_detected = False
        self._fail_reason = ""
        self._fail_step = -1

        self._z_min = float(options.get("z_min", 0.55))
        self._angle_max_rad = float(options.get("angle_max_deg", 60.0)) * math.pi / 180.0
        self._disable_termination = bool(options.get("disable_termination", True))

        sim_seed = int(options.get("sim_seed", 0))
        self.sim.reset(sim_seed=sim_seed)

        if self.sim.obs_mode == "teacher_priv_mimic":
            if self._motion is None:
                raise ValueError("teacher_priv_mimic requires passing `motion` in reset(options=...)")
            self._teacher_ref = self.sim._prepare_teacher_ref(
                self._motion,
                publish_hz=float(self.sim.policy_frequency),
                future_step=int(options.get("future_step", 1)),
                idle_s=float(options.get("idle_s", 0.5)),
                tail_s=float(options.get("tail_s", 0.5)),
                transition_s=float(options.get("transition_s", 0.4)),
                loop=bool(options.get("loop", False)),
            )
        else:
            self._teacher_ref = None

        obs0 = self._build_obs_at(0)
        info0 = self._info()
        if _GYMNASIUM:
            return obs0, info0
        return obs0, info0

    def step(self, action):  # type: ignore[override]
        if self._mimic is None:
            raise RuntimeError("env.step() called before reset()")
        if self._terminated:
            raise RuntimeError("env.step() called after termination")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != int(self.sim.num_actions):
            raise ValueError(f"action must be ({self.sim.num_actions},), got {action.shape}")

        raw_action = action.astype(np.float32, copy=True)
        self.sim._last_action = raw_action  # matches deploy_real behavior (store before clip)

        clipped = np.clip(raw_action, -10.0, 10.0).astype(np.float32, copy=False)
        pd_target = clipped * self.sim.action_scale + self.sim.default_dof_pos

        for _ in range(int(self.sim.sim_decimation)):
            dof_pos = self.sim.data.qpos[7 : 7 + self.sim.num_actions]
            dof_vel = self.sim.data.qvel[6 : 6 + self.sim.num_actions]
            tau = (pd_target - dof_pos) * self.sim.stiffness - dof_vel * self.sim.damping
            tau = np.clip(tau, -self.sim.torque_limits, self.sim.torque_limits)
            self.sim.data.ctrl[:] = tau
            base.mujoco.mj_step(self.sim.model, self.sim.data)

        # Advance to next policy frame (frame index, not sim substep index).
        self._t += 1
        self._maybe_mark_failure(frame_idx=self._t)

        # Episode truncation: last valid observation is at t=T-1; there are only (T-1) actions.
        truncated = bool(self._t >= int(self._mimic.shape[0]) - 1)
        terminated = bool(self._terminated)

        obs = self._build_obs_at(self._t) if not terminated else self._build_obs_at(min(self._t, int(self._mimic.shape[0]) - 1))
        info = self._info()
        reward = 0.0

        if _GYMNASIUM:
            return obs, float(reward), bool(terminated), bool(truncated), info
        done = bool(terminated or truncated)
        return obs, float(reward), done, info


class Twist2GymSimRunner:
    """
    Adapter that exposes the same `.run()` contract as `base.Twist2SimRunner`,
    but drives the rollout through a Gym environment.

    This keeps `base.MotionEvaluator` unchanged.
    """

    def __init__(self, cfg: base.Twist2SimConfig) -> None:
        self.env = Twist2MujocoGymEnv(cfg)
        self.policy_frequency = float(self.env.sim.policy_frequency)
        self.default_dof_pos = self.env.sim.default_dof_pos

    def run(
        self,
        mimic_target: np.ndarray,
        *,
        motion: base.Motion | None = None,
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
        mimic_target = np.asarray(mimic_target, dtype=np.float32)
        T = int(mimic_target.shape[0])

        obs, info0 = self.env.reset(
            options={
                "mimic_target": mimic_target,
                "motion": motion,
                "loop": bool(loop),
                "future_step": int(future_step),
                "idle_s": float(idle_s),
                "tail_s": float(tail_s),
                "transition_s": float(transition_s),
                "sim_seed": int(sim_seed),
                "z_min": float(z_min),
                "angle_max_deg": float(angle_max_deg),
                "disable_termination": bool(disable_termination),
            }
        )

        qpos_list = [np.asarray(info0["qpos"], dtype=np.float32)]
        qvel_list = [np.asarray(info0["qvel"], dtype=np.float32)]
        torque_list = [np.asarray(info0["torque"], dtype=np.float32)]
        pelvis_z_list = [float(info0["pelvis_z"])]
        roll_list = [float(info0["roll"])]
        pitch_list = [float(info0["pitch"])]

        terminated = False
        fail_detected = bool(info0.get("fail_detected", False))
        fail_reason = str(info0.get("fail_reason", ""))
        fail_step = int(info0.get("fail_step", -1))

        # The episode has (T-1) actions; stop once truncated or terminated.
        for _ in range(T - 1):
            action = self.env.sim.policy(obs).reshape(-1).astype(np.float32, copy=False)
            step_ret = self.env.step(action)
            if _GYMNASIUM:
                obs, _rew, term, trunc, info = step_ret
                done = bool(term or trunc)
            else:
                obs, _rew, done, info = step_ret
                term = bool(info.get("terminated", False))
                trunc = bool(done and not term)

            qpos_list.append(np.asarray(info["qpos"], dtype=np.float32))
            qvel_list.append(np.asarray(info["qvel"], dtype=np.float32))
            torque_list.append(np.asarray(info["torque"], dtype=np.float32))
            pelvis_z_list.append(float(info["pelvis_z"]))
            roll_list.append(float(info["roll"]))
            pitch_list.append(float(info["pitch"]))

            if bool(info.get("fail_detected", False)) and not fail_detected:
                fail_detected = True
                fail_reason = str(info.get("fail_reason", ""))
                fail_step = int(info.get("fail_step", -1))

            if bool(term):
                terminated = True
                break
            if bool(done):
                break

        qpos = np.stack(qpos_list, axis=0).astype(np.float32, copy=False)
        qvel = np.stack(qvel_list, axis=0).astype(np.float32, copy=False)
        torque = np.stack(torque_list, axis=0).astype(np.float32, copy=False)

        T_exec = int(qpos.shape[0])
        mimic_used = mimic_target[:T_exec]

        return {
            "qpos": qpos,
            "qvel": qvel,
            "torque": torque,
            "mimic_target": mimic_used,
            "terminated": bool(terminated),
            "fail_detected": bool(fail_detected),
            "fail_reason": str(fail_reason),
            "fail_step": int(fail_step),
            "pelvis_z": np.asarray(pelvis_z_list, dtype=np.float32),
            "roll": np.asarray(roll_list, dtype=np.float32),
            "pitch": np.asarray(pitch_list, dtype=np.float32),
        }


_WORKER_SIM: Twist2GymSimRunner | None = None
_WORKER_EVAL: base.MotionEvaluator | None = None
_WORKER_ARGS: dict[str, Any] | None = None


def _worker_init(cfg: dict[str, Any]) -> None:
    global _WORKER_SIM, _WORKER_EVAL, _WORKER_ARGS
    _WORKER_ARGS = cfg
    _WORKER_SIM = Twist2GymSimRunner(
        base.Twist2SimConfig(
            xml_path=Path(cfg["xml_path"]).expanduser().resolve(),
            policy_path=Path(cfg["policy_path"]).expanduser().resolve(),
            device=str(cfg["device"]),
            policy_frequency=float(cfg["policy_frequency"]),
            sim_dt=float(cfg["sim_dt"]),
            smooth_body=float(cfg["smooth_body"]),
            obs_mode=str(cfg.get("obs_mode", "auto")),
        )
    )
    _WORKER_EVAL = base.MotionEvaluator(sim=_WORKER_SIM, xml_path=Path(cfg["xml_path"]), body_set=str(cfg["body_set"]))


def _worker_eval_entry(entry: base.MotionEntry) -> dict[str, Any]:
    if _WORKER_EVAL is None or _WORKER_ARGS is None:
        raise RuntimeError("Worker not initialized")
    cfg = _WORKER_ARGS

    try:
        motion = base.load_motion_pkl_or_npz(Path(entry.file_abs), quat_order=str(cfg["quat_order"]))
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
        res = base.EvalResult(
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
        description="Gym-based evaluator for TWIST2 MuJoCo sim2sim execution over a training motion YAML; outputs per-motion CSV metrics."
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

    if base.torch is None:
        raise RuntimeError("torch is required (pose utils dependency)")
    if base.mujoco is None:
        raise RuntimeError("mujoco is required")
    if gym is None or spaces is None:  # pragma: no cover
        raise RuntimeError("gym/gymnasium is required")

    policy_path = Path(args.policy_path).expanduser().resolve()
    if policy_path.suffix.lower() in (".pt", ".pth"):
        cache_dir = base._default_pt_to_onnx_cache_dir() if not str(args.onnx_cache_dir).strip() else Path(args.onnx_cache_dir).expanduser().resolve()
        exported = base.export_actor_critic_mimic_ckpt_to_onnx(policy_path, out_dir=cache_dir)
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

    entries = list(
        base.iter_motion_config_files(
            args.motion_yaml,
            motion_ids=str(args.motion_ids),
            max_motions=int(args.max_motions),
            shuffle=bool(args.shuffle),
            shuffle_seed=int(args.shuffle_seed),
            shard_idx=int(args.shard_idx),
            num_shards=int(args.num_shards),
        )
    )
    workers = int(max(1, args.workers))
    print(f"[{_now()}] motions={len(entries)} workers={workers} obs_mode={args.obs_mode}", flush=True)

    worker_cfg: dict[str, Any] = {
        "xml_path": str(args.xml_path),
        "policy_path": str(args.policy_path),
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

    write_header = (mode == "w") or (not out_csv.exists()) or (out_csv.stat().st_size == 0)
    n_total = 0
    n_ok = 0
    n_err = 0
    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        if workers == 1:
            sim = Twist2GymSimRunner(
                base.Twist2SimConfig(
                    xml_path=Path(args.xml_path).expanduser().resolve(),
                    policy_path=Path(args.policy_path).expanduser().resolve(),
                    device=str(args.device),
                    policy_frequency=float(args.policy_frequency),
                    sim_dt=0.001,
                    smooth_body=float(args.smooth_body),
                    obs_mode=str(args.obs_mode),
                )
            )
            evaluator = base.MotionEvaluator(sim=sim, xml_path=Path(args.xml_path), body_set=str(args.body_set))
            for entry in entries:
                n_total += 1
                try:
                    motion = base.load_motion_pkl_or_npz(entry.file_abs, quat_order=str(args.quat_order))
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
                    res = base.EvalResult(
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
                w.writerow({k: res.to_flat_dict().get(k) for k in fieldnames})
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

