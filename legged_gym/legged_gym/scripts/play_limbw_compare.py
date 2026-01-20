#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import json
import math
import os
import random
import re
import sys
from typing import List, Optional, Sequence

import faulthandler
import numpy as np
from termcolor import cprint
from tqdm import tqdm

# Isaac Gym requires importing isaacgym modules before torch. Some environments preload torch
# (e.g., via sitecustomize/usercustomize). If that happened, drop torch from sys.modules so
# isaacgym can import cleanly, then import torch afterwards.
if "torch" in sys.modules:
    for _k in list(sys.modules.keys()):
        if _k == "torch" or _k.startswith("torch."):
            del sys.modules[_k]

from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.gym_utils import get_args, task_registry


DEFAULT_LIMBW_CASES = [
    [1, 1, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
]


def _pop_argv_value(flag: str) -> Optional[str]:
    for i, a in enumerate(list(sys.argv)):
        if a == flag and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            return val
        if a.startswith(flag + "="):
            val = a.split("=", 1)[1]
            del sys.argv[i]
            return val
    return None


def _parse_limbw_cases_json(raw: Optional[str]) -> List[List[float]]:
    if raw is None or str(raw).strip() == "":
        return [list(map(float, x)) for x in DEFAULT_LIMBW_CASES]
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse --limbw_cases_json as JSON: {e}") from e
    if not isinstance(data, list) or not data:
        raise ValueError("--limbw_cases_json must be a non-empty JSON list of 4-element lists")
    out: List[List[float]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError(f"--limbw_cases_json[{idx}] must be a 4-element list, got: {item}")
        vec = [float(x) for x in item]
        for v in vec:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"limb weight values must be in [0,1], got {vec}")
        out.append(vec)
    return out


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_index_spec(spec: str, n: int):
    spec = (spec or "").strip()
    if not spec:
        return list(range(n))
    indices = []
    parts = [p for p in re.split(r"\s*,\s*", spec) if p]
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            if a_s == "" or b_s == "":
                raise ValueError(f"Invalid range token '{part}' in record_motion_ids='{spec}'")
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
                    raise IndexError(f"record_motion_ids index {i} out of range [0, {n-1}]")
                indices.append(i)
        else:
            i = int(part)
            if i < 0:
                i = n + i
            if i < 0 or i >= n:
                raise IndexError(f"record_motion_ids index {i} out of range [0, {n-1}]")
            indices.append(i)
    seen = set()
    out = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _refresh_obs(env):
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


def _overlay_text(img, text: str, disable: bool):
    if disable:
        return img
    try:
        import cv2
    except Exception:
        return img

    if img is None:
        return img
    frame = img
    if hasattr(frame, "dtype") and frame.dtype != "uint8":
        frame = frame.astype("uint8")
    if len(frame.shape) == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    if len(frame.shape) != 3 or frame.shape[-1] != 3:
        return img

    bgr = frame[..., ::-1].copy()
    h, w = bgr.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    margin = 10
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x0, y0 = margin, margin + th

    pad = 6
    x1 = min(w - 1, x0 + tw + pad * 2)
    y1 = min(h - 1, y0 + baseline + pad)
    cv2.rectangle(bgr, (x0 - pad, y0 - th - pad), (x1, y1), (0, 0, 0), -1)
    cv2.putText(bgr, text, (x0, y0), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    rgb = bgr[..., ::-1]
    return rgb


def _set_play_cfg(env_cfg, args):
    env_cfg.env.num_envs = 2 if not args.num_envs else args.num_envs
    env_cfg.env.debug_viz = True
    env_cfg.env.episode_length_s = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False

    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False

    if hasattr(env_cfg.env, "obs_type") and env_cfg.env.obs_type == "student_future":
        env_cfg.env.evaluation_mode = False
        env_cfg.env.force_full_masking = False


def play_limbw_compare(args, limbw_cases: Sequence[Sequence[float]], base_seed: int):
    faulthandler.enable()

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", str(args.proj_name), str(args.exptid))
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _set_play_cfg(env_cfg, args)

    args.record_video = True
    env_cfg.env.record_video = True
    env_cfg.env.rand_reset = False

    if args.record_video:
        env_cfg.env.viz_keypoints = True
        env_cfg.env.viz_keypoints_gt_local = True

    if args.num_envs is None:
        env_cfg.env.num_envs = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not bool(getattr(env.cfg.env, "use_limb_weights", False)):
        raise RuntimeError(
            f"Task '{args.task}' does not enable limb weights. Use `g1_priv_mimic_limbw` or `g1_stu_mimic_limbw`."
        )

    obs = env.get_observations()
    record_video_enabled = bool(args.record_video) and getattr(env, "_rendering_camera_handles", None) is not None
    if args.record_video and not record_video_enabled:
        raise RuntimeError(
            "Video recording requested, but camera sensors are unavailable (create_camera_sensor returned -1). "
            "Try running with a valid graphics context (X/Wayland or `xvfb-run`) and/or set `--graphics_device_id` "
            "to your GPU."
        )

    if_normalize = env_cfg.env.normalize_obs
    if not args.use_jit:
        train_cfg.runner.resume = True
    else:
        train_cfg.runner.resume = False

    ppo_runner, _ = task_registry.make_alg_runner(log_root=log_root, env=env, name=args.task, args=args, train_cfg=train_cfg)

    if args.use_jit and args.jit_path is not None:
        cprint(f"Loading jit for policy: {args.jit_path}", "green")
        policy_jit = torch.jit.load(args.jit_path, map_location=env.device)
        policy = None
        normalizer = None
    else:
        policy_jit = None
        policy = ppo_runner.get_inference_policy(device=env.device)
        if if_normalize:
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
                cprint("Normalizer found", "green")
            except Exception:
                cprint("No normalizer found", "yellow")
                normalizer = None
        else:
            normalizer = None

    import imageio

    num_motions = int(env._motion_lib.num_motions())
    if num_motions <= 0:
        raise RuntimeError("No motions loaded; cannot record video.")

    if getattr(args, "record_motion_ids", ""):
        motion_ids_to_record = _parse_index_spec(args.record_motion_ids, num_motions)
        if getattr(args, "random", False):
            rng = random.Random(int(getattr(args, "record_seed", 0) or base_seed))
            rng.shuffle(motion_ids_to_record)
    else:
        motion_ids_to_record = list(range(num_motions))
        if getattr(args, "random", False):
            rng = random.Random(int(getattr(args, "record_seed", 0) or base_seed))
            rng.shuffle(motion_ids_to_record)
        nrec = int(getattr(args, "record_num_motions", 1) or 1)
        if nrec <= 0:
            raise ValueError(f"--record_num_motions must be >= 1, got {nrec}")
        motion_ids_to_record = motion_ids_to_record[: min(nrec, len(motion_ids_to_record))]

    motion_names = None
    try:
        motion_names = env._motion_lib.get_motion_names()
    except Exception:
        motion_names = None

    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "videos_retarget", str(args.exptid))
    os.makedirs(out_dir, exist_ok=True)

    video_name = getattr(args, "record_video_name", "") or f"{args.proj_name}-{args.exptid}-limbw_compare.mp4"
    if not video_name.endswith(".mp4"):
        video_name += ".mp4"
    video_path = os.path.join(out_dir, video_name)
    writer = imageio.get_writer(video_path, fps=int(1 / env.dt))
    cprint(f"Recording video to {video_path}", "green")

    overlay_disable = bool(getattr(args, "record_no_overlay", False))
    env.enable_viewer_sync = True

    nsamp = len(motion_ids_to_record)
    nc = len(limbw_cases)
    for sample_idx, motion_id in enumerate(motion_ids_to_record):
        motion_name = None
        if motion_names is not None and 0 <= int(motion_id) < len(motion_names):
            motion_name = os.path.splitext(os.path.basename(motion_names[int(motion_id)]))[0]
        if not motion_name:
            motion_name = f"motion_{int(motion_id):06d}"

        for case_idx, limbw in enumerate(limbw_cases):
            seed = int(base_seed) + int(motion_id)
            _set_all_seeds(seed)

            env.set_limb_weights_fixed(limbw)

            motion_id_tensor = torch.full((env.num_envs,), int(motion_id), device=env.device, dtype=torch.long)
            env.reset_idx(torch.arange(env.num_envs, device=env.device), motion_ids=motion_id_tensor)
            obs = _refresh_obs(env)

            motion_len_s = env._motion_lib.get_motion_length(motion_id_tensor[:1]).item()
            steps = max(1, int(math.ceil(motion_len_s / env.dt)))

            wtxt = "[" + ",".join(f"{float(x):.2f}" for x in limbw) + "]"
            overlay = (
                f"samp {sample_idx+1}/{nsamp}  id={int(motion_id)}  {motion_name}  "
                f"case {case_idx+1}/{nc}  w={wtxt}"
            )

            for _ in tqdm(range(steps), desc=f"[limbw] samp {sample_idx+1}/{nsamp} id={motion_id} case {case_idx+1}/{nc}"):
                if policy_jit is not None:
                    actions = policy_jit(obs.detach())
                else:
                    if if_normalize and normalizer is not None:
                        normalized_obs = normalizer.normalize(obs.detach())
                    else:
                        normalized_obs = obs.detach()
                    actions = policy(normalized_obs, hist_encoding=True)

                if "AMP" in env.__class__.__name__:
                    obs, _, _, _, _, _, _ = env.step(actions.detach())
                else:
                    obs, _, _, _, _ = env.step(actions.detach())

                imgs = env.render_record(mode="rgb_array")
                if imgs is None:
                    continue
                frame = _overlay_text(imgs[0], overlay, overlay_disable)
                writer.append_data(frame)

    writer.close()
    cprint(f"Done: {video_path}", "green")


if __name__ == "__main__":
    limbw_cases_json = _pop_argv_value("--limbw_cases_json")
    limbw_cases = _parse_limbw_cases_json(limbw_cases_json)

    args = get_args()
    base_seed = int(getattr(args, "seed", 0) or 0)
    play_limbw_compare(args, limbw_cases=limbw_cases, base_seed=base_seed)
