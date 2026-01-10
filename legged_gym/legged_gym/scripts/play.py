# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry
import torch
import faulthandler
from tqdm import tqdm
from termcolor import cprint
import random
import re
import math
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.base.legged_robot import euler_from_quaternion

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def set_play_cfg(env_cfg):
    env_cfg.env.num_envs = 2#2 if not args.num_envs else args.num_envs
    env_cfg.env.debug_viz = True
    env_cfg.env.episode_length_s = 60
    # env_cfg.commands.resampling_time = 60
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
    
    # Set evaluation mode with full masking for student future policies
    if hasattr(env_cfg.env, 'obs_type') and env_cfg.env.obs_type == 'student_future':
        # env_cfg.env.evaluation_mode = True
        # env_cfg.env.force_full_masking = True
        env_cfg.env.evaluation_mode = False
        env_cfg.env.force_full_masking = False


def play(args):
    faulthandler.enable()
    if args.jit_path is not None:
        args.use_jit = True
        args.proj_name = "g1_stu_future_single"
        args.exptid = "g1_stu_future_single"

    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    set_play_cfg(env_cfg)

    env_cfg.env.record_video = args.record_video
    env_cfg.env.rand_reset = False

    # Recording usually wants a single env unless the user explicitly overrides it.
    if args.record_video and args.num_envs is None:
        env_cfg.env.num_envs = 1
    
    if_normalize = env_cfg.env.normalize_obs
    cprint(f"if_normalize: {if_normalize}", "green")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    record_video_enabled = bool(args.record_video) and getattr(env, "_rendering_camera_handles", None) is not None
    if args.record_video and not record_video_enabled:
        cprint(
            "Video recording requested, but camera sensors are unavailable (create_camera_sensor returned -1). "
            "Try running with a valid graphics context (X/Wayland or `xvfb-run`) and/or set `--graphics_device_id` "
            "to your GPU. Continuing without recording.",
            "red",
        )

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
        # Build a valid observation after a manual reset_idx without advancing the sim.
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

    def _overlay_text(img, text: str):
        if getattr(args, "record_no_overlay", False):
            return img
        try:
            import cv2
        except Exception:
            return img

        if img is None:
            return img
        frame = img
        if hasattr(frame, "dtype"):
            if frame.dtype != "uint8":
                frame = frame.astype("uint8")
        if len(frame.shape) == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if len(frame.shape) != 3 or frame.shape[-1] != 3:
            return img

        # cv2 uses BGR
        bgr = frame[..., ::-1].copy()
        h, w = bgr.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        margin = 10
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x0, y0 = margin, margin + th

        # Background box for readability
        pad = 6
        x1 = min(w - 1, x0 + tw + pad * 2)
        y1 = min(h - 1, y0 + baseline + pad)
        cv2.rectangle(bgr, (x0 - pad, y0 - th - pad), (x1, y1), (0, 0, 0), -1)
        cv2.putText(bgr, text, (x0, y0), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        rgb = bgr[..., ::-1]
        return rgb

    # load policy
    if not args.use_jit:
        train_cfg.runner.resume = True
    else:
        train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)

    if args.use_jit and args.jit_path is not None:
        print("Loading jit for policy: ", args.jit_path)
        policy_jit = torch.jit.load(args.jit_path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        if if_normalize:
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
                print("Normalizer found")
            except:
                print("No normalizer found")
                normalizer = None

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    if args.record_log:
        import json
        from pathlib import Path
        run_name = log_pth.split("/")[-1]
        logs_dict = []
        dict_name = args.proj_name + "-" + args.exptid + ".json"
        record_log_dir = (args.record_log_dir or "").strip()
        if not record_log_dir:
            record_log_dir = f"../../logs/env_logs/{run_name}"
        Path(record_log_dir).mkdir(parents=True, exist_ok=True)
        dict_name = os.path.join(record_log_dir, dict_name)
        meta_name = dict_name.replace(".json", "_meta.json")

        stride = int(getattr(args, "record_log_stride", 1) or 1)
        stride = max(1, stride)
        env_id_to_log = int(getattr(args, "record_log_env_id", -1))
        if env_id_to_log < 0:
            env_id_to_log = int(getattr(env, "lookat_id", 0))
        env_id_to_log = max(0, min(env_id_to_log, env.num_envs - 1))

        meta = {
            "task": args.task,
            "proj_name": args.proj_name,
            "exptid": args.exptid,
            "log_root": log_pth,
            "motion_file": getattr(getattr(env.cfg, "motion", None), "motion_file", ""),
            "dt": float(getattr(env, "dt", 0.0)),
            "decimation": int(getattr(getattr(env.cfg, "control", None), "decimation", 0) or 0),
            "num_actions": int(getattr(env, "num_actions", 0) or 0),
            "num_dof": int(getattr(env, "num_dof", 0) or 0),
            "env_id": int(env_id_to_log),
            "stride": int(stride),
        }
        with open(meta_name, "w") as f:
            json.dump(meta, f, indent=2)
        
    
    if not (args.record_video or args.record_log):
        traj_length = 100*int(env.max_episode_length)
    else:
        traj_length = 1 * int(env.max_episode_length)
    
    # traj_length = 2000
    
    env_id = env.lookat_id

    if record_video_enabled:
        import imageio

        split_videos = bool(getattr(args, "record_split_videos", False))
        num_motions = int(env._motion_lib.num_motions())
        if num_motions <= 0:
            raise RuntimeError("No motions loaded; cannot record video.")

        if args.record_motion_ids:
            motion_ids_to_record = _parse_index_spec(args.record_motion_ids, num_motions)
        else:
            motion_ids_to_record = list(range(num_motions))
            if args.record_shuffle:
                rng = random.Random(int(args.record_seed))
                rng.shuffle(motion_ids_to_record)
            nrec = int(args.record_num_motions) if args.record_num_motions is not None else 1
            if nrec <= 0:
                raise ValueError(f"--record_num_motions must be >= 1, got {nrec}")
            if nrec > len(motion_ids_to_record):
                cprint(
                    f"--record_num_motions={nrec} > num_motions={len(motion_ids_to_record)}; recording all available motions.",
                    "yellow",
                )
                nrec = len(motion_ids_to_record)
            motion_ids_to_record = motion_ids_to_record[:nrec]

        motion_names = None
        try:
            motion_names = env._motion_lib.get_motion_names()
        except Exception:
            motion_names = None

        run_name = log_pth.split("/")[-1]
        out_dir = f"../../logs/videos_retarget/{run_name}"
        os.makedirs(out_dir, exist_ok=True)

        env.enable_viewer_sync = True

        if env.num_envs != 1 and not split_videos:
            cprint(
                f"Concatenated recording uses env0 only (num_envs={env.num_envs}). "
                "Set `--num_envs 1` for deterministic video.",
                "yellow",
            )

        single_writer = None
        single_video_path = None
        if not split_videos:
            video_name = getattr(args, "record_video_name", "") or f"{args.proj_name}-{args.exptid}.mp4"
            if not video_name.endswith(".mp4"):
                video_name += ".mp4"
            single_video_path = os.path.join(out_dir, video_name)
            single_writer = imageio.get_writer(single_video_path, fps=int(1 / env.dt))
            cprint(f"Recording video to {single_video_path}", "green")

        for clip_idx, motion_id in enumerate(motion_ids_to_record):
            motion_id_tensor = torch.full((env.num_envs,), int(motion_id), device=env.device, dtype=torch.long)
            env.reset_idx(torch.arange(env.num_envs, device=env.device), motion_ids=motion_id_tensor)
            obs = _refresh_obs(env)

            motion_len_s = env._motion_lib.get_motion_length(motion_id_tensor[:1]).item()
            steps = max(1, int(math.ceil(motion_len_s / env.dt)))

            motion_name = None
            if motion_names is not None and 0 <= int(motion_id) < len(motion_names):
                motion_name = os.path.splitext(os.path.basename(motion_names[int(motion_id)]))[0]
            if not motion_name:
                motion_name = f"motion_{int(motion_id):06d}"
            safe_motion_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", motion_name)
            overlay = f"{clip_idx+1}/{len(motion_ids_to_record)}  id={int(motion_id)}  {safe_motion_name}"

            mp4_writers = None
            if split_videos:
                mp4_writers = []
                for env_i in range(env.num_envs):
                    video_name = f"{args.proj_name}-{args.exptid}-clip{clip_idx:03d}-{safe_motion_name}-env{env_i}.mp4"
                    video_path = os.path.join(out_dir, video_name)
                    mp4_writer = imageio.get_writer(video_path, fps=int(1 / env.dt))
                    cprint(f"Recording video to {video_path}", "green")
                    mp4_writers.append(mp4_writer)

            for _ in tqdm(range(steps), desc=f"[play] clip {clip_idx} motion {motion_id}"):
                if args.use_jit:
                    actions = policy_jit(obs.detach())
                else:
                    if if_normalize and normalizer is not None:
                        normalized_obs = normalizer.normalize(obs.detach())
                    else:
                        normalized_obs = obs.detach()
                    actions = policy(normalized_obs, hist_encoding=True)

                if "AMP" in env.__class__.__name__:
                    obs, _, rews, dones, info0s, _, _ = env.step(actions.detach())
                else:
                    obs, _, rews, dones, infos = env.step(actions.detach())

                if args.record_log and ((_ % stride) == 0):
                    log_dict = env.get_episode_log(env_id_to_log)
                    log_dict["_mode"] = "record_video"
                    log_dict["_clip_idx"] = int(clip_idx)
                    log_dict["_motion_id"] = int(motion_id)
                    log_dict["_frame_in_clip"] = int(_)
                    logs_dict.append(log_dict)

                imgs = env.render_record(mode="rgb_array")
                if imgs is not None:
                    if split_videos:
                        for env_i in range(env.num_envs):
                            frame = _overlay_text(imgs[env_i], overlay)
                            mp4_writers[env_i].append_data(frame)
                    else:
                        frame = _overlay_text(imgs[0], overlay)
                        single_writer.append_data(frame)

                if env.button_pressed:
                    print(f"env_id: {env.lookat_id:<{5}}")

            if split_videos:
                for mp4_writer in mp4_writers:
                    mp4_writer.close()

        if single_writer is not None:
            single_writer.close()

        if args.record_log:
            with open(dict_name, 'w') as f:
                json.dump(logs_dict, f)
        return

    for i in tqdm(range(traj_length)):
        if args.use_jit:
            actions = policy_jit(obs.detach())
        else:
            if if_normalize and normalizer is not None:
                normalized_obs = normalizer.normalize(obs.detach())
            else:
                normalized_obs = obs.detach()
            actions = policy(normalized_obs, hist_encoding=True)
            
        if "AMP" in env.__class__.__name__:
            obs, _, rews, dones, info0s, _, _ = env.step(actions.detach())
        else:
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            
        if record_video_enabled:
            # Single-clip video path is handled above. Keep legacy loop as no-op here.
            pass
                    
        if args.record_log:
            if (i % stride) == 0:
                log_dict = env.get_episode_log(env_id_to_log)
                log_dict["_mode"] = "play"
                log_dict["_frame"] = int(i)
                logs_dict.append(log_dict)
        
        # Interaction
        if env.button_pressed:
            print(f"env_id: {env.lookat_id:<{5}}")
    
    if args.record_log:
        with open(dict_name, 'w') as f:
            json.dump(logs_dict, f)
    

if __name__ == '__main__':
    args = get_args()
    play(args)
