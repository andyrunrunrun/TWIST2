import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from .humanoid_mimic_config import HumanoidMimicCfg
from .humanoid_char import HumanoidChar, convert_to_global_root_body_pos, convert_to_local_root_body_pos

from pose.utils import torch_utils
from pose.utils.motion_lib_pkl import MotionLib
from legged_gym.gym_utils.helpers import class_to_dict

import time
from termcolor import cprint
import os

import torch


class HumanoidMimic(HumanoidChar):
    def __init__(self, cfg: HumanoidMimicCfg, sim_params, physics_engine, sim_device, headless):
        self._enable_early_termination = cfg.env.enable_early_termination
        self._pose_termination = cfg.env.pose_termination
        self._pose_termination_dist = cfg.env.pose_termination_dist
        self._root_tracking_termination_dist = cfg.env.root_tracking_termination_dist
        self._tar_motion_steps_priv = cfg.env.tar_motion_steps_priv
        self._tar_motion_steps_priv = torch.tensor(self._tar_motion_steps_priv, device=sim_device, dtype=torch.int)
        self._tar_motion_steps = cfg.env.tar_motion_steps
        self._tar_motion_steps = torch.tensor(self._tar_motion_steps, device=sim_device, dtype=torch.int)
        self._tar_motion_steps_idx_in_teacher = []
        for step in self._tar_motion_steps:
            idx = torch.where(self._tar_motion_steps_priv == step)[0]
            if len(idx) > 0:
                self._tar_motion_steps_idx_in_teacher.append(idx[0].item())
            else:
                self._tar_motion_steps_idx_in_teacher.append(0)
        self._tar_motion_steps_idx_in_teacher = torch.tensor(self._tar_motion_steps_idx_in_teacher, device=sim_device, dtype=torch.int)
        self._rand_reset = cfg.env.rand_reset
        self._ref_char_offset = torch.tensor(cfg.env.ref_char_offset, device=sim_device, dtype=torch.float)
        self._track_root = cfg.env.track_root
        self.global_obs = cfg.env.global_obs
        cprint(f"[HumanoidMimic] global_obs: {self.global_obs}")
        
        
        self.evaluations = class_to_dict(self.cfg.evaluations)
        self.eval_functions = []
        self.eval_names = []
        for name, scale in self.evaluations.items():
            self.eval_names.append(name)
            name = '_error_' + name
            self.eval_functions.append(getattr(self, name))
        
        self.episode_means = {
            name: torch.zeros(cfg.env.num_envs, dtype=torch.float, device=sim_device, requires_grad=False)
            for name in self.evaluations.keys()}
            
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        num_motions = self._motion_lib.num_motions()
        self.motion_difficulty = torch.ones((num_motions), device=self.device, dtype=torch.float32, requires_grad=False) * 100.0
        self.mean_motion_difficulty = 100.
        self.motion_termination_dist = torch.ones((num_motions), device=self.device, dtype=torch.float32, requires_grad=False) * self._pose_termination_dist
        self.motion_names = self._motion_lib.get_motion_names()

        # Error aware sampling: track max key body error for each motion
        self.max_key_body_error = torch.zeros((num_motions), device=self.device, dtype=torch.float32, requires_grad=False)
        
        self.deviate_tracking_frames = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self.deviate_vel_tracking_frames = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        # Resume motion difficulty from previous experiment if specified
        resume_difficulty_from = getattr(self.cfg.motion, "resume_difficulty_from", None)
        if resume_difficulty_from is not None:
            self._resume_motion_difficulty(resume_difficulty_from)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _resume_motion_difficulty(self, resume_exptid: str):
        """Resume motion difficulty from previous training CSV files.

        Args:
            resume_exptid: Previous experiment ID to load difficulty from

        Raises:
            FileNotFoundError: If previous experiment log_dir not found
            FileNotFoundError: If no difficulty CSV files found
            RuntimeError: If any current motion not found in CSV
        """
        from termcolor import cprint
        import glob

        # Step 1: Build previous experiment log directory path
        # Assuming logs are in LEGGED_GYM_ROOT_DIR/logs/<proj_name>/<exptid>/
        # We need to find the previous experiment's log directory

        # Try to find the experiment directory by searching common patterns
        possible_log_dirs = [
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "g1_priv_mimic", resume_exptid),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "g1_stu_future", resume_exptid),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs", resume_exptid),
        ]

        prev_log_dir = None
        for log_dir_candidate in possible_log_dirs:
            if os.path.isdir(log_dir_candidate):
                prev_log_dir = log_dir_candidate
                break

        if prev_log_dir is None:
            raise FileNotFoundError(
                f"Cannot find previous experiment directory for exptid '{resume_exptid}'. "
                f"Searched: {possible_log_dirs}"
            )

        # Step 2: Find the difficulty directory
        difficulty_dir = os.path.join(prev_log_dir, "difficulty")
        if not os.path.isdir(difficulty_dir):
            raise FileNotFoundError(
                f"No difficulty directory found in previous experiment: {difficulty_dir}"
            )

        # Step 3: Find the latest difficulty CSV files (max iteration)
        # Pattern: difficulty_iter_{iteration:07d}.csv or difficulty_iter_{iteration:07d}_rank*.csv

        # First, try to find CSVs without rank suffix (resample mode or single GPU)
        csv_files_no_rank = glob.glob(os.path.join(difficulty_dir, "difficulty_iter_*.csv"))
        csv_files_no_rank = [f for f in csv_files_no_rank if "_rank" not in os.path.basename(f)]

        # Also find CSVs with rank suffix (DDP sharded mode)
        csv_files_with_rank = glob.glob(os.path.join(difficulty_dir, "difficulty_iter_*_rank*.csv"))

        if not csv_files_no_rank and not csv_files_with_rank:
            raise FileNotFoundError(
                f"No difficulty CSV files found in {difficulty_dir}"
            )

        # Determine the latest iteration
        def extract_iteration(filepath):
            """Extract iteration number from filename like difficulty_iter_0002500.csv"""
            basename = os.path.basename(filepath)
            # Remove prefix and suffix
            name = basename.replace("difficulty_iter_", "").replace(".csv", "")
            # Remove rank suffix if present
            if "_rank" in name:
                name = name.split("_rank")[0]
            return int(name)

        if csv_files_no_rank:
            # Resample mode: use files without rank suffix
            latest_iter = max(extract_iteration(f) for f in csv_files_no_rank)
            csv_files = [f for f in csv_files_no_rank if extract_iteration(f) == latest_iter]
        else:
            # DDP shard mode: collect all rank files for the latest iteration
            latest_iter = max(extract_iteration(f) for f in csv_files_with_rank)
            csv_files = [f for f in csv_files_with_rank if extract_iteration(f) == latest_iter]

        # Step 4: Load and merge difficulty from CSVs using MotionLib instance
        # This uses self._motion_lib._motion_files which is the FINAL list after
        # all preprocessing (shuffle, motion_ids filter, max_motions, DDP sharding)
        try:
            resumed_difficulty = self._motion_lib.load_difficulty_from_csvs(csv_files=csv_files)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to resume difficulty from {resume_exptid}: {e}"
            ) from e

        # Step 5: Update motion difficulty
        if len(resumed_difficulty) != len(self.motion_difficulty):
            raise RuntimeError(
                f"Difficulty length mismatch: loaded {len(resumed_difficulty)} values, "
                f"but current motion set has {len(self.motion_difficulty)} motions. "
                f"Make sure the motion config matches the one used in previous training."
            )

        self.motion_difficulty[:] = resumed_difficulty
        self.mean_motion_difficulty = torch.mean(self.motion_difficulty).item()

        # Step 6: Print summary with fancy output
        rank = getattr(self, 'rank', 0)
        if rank == 0:
            num_easy = int((self.motion_difficulty < 2.0).sum().item())
            num_medium = int(((self.motion_difficulty >= 2.0) & (self.motion_difficulty < 5.0)).sum().item())
            num_hard = int((self.motion_difficulty >= 5.0).sum().item())

            cprint("\n", "white", end="")
            cprint("╔" + "═" * 68 + "╗", "cyan", attrs=["bold"])
            cprint("║" + " " * 15 + "🎯  DIFFICULTY RESUMED  🎯" + " " * 26 + "║", "yellow", attrs=["bold"])
            cprint("╠" + "═" * 68 + "╣", "cyan", attrs=["bold"])
            cprint(f"║  Source:  {resume_exptid:<55}║", "white")
            cprint(f"║  Iter:    {latest_iter:<55}║", "white")
            cprint(f"║  Motions: {len(resumed_difficulty):<55}║", "white")
            cprint("╠" + "═" * 68 + "╣", "cyan", attrs=["bold"])
            cprint(f"║  📊 Mean: {self.mean_motion_difficulty:.2f}", "white", end="")
            cprint(" " * (60 - len(f"║  📊 Mean: {self.mean_motion_difficulty:.2f}")) + "║", "white")
            cprint(f"║  📉 Min:  {self.motion_difficulty.min().item():.2f}", "green", end="")
            cprint(" " * (60 - len(f"║  📉 Min:  {self.motion_difficulty.min().item():.2f}")) + "║", "green")
            cprint(f"║  📈 Max:  {self.motion_difficulty.max().item():.2f}", "red", end="")
            cprint(" " * (60 - len(f"║  📈 Max:  {self.motion_difficulty.max().item():.2f}")) + "║", "red")
            cprint("╠" + "═" * 68 + "╣", "cyan", attrs=["bold"])
            cprint(f"║  🟢 Easy ( < 2.0):   {num_easy:>5}  " + "█" * min(num_easy // 10, 30) + " " * (30 - min(num_easy // 10, 30)) + "║", "green")
            cprint(f"║  🟡 Medium(2.0-5.0): {num_medium:>5}  " + "█" * min(num_medium // 10, 30) + " " * (30 - min(num_medium // 10, 30)) + "║", "yellow")
            cprint(f"║  🔴 Hard ( > 5.0):   {num_hard:>5}  " + "█" * min(num_hard // 10, 30) + " " * (30 - min(num_hard // 10, 30)) + "║", "red")
            cprint("╚" + "═" * 68 + "╝", "cyan", attrs=["bold"])
            cprint("", "white", end="")


    def _get_max_motion_len(self):
        max_len = 0
        num_motions = self._motion_lib.num_motions()
        for i in range(num_motions):
            curr_len = self._motion_lib.get_motion_length(i)
            max_len = max(max_len, curr_len)
            
        return max_len
        
    def _init_buffers(self):
        self._load_motions()
        # if self.viewer is None:
        self.max_episode_length_s = self._get_max_motion_len().item()
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        super()._init_buffers()
        # Anti-shuffle reward state: previous per-foot contact for switch-rate penalty.
        self._anti_shuffle_last_contact = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=torch.bool,
        )
        self._init_motion_buffers()
        
    def _load_motions(self):
        # Check if resample mode is enabled (resample_interval > 0)
        resample_interval = getattr(self.cfg.motion, "resample_interval", 0)
        skip_ddp_sharding = (resample_interval > 0)

        self._motion_lib = MotionLib(
            motion_file=self.cfg.motion.motion_file,
            device=self.device,
            sample_ratio=self.cfg.motion.sample_ratio,
            motion_decompose=self.cfg.motion.motion_decompose,
            motion_smooth=self.cfg.motion.motion_smooth,
            max_motions=getattr(self.cfg.motion, "max_motions", -1),
            motion_ids=getattr(self.cfg.motion, "motion_ids", ""),
            shuffle_motions=getattr(self.cfg.motion, "shuffle_motions", False),
            shuffle_seed=getattr(self.cfg.motion, "shuffle_seed", 0),
            hy_feat_cache_motions=getattr(self.cfg.motion, "hy_feat_cache_motions", 0),
            gpu_cache_gib=getattr(self.cfg.motion, "gpu_cache_gib", 4.0),
            # Memory optimization parameters
            lazy_load=getattr(self.cfg.motion, "lazy_load", False),
            cpu_cache_gib=getattr(self.cfg.motion, "cpu_cache_gib", 50.0),
            storage_dtype=getattr(self.cfg.motion, "storage_dtype", "float32"),
            # Skip DDP sharding in resample mode to allow sampling from full dataset
            skip_ddp_sharding=skip_ddp_sharding,
        )
        return
    
    def _init_motion_buffers(self):
        self._motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self._ref_root_pos = torch.zeros_like(self.root_states[:, 0:3])
        self._ref_root_rot = torch.zeros_like(self.root_states[:, 3:7])
        self._ref_root_vel = torch.zeros_like(self.root_states[:, 7:10])
        self._ref_root_ang_vel = torch.zeros_like(self.root_states[:, 10:13])
        self._ref_body_pos = torch.zeros_like(self.rigid_body_states[..., 0:3])
        self._ref_dof_pos = torch.zeros_like(self.dof_pos)
        self._ref_dof_vel = torch.zeros_like(self.dof_vel)
        self._ref_root_pos_delta_local = torch.zeros_like(self.root_states[:, 0:3])
        self._ref_root_rot_delta_local = torch.zeros_like(self.root_states[:, 3:6]) # euler angle
        
        self._dof_err_w = self.cfg.env.dof_err_w
        if self._dof_err_w is None:
            self._dof_err_w = torch.ones(self.num_dof, device=self.device, dtype=torch.float)
        else:
            self._dof_err_w = torch.tensor(self._dof_err_w, device=self.device, dtype=torch.float)
        
        self._key_body_ids_motion = self._motion_lib.get_key_body_idx(key_body_names=self.cfg.motion.key_bodies)
        # compare two tensors are same
        # assert torch.equal(self._key_body_ids, torch.tensor(key_body_ids_motion, device=self.device, dtype=torch.long)), \
        #     f"Key body ids mismatch: {self._key_body_ids} vs {key_body_ids_motion}"

        # Store resample config for training loop to access
        self._motion_resample_interval = getattr(self.cfg.motion, "resample_interval", 0)
        self._motion_resample_per_gpu = getattr(self.cfg.motion, "resample_per_gpu", 15000)
        # GPU memory budget for resample (in GB). If specified, overrides resample_per_gpu
        self._motion_resample_gpu_memory_gb = getattr(self.cfg.motion, "resample_gpu_memory_gb", None)
        # Async resample flag (only enables when explicitly set via --motion_async_resample)
        self._motion_async_resample = getattr(self.cfg.motion, "async_resample", False)

        # Debug: print loaded config values
        if self._motion_resample_interval > 0:
            from termcolor import cprint
            async_str = " [ASYNC]" if self._motion_async_resample else " [SYNC]"
            cprint(f"[HumanoidMimic] Resample config loaded{async_str}: interval={self._motion_resample_interval}, "
                   f"per_gpu={self._motion_resample_per_gpu}, gpu_memory_budget={self._motion_resample_gpu_memory_gb}", "cyan")
    
    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            # Check if error aware sampling is enabled
            if (hasattr(self.cfg.motion, 'use_error_aware_sampling') and
                self.cfg.motion.use_error_aware_sampling):
                motion_ids = self._motion_lib.sample_motions(
                    n,
                    motion_difficulty=self.motion_difficulty,
                    max_key_body_error=self.max_key_body_error,
                    use_error_aware_sampling=True,
                    error_sampling_power=self.cfg.motion.error_sampling_power,
                    error_sampling_threshold=self.cfg.motion.error_sampling_threshold
                )
            else:
                motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)

        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)

        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        if hasattr(self._motion_lib, "prefetch"):
            self._motion_lib.prefetch(motion_ids)

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
    
    def _get_motion_times(self, env_ids=None):
        if env_ids is None:
            motion_times = self.episode_length_buf * self.dt + self._motion_time_offsets
        else:
            motion_times = self.episode_length_buf[env_ids] * self.dt + self._motion_time_offsets[env_ids]
        return motion_times

    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
        self.dof_pos[env_ids] = dof_pos[env_ids] * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        if hasattr(self._motion_lib, "prefetch"):
            self._motion_lib.prefetch(motion_ids)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, root_pos_delta_local, root_rot_delta_local = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]

        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
            
    def _reset_root_states(self, env_ids, root_vel=None, root_quat=None, root_pos=None, root_ang_vel=None):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                rand_pos = torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device)
                self.root_states[env_ids, :2] += rand_pos # xy position within 1m of the center
                self.episode_init_origin[env_ids, :2] = self.env_origins[env_ids, :2] + rand_pos
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                quat = quat_from_euler_xyz(0*rand_yaw, 0*rand_yaw, rand_yaw)
                self.root_states[env_ids, 3:7] = quat[:, :]

            if root_vel is not None:
                self.root_states[env_ids, 7:10] = root_vel[env_ids, :]
            if root_quat is not None:
                self.root_states[env_ids, 3:7] = root_quat[env_ids, :]

            if root_pos is not None:
                self.root_states[env_ids, 2] = root_pos[env_ids, 2] + 0.05 # always higher a bit to avoid foot penetration
                self.root_states[env_ids, :2] += root_pos[env_ids, :2]
            if root_ang_vel is not None:
                self.root_states[env_ids, 10:13] = root_ang_vel[env_ids, :]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            
    def reset_idx(self, env_ids, motion_ids=None):
        if len(env_ids) == 0:
            return

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_sums[key][env_ids] / self._motion_lib.get_motion_length(self._motion_ids[env_ids]))
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] * self.reward_scales[key] / self._motion_lib.get_motion_length(self._motion_ids[env_ids]))
            self.episode_sums[key][env_ids] = 0.

        for key in self.episode_means.keys():
            self.extras["episode"]['error_' + key] = torch.mean(self.episode_means[key][env_ids])
            self.episode_means[key][env_ids] = 0.

        if self.cfg.motion.motion_curriculum:
            self._update_motion_difficulty(env_ids)
        self._reset_ref_motion(env_ids=env_ids, motion_ids=motion_ids)


        # vel_factor = 1.0
        vel_factor = 0.8

        # RSI
        self._reset_dofs(env_ids, self._ref_dof_pos, self._ref_dof_vel*vel_factor)
        self._reset_root_states(env_ids=env_ids, root_vel=self._ref_root_vel*vel_factor, root_quat=self._ref_root_rot,
                                root_pos=self._ref_root_pos, root_ang_vel=self._ref_root_ang_vel*vel_factor)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        anti_shuffle_contact = self.contact_forces[env_ids][:, self.feet_indices, 2] > getattr(
            self.cfg.rewards, "anti_shuffle_contact_force_th", 5.0
        )
        self._anti_shuffle_last_contact[env_ids] = anti_shuffle_contact
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        self.feet_land_time[env_ids] = 0.
        self.deviate_tracking_frames[env_ids] = 0.
        self.deviate_vel_tracking_frames[env_ids] = 0.
        self._reset_buffers_extra(env_ids)

        self.episode_length_buf[env_ids] = 0

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.cfg.motion.motion_curriculum:
            self.mean_motion_difficulty = torch.mean(self.motion_difficulty)

        _, _, y = euler_from_quaternion(self.root_states[:, 3:7])
        self.init_yaw[env_ids] = y[env_ids]
        return
    
    def _hard_sync_motion_loop(self):
        motion_times = self._get_motion_times()
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        hard_sync_envs = (motion_times >= motion_lengths) & (torch.abs(motion_times - motion_lengths) < self.dt)
        hard_sync_env_ids = hard_sync_envs.nonzero(as_tuple=False).flatten()
        if len(hard_sync_env_ids) == 0:
            return
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, _, _ = self._motion_lib.calc_motion_frame(self._motion_ids, motion_times*0)
        self._reset_dofs(hard_sync_env_ids, dof_pos, dof_vel*0.8)
        self._reset_root_states(env_ids=hard_sync_env_ids, root_vel=root_vel*0.8, root_quat=root_rot, root_pos=root_pos, root_ang_vel=root_ang_vel*0.8)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
    
    def _update_motion_difficulty(self, env_ids):
        """
        Update the difficulty of motions for adaptive curriculum learning.

        This function adjusts the difficulty of each motion based on how well the robot can track it.
        Motions that are harder to complete will have their difficulty increased, while motions that
        are easy to complete will have their difficulty decreased.

        Args:
            env_ids (torch.Tensor): Indices of environments being reset
        """
        # Skip if this is initial reset (episode_length_buf == 0 for all env_ids)
        # Initial reset doesn't represent actual training progress
        if torch.all(self.episode_length_buf[env_ids] == 0):
            return

        # Debug mode: enable via DEBUG_DIFFICULTY=1 environment variable
        debug_difficulty = os.environ.get('DEBUG_DIFFICULTY', '0') == '1'
        rank = getattr(self, 'rank', 0)

        # Store old difficulty for comparison
        if debug_difficulty:
            old_difficulty = self.motion_difficulty.clone()
            old_mean = torch.mean(old_difficulty).item()

        # Get motion IDs of environments being reset
        reset_motion_ids = self._motion_ids[env_ids]

        # Safety check: ensure motion IDs are within valid range
        num_motions = self._motion_lib.num_motions()
        if torch.any(reset_motion_ids >= num_motions) or torch.any(reset_motion_ids < 0):
            import warnings
            warnings.warn(f"Motion ID out of bounds: motion_ids in [0, {num_motions-1}], "
                         f"but got min={reset_motion_ids.min()}, max={reset_motion_ids.max()}. "
                         f"This may indicate a bug in DDP sharding or resample mode.")
            # Clamp to valid range to prevent crash
            reset_motion_ids = torch.clamp(reset_motion_ids, 0, num_motions - 1)

        # Calculate completion rate for each environment (how far the robot got through the motion)
        completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lib.get_motion_length(reset_motion_ids)

        # Aggregate completion rates for each unique motion
        motion_completion_rate_sum = torch.zeros(num_motions, device=self.device, dtype=torch.float).scatter_add(0, reset_motion_ids, completion_rate)
        motion_completion_rate_count = torch.zeros(num_motions, device=self.device, dtype=torch.float).scatter_add(0, reset_motion_ids, torch.ones_like(completion_rate, dtype=torch.float))
        
        # Calculate mean completion rate for each motion
        motion_completion_rate = motion_completion_rate_sum / torch.clamp(motion_completion_rate_count, min=1)
        # Set default completion rate for motions that weren't used (use torch.where to avoid inference mode error)
        motion_completion_rate = torch.where(motion_completion_rate_count == 0,
                                              torch.tensor(0.7, device=self.device, dtype=torch.float),
                                              motion_completion_rate)
        
        # Update motion difficulty based on completion rates
        # If completion rate is low (≤ 0.5), increase difficulty
        add_idx = motion_completion_rate <= 0.5
        # If completion rate is high (≥ 0.95), decrease difficulty
        sub_idx = (motion_completion_rate >= 0.95) & (motion_completion_rate < 0.99)
        # If completion rate is high (≥ 0.99), decrease difficulty 4 times
        super_sub_idx = motion_completion_rate >= 0.99

        gamma = self.cfg.motion.motion_curriculum_gamma

        # Create new difficulty tensor based on conditions
        new_difficulty = self.motion_difficulty.clone()
        new_difficulty = torch.where(add_idx, new_difficulty * (1 + gamma), new_difficulty)
        new_difficulty = torch.where(sub_idx, new_difficulty * (1 - gamma), new_difficulty)
        new_difficulty = torch.where(super_sub_idx, new_difficulty * (1 - gamma * 20), new_difficulty)

        # Ensure difficulty stays within valid range
        # motion_difficulty_max = 100.
        MOTION_DIFFICULTY_MAX = 10.
        MOTION_DIFFICULTY_MIN = 1.
        self.motion_difficulty = torch.clamp(new_difficulty, min=MOTION_DIFFICULTY_MIN, max=MOTION_DIFFICULTY_MAX)

        # NOTE: motion_difficulty synchronization is now done during resample, not here.
        # This avoids deadlock when different ranks call _update_motion_difficulty different times.
        # The sync is handled in _sync_motion_difficulty() which is called during resample.

        # way 1: Calculate motion difficulty ratio (normalized to 0-1 range)
        # motion_difficulty_ratio = self.motion_difficulty / 100.
        motion_difficulty_ratio = self.motion_difficulty / MOTION_DIFFICULTY_MAX

        # way 2: only use 5 levels of pose termination distance
        # motion_difficulty_ratio = torch.floor(self.motion_difficulty / 20.) / 5.

        # Adjust termination distance threshold based on difficulty
        # Higher difficulty -> larger termination distance (more lenient)
        # Lower difficulty -> smaller termination distance (more strict)
        # update pose termination distance
        self.motion_termination_dist = (self._pose_termination_dist - 0.2) * motion_difficulty_ratio + 0.2 # (num_motions)
        # use min to avoid jittering of motion termination distance
        # self.motion_termination_dist = torch.min(new_motion_termination_dist, self.motion_termination_dist) # (num_motions)

        # Save motion difficulties to CSV every N iterations
        # Use global_counter to track actual iterations (not number of resets)
        current_iter = self.global_counter // 24
        save_interval_iters = getattr(self.cfg.motion, 'difficulty_save_interval', 2500)

        # Get rank info for multi-GPU training
        rank = getattr(self, 'rank', 0)
        world_size = getattr(self, 'world_size', 1)

        if hasattr(self, 'log_dir') and self.log_dir is not None:
            # Only save once per iteration (use a flag to avoid duplicate saves)
            if not hasattr(self, '_last_saved_iter'):
                self._last_saved_iter = -1

            should_save = False
            # Skip iteration 0 - only save starting from save_interval_iters
            # Save when current iteration is a positive multiple of save_interval
            if current_iter > 0 and current_iter % save_interval_iters == 0 and current_iter != self._last_saved_iter:
                should_save = True

            if should_save:
                # Check if in resample mode using skip_ddp_sharding flag
                is_resample_mode = getattr(self._motion_lib, '_skip_ddp_sharding', False)

                if is_resample_mode and world_size > 1:
                    # Resample mode: all ranks have the same synced data, only rank0 saves
                    if rank == 0:
                        self._motion_lib.save_difficulty_to_csv(self.log_dir, current_iter, self.motion_difficulty, rank=None)
                elif world_size > 1:
                    # DDP sharded mode: each rank saves its own shard
                    self._motion_lib.save_difficulty_to_csv(self.log_dir, current_iter, self.motion_difficulty, rank=rank)
                else:
                    # Single GPU: save without rank suffix
                    self._motion_lib.save_difficulty_to_csv(self.log_dir, current_iter, self.motion_difficulty, rank=None)

                self._last_saved_iter = current_iter

    def _sync_motion_difficulty(self):
        """Synchronize motion_difficulty across all GPUs using all_reduce MIN operation.

        This is called during resample, ensuring all ranks call it at the same time,
        avoiding deadlock that would occur if called from _update_motion_difficulty during rollout.

        Uses MIN operation: if any rank trained a motion well (lower difficulty),
        all ranks adopt that lower difficulty.
        """
        import torch.distributed as dist

        world_size = getattr(self, 'world_size', 1)
        rank = getattr(self, 'local_rank', 0)

        if world_size > 1:
            try:
                if dist.is_available() and dist.is_initialized():
                    # Only sync in resample mode (all ranks have full dataset)
                    is_resample_mode = getattr(self._motion_lib, '_skip_ddp_sharding', False)

                    if is_resample_mode:
                        # Stats before sync (only rank 0 prints to avoid clutter)
                        min_diff_before = float(self.motion_difficulty.min().item())
                        easy_motions_before = int((self.motion_difficulty < 0.5).sum().item())

                        dist.all_reduce(self.motion_difficulty, op=dist.ReduceOp.MIN)

                        # Stats after sync
                        min_diff_after = float(self.motion_difficulty.min().item())
                        easy_motions_after = int((self.motion_difficulty < 0.5).sum().item())

                        changed = easy_motions_after != easy_motions_before

                        if rank == 0:
                            print(f"[Difficulty Sync] min={min_diff_after:.3f} (<0.5: {easy_motions_after}/{len(self.motion_difficulty)}){' [CHANGED]' if changed else ''}", flush=True)
                    # In DDP shard mode: no sync needed (each rank has different subset)
            except Exception as e:
                import warnings
                warnings.warn(f"Motion difficulty sync failed: {e}")

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self._update_ref_motion()
        # self._hard_sync_motion_loop()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        
        if self.cfg.domain_rand.push_end_effector and (self.common_step_counter % self.cfg.domain_rand.push_end_effector_interval == 0):
            self._push_end_effector()
            
        for i in range(len(self.eval_functions)):
            name = self.eval_names[i]
            error = self.eval_functions[i]()
            # running mean
            self.episode_means[name] += (-self.episode_means[name] + error) / (self.episode_length_buf + 1.0)
        
        # Update max key body error for error aware sampling
        if hasattr(self.cfg.motion, 'use_error_aware_sampling') and self.cfg.motion.use_error_aware_sampling:
            self._update_max_key_body_error()
            
    def check_termination(self):
        contact_force_termination = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf = contact_force_termination.clone()
        
        # height_cutoff = self.root_states[:, 2] < self.cfg.rewards.termination_height
        root_height_diff = torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2])
        height_cutoff = root_height_diff > self.cfg.rewards.root_height_diff_threshold


        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
        self.reset_buf |= roll_cut
        self.reset_buf |= pitch_cut
        motion_end = self.episode_length_buf * self.dt >= self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf |= height_cutoff
        
        if self.viewer is None:
            self.reset_buf |= motion_end
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        if self.viewer is None:
            self.time_out_buf |= motion_end
        
        self.reset_buf |= self.time_out_buf
        
        vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 6.5
        self.reset_buf |= vel_too_large
        
        if self._pose_termination:
            body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] - self.rigid_body_states[:, 0:1, 0:3]
            tar_body_pos = self._ref_body_pos[:, self._key_body_ids] - self._ref_root_pos[:, None, :] 
            
            if not self.global_obs:
                body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], body_pos)
                tar_body_pos = convert_to_local_root_body_pos(self._ref_root_rot, tar_body_pos)
            
            body_pos_diff = tar_body_pos - body_pos # (envs, bodies, 3)
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1) # (envs, bodies)

            body_pos_dist = torch.max(body_pos_dist, dim=-1)[0] # (envs)
            
            # if lose tracking for 50 frames continuously, reset (corresponds to 1 second)
            # lose_tracking = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
            # self.deviate_tracking_frames[lose_tracking] += 1
            # self.deviate_tracking_frames[~lose_tracking] = 0
            # pose_fail = self.deviate_tracking_frames >= self.cfg.motion.reset_consec_frames # 50 frames = 1 second
            
            # use config to determine whether to use fixed or adaptive pose termination distance
            if self.cfg.motion.use_adaptive_pose_termination:
                # use an adaptive pose termination distance
                pose_fail = body_pos_dist > self.motion_termination_dist[self._motion_ids] ** 2
            else:
                # use a fixed pose termination distance
                pose_fail = body_pos_dist > self._pose_termination_dist ** 2
            
            if self._track_root:
                root_pos_diff = self._ref_root_pos[:, 0:2] - self.root_states[:, 0:2]
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
                root_pos_fail = root_pos_dist > self._root_tracking_termination_dist ** 2
                root_pos_fail = root_pos_fail.squeeze(-1)
                pose_fail |= root_pos_fail
            self.reset_buf |= pose_fail
        
        first_step = self.episode_length_buf == 0

        self.reset_buf[first_step] = 0 # Do not reset on first step
        
        # if self.viewer is not None:
        #     # if use viewer, just not reset.
        #     self.reset_buf = torch.zeros_like(self.reset_buf)
        
        # print reset reason
        if self.viewer is not None and self.reset_buf.any():
            reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            for id in reset_ids:
                reset_reason = ""
                if contact_force_termination[id]:
                    reset_reason = "contact force"
                elif height_cutoff[id]:
                    reset_reason = "height cutoff"
                    print("height diff: ", root_height_diff[id])
                elif roll_cut[id]:
                    reset_reason = "roll limit"
                    print("roll diff: ", self.roll[id])
                elif pitch_cut[id]:
                    reset_reason = "pitch limit"
                    print("pitch diff: ", self.pitch[id])
                elif motion_end[id]:
                    reset_reason = "motion end"
                elif vel_too_large[id]:
                    reset_reason = "velocity too large"
                elif self._pose_termination and pose_fail[id]:
                    reset_reason = "pose tracking failure"
                print(f"Env {id} reset due to: {reset_reason}")
            
            # not reset if we are using viewer
            # if self.viewer is not None:
            #     self.reset_buf = torch.zeros_like(self.reset_buf)
            #     print("not reset")
        

    def _get_mimic_obs(self):
        num_steps = self._tar_motion_steps_priv.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_motion_steps_priv * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, _, _ = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        # Apply motion domain randomization noise
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self._apply_motion_domain_randomization(
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)
        
        roll, pitch, _ = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
        
        mimic_obs_buf = torch.cat((
            root_pos[..., 0:3], # 3 dims @Yanjie: for tracking human root position
            roll, pitch, # 2 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
        ), dim=-1) # shape: (num_envs, num_steps, 7 + num_dof)
        
        return mimic_obs_buf.reshape(self.num_envs, -1)
    
    def _apply_motion_domain_randomization(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel):
        """Apply motion domain randomization noise to reference motion data.
        
        Based on the tracking_implementation reference with ranges:
        - Root Position: ±1-5cm noise range 
        - Root Orientation: ±5.7-11.4° noise range (±0.1-0.2 rad)
        - Root Velocity: ±0.05-0.1 noise range
        - Joint Positions: ±0.05-0.1 rad noise range
        """
        if not self.cfg.motion.motion_dr_enabled:
            return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel
        
        # Get noise ranges from config
        pos_noise_range = torch.tensor(self.cfg.motion.root_position_noise, device=self.device)
        ori_noise_range = torch.tensor(self.cfg.motion.root_orientation_noise, device=self.device) 
        vel_noise_range = torch.tensor(self.cfg.motion.root_velocity_noise, device=self.device)
        joint_noise_range = torch.tensor(self.cfg.motion.joint_position_noise, device=self.device)
        
        batch_size = root_pos.shape[0]
        
        if self.cfg.motion.motion_dr_resampling:
            # Sample noise magnitudes from uniform distribution within ranges
            pos_noise_mag = torch.rand(batch_size, 1, device=self.device) * (pos_noise_range[1] - pos_noise_range[0]) + pos_noise_range[0]
            ori_noise_mag = torch.rand(batch_size, 1, device=self.device) * (ori_noise_range[1] - ori_noise_range[0]) + ori_noise_range[0] 
            vel_noise_mag = torch.rand(batch_size, 1, device=self.device) * (vel_noise_range[1] - vel_noise_range[0]) + vel_noise_range[0]
            joint_noise_mag = torch.rand(batch_size, 1, device=self.device) * (joint_noise_range[1] - joint_noise_range[0]) + joint_noise_range[0]
            
            # Root position noise: uniform ±noise_mag for each axis
            pos_noise = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * pos_noise_mag
            root_pos_noisy = root_pos + pos_noise
            
            # Root orientation noise: small axis-angle perturbations converted to quaternion
            # Generate uniform random 3D directions
            axis_noise = torch.randn(batch_size, 3, device=self.device)
            axis_noise = axis_noise / (torch.norm(axis_noise, dim=1, keepdim=True) + 1e-8)
            
            # Sample rotation angles uniformly within noise magnitude  
            angle_noise = (torch.rand(batch_size, 1, device=self.device) * 2 - 1) * ori_noise_mag
            
            # Convert axis-angle to quaternion: q = [sin(θ/2) * axis, cos(θ/2)]
            half_angle = angle_noise / 2
            sin_half = torch.sin(half_angle) 
            cos_half = torch.cos(half_angle)
            
            quat_noise = torch.zeros(batch_size, 4, device=self.device)
            quat_noise[:, :3] = axis_noise * sin_half  # xyz components
            quat_noise[:, 3:4] = cos_half             # w component
            
            # Apply rotation noise by quaternion multiplication: q_new = q_orig * q_noise
            root_rot_noisy = torch_utils.quat_mul(root_rot, quat_noise)
            
            # Root velocity noise: uniform ±noise_mag for each axis
            vel_noise = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * vel_noise_mag
            root_vel_noisy = root_vel + vel_noise
            
            # Root angular velocity noise: uniform ±noise_mag for each axis
            ang_vel_noise = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * vel_noise_mag
            root_ang_vel_noisy = root_ang_vel + ang_vel_noise
            
            # Joint position noise: uniform ±noise_mag for each joint
            joint_noise = (torch.rand(batch_size, dof_pos.shape[1], device=self.device) * 2 - 1) * joint_noise_mag
            dof_pos_noisy = dof_pos + joint_noise
            
        else:
            # Fixed noise per episode - simplified implementation
            pos_noise = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * pos_noise_range[0]
            root_pos_noisy = root_pos + pos_noise
            root_rot_noisy = root_rot  # No rotation noise for fixed mode to keep it simple
            root_vel_noisy = root_vel + (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * vel_noise_range[0]
            root_ang_vel_noisy = root_ang_vel + (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * vel_noise_range[0]
            dof_pos_noisy = dof_pos + (torch.rand(batch_size, dof_pos.shape[1], device=self.device) * 2 - 1) * joint_noise_range[0]
        
        return root_pos_noisy, root_rot_noisy, root_vel_noisy, root_ang_vel_noisy, dof_pos_noisy, dof_vel
        
    def compute_observations(self):
        # imu_obs = torch.stack((self.roll, self.pitch, self.yaw - self.init_yaw), dim=1)
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        mimic_obs = self._get_mimic_obs()
        obs_buf = torch.cat((
                            mimic_obs, # (11 + num_dof) * num_steps
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 3 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        if self.cfg.noise.add_noise and self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        else:
            obs_buf += 0.

        if self.cfg.domain_rand.domain_rand_general:
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
                self.base_lin_vel,
            ), dim=-1)
        else:
            priv_latent = torch.zeros((self.num_envs, self.cfg.env.n_priv_latent), device=self.device)
            priv_latent = torch.cat((priv_latent, self.base_lin_vel), dim=-1)

 
        self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            
        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)
            )
        
            
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2
        noise_start_dim = self.cfg.env.n_mimic_obs * len(self._tar_motion_steps_priv)
        noise_scale_vec[:, noise_start_dim:noise_start_dim+ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, noise_start_dim+ang_vel_dim:noise_start_dim+ang_vel_dim+imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, noise_start_dim+(ang_vel_dim+imu_dim):noise_start_dim+(ang_vel_dim+imu_dim)+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, noise_start_dim+(ang_vel_dim+imu_dim)+self.num_dof:noise_start_dim+(ang_vel_dim+imu_dim)+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec

    def get_episode_log(self, env_ids=0):
        log = super().get_episode_log(env_ids=env_ids)

        try:
            env_id_int = int(env_ids) if not torch.is_tensor(env_ids) else int(env_ids.item())
        except Exception:
            env_id_int = 0

        # Basic time/motion context
        log["env_id"] = env_id_int
        log["step"] = int(self.episode_length_buf[env_id_int].item())
        log["time_s"] = float(self.episode_length_buf[env_id_int].item() * self.dt)
        if hasattr(self, "_motion_ids"):
            log["motion_id"] = int(self._motion_ids[env_id_int].item())
        try:
            log["motion_time_s"] = float(self._get_motion_times()[env_id_int].item())
        except Exception:
            pass

        # Root tracking errors (pose and velocity)
        try:
            root_pos_err = self._ref_root_pos[env_id_int] - self.root_states[env_id_int, 0:3]
            log["err_root_pos_l2"] = float(torch.norm(root_pos_err).item())
            root_rot_err = torch_utils.quat_diff_angle(
                self.root_states[env_id_int, 3:7].unsqueeze(0),
                self._ref_root_rot[env_id_int].unsqueeze(0),
            )
            log["err_root_rot_rad"] = float(root_rot_err.squeeze(0).item())

            local_ref_root_vel = quat_rotate_inverse(
                self._ref_root_rot[env_id_int].unsqueeze(0),
                self._ref_root_vel[env_id_int].unsqueeze(0),
            ).squeeze(0)
            root_vel_err = local_ref_root_vel - self.base_lin_vel[env_id_int]
            log["err_root_lin_vel_l2"] = float(torch.norm(root_vel_err).item())

            local_ref_root_ang_vel = quat_rotate_inverse(
                self._ref_root_rot[env_id_int].unsqueeze(0),
                self._ref_root_ang_vel[env_id_int].unsqueeze(0),
            ).squeeze(0)
            root_ang_vel_err = local_ref_root_ang_vel - self.base_ang_vel[env_id_int]
            log["err_root_ang_vel_l2"] = float(torch.norm(root_ang_vel_err).item())
        except Exception:
            pass

        # Joint tracking errors
        try:
            dof_pos_err = self._ref_dof_pos[env_id_int] - self.dof_pos[env_id_int]
            dof_vel_err = self._ref_dof_vel[env_id_int] - self.dof_vel[env_id_int]
            log["err_dof_pos_l2"] = float(torch.norm(dof_pos_err).item())
            log["err_dof_vel_l2"] = float(torch.norm(dof_vel_err).item())
        except Exception:
            pass

        # Key body position error (L1 mean), consistent with _error_tracking_keybody_pos
        try:
            key_body_pos = self.rigid_body_states[env_id_int, self._key_body_ids, 0:3]
            key_body_pos = key_body_pos - self.root_states[env_id_int, 0:3].unsqueeze(0)
            if not self.global_obs:
                yaw = self.yaw[env_id_int:env_id_int + 1]
                base_yaw_quat = quat_from_euler_xyz(0 * yaw, 0 * yaw, yaw)
                key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos.unsqueeze(0)).squeeze(0)

            tar_key_body_pos = self._ref_body_pos[env_id_int, self._key_body_ids, :]
            tar_key_body_pos = tar_key_body_pos - self._ref_root_pos[env_id_int].unsqueeze(0)
            if not self.global_obs:
                _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot[env_id_int:env_id_int + 1])
                ref_yaw_quat = quat_from_euler_xyz(0 * ref_yaw, 0 * ref_yaw, ref_yaw)
                tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos.unsqueeze(0)).squeeze(0)

            key_body_pos_err = torch.mean(torch.abs(key_body_pos - tar_key_body_pos))
            log["err_keybody_pos_l1"] = float(key_body_pos_err.item())
        except Exception:
            pass

        # Contacts (feet)
        try:
            feet_contact = (self.contact_forces[env_id_int, self.feet_indices, 2] > 5.0)
            log["feet_contact"] = feet_contact.cpu().numpy().astype(int).tolist()
            log["feet_contact_forces_z"] = self.contact_forces[env_id_int, self.feet_indices, 2].cpu().numpy().tolist()
        except Exception:
            pass

        return log
    
    
    # ================== rewards ==================
    def _reward_alive(self):
        return 1.
    
    def _reward_tracking_joint_dof(self):
        dof_diff = self._ref_dof_pos - self.dof_pos
        dof_err = torch.sum(self._dof_err_w * dof_diff * dof_diff, dim=-1)
        
        pos_scale = 0.15
        return torch.exp(-pos_scale * dof_err)
    
    def _reward_tracking_joint_vel(self):
        vel_diff = self._ref_dof_vel - self.dof_vel
        vel_err = torch.sum(self._dof_err_w * vel_diff * vel_diff, dim=-1)
        
        vel_scale = 0.01
        return torch.exp(-vel_scale * vel_err)
    
    def _reward_tracking_root_pose(self):
        """jointly reward translation and rotation"""

        root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        
        root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
        
        root_pose_scale = 1.0
        
        root_rot_err = torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
        root_rot_err *= root_rot_err
        
        return torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))

    def _reward_tracking_root_pose_delta_local(self):
        """reward translation and rotation"""
        root_pose_delta_local = self.root_states[:, 0:3] - self.last_root_pos
        root_pose_delta_local = quat_rotate_inverse(self.last_root_rot, root_pose_delta_local)
        diff = self._ref_root_pos_delta_local - root_pose_delta_local
        root_pos_err = torch.sum(diff * diff, dim=-1)
        
        root_pose_scale = 1.0
        
        return torch.exp(-root_pose_scale * root_pos_err)
    
    def _reward_tracking_root_rotation_delta_local(self):
        root_rot_delta_local = self.root_states[:, 3:7] - self.last_root_rot
        # to eluer
        root_rot_delta_local = torch.stack(euler_from_quaternion(root_rot_delta_local), dim=-1)
        # to local
        root_rot_delta_local = quat_rotate_inverse(self.last_root_rot, root_rot_delta_local)
        diff = self._ref_root_rot_delta_local - root_rot_delta_local
        root_rot_err = torch.sum(diff * diff, dim=-1)
        
        root_pose_scale = 1.0
        
        return torch.exp(-root_pose_scale * root_rot_err)
        
    def _reward_tracking_root_translation(self):
        """reward translation only"""
        root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        
        root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
        
        root_pose_scale = 5.0
        
        return torch.exp(-root_pose_scale * root_pos_err)

    def _reward_tracking_root_translation_xy(self):
        """reward translation xy only"""
        root_pos_diff = self._ref_root_pos[:, :2] - self.root_states[:, :2]
        
        root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
        
        root_pose_scale = 5.0
        
        return torch.exp(-root_pose_scale * root_pos_err)
    
    
    def _reward_tracking_root_translation_z(self):
        """reward translation z only"""
        root_pos_diff = self._ref_root_pos[:, 2:3] - self.root_states[:, 2:3]
        
        root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
        
        root_pose_scale = 5.0
        
        return torch.exp(-root_pose_scale * root_pos_err)
    
    def _reward_tracking_root_rotation(self):
        """reward rotation only"""
        root_rot_err = torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
        root_rot_err *= root_rot_err
        
        root_pose_scale = 5.0
        
        return torch.exp(-root_pose_scale * root_rot_err)
    
    def _reward_tracking_root_vel(self):
        """jointly reward angular and linear velocity"""
        if self.global_obs:
            root_vel_diff = self._ref_root_vel - self.root_states[:, 7:10]
            root_ang_vel_diff = self._ref_root_ang_vel - self.root_states[:, 10:13]
        else:
            local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
            root_vel_diff = local_ref_root_vel - self.base_lin_vel
            local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
            root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
        
        
        root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
        root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
        root_vel_scale = 1.0
        
        # return torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
        return torch.exp(-root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))
    
    
    def _reward_tracking_root_linear_vel(self):
        """reward linear velocity only"""
        if self.global_obs:
            root_vel_diff = self._ref_root_vel - self.root_states[:, 7:10]
        else:
            local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
            root_vel_diff = local_ref_root_vel - self.base_lin_vel
        root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
        root_vel_scale = 1.0
        return torch.exp(-root_vel_scale * root_vel_err)
    
    def _reward_tracking_root_angular_vel(self):
        """reward angular velocity only"""
        if self.global_obs:
            root_ang_vel_diff = self._ref_root_ang_vel - self.root_states[:, 10:13]
        else:
            local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
            root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
        root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
        root_ang_vel_scale = 1.0
        return torch.exp(-root_ang_vel_scale * root_ang_vel_err)
    
    def _reward_tracking_keybody_pos(self): # local body pos
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] # (num_envs, num_key_bodies, 3)
        key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        
        # key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)
        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
        ref_yaw_quat = quat_from_euler_xyz(0*ref_yaw, 0*ref_yaw, ref_yaw)
        
        # tar_key_body_pos = convert_to_local_root_body_pos(self._ref_root_rot, tar_key_body_pos)
        tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)
        key_body_pos_diff = key_body_pos - tar_key_body_pos
        key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # (num_envs, num_key_bodies)
        if bool(getattr(self.cfg.env, "use_limb_weights", False)) and hasattr(self, "limb_weights") and self.limb_weights is not None and hasattr(self, "_key_body_limb_src") and self._key_body_limb_src is not None:
            src = self._key_body_limb_src.to(device=key_body_pos_err.device)
            gather_idx = src.clamp(min=0)
            body_w = self.limb_weights[:, gather_idx]
            body_w = torch.where(src.unsqueeze(0) < 0, torch.ones_like(body_w), body_w)
            key_body_pos_err = key_body_pos_err * body_w
        key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)
        
        key_body_pos_scale = 10.0
        return torch.exp(-key_body_pos_scale * key_body_pos_err)
    
    def _reward_tracking_keybody_pos_global(self):
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] # (num_envs, num_key_bodies, 3)
        # key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        
        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        # tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        
        key_body_pos_diff = key_body_pos - tar_key_body_pos
        key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)  # (num_envs, num_key_bodies)
        if bool(getattr(self.cfg.env, "use_limb_weights", False)) and hasattr(self, "limb_weights") and self.limb_weights is not None and hasattr(self, "_key_body_limb_src") and self._key_body_limb_src is not None:
            src = self._key_body_limb_src.to(device=key_body_pos_err.device)
            gather_idx = src.clamp(min=0)
            body_w = self.limb_weights[:, gather_idx]
            body_w = torch.where(src.unsqueeze(0) < 0, torch.ones_like(body_w), body_w)
            key_body_pos_err = key_body_pos_err * body_w
        key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)
        
        key_body_pos_scale = 10.0
        return torch.exp(-key_body_pos_scale * key_body_pos_err)
    
    def _reward_tracking_feet_height(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        ref_feet_height = self._ref_body_pos[:, self.feet_indices, 2]
        feet_z = self.rigid_body_states[:, self.feet_indices, 2]
        
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        
        rew_pos = torch.abs(self.feet_height - ref_feet_height) < 0.05
        
        in_place_flag = torch.norm(self._ref_root_vel[:, :2], dim=1) < 0.1
        rew_pos[in_place_flag] = 0.
        self.feet_height *= ~contact
        return torch.sum(rew_pos, dim=1)
    
    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_torque_limits(self):
        out_of_limits = torch.sum((torch.abs(self.torques) / self.torque_limits - self.cfg.rewards.soft_torque_limit).clip(min=0), dim=1)
        return out_of_limits
    
    def _reward_feet_stumble(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_feet_contact_forces(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < self.cfg.rewards.max_contact_force] = 0
        rew[rew > self.cfg.rewards.max_contact_force] -= self.cfg.rewards.max_contact_force
        return rew
    
    def _reward_feet_height(self):
        # from OmniH2O
        feet_height = self.rigid_body_states[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.cfg.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
        
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return rew
    
    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        return torch.norm(self.last_actions - self.actions, dim=1)
    
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_base_acc(self):
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=1)
    
    def _reward_torque_penalty(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _anti_shuffle_stable_gate(self):
        """Gate anti-shuffle penalties to only act in stable/slow reference phases."""
        ref_speed_th = getattr(self.cfg.rewards, "anti_shuffle_ref_vel_th", 0.12)
        tilt_th = getattr(self.cfg.rewards, "anti_shuffle_tilt_th", 0.25)

        ref_speed = torch.norm(self._ref_root_vel[:, :2], dim=1)
        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        return ((ref_speed < ref_speed_th) & (tilt < tilt_th)).float()

    def _reward_step_switch_rate(self):
        if not getattr(self.cfg.rewards, "enable_anti_shuffle_reward", False):
            return torch.zeros(self.num_envs, device=self.device)

        contact_th = getattr(self.cfg.rewards, "anti_shuffle_contact_force_th", 5.0)
        contact = self.contact_forces[:, self.feet_indices, 2] > contact_th

        # Penalize frequent contact-state toggling (small-step behavior).
        switch_cnt = torch.logical_xor(contact, self._anti_shuffle_last_contact).float().sum(dim=1)
        self._anti_shuffle_last_contact[:] = contact

        return switch_cnt * self._anti_shuffle_stable_gate()

    def _reward_stance_foot_speed(self):
        if not getattr(self.cfg.rewards, "enable_anti_shuffle_reward", False):
            return torch.zeros(self.num_envs, device=self.device)

        contact_th = getattr(self.cfg.rewards, "anti_shuffle_contact_force_th", 5.0)
        contact = (self.contact_forces[:, self.feet_indices, 2] > contact_th).float()
        foot_speed_xy = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        stance_speed = (foot_speed_xy * contact).sum(dim=1)

        return stance_speed * self._anti_shuffle_stable_gate()

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        tgt_air_time = self.cfg.rewards.feet_air_time_target
        air_time = (self.feet_air_time - tgt_air_time) * first_contact
        air_time = air_time.clamp(max=0.)
        self.feet_air_time *= ~self.contact_filt
        rew_airtime = air_time.sum(dim=1)
        rew_airtime *= torch.norm(self._ref_root_vel[:, :2], dim=1) > 0.05
        return rew_airtime

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma_ang)
    
    def _error_tracking_joint_dof(self):
        dof_diff = self._ref_dof_pos - self.dof_pos
        # compute L1 error
        dof_err = torch.mean(torch.abs(dof_diff), dim=-1)
        return dof_err
    
    def _error_tracking_joint_vel(self):
        vel_diff = self._ref_dof_vel - self.dof_vel
        # compute L1 error
        vel_err = torch.mean(torch.abs(vel_diff), dim=-1)
        return vel_err
    
    def _error_tracking_root_translation(self):
        root_pos_diff = self._ref_root_pos - self.root_states[:, 0:3]
        # compute L1 error
        root_pos_err = torch.mean(torch.abs(root_pos_diff), dim=-1)
        return root_pos_err
    
    def _error_tracking_root_rotation(self):
        root_rot_err = torch_utils.quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
        # compute L1 error
        root_rot_err = torch.mean(torch.abs(root_rot_err), dim=-1)
        return root_rot_err
    
    def _error_tracking_root_vel(self):
        local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
        root_vel_diff = local_ref_root_vel - self.base_lin_vel
        # compute L1 error
        root_vel_err = torch.mean(torch.abs(root_vel_diff), dim=-1)
        return root_vel_err
    
    def _error_tracking_root_ang_vel(self):
        local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
        root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
        # compute L1 error
        root_ang_vel_err = torch.mean(torch.abs(root_ang_vel_diff), dim=-1)
        return root_ang_vel_err
    
    def _error_tracking_keybody_pos(self):
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] # (num_envs, num_key_bodies, 3)
        key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        if not self.global_obs:
            base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
            # key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
            key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)
        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        if not self.global_obs:
            _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
            ref_yaw_quat = quat_from_euler_xyz(0*ref_yaw, 0*ref_yaw, ref_yaw)
            tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)
        key_body_pos_diff = torch.mean(torch.abs(key_body_pos - tar_key_body_pos), dim=-1)
        # compute L1 error
        key_body_pos_err = torch.mean(key_body_pos_diff, dim=-1)
        return key_body_pos_err
    
    def _update_max_key_body_error(self):
        """Update max key body error for each motion for error aware sampling."""
        # Calculate current key body position error for all environments
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3] # (num_envs, num_key_bodies, 3)
        key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
        if not self.global_obs:
            base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
            key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)
        
        tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
        tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
        if not self.global_obs:
            _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
            ref_yaw_quat = quat_from_euler_xyz(0*ref_yaw, 0*ref_yaw, ref_yaw)
            tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)
        
        # Calculate L1 error for each key body part and take maximum across all key body parts
        key_body_pos_diff = torch.abs(key_body_pos - tar_key_body_pos) # (num_envs, num_key_bodies, 3)
        key_body_pos_error_per_part = torch.mean(key_body_pos_diff, dim=-1) # (num_envs, num_key_bodies)
        current_max_error = torch.max(key_body_pos_error_per_part, dim=-1)[0] # (num_envs,)

        # Update max error for each motion using scatter_reduce (more efficient, avoids inference mode issues)
        # Create a buffer to hold current max values per motion
        motion_max_errors = torch.zeros_like(self.max_key_body_error)
        motion_max_errors.scatter_reduce_(0, self._motion_ids, current_max_error, reduce='amax', include_self=False)

        # Update the max error tensor by taking element-wise max
        self.max_key_body_error = torch.max(self.max_key_body_error, motion_max_errors)
    
    def _error_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    
    def _error_tracking_root_pose_delta_local(self):
        """reward translation and rotation"""
        root_pose_delta_local = self.root_states[:, 0:3] - self.last_root_pos
        root_pose_delta_local = quat_rotate_inverse(self.last_root_rot, root_pose_delta_local)
        diff = self._ref_root_pos_delta_local - root_pose_delta_local
        root_pos_err = torch.sum(diff * diff, dim=-1)
        
        return root_pos_err
    
    def _error_tracking_root_rotation_delta_local(self):
        root_rot_delta_local = self.root_states[:, 3:7] - self.last_root_rot
        # to eluer
        root_rot_delta_local = torch.stack(euler_from_quaternion(root_rot_delta_local), dim=-1)
        # to local
        root_rot_delta_local = quat_rotate_inverse(self.last_root_rot, root_rot_delta_local)
        diff = self._ref_root_rot_delta_local - root_rot_delta_local
        root_rot_err = torch.sum(diff * diff, dim=-1)
        
        return root_rot_err
