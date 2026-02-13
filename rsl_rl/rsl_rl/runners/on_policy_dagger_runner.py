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

import time
import os
from collections import deque
import statistics
from rich import print

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

import numpy as np
from rsl_rl.algorithms import *
from rsl_rl.modules import *
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
# from rsl_rl.utils.running_mean_std import RunningMeanStd
from rsl_rl.utils.normalizer import Normalizer
from rsl_rl.utils.utils import enable_mp, is_root_proc, maybe_wrap_ddp, unwrap_model

from legged_gym import LEGGED_GYM_ROOT_DIR


def get_policy_path(proj_name, exptid, checkpoint=-1):
    policy_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid)
    if checkpoint == -1:
        models = [file for file in os.listdir(policy_dir) if "model" in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    
    return os.path.join(policy_dir, model)


class OnPolicyDaggerRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu', **kwargs):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.normalize_obs = env.cfg.env.normalize_obs

        # Store args for config override access
        self.args = kwargs.get('args', None)

        # Teacher configuration
        self.teacher_cfg = train_cfg["teachercfg"]["runner"]
        self.teacher_policy_cfg = train_cfg["teachercfg"]["policy"]
        self.teacher_alg_cfg = train_cfg["teachercfg"]["algorithm"]
        self.warm_iters = self.cfg["warm_iters"]
        self.eval_student = self.cfg["eval_student"]

        # Initialize teacher policy
        teacher_policy_class = eval(self.teacher_cfg["policy_class_name"])
        self.teacher_actor_critic = teacher_policy_class(num_observations=self.env.num_privileged_obs,
                                    num_critic_observations=self.env.num_privileged_obs,
                                    num_motion_observations=self.env.cfg.env.n_priv_mimic_obs,
                                    num_motion_steps=len(self.env.cfg.env.tar_motion_steps_priv),
                                    num_actions=self.env.num_actions,
                                    **self.teacher_policy_cfg).to(self.device)

        if self.normalize_obs:
            self.teacher_normalizer = Normalizer(shape=self.env.num_privileged_obs, device=self.device, dtype=env.obs_buf.dtype)
        else:
            self.teacher_normalizer = None
        self.teacher_actor = self.teacher_actor_critic.act_inference
        
        # Initialize teacher loaded flag
        self.teacher_loaded = False
        
        if not self.eval_student and self.cfg["teacher_experiment_name"] not in ["None", "dummy", None]:
            teacher_policy_pth = get_policy_path(self.cfg["teacher_proj_name"], exptid=self.cfg["teacher_experiment_name"], checkpoint=self.cfg["teacher_checkpoint"])
            self.load_teacher(teacher_policy_pth)
            self.teacher_loaded = True
            print(f"Teacher policy loaded: {teacher_policy_pth}")
        else:
            print("Evaluating student policy only, not loading teacher policy. KL loss will be disabled.")
        
        policy_class = eval(self.cfg["policy_class_name"])
        if "Teleop" in self.cfg["policy_class_name"] or "Tracking" in self.cfg["policy_class_name"]:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
                                        num_priop_observations=self.env.cfg.env.n_proprio,
                                        num_history_steps=self.env.cfg.env.history_len,
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
        elif "HyFeat" in self.cfg["policy_class_name"]:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
                                        num_priop_observations=self.env.cfg.env.n_proprio,
                                        num_history_steps=self.env.cfg.env.history_len,
                                        num_feature_dim=getattr(self.env.cfg.env, "hy_feat_dim", 0),
                                        num_feature_steps=getattr(self.env.cfg.env, "hy_feat_history_steps", self.env.cfg.env.history_len + 1),
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
        elif "Future" in self.cfg["policy_class_name"]:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
                                        num_priop_observations=self.env.cfg.env.n_proprio,
                                        num_history_steps=self.env.cfg.env.history_len,
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)
        else:
            actor_critic = policy_class(num_observations=self.env.num_obs,
                                        num_critic_observations=self.env.num_privileged_obs,
                                        num_motion_observations=self.env.cfg.env.n_mimic_obs,
                                        num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
                                        num_actions=self.env.num_actions,
                                        **self.policy_cfg).to(self.device)

        # Wrap student policy for distributed training (torchrun + DDP).
        actor_critic = maybe_wrap_ddp(actor_critic, self.device, find_unused_parameters=True)
                
        share_normalizer = (self.env.num_obs == self.env.num_privileged_obs) or self.env.num_privileged_obs is None
            
        if self.normalize_obs:
            # DAgger Runner: Initializing normalizer
            if share_normalizer:
                self.normalizer = Normalizer(shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype)
                self.critic_normalizer = None
            else:
                self.normalizer = Normalizer(shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype)
                self.critic_normalizer = Normalizer(shape=self.env.num_privileged_obs, device=self.device, dtype=env.obs_buf.dtype)
        else:
            self.normalizer = None
            self.critic_normalizer = None
        
        alg_class = eval(self.cfg["algorithm_class_name"]) # DaggerPPO
        # 获取训练精度配置（从 train_cfg["precision"] 或 train_cfg["train"]["precision"]，默认 float32）
        precision = train_cfg.get("precision", train_cfg.get("train", {}).get("precision", "float32"))
        self.alg = alg_class(self.env, 
                                  actor_critic,
                                  self.teacher_actor_critic,
                                  teacher_loaded=self.teacher_loaded,
                                  device=self.device,
                                  precision=precision,
                                  **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        if "Transformer" in self.cfg["policy_class_name"]:
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.policy_cfg["obs_context_len"], self.env.num_obs],
                [self.policy_cfg["obs_context_len"], self.env.num_privileged_obs],
                [self.env.num_actions],
            )
        else:
            self.alg.init_storage(
                self.env.num_envs, 
                self.num_steps_per_env, 
                [self.env.num_obs], 
                [self.env.num_privileged_obs], 
                [self.env.num_actions],
            )

        self.learn = self.learn_RL

        # Log
        self.log_dir = log_dir
        if self.log_dir is not None:
            self.env.log_dir = self.log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Set rank info for environment (for multi-GPU CSV saving)
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.env.rank = dist.get_rank()
                self.env.world_size = dist.get_world_size()
            else:
                self.env.rank = 0
                self.env.world_size = 1
        except:
            self.env.rank = 0
            self.env.world_size = 1

        # Motion resample mode initialization
        self._motion_resample_interval = getattr(self.env, "_motion_resample_interval", 0)
        self._motion_resample_per_gpu = getattr(self.env, "_motion_resample_per_gpu", 15000)
        # Initialize to -1 so that resample triggers at iteration 30, 60, 90... (not 29, 59, 89...)
        # This aligns with check_and_resample_async() condition: iteration % interval == 0
        self._resample_counter = -1

        # Debug: print values read from env
        if self._motion_resample_interval > 0:
            gpu_mem = getattr(self.env, "_motion_resample_gpu_memory_gb", None)
            print(f"[Runner DEBUG] Resample config from env: interval={self._motion_resample_interval}, "
                  f"per_gpu={self._motion_resample_per_gpu}, gpu_memory_budget={gpu_mem}")

        # Initialize resample mode if enabled
        if self._motion_resample_interval > 0:
            self._init_resample_mode()

    def _init_resample_mode(self):
        """Initialize motion resample mode with a subset of motions."""
        import sys
        try:
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except:
            world_size = 1
            rank = 0

        # Get GPU memory budget if specified (overrides num_motions)
        gpu_memory_budget_gb = getattr(self.env, "_motion_resample_gpu_memory_gb", None)

        # Use rank-dependent seed to ensure each rank samples different subset
        seed = self.current_learning_iteration + getattr(self.env.cfg, "seed", 0) + rank * 10000

        # Get motion difficulty for consistent sampling
        motion_difficulty = getattr(self.env, "motion_difficulty", None)

        # Print which mode we're using
        if gpu_memory_budget_gb is not None:
            print(f"[Motion Resample] Rank {rank}/{world_size}: using GPU memory budget {gpu_memory_budget_gb}GB", file=sys.stderr, flush=True)
        else:
            print(f"[Motion Resample] Rank {rank}/{world_size}: sampling {self._motion_resample_per_gpu} motions", flush=True)

        sampled_ids = self.env._motion_lib.resample_subset(
            num_motions=self._motion_resample_per_gpu,  # Fallback if gpu_memory_budget_gb fails
            seed=seed,
            motion_difficulty=motion_difficulty,  # Use difficulty for sampling
            preload=True,  # Preload with progress bar
            gpu_memory_budget_gb=gpu_memory_budget_gb  # Use GPU budget instead of num_motions
        )
        print(f"[Motion Resample] Rank {rank}/{world_size}: ready with {len(sampled_ids)} motions", flush=True)

        # Re-sample environment motion_ids to match the current rank's subset
        # This ensures env._motion_ids are all in _resample_gpu_storage
        print(f"[Motion Resample] Rank {rank}/{world_size}: re-sampling environment motion IDs...", flush=True)
        self.env.reset_idx(torch.arange(self.env.num_envs, device=self.env.device))
        print(f"[Motion Resample] Rank {rank}/{world_size}: environment motion IDs re-sampled", flush=True)

        # Start async resample thread for next iteration (only if explicitly enabled)
        if getattr(self.env, "_motion_async_resample", False):
            self.env._motion_lib.enable_async_resample(self._motion_resample_interval)

    def _maybe_resample_motions(self, iteration):
        """Check if we need to resample motions and do it if needed."""
        if self._motion_resample_interval <= 0:
            return

        self._resample_counter += 1
        if self._resample_counter >= self._motion_resample_interval:
            self._resample_counter = 0

            try:
                import torch.distributed as dist
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            except:
                world_size = 1
                rank = 0

            # Try async resample first (fast switch to pre-loaded data)
            if self.env._motion_lib._async_resample_enabled:
                if self.env._motion_lib.check_and_resample_async(iteration):
                    # Sync motion_difficulty across all ranks
                    if hasattr(self.env, '_sync_motion_difficulty'):
                        self.env._sync_motion_difficulty()
                    # Re-sample environment motion_ids to match the new subset
                    self.env.reset_idx(torch.arange(self.env.num_envs, device=self.env.device))

                    # Fancy success output
                    num_motions = len(self.env._motion_lib._loaded_subset_ids)
                    print(f"\033[1;32m" + "═" * 60 + "\033[0m", flush=True)
                    print(f"\033[1;32m║\033[0m \033[1;33m⚡  ASYNC RESAMPLE SUCCESS  ⚡\033[0m", flush=True)
                    print(f"\033[1;32m║\033[0m  Iteration: {iteration:<10}  Rank: {rank}/{world_size:<10}  Motions: {num_motions:<8}", flush=True)
                    print(f"\033[1;32m" + "═" * 60 + "\033[0m", flush=True)
                    return

            # Fall back to synchronous resample
            # IMPORTANT: Clear the ready_event so async worker immediately prepares next batch
            if hasattr(self.env._motion_lib, '_async_resample_ready_event'):
                self.env._motion_lib._async_resample_ready_event.clear()

            # Get GPU memory budget if specified (overrides num_motions)
            gpu_memory_budget_gb = getattr(self.env, "_motion_resample_gpu_memory_gb", None)

            # Warning output for sync fallback
            print(f"\033[1;33m" + "═" * 60 + "\033[0m", flush=True)
            print(f"\033[1;33m║\033[0m \033[1;31m⚠  SYNC RESAMPLE FALLBACK  ⚠\033[0m", flush=True)
            print(f"\033[1;33m║\033[0m  Iteration: {iteration:<10}  Rank: {rank}/{world_size:<10}", flush=True)
            print(f"\033[1;33m║\033[0m  Async data not ready, using synchronous load", flush=True)
            print(f"\033[1;33m" + "═" * 60 + "\033[0m", flush=True)

            # Use rank-dependent seed to ensure each rank samples different subset
            seed = iteration + getattr(self.env.cfg, "seed", 0) + rank * 10000

            # Get motion difficulty if available
            motion_difficulty = getattr(self.env, "motion_difficulty", None)

            # Print which mode we're using
            if gpu_memory_budget_gb is not None:
                print(f"[Iteration {iteration}] Rank {rank}/{world_size}: resampling with GPU budget {gpu_memory_budget_gb}GB...", flush=True)
            else:
                print(f"[Iteration {iteration}] Rank {rank}/{world_size}: resampling {self._motion_resample_per_gpu} motions...", flush=True)

            try:
                # Sync motion_difficulty across all ranks BEFORE resampling
                # This ensures all ranks use the same difficulty values when sampling
                if hasattr(self.env, '_sync_motion_difficulty'):
                    self.env._sync_motion_difficulty()

                sampled_ids = self.env._motion_lib.resample_subset(
                    num_motions=self._motion_resample_per_gpu,  # Fallback if gpu_memory_budget_gb fails
                    seed=seed,
                    motion_difficulty=motion_difficulty,
                    preload=True,  # Preload with progress bar
                    gpu_memory_budget_gb=gpu_memory_budget_gb  # Use GPU budget instead of num_motions
                )
                print(f"[Iteration {iteration}] Rank {rank}/{world_size}: resample complete with {len(sampled_ids)} motions", flush=True)

                # Re-sample environment motion_ids to match the new subset
                print(f"[Iteration {iteration}] Rank {rank}/{world_size}: re-sampling environment motion IDs...", flush=True)
                self.env.reset_idx(torch.arange(self.env.num_envs, device=self.env.device))
                print(f"[Iteration {iteration}] Rank {rank}/{world_size}: environment motion IDs re-sampled", flush=True)
            except Exception as e:
                # Print error but don't crash - this could cause DDP deadlock
                import traceback
                print(f"[Iteration {iteration}] Rank {rank}/{world_size}: ERROR during resample - {e}")
                traceback.print_exc()
                # In DDP, we should probably abort to avoid deadlock
                if world_size > 1:
                    raise  # Re-raise to ensure all ranks fail together

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        grad_penalty_coef = 0.

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if self.normalize_obs:
            obs = self.normalizer.normalize(obs)
            critic_obs = self.teacher_normalizer.normalize(critic_obs)
        infos = {}
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        root_only = (not enable_mp()) or is_root_proc()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        task_rew_buf = deque(maxlen=100)
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0

            # Motion resample: check and resample if needed
            self._maybe_resample_motions(it)

            # Rollout
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    # if it < self.warm_iters:
                    #     actions = self.teacher_actor(critic_obs)
                    # else:
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    if self.normalize_obs:
                        before_norm_obs = obs.clone()
                        before_norm_critic_obs = critic_obs.clone()
                        # DAgger Runner: Normalizing obs
                        obs = self.normalizer.normalize(obs)
                        critic_obs = self.teacher_normalizer.normalize(critic_obs)
                        if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                            self.normalizer.record(before_norm_obs)
                            if self.critic_normalizer is not None:
                                self.critic_normalizer.record(before_norm_critic_obs)
                    
                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start
                if self.normalize_obs:
                    if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                        self.normalizer.update()
                        if self.critic_normalizer is not None:
                            self.critic_normalizer.update()

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            regularization_scale = self.env.cfg.rewards.regularization_scale if hasattr(self.env.cfg.rewards, "regularization_scale") else 1
            average_episode_length = torch.mean(self.env.episode_length.float()).item() if hasattr(self.env, "episode_length") else 0
            mean_motion_difficulty = self.env.mean_motion_difficulty if hasattr(self.env, "mean_motion_difficulty") else 0
            mean_value_loss, mean_surrogate_loss, mean_priv_reg_loss, priv_reg_coef, mean_grad_penalty_loss, grad_penalty_coef, kl_teacher_student_loss = self.alg.update()
    
            stop = time.time()
            learn_time = stop - start
            if root_only and self.log_dir is not None:
                self.log(locals())

            if root_only and self.log_dir is not None:
                if it < 2500:
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                elif it <= 10000:
                    if it % (2*self.save_interval) == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                else:
                    if it % (5*self.save_interval) == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # Save the final checkpoint even if it doesn't land on a save interval.
        # Iterations are zero-based, so the last iteration is tot_iter - 1.
        if tot_iter > self.current_learning_iteration:
            self.current_learning_iteration = tot_iter - 1
        if root_only and self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
    
    def _need_normalizer_update(self, iterations, update_iterations):
        return iterations < update_iterations

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb_dict['Episode_rew/' + key] = value
                if "metric" in key:
                    wandb_dict['Episode_rew_metrics/' + key] = value
                else:
                    if "tracking" in key:
                        wandb_dict['Episode_rew_tracking/' + key] = value
                    elif "curriculum" in key:
                        wandb_dict['Episode_curriculum/' + key] = value
                    else:
                        wandb_dict['Episode_rew_regularization/' + key] = value
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n""" # dont print metrics
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_func'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/kl_teacher_student'] = locs['kl_teacher_student_loss']
        wandb_dict['Adaptation/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Adaptation/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Adaptation/priv_ref_lambda'] = locs['priv_reg_coef']

        wandb_dict['Scale/regularization_scale'] = locs["regularization_scale"]
        if locs['grad_penalty_coef'] != 0:
            wandb_dict['Loss/grad_penalty_loss'] = locs['mean_grad_penalty_loss']
            wandb_dict['Scale/grad_penalty_coef'] = locs["grad_penalty_coef"]
        
        if locs['mean_motion_difficulty'] != 0:
            wandb_dict['Scale/motion_difficulty'] = locs["mean_motion_difficulty"]

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        scale_str = f"""{'Regularization_scale:':>{pad}} {locs['regularization_scale']:.4f}\n"""
        average_episode_length = f"""{'Average_episode_length:':>{pad}} {locs['average_episode_length']:.4f}\n"""
        gp_scale_str = f"""{'Grad_penalty_coef:':>{pad}} {locs['grad_penalty_coef']:.4f}\n"""
        motion_difficulty_str = f"""{'Mean_motion_difficulty:':>{pad}} {locs['mean_motion_difficulty']:.4f}\n"""
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        log_string += f"""{'-' * width}\n"""
        log_string += scale_str
        log_string += average_episode_length
        log_string += gp_scale_str
        log_string += motion_difficulty_str
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        model = unwrap_model(self.alg.actor_critic)
        if self.normalize_obs:
            state_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'normalizer': self.normalizer,
            'critic_normalizer': self.critic_normalizer,
            'infos': infos,
            }
        else:
            state_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        torch.save(state_dict, path)
        
        # Save to wandb only if enabled in config
        if getattr(self.cfg, 'save_to_wandb', True):  # Default to True for backward compatibility
            wandb.save(path, base_path=os.path.dirname(path))
            print(f"Saved model to {path} as well as to wandb")
        else:
            print(f"Saved model to {path} (wandb saving disabled)")

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        unwrap_model(self.alg.actor_critic).load_state_dict(loaded_dict['model_state_dict'])
        if self.normalize_obs:
            self.normalizer = loaded_dict['normalizer']
            self.critic_normalizer = loaded_dict['critic_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        self.current_learning_iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
        self.env.global_counter = self.current_learning_iteration * 24
        self.env.total_env_steps_counter = self.current_learning_iteration * 24
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_normalizer(self, device=None):
        if device is not None:
            self.normalizer.to(device)
        return self.normalizer
    
    def load_teacher(self, path):
        print("*" * 80)
        print("Loading teacher policy from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.teacher_actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if self.normalize_obs:
            self.teacher_normalizer = loaded_dict['normalizer']
        print("*" * 80)

       
    def get_teacher_inference_policy(self, device=None):
        self.teacher_actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.teacher_actor_critic.to(device)
        return self.teacher_actor_critic.act_inference
