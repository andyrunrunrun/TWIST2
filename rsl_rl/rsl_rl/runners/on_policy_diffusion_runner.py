import os
import statistics
import time
from collections import deque
from copy import copy

import torch
import wandb
from rich import print

from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.algorithms import *
from rsl_rl.env import VecEnv
from rsl_rl.modules import *
from rsl_rl.utils.normalizer import Normalizer
from rsl_rl.utils.utils import (
    enable_mp,
    is_root_proc,
    maybe_wrap_ddp,
    unwrap_model,
)


class OnPolicyDiffusionRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu", **kwargs):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.normalize_obs = env.cfg.env.normalize_obs
        self.args = kwargs.get("args", None)

        policy_class = eval(self.cfg["policy_class_name"])
        actor_critic = policy_class(
            num_observations=self.env.num_obs,
            num_critic_observations=self.env.num_privileged_obs,
            num_motion_observations=self.env.cfg.env.n_mimic_obs,
            num_motion_steps=len(self.env.cfg.env.tar_motion_steps),
            num_priop_observations=self.env.cfg.env.n_proprio,
            num_history_steps=self.env.cfg.env.history_len,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)
        actor_critic = maybe_wrap_ddp(actor_critic, self.device, find_unused_parameters=True)

        if self.normalize_obs:
            self.normalizer = Normalizer(
                shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype
            )
        else:
            self.normalizer = None

        alg_class = eval(self.cfg["algorithm_class_name"])
        precision = train_cfg.get("precision", train_cfg.get("train", {}).get("precision", "float32"))
        self.alg = alg_class(
            self.env, actor_critic, device=self.device, precision=precision, **self.alg_cfg
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_actions],
        )
        self.learn = self.learn_RL

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.env.log_dir = self.log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                self.env.rank = dist.get_rank()
                self.env.world_size = dist.get_world_size()
            else:
                self.env.rank = 0
                self.env.world_size = 1
        except Exception:
            self.env.rank = 0
            self.env.world_size = 1

        self._motion_resample_interval = getattr(self.env, "_motion_resample_interval", 0)
        self._motion_resample_per_gpu = getattr(self.env, "_motion_resample_per_gpu", 15000)
        self._resample_counter = -1
        if self._motion_resample_interval > 0:
            gpu_mem = getattr(self.env, "_motion_resample_gpu_memory_gb", None)
            print(
                f"[Runner DEBUG] Resample config from env: interval={self._motion_resample_interval}, "
                f"per_gpu={self._motion_resample_per_gpu}, gpu_memory_budget={gpu_mem}"
            )
            self._init_resample_mode()

    def _init_resample_mode(self):
        import sys

        try:
            import torch.distributed as dist

            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            world_size = 1
            rank = 0

        gpu_memory_budget_gb = getattr(self.env, "_motion_resample_gpu_memory_gb", None)
        seed = self.current_learning_iteration + getattr(self.env.cfg, "seed", 0) + rank * 10000
        motion_difficulty = getattr(self.env, "motion_difficulty", None)

        if gpu_memory_budget_gb is not None:
            print(
                f"[Motion Resample] Rank {rank}/{world_size}: using GPU memory budget {gpu_memory_budget_gb}GB",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[Motion Resample] Rank {rank}/{world_size}: sampling {self._motion_resample_per_gpu} motions",
                flush=True,
            )

        sampled_ids = self.env._motion_lib.resample_subset(
            num_motions=self._motion_resample_per_gpu,
            seed=seed,
            motion_difficulty=motion_difficulty,
            preload=True,
            gpu_memory_budget_gb=gpu_memory_budget_gb,
        )
        print(
            f"[Motion Resample] Rank {rank}/{world_size}: ready with {len(sampled_ids)} motions",
            flush=True,
        )
        print(f"[Motion Resample] Rank {rank}/{world_size}: re-sampling environment motion IDs...", flush=True)
        self.env.reset_idx(torch.arange(self.env.num_envs, device=self.env.device))
        print(
            f"[Motion Resample] Rank {rank}/{world_size}: environment motion IDs re-sampled",
            flush=True,
        )

        if getattr(self.env._motion_lib, "_async_resample_enabled", False):
            self.env._motion_lib.enable_async_resample(self._motion_resample_interval)

    def _maybe_resample_motions(self, iteration):
        if self._motion_resample_interval <= 0:
            return

        self._resample_counter += 1
        if self._resample_counter < self._motion_resample_interval:
            return

        self._resample_counter = 0
        try:
            import torch.distributed as dist

            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            world_size = 1
            rank = 0

        if getattr(self.env._motion_lib, "_async_resample_enabled", False):
            if self.env._motion_lib.check_and_resample_async(iteration):
                if hasattr(self.env, "_sync_motion_difficulty"):
                    self.env._sync_motion_difficulty()
                self.env.reset_idx(torch.arange(self.env.num_envs, device=self.env.device))
                print(
                    f"[Motion Resample] Iteration {iteration}: async switch complete on rank {rank}/{world_size}",
                    flush=True,
                )
                return

        gpu_memory_budget_gb = getattr(self.env, "_motion_resample_gpu_memory_gb", None)
        seed = iteration + getattr(self.env.cfg, "seed", 0) + rank * 10000
        motion_difficulty = getattr(self.env, "motion_difficulty", None)

        if hasattr(self.env, "_sync_motion_difficulty"):
            self.env._sync_motion_difficulty()

        sampled_ids = self.env._motion_lib.resample_subset(
            num_motions=self._motion_resample_per_gpu,
            seed=seed,
            motion_difficulty=motion_difficulty,
            preload=True,
            gpu_memory_budget_gb=gpu_memory_budget_gb,
        )
        print(
            f"[Motion Resample] Iteration {iteration}: rank {rank}/{world_size} loaded {len(sampled_ids)} motions",
            flush=True,
        )
        self.env.reset_idx(torch.arange(self.env.num_envs, device=self.env.device))

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_total_loss = 0.0
        mean_denoise_loss = 0.0
        mean_recon_loss = 0.0
        regularization_scale = 1.0
        average_episode_length = 0.0
        mean_motion_difficulty = 0.0

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        if self.normalize_obs:
            obs = self.normalizer.normalize(obs)
        infos = {}

        self.alg.actor_critic.train()
        root_only = (not enable_mp()) or is_root_proc()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            self._maybe_resample_motions(it)

            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, None, infos)
                    obs, _, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    if self.normalize_obs:
                        before_norm_obs = obs.clone()
                        obs = self.normalizer.normalize(obs)
                        if self._need_normalizer_update(
                            it, self.alg_cfg["normalizer_update_iterations"]
                        ):
                            self.normalizer.record(before_norm_obs)

                    total_rew = self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += total_rew
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                if self.normalize_obs and self._need_normalizer_update(
                    it, self.alg_cfg["normalizer_update_iterations"]
                ):
                    self.normalizer.update()

                start = stop
                self.alg.compute_returns(None)

            regularization_scale = (
                self.env.cfg.rewards.regularization_scale
                if hasattr(self.env.cfg.rewards, "regularization_scale")
                else 1.0
            )
            average_episode_length = (
                torch.mean(self.env.episode_length.float()).item()
                if hasattr(self.env, "episode_length")
                else 0.0
            )
            mean_motion_difficulty = (
                self.env.mean_motion_difficulty
                if hasattr(self.env, "mean_motion_difficulty")
                else 0.0
            )
            mean_total_loss, mean_denoise_loss, mean_recon_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            if root_only and self.log_dir is not None:
                self.log(locals())

            if root_only and self.log_dir is not None:
                if it < 2500:
                    should_save = it % self.save_interval == 0
                elif it <= 10000:
                    should_save = it % (2 * self.save_interval) == 0
                else:
                    should_save = it % (5 * self.save_interval) == 0
                if should_save:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        if tot_iter > self.current_learning_iteration:
            self.current_learning_iteration = tot_iter - 1
        if root_only and self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _need_normalizer_update(self, iterations, update_iterations):
        return iterations < update_iterations

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        fps = int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-6))

        wandb_dict = {
            "Loss/total": locs["mean_total_loss"],
            "Loss/denoise": locs["mean_denoise_loss"],
            "Loss/reconstruction": locs["mean_recon_loss"],
            "Loss/learning_rate": self.alg.learning_rate,
            "Perf/fps": fps,
            "Scale/regularization_scale": locs["regularization_scale"],
        }
        if locs["mean_motion_difficulty"] != 0:
            wandb_dict["Scale/motion_difficulty"] = locs["mean_motion_difficulty"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "metric" in key:
                    wandb_dict[f"Episode_rew_metrics/{key}"] = value
                else:
                    if "tracking" in key:
                        wandb_dict[f"Episode_rew_tracking/{key}"] = value
                    elif "curriculum" in key:
                        wandb_dict[f"Episode_curriculum/{key}"] = value
                    else:
                        wandb_dict[f"Episode_rew_regularization/{key}"] = value
                    ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        if len(locs["rewbuffer"]) > 0:
            mean_reward = statistics.mean(locs["rewbuffer"])
            mean_length = statistics.mean(locs["lenbuffer"])
            wandb_dict["Train/mean_reward"] = mean_reward
            wandb_dict["Train/mean_episode_length"] = mean_length
        else:
            mean_reward = 0.0
            mean_length = 0.0

        if self.log_dir is not None:
            wandb.log(wandb_dict, step=locs["it"])

        iterations_done = locs["it"] - self.start_learning_iteration + 1
        eta_seconds = (self.cfg["max_iterations"] - locs["it"] - 1) * (
            self.tot_time / max(iterations_done, 1)
        )
        mins = eta_seconds // 60
        secs = eta_seconds % 60
        title = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"{'#' * width}\n"
                f"{title.center(width, ' ')}\n\n"
                f"{f'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"
                f"{f'Computation:':>{pad}} {fps:.0f} steps/s "
                f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{f'Mean reward:':>{pad}} {mean_reward:.4f}\n"
                f"{f'Mean episode length:':>{pad}} {mean_length:.2f}\n"
                f"{f'Total loss:':>{pad}} {locs['mean_total_loss']:.6f}\n"
                f"{f'Denoise loss:':>{pad}} {locs['mean_denoise_loss']:.6f}\n"
                f"{f'Reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.6f}\n"
            )
        else:
            log_string = (
                f"{'#' * width}\n"
                f"{title.center(width, ' ')}\n\n"
                f"{f'Computation:':>{pad}} {fps:.0f} steps/s "
                f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
                f"{f'Total loss:':>{pad}} {locs['mean_total_loss']:.6f}\n"
                f"{f'Denoise loss:':>{pad}} {locs['mean_denoise_loss']:.6f}\n"
                f"{f'Reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.6f}\n"
            )

        log_string += f"{'-' * width}\n"
        log_string += ep_string
        log_string += f"{'-' * width}\n"
        log_string += f"{f'Regularization_scale:':>{pad}} {locs['regularization_scale']:.4f}\n"
        log_string += f"{f'Average_episode_length:':>{pad}} {locs['average_episode_length']:.4f}\n"
        log_string += f"{f'Grad_penalty_coef:':>{pad}} {0.0:.4f}\n"
        log_string += f"{f'Mean_motion_difficulty:':>{pad}} {locs['mean_motion_difficulty']:.4f}\n"
        log_string += (
            f"{'-' * width}\n"
            f"{f'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{f'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{f'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{f'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        model = unwrap_model(self.alg.actor_critic)
        if self.normalize_obs:
            state_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "normalizer": self.normalizer,
                "infos": infos,
            }
        else:
            state_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            }
        torch.save(state_dict, path)
        if getattr(self.cfg, "save_to_wandb", True):
            wandb.save(path, base_path=os.path.dirname(path))
            print(f"Saved model to {path} as well as to wandb")
        else:
            print(f"Saved model to {path} (wandb saving disabled)")

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print(f"Loading model from {path}...")
        loaded_dict = torch.load(path, map_location=self.device)
        unwrap_model(self.alg.actor_critic).load_state_dict(loaded_dict["model_state_dict"])
        if self.normalize_obs and "normalizer" in loaded_dict:
            self.normalizer = loaded_dict["normalizer"]
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
        self.env.global_counter = self.current_learning_iteration * self.num_steps_per_env
        self.env.total_env_steps_counter = self.current_learning_iteration * self.num_steps_per_env
        print("*" * 80)
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    def get_normalizer(self, device=None):
        if device is not None and self.normalizer is not None:
            self.normalizer.to(device)
        return self.normalizer
