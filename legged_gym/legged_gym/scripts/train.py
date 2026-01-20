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
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry

import torch
import wandb

def _get_distributed_env():
    """Return (enabled, rank, local_rank, world_size) from torchrun-style env vars."""

    def _get_int(name: str, default: int) -> int:
        val = os.environ.get(name, None)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    world_size = _get_int("WORLD_SIZE", 1)
    rank = _get_int("RANK", 0)
    local_rank = _get_int("LOCAL_RANK", 0)
    return world_size > 1, rank, local_rank, world_size


# The `_setup_distributed` function is responsible for setting up distributed training if multiple
# GPUs are available. Here is a breakdown of what the function does:
def _setup_distributed(args):
    enabled, rank, local_rank, world_size = _get_distributed_env()
    if not enabled:
        return False, 0, 0, 1

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1 was set.")

    if not torch.cuda.is_available():
        raise RuntimeError("DDP multi-GPU training requires CUDA, but torch.cuda.is_available() is False.")

    torch.cuda.set_device(local_rank)

    # Ensure sim + RL both bind to this process GPU.
    device = f"cuda:{local_rank}"
    args.device = device
    args.sim_device = device
    args.rl_device = device

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )

    # rsl_rl's distributed reduce helpers need a CUDA device for scalar reductions under NCCL.
    try:
        from rsl_rl.utils import utils as rsl_dist_utils
        rsl_dist_utils.global_mp_device = device
    except Exception:
        pass

    # Reduce per-rank stdout noise.
    if rank != 0:
        os.environ.setdefault("WANDB_SILENT", "true")

    return True, rank, local_rank, world_size

def train(args):
    args.headless = True
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    
    wandb_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs")
    os.makedirs(wandb_dir, exist_ok=True)
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 5
        args.num_envs = 4
        args.headless = False
        # args.headless = True
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
        
    print("====================================")
    print("mode: ", mode)
    print("====================================")
        
    robot_type = args.task.split("_")[0]

    is_dist, rank, _, _ = _get_distributed_env()
    is_root = (not is_dist) or rank == 0

    if is_root:
        try:
            wandb.init(entity="far-wandb", project="twist", name=args.exptid, mode=mode, dir=wandb_dir)
        except:
            wandb.init(project="g1_mimic", name=args.exptid, mode=mode, dir=wandb_dir)
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid.py", policy="now")
    if robot_type == "g1":
        if is_root:
            wandb.save(LEGGED_GYM_ENVS_DIR + "/g1/g1_mimic_distill_config.py", policy="now")
    
    env, _ = task_registry.make_env(name=args.task, args=args)
    print(f"Using motion file: {env.cfg.motion.motion_file}")
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    

if __name__ == "__main__":
    args = get_args()
    _setup_distributed(args)
    train(args)
