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

import os                                 # 导入标准库os，用于操作系统相关功能（如环境变量、文件路径）
from datetime import datetime             # 从datetime模块导入datetime类，用于获取当前日期时间（当前代码中未直接使用）

import isaacgym                           # 导入Isaac Gym物理仿真库，初始化其运行时环境
from legged_gym.envs import *             # 从legged_gym.envs导入所有环境类，注册各个机器人任务环境
from legged_gym.gym_utils import get_args, task_registry  # 导入命令行参数解析器get_args和任务注册器task_registry

import torch                              # 导入PyTorch深度学习框架，用于模型训练和分布式通信
import wandb                              # 导入Weights & Biases库，用于实验日志记录和可视化

def _get_distributed_env():
    """Return (enabled, rank, local_rank, world_size) from torchrun-style env vars."""
    # 该函数用于从环境变量中读取分布式训练配置，返回是否启用分布式、全局rank、本地rank和世界大小

    def _get_int(name: str, default: int) -> int:
        # 定义内部辅助函数，用于安全地将环境变量转换为整数
        val = os.environ.get(name, None)   # 从系统环境变量中获取指定名称的值，若不存在则返回None
        if val is None:                     # 判断是否未找到该环境变量
            return default                  # 未找到时返回提供的默认值
        try:                                # 尝试将字符串值转换为整数
            return int(val)                 # 转换成功，返回整数值
        except ValueError:                  # 若值无法转换为整数（格式错误）
            return default                  # 捕获异常后返回默认值

    world_size = _get_int("WORLD_SIZE", 1) # 读取WORLD_SIZE环境变量，默认值为1（单卡训练）
    rank = _get_int("RANK", 0)             # 读取RANK环境变量，默认值为0（主进程）
    local_rank = _get_int("LOCAL_RANK", 0) # 读取LOCAL_RANK环境变量，默认值为0（本地第一块GPU）
    return world_size > 1, rank, local_rank, world_size  # 当world_size大于1时认为启用了分布式训练，返回四元组


# The `_setup_distributed` function is responsible for setting up distributed training if multiple
# GPUs are available. Here is a breakdown of what the function does:
def _setup_distributed(args):
    # 定义设置分布式训练环境的函数，接收命令行参数对象args
    enabled, rank, local_rank, world_size = _get_distributed_env()
    # 调用_get_distributed_env获取分布式配置信息
    if not enabled:
        # 判断是否未启用分布式训练
        return False, 0, 0, 1              # 返回单卡模式的默认值：未启用、rank=0、local_rank=0、world_size=1

    if not torch.distributed.is_available():
        # 检查当前PyTorch编译版本是否支持分布式训练
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1 was set.")
        # 不支持分布式但环境变量要求多卡时抛出运行时错误

    if not torch.cuda.is_available():
        # 检查当前系统是否可用CUDA GPU
        raise RuntimeError("DDP multi-GPU training requires CUDA, but torch.cuda.is_available() is False.")
        # 多卡训练必须依赖CUDA，无CUDA时报错

    torch.cuda.set_device(local_rank)      # 将当前进程绑定到指定的本地GPU设备，避免多进程竞争同一块GPU

    # Ensure sim + RL both bind to this process GPU.
    device = f"cuda:{local_rank}"          # 构造当前进程使用的CUDA设备字符串，如"cuda:0"
    args.device = device                   # 将计算设备设置到命令行参数中，供后续模型和网络使用
    args.sim_device = device               # 将物理仿真设备也绑定到当前进程的GPU
    args.rl_device = device                # 将强化学习算法设备同样绑定到当前进程的GPU

    if not torch.distributed.is_initialized():
        # 判断PyTorch分布式进程组是否尚未初始化
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        # 使用NCCL后端、环境变量初始化方式创建分布式进程组，用于多GPU间梯度同步

    # rsl_rl's distributed reduce helpers need a CUDA device for scalar reductions under NCCL.
    try:                                   # 尝试导入并设置rsl_rl分布式工具的全局CUDA设备
        from rsl_rl.utils import utils as rsl_dist_utils  # 导入rsl_rl框架中的分布式工具模块
        rsl_dist_utils.global_mp_device = device          # 设置全局多进程归约操作使用的CUDA设备
    except Exception:                      # 若导入失败（模块不存在或版本不兼容）则静默忽略
        pass                               # 忽略异常，继续执行后续代码

    # Reduce per-rank stdout noise.
    if rank != 0:                          # 判断是否不是主进程（rank不为0）
        os.environ.setdefault("WANDB_SILENT", "true")  # 为非主进程设置WANDB_SILENT环境变量，抑制wandb输出

    return True, rank, local_rank, world_size  # 返回分布式已启用标志及当前的rank/local_rank/world_size

def train(args):
    # 定义主训练函数，接收解析后的命令行参数对象args
    args.headless = True                   # 强制设置无头模式（不渲染可视化窗口），避免训练时弹出GUI影响性能
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    # 构造实验日志保存路径，格式为：<legged_gym根目录>/logs/<项目名称>/<实验ID>
    try:                                   # 尝试创建日志目录
        os.makedirs(log_pth)               # 递归创建日志目录，若父目录不存在也会一并创建
    except:                                # 捕获任何异常（如目录已存在或权限问题）
        pass                               # 忽略异常，确保训练流程不被打断
    
    wandb_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs")  # 构造wandb本地缓存目录路径
    os.makedirs(wandb_dir, exist_ok=True)  # 确保wandb目录存在，exist_ok=True表示目录已存在时不报错
    if args.debug:                         # 判断是否开启了调试模式
        mode = "disabled"                  # 调试模式下禁用wandb日志上传，避免污染线上实验记录
        args.rows = 10                     # 调试模式：将环境行数设置为10，减小规模
        args.cols = 5                      # 调试模式：将环境列数设置为5，减小规模
        args.num_envs = 4                  # 调试模式：将并行环境数减少到4个，便于快速验证
        args.headless = False              # 调试模式：关闭无头模式，启用可视化窗口以便观察
        # args.headless = True             # 注释掉的备用选项：调试时仍可选择无头模式
    else:                                  # 非调试模式（正常训练）
        mode = "online"                    # 设置wandb为在线模式，实时上传训练指标到云端
    
    if args.no_wandb:                      # 判断是否通过命令行显式禁用了wandb
        mode = "disabled"                  # 强制将wandb模式设为disabled，完全不记录日志
        
    print("====================================")  # 打印分隔线，增强日志可读性
    print("mode: ", mode)                   # 打印当前wandb运行模式（online/disabled）
    print("====================================")  # 打印结束分隔线
        
    robot_type = args.task.split("_")[0]   # 从任务名称中提取机器人类型，如"g1_stu_future"得到"g1"

    is_dist, rank, _, _ = _get_distributed_env()  # 再次获取分布式环境信息
    is_root = (not is_dist) or rank == 0   # 判断当前进程是否为主进程：单卡时恒为True，分布式时仅rank 0为True

    if is_root:                            # 仅在主进程中初始化wandb，防止多进程重复初始化
        try:                               # 尝试使用指定的wandb实体（团队/组织）初始化项目
            wandb.init(entity="far-wandb", project="twist", name=args.exptid, mode=mode, dir=wandb_dir)
            # 初始化wandb运行，实体为far-wandb，项目名为twist，运行名称为实验ID
        except:                            # 若指定实体不可用或网络异常
            wandb.init(project="g1_mimic", name=args.exptid, mode=mode, dir=wandb_dir)
            # 降级使用默认实体，项目名为g1_mimic，确保wandb仍能启动
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    # 以下四行为注释掉的代码，原本用于自动保存基础环境配置文件到wandb
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid.py", policy="now")
    if robot_type == "g1":                 # 判断当前机器人类型是否为G1人形机器人
        if is_root:                        # 仅在主进程中保存配置文件到wandb
            wandb.save(LEGGED_GYM_ENVS_DIR + "/g1/g1_mimic_distill_config.py", policy="now")
            # 将G1蒸馏配置文件上传到wandb，便于后续复现实验
    
    env, _ = task_registry.make_env(name=args.task, args=args)
    # 通过任务注册器创建训练环境，name指定任务名称，args传入超参数和环境配置
    print(f"Using motion file: {env.cfg.motion.motion_file}")
    # 打印当前使用的动作捕捉数据文件路径，确认加载了正确的参考动作数据
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
    # 通过任务注册器创建PPO训练runner，log_root指定日志根目录，env传入创建好的环境
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    # 启动PPO训练循环，num_learning_iterations设置最大迭代次数，init_at_random_ep_len表示在随机回合长度初始化
    

if __name__ == "__main__":                 # 当该脚本被直接运行时（而非被导入为模块）执行以下代码
    args = get_args()                      # 解析命令行参数，获取训练配置、任务名、设备等信息
    _setup_distributed(args)               # 根据命令行参数和环境变量设置分布式训练环境
    train(args)                            # 调用主训练函数，传入解析后的参数，开始训练流程
