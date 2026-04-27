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

from copy import deepcopy                 # 从copy模块导入deepcopy函数，用于深拷贝对象（当前文件中未直接使用）
import os                                 # 导入标准库os，用于操作系统路径拼接和目录操作
from datetime import datetime             # 导入datetime类，用于生成带时间戳的日志目录名
from typing import Tuple                  # 从typing模块导入Tuple类型提示，用于函数返回类型注解
import torch                              # 导入PyTorch框架，用于分布式训练中的随机种子设置
import numpy as np                        # 导入NumPy库，用于数值计算（当前文件中未直接使用）

from rsl_rl.env import VecEnv             # 从rsl_rl导入VecEnv基类，作为环境类的类型提示
from rsl_rl.runners import *              # 导入rsl_rl中的所有runner类（如OnPolicyRunner等），通过eval动态实例化

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR  # 导入legged_gym的根目录和环境目录常量
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params  # 从同级helpers模块导入辅助函数
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO  # 导入机器人环境配置和PPO训练配置类

class TaskRegistry():                     # 定义任务注册器类，负责统一管理环境类、环境配置和训练配置的注册与创建
    def __init__(self):                   # 构造方法，初始化三个内部字典
        self.task_classes = {}            # 创建空字典，用于按名称存储已注册的环境类（如G1人形机器人环境类）
        self.env_cfgs = {}                # 创建空字典，用于按名称存储环境配置对象（LeggedRobotCfg实例）
        self.train_cfgs = {}              # 创建空字典，用于按名称存储训练配置对象（LeggedRobotCfgPPO实例）
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        # 定义注册方法，接收任务名称、环境类、环境配置和训练配置四个参数
        self.task_classes[name] = task_class  # 将任务名称映射到对应的环境类，存入task_classes字典
        self.env_cfgs[name] = env_cfg         # 将任务名称映射到对应的环境配置，存入env_cfgs字典
        self.train_cfgs[name] = train_cfg     # 将任务名称映射到对应的训练配置，存入train_cfgs字典
    
    def get_task_class(self, name: str) -> VecEnv:
        # 定义获取环境类的方法，接收任务名称参数，返回类型为VecEnv
        return self.task_classes[name]        # 根据任务名称从task_classes字典中查找并返回对应的环境类
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        # 定义获取配置对的方法，接收任务名称参数，返回环境配置和训练配置的二元组
        # Return isolated config copies so CLI overrides never mutate the registered defaults.
        train_cfg = deepcopy(self.train_cfgs[name])  # 从train_cfgs字典中获取指定任务名称的训练配置副本
        env_cfg = deepcopy(self.env_cfgs[name])      # 从env_cfgs字典中获取指定任务名称的环境配置副本
        # copy seed
        env_cfg.seed = train_cfg.seed         # 将训练配置中的随机种子同步到环境配置，确保环境随机性和训练一致
        return env_cfg, train_cfg             # 返回环境配置和训练配置的二元组
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        # 定义创建环境的方法，接收任务名称、命令行参数和环境配置，返回环境实例和配置对象
        """ Creates an environment either from a registered namme or from the provided config file.
        # 文档字符串说明：根据已注册名称或提供的配置文件创建环境

        Args:
            name (string): Name of a registered env.
            # name参数：已注册环境的名称字符串
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            # args参数：Isaac Gym命令行参数对象，若未提供则自动调用get_args()获取
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.
            # env_cfg参数：可选的环境配置对象，用于覆盖已注册的默认配置

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 
            # 若name未在注册表中则抛出ValueError

        Returns:
            isaacgym.VecTaskPython: The created environment
            # 返回创建好的Isaac Gym向量化环境实例
            Dict: the corresponding config file
            # 返回对应的环境配置对象
        """
        # if no args passed get command line arguments
        if args is None:                      # 判断调用者是否未传入args参数
            args = get_args()                 # 自动解析命令行参数，获取Isaac Gym相关的默认参数和超参数
        # check if there is a registered env with that name
        if name in self.task_classes:         # 检查任务名称是否存在于已注册的环境类字典中
            task_class = self.get_task_class(name)  # 若存在，调用get_task_class获取对应的环境类
        else:                                 # 若任务名称未注册
            raise ValueError(f"Task with name: {name} was not registered")  # 抛出ValueError，提示用户该任务未注册
        if env_cfg is None:                   # 判断调用者是否未显式传入环境配置
            # load config files
            env_cfg, _ = self.get_cfgs(name)  # 从注册表中获取该任务的默认环境配置（忽略训练配置）
        else:
            env_cfg = deepcopy(env_cfg)       # 复制调用方传入的配置，避免make_env内的覆盖修改回写到调用方对象
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)  # 使用命令行参数覆盖环境配置中的对应字段（如num_envs、headless等）
        # In distributed training, offset the seed per-rank so each process generates distinct rollouts.
        try:                                  # 尝试导入并检查分布式训练状态
            import torch.distributed as dist  # 导入PyTorch分布式模块
            if dist.is_available() and dist.is_initialized():  # 检查分布式是否可用且已初始化
                env_cfg.seed = int(env_cfg.seed) + int(dist.get_rank())  # 根据进程rank偏移随机种子，保证各进程数据不同
        except Exception:                     # 捕获任何导入或分布式相关的异常
            pass                              # 忽略异常，避免在非分布式场景下报错
        set_seed(env_cfg.seed)                # 设置PyTorch、NumPy和环境内部的随机种子，确保实验可复现
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}  # 将环境配置中的sim对象（嵌套配置类）转换为普通字典
        sim_params = parse_sim_params(args, sim_params)   # 根据命令行args进一步解析并填充仿真参数字典
        env = task_class(   cfg=env_cfg,     # 实例化环境类，传入环境配置对象
                            sim_params=sim_params,  # 传入解析后的仿真参数字典
                            physics_engine=args.physics_engine,  # 传入物理引擎类型（如PhysX）
                            sim_device=args.sim_device,          # 传入仿真运行的计算设备（如cuda:0）
                            headless=args.headless)              # 传入是否无头模式（是否渲染GUI）
        return env, env_cfg                   # 返回创建好的环境实例和对应的环境配置对象

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default", **kwargs):
        # 定义创建算法训练runner的方法，接收环境实例、任务名、命令行参数、训练配置、日志根路径及额外关键字参数
        """ Creates the training algorithm  either from a registered namme or from the provided config file.
        # 文档字符串说明：根据已注册名称或提供的配置文件创建训练算法runner

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            # env参数：用于训练的Isaac Gym环境实例
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            # name参数：已注册环境名称，可选
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            # args参数：命令行参数对象
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            # train_cfg参数：训练配置对象，可选
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.
            # log_root参数：Tensorboard日志根目录，default表示使用默认路径

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            # 若name和train_cfg都未提供则报错
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored
            # 若两者都提供，则忽略name，使用train_cfg

        Returns:
            PPO: The created algorithm
            # 返回创建好的算法runner实例（如PPO runner）
            Dict: the corresponding config file
            # 返回对应的训练配置对象
        """
        # if no args passed get command line arguments
        if args is None:                      # 判断是否未传入args参数
            args = get_args()                 # 自动解析命令行参数
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:                 # 判断是否未显式传入训练配置
            if name is None:                  # 若同时未传入name，则无法获取配置
                raise ValueError("Either 'name' or 'train_cfg' must be not None")  # 抛出错误，提示必须提供其中之一
            # load config files
            _, train_cfg = self.get_cfgs(name)  # 通过name从注册表中获取训练配置（忽略环境配置）
        else:                                 # 若train_cfg已提供
            if name is not None:              # 且name也提供了
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")  # 打印提示信息，说明忽略name，使用train_cfg
            train_cfg = deepcopy(train_cfg)   # 复制调用方传入的训练配置，避免命令行覆盖污染原配置对象
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)  # 使用命令行参数覆盖训练配置中的对应字段（如学习率、迭代次数等）
        
        if log_root=="default":               # 判断log_root是否为默认字符串"default"
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)  # 构造默认日志根目录：<根目录>/logs/<实验名>
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)  # 在根目录下添加带时间戳和运行名的子目录
        elif log_root is None:                # 若log_root显式设为None
            log_dir = None                    # 则不创建日志目录（常用于测试时不记录日志）
        else:                                 # 若log_root是用户自定义路径
            log_dir = log_root                # 直接使用用户指定的路径作为日志目录（注释掉的代码原本会附加时间戳）
        
        train_cfg_dict = class_to_dict(train_cfg)  # 将训练配置对象（嵌套类结构）转换为普通字典，便于runner内部序列化
        runner_class = eval(train_cfg.runner.runner_class_name)  # 通过eval字符串动态获取runner类名对应的类对象（如OnPolicyRunner）
        runner = runner_class(env,            # 实例化runner类，传入环境实例
                                train_cfg_dict,  # 传入训练配置字典
                                log_dir,      # 传入日志目录路径
                                device=args.rl_device,  # 传入RL算法运行的设备（如cuda:0）
                                args=args,    # 将原始命令行参数也传给runner，方便内部访问config_overrides等字段
                                **kwargs)     # 传入额外的关键字参数（如resume标志等）

        # Sync motion config from env to runner's stored values
        # This is needed because config overrides applied in make_env may not be
        # reflected in runner's internal state if it was initialized with old values
        if hasattr(env, '_motion_resample_gpu_memory_gb'):  # 检查环境实例是否有动作重采样的显存限制属性
            if not hasattr(runner, '_motion_resample_gpu_memory_gb') or runner._motion_resample_gpu_memory_gb is None:  # 检查runner是否缺失该属性或值为None
                runner._motion_resample_gpu_memory_gb = env._motion_resample_gpu_memory_gb  # 将环境属性同步到runner，保证动作重采样显存限制一致
        if hasattr(env, '_motion_resample_per_gpu'):  # 检查环境实例是否有每GPU动作重采样数量属性
            if not hasattr(runner, '_motion_resample_per_gpu'):  # 检查runner是否缺失该属性
                runner._motion_resample_per_gpu = env._motion_resample_per_gpu  # 同步每GPU重采样数量
        if hasattr(env, '_motion_resample_interval'):  # 检查环境实例是否有动作重采样间隔属性
            if not hasattr(runner, '_motion_resample_interval'):  # 检查runner是否缺失该属性
                runner._motion_resample_interval = env._motion_resample_interval  # 同步重采样间隔
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume      # 从训练配置中读取是否需要恢复训练（加载已有模型）
        if args.resumeid:                     # 判断命令行参数是否提供了恢复训练的实验ID
            log_root = LEGGED_GYM_ROOT_DIR + f"/logs/{args.proj_name}/" + args.resumeid  # 根据proj_name和resumeid构造恢复路径
            resume = True                     # 显式设置resume标志为True
        if resume:                            # 判断是否需要加载之前保存的模型权重
            # load previously trained model
            print(log_root)                   # 打印恢复路径的根目录，便于调试确认
            print(train_cfg.runner.load_run)  # 打印配置中指定的要加载的运行目录名
            # load_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', "rough_a1", train_cfg.runner.load_run)
            # 注释掉的硬编码路径：原本用于rough_a1任务的恢复
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)  # 根据log_root、load_run和checkpoint自动查找最新的模型文件路径
            runner.load(resume_path)          # 调用runner的load方法加载预训练模型权重到网络中
            # if not train_cfg.policy.continue_from_last_std:
            #     runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 19, device=runner.device)
            # 注释掉的代码：若不从上次的标准差继续，则重置策略网络的标准差

        if "return_log_dir" in kwargs:        # 检查关键字参数中是否包含return_log_dir标志
            return runner, train_cfg, os.path.dirname(resume_path)  # 若需要，额外返回恢复模型所在目录的父目录路径
        else:                                 # 默认情况
            return runner, train_cfg          # 返回算法runner实例和训练配置对象

# make global task registry
task_registry = TaskRegistry()            # 实例化TaskRegistry，创建全局任务注册器对象task_registry，供各环境脚本导入使用
