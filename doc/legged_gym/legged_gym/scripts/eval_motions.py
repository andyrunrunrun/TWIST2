#!/usr/bin/env python3
"""
TWIST2 动作评估脚本 (v6 - 多进程 + 断点续评 + Debug模式)

核心设计：
1. 主进程：解析配置，调度子进程，收集结果
2. 子进程：每个子进程只处理一个文件夹，完成后退出彻底释放资源
3. 断点续评：指定 --resume 和 --output 后，会跳过已完成的文件夹
4. Debug模式：测试单个动作文件，输出详细失败原因

使用方法：
    python eval_motions.py --policy <onnx路径> --motion_config <yaml路径> [其他选项]
    
断点续评：
    python eval_motions.py --policy xxx.onnx --motion_config xxx.yaml --output results.json --resume

Debug模式（测试单个动作）：
    python eval_motions.py --policy xxx.onnx --motion_file xxx.pkl --debug
"""

import argparse
import json
import os
import sys
import time
import tempfile
import subprocess
from collections import defaultdict
from datetime import datetime
import yaml
import numpy as np

from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)


# ============================================================================
# 公共工具函数
# ============================================================================

def parse_motion_config(config_path: str) -> tuple:
    """解析 YAML 配置文件，按第一级目录分组"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    root_path = config.get('root_path', '')
    motions = config.get('motions', [])
    
    motions_by_folder = defaultdict(list)
    for motion in motions:
        file_path = motion.get('file', '')
        if file_path:
            parts = file_path.split('/')
            if len(parts) > 0:
                folder_name = parts[0]
                motions_by_folder[folder_name].append(motion)
    
    return root_path, dict(motions_by_folder)


# ============================================================================
# 子进程工作逻辑 (Worker Mode)
# ============================================================================

def worker_main(args):
    """子进程入口：评估单个文件夹"""
    # 必须先导入 isaacgym，再导入 torch
    from isaacgym import gymapi
    import torch
    import gc
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    # 注意：不能在函数内部使用 import *，需要显式导入
    import legged_gym.envs  # 这会触发 __init__.py 中的注册
    from legged_gym.gym_utils import task_registry
    from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
    
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    
    # ONNX Policy 包装器
    class OnnxPolicyWrapper:
        def __init__(self, session, input_name, output_index=0):
            self.session = session
            self.input_name = input_name
            self.output_index = output_index

        def __call__(self, obs_tensor):
            obs_np = obs_tensor.cpu().numpy().astype(np.float32)
            outputs = self.session.run(None, {self.input_name: obs_np})
            return torch.from_numpy(outputs[self.output_index])

    def load_onnx_policy(policy_path, device):
        providers = []
        if device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        session = ort.InferenceSession(policy_path, providers=providers)
        input_name = session.get_inputs()[0].name
        return OnnxPolicyWrapper(session, input_name)

    # 1. 设置环境配置
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.motion.motion_file = args.motion_config
    env_cfg.env.num_envs = args.num_envs
    
    # 评估模式配置
    env_cfg.env.debug_viz = False
    env_cfg.env.episode_length_s = 60
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    env_cfg.domain_rand.randomize_motor = False
    if hasattr(env_cfg, 'motion'):
        env_cfg.motion.motion_curriculum = False
    env_cfg.env.rand_reset = False

    # 2. 创建环境参数对象
    class Args:
        def __init__(self):
            self.task = args.task
            self.device = args.device
            self.headless = True
            self.use_jit = False
            self.record_video = False
            self.record_log = False
            self.jit_path = None
            self.teleop_mode = False
            self.num_envs = args.num_envs
            self.seed = None
            self.rows = None
            self.cols = None
            self.no_rand = True
            self.max_iterations = None
            self.resume = False
            self.experiment_name = None
            self.run_name = None
            self.load_run = None
            self.checkpoint = None
            self.fix_action_std = False
            self.teacher_exptid = None
            self.teacher_checkpoint = None
            self.eval_student = False
            self.config_overrides = {}
            self.physics_engine = gymapi.SIM_PHYSX
            self.use_gpu = True
            self.subscenes = 0
            self.use_gpu_pipeline = True
            self.num_threads = 0
            self.sim_device = args.device
            self.sim_device_type = 'cuda'
            self.compute_device_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
            self.graphics_device_id = self.compute_device_id
            self.pipeline = 'gpu'
            self.rl_device = args.device
    
    env_args = Args()

    # 3. 加载 Policy
    policy = load_onnx_policy(args.policy, args.device)

    # 4. 创建环境
    try:
        env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=env_cfg)
    except Exception as e:
        with open(args.output, 'w') as f:
            json.dump({'error': str(e)}, f)
        return

    # 5. 评估循环
    motion_lib = env._motion_lib
    num_motions = motion_lib.num_motions()
    motion_names = motion_lib.get_motion_names()
    motion_files = motion_lib._motion_files
    num_envs = env.num_envs
    
    full_results = []

    for batch_start in range(0, num_motions, num_envs):
        batch_end = min(batch_start + num_envs, num_motions)
        batch_motion_ids = list(range(batch_start, batch_end))
        batch_size = len(batch_motion_ids)
        
        # 准备数据
        motion_ids_tensor = torch.tensor(batch_motion_ids, device=args.device, dtype=torch.int64)
        
        # 填充（使用最短动作）
        if batch_size < num_envs:
            current_lens = motion_lib.get_motion_length(motion_ids_tensor)
            min_idx = torch.argmin(current_lens).item()
            padding_id = batch_motion_ids[min_idx]
            padding = torch.full((num_envs - batch_size,), padding_id, device=args.device, dtype=torch.int64)
            motion_ids_tensor = torch.cat([motion_ids_tensor, padding])
        
        # 重置
        all_env_ids = torch.arange(num_envs, device=args.device)
        HumanoidMimic.reset_idx(env, all_env_ids, motion_ids=motion_ids_tensor)
        obs = env.get_observations()
        
        # 追踪
        episode_lengths = torch.zeros(num_envs, device=args.device)
        motion_lengths = motion_lib.get_motion_length(motion_ids_tensor)
        done_mask = torch.zeros(num_envs, device=args.device, dtype=torch.bool)
        
        # 动态 max_steps
        max_motion_length = motion_lengths.max().item()
        actual_max_steps = min(args.max_steps, int(max_motion_length / env.dt * 1.2) + 10)
        
        # 运行
        for _ in range(actual_max_steps):
            with torch.no_grad():
                valid_obs = obs[:batch_size]
                valid_actions = policy(valid_obs.detach())
                
                if isinstance(valid_actions, torch.Tensor):
                    valid_actions = valid_actions.to(args.device)
                else:
                    valid_actions = torch.from_numpy(valid_actions).to(args.device)
                
                if batch_size < num_envs:
                    actions = torch.zeros((num_envs, valid_actions.shape[1]), device=args.device, dtype=valid_actions.dtype)
                    actions[:batch_size] = valid_actions
                else:
                    actions = valid_actions
            
            obs, _, _, dones, _ = env.step(actions)
            episode_lengths[~done_mask] += 1
            done_mask = done_mask | dones
            
            # 只检查有效 batch
            if done_mask[:batch_size].all():
                break
        
        # 结果
        completion_rates = (episode_lengths * env.dt) / motion_lengths
        completion_rates = torch.clamp(completion_rates, 0.0, 1.0)
        
        for i, motion_idx in enumerate(batch_motion_ids):
            full_results.append({
                'motion_idx': motion_idx,
                'motion_name': motion_names[motion_idx],
                'motion_file': motion_files[motion_idx],
                'completion_rate': completion_rates[i].item()
            })
            
    # 6. 保存结果
    with open(args.output, 'w') as f:
        json.dump({'results': full_results}, f)
    
    # 显式清理
    del env
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# Debug 模式：单个动作文件详细测试
# ============================================================================

def debug_main(args):
    """Debug模式：测试单个 pkl 文件，输出详细失败原因"""
    # 必须先导入 isaacgym
    from isaacgym import gymapi
    import torch
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import legged_gym.envs
    from legged_gym.gym_utils import task_registry
    from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
    
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    
    console = Console()
    
    try:
        import onnxruntime as ort
    except ImportError:
        console.print("[red]Error: onnxruntime not installed[/red]")
        return
    
    # ONNX Policy 包装器
    class OnnxPolicyWrapper:
        def __init__(self, session, input_name, output_index=0):
            self.session = session
            self.input_name = input_name
            self.output_index = output_index

        def __call__(self, obs_tensor):
            obs_np = obs_tensor.cpu().numpy().astype(np.float32)
            outputs = self.session.run(None, {self.input_name: obs_np})
            return torch.from_numpy(outputs[self.output_index])

    def load_onnx_policy(policy_path, device):
        providers = []
        if device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        session = ort.InferenceSession(policy_path, providers=providers)
        input_name = session.get_inputs()[0].name
        return OnnxPolicyWrapper(session, input_name)
    
    # 验证文件
    if not os.path.exists(args.policy):
        console.print(f"[red]Error: Policy file {args.policy} not found[/red]")
        return
    if not os.path.exists(args.motion_file):
        console.print(f"[red]Error: Motion file {args.motion_file} not found[/red]")
        return
    
    console.print(f"\n[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]        TWIST2 Motion Debug Mode - 动作失败原因分析       [/bold cyan]")
    console.print(f"[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]\n")
    console.print(f"[green]Policy:[/green] {args.policy}")
    console.print(f"[green]Motion file:[/green] {args.motion_file}\n")
    
    # 创建临时 yaml 配置
    import tempfile
    motion_dir = os.path.dirname(args.motion_file)
    motion_basename = os.path.basename(args.motion_file)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_config = {
            'root_path': motion_dir,
            'motions': [{'file': motion_basename, 'weight': 1.0}]
        }
        yaml.dump(temp_config, f)
        temp_yaml_path = f.name
    
    try:
        # 设置环境配置
        env_cfg, _ = task_registry.get_cfgs(name=args.task)
        env_cfg.motion.motion_file = temp_yaml_path
        env_cfg.env.num_envs = 1  # Debug 模式只用1个环境
        
        # 评估模式配置
        env_cfg.env.debug_viz = True  # 可视化
        env_cfg.env.episode_length_s = 120
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.action_delay = False
        env_cfg.domain_rand.randomize_motor = False
        if hasattr(env_cfg, 'motion'):
            env_cfg.motion.motion_curriculum = False
        env_cfg.env.rand_reset = False
        
        # 如果指定了 --no_pose_term，禁用姿态终止检测
        if hasattr(args, 'no_pose_term') and args.no_pose_term:
            env_cfg.env.pose_termination = False
            console.print("[yellow]⚠ 已禁用姿态终止检测 (--no_pose_term)[/yellow]\n")
        
        # 环境参数
        class Args:
            def __init__(self):
                self.task = args.task
                self.device = args.device
                self.headless = True  # Debug 模式也使用 headless 避免崩溃
                self.use_jit = False
                self.record_video = False
                self.record_log = False
                self.jit_path = None
                self.teleop_mode = False
                self.num_envs = 1
                self.seed = None
                self.rows = None
                self.cols = None
                self.no_rand = True
                self.max_iterations = None
                self.resume = False
                self.experiment_name = None
                self.run_name = None
                self.load_run = None
                self.checkpoint = None
                self.fix_action_std = False
                self.teacher_exptid = None
                self.teacher_checkpoint = None
                self.eval_student = False
                self.config_overrides = {}
                self.physics_engine = gymapi.SIM_PHYSX
                self.use_gpu = True
                self.subscenes = 0
                self.use_gpu_pipeline = True
                self.num_threads = 0
                self.sim_device = args.device
                self.sim_device_type = 'cuda'
                self.compute_device_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
                self.graphics_device_id = self.compute_device_id
                self.pipeline = 'gpu'
                self.rl_device = args.device
        
        env_args = Args()
        
        # 加载 Policy
        console.print("[cyan]Loading policy...[/cyan]")
        policy = load_onnx_policy(args.policy, args.device)
        
        # 创建环境
        console.print("[cyan]Creating environment...[/cyan]")
        env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=env_cfg)
        
        # 获取动作信息
        motion_lib = env._motion_lib
        motion_length = motion_lib.get_motion_length(torch.tensor([0], device=args.device)).item()
        motion_name = motion_lib.get_motion_names()[0]
        
        console.print(f"\n[bold green]Motion Info:[/bold green]")
        console.print(f"  Name: {motion_name}")
        console.print(f"  Duration: {motion_length:.2f}s")
        console.print(f"  Max steps: {int(motion_length / env.dt) + 10}")
        
        # 重置环境
        all_env_ids = torch.arange(1, device=args.device)
        motion_ids = torch.tensor([0], device=args.device, dtype=torch.int64)
        HumanoidMimic.reset_idx(env, all_env_ids, motion_ids=motion_ids)
        obs = env.get_observations()
        
        # 追踪变量
        step = 0
        max_steps = int(motion_length / env.dt * 1.5) + 20
        done = False
        failure_reasons = []
        
        # 记录历史数据用于分析
        history = {
            'base_height': [],
            'base_orientation': [],  # roll, pitch
            'tracking_error': [],
            'joint_error': [],
            'velocity': [],
            'contact_forces': []
        }
        
        console.print(f"\n[bold yellow]Running simulation...[/bold yellow]")
        console.print("[dim]Press Ctrl+C to stop early[/dim]\n")
        
        try:
            while step < max_steps and not done:
                with torch.no_grad():
                    actions = policy(obs.detach())
                    if isinstance(actions, torch.Tensor):
                        actions = actions.to(args.device)
                    else:
                        actions = torch.from_numpy(actions).to(args.device)
                
                obs, _, _, dones, info = env.step(actions)
                step += 1
                current_time = step * env.dt
                
                # ========================
                # 收集实时监控数据
                # ========================
                
                # 1. 基座高度
                base_pos = env.root_states[:, :3]
                base_height = base_pos[0, 2].item()
                history['base_height'].append(base_height)
                
                # 2. 基座姿态 (roll, pitch)
                base_quat = env.root_states[:, 3:7]
                # 计算 roll 和 pitch
                qx, qy, qz, qw = base_quat[0, 0].item(), base_quat[0, 1].item(), base_quat[0, 2].item(), base_quat[0, 3].item()
                # Roll (x-axis rotation)
                sinr_cosp = 2 * (qw * qx + qy * qz)
                cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                roll = np.arctan2(sinr_cosp, cosr_cosp)
                # Pitch (y-axis rotation)
                sinp = 2 * (qw * qy - qz * qx)
                pitch = np.arcsin(np.clip(sinp, -1, 1))
                history['base_orientation'].append((np.degrees(roll), np.degrees(pitch)))
                
                # 3. 使用环境的 tracking reward 作为误差指标
                if hasattr(env, 'extras') and 'episode' in env.extras and 'rew_imitation' in env.extras['episode']:
                    tracking_reward = env.extras['episode']['rew_imitation']
                    history['tracking_error'].append(1.0 - min(tracking_reward, 1.0))
                elif hasattr(env, '_reward_tracking') or hasattr(env, 'rew_buf'):
                    # 使用 tracking error 相关信息
                    history['tracking_error'].append(0.0)
                
                # 4. 关节误差
                if hasattr(env, 'dof_pos') and hasattr(env, '_target_dof_pos'):
                    joint_err = torch.mean(torch.abs(env.dof_pos - env._target_dof_pos)).item()
                    history['joint_error'].append(joint_err)
                elif hasattr(env, 'dof_pos'):
                    history['joint_error'].append(0.0)
                
                # 5. 基座速度
                base_vel = env.root_states[:, 7:10]
                vel_magnitude = torch.norm(base_vel[0]).item()
                history['velocity'].append(vel_magnitude)
                
                # ========================
                # 失败检测 - 使用环境内部的终止条件
                # ========================
                
                if dones[0]:
                    done = True
                    completion_rate = current_time / motion_length
                    
                    console.print(f"\n[bold red]━━━ 动作在 {current_time:.2f}s 处失败 ━━━[/bold red]")
                    console.print(f"[yellow]完成率: {completion_rate*100:.1f}%[/yellow]\n")
                    
                    # 分析失败原因 - 使用环境内部的终止条件
                    console.print("[bold cyan]失败原因分析 (环境内部检测):[/bold cyan]\n")
                    
                    # 1. 检查接触力终止
                    if hasattr(env, 'termination_contact_indices'):
                        contact_forces_norm = torch.norm(env.contact_forces[0, env.termination_contact_indices, :], dim=-1)
                        contact_force_term = torch.any(contact_forces_norm > 1.).item()
                        if contact_force_term:
                            max_contact = contact_forces_norm.max().item()
                            console.print(f"  [red]✗ 非法接触力[/red]: 终止部位接触力 = {max_contact:.2f}N (阈值 > 1.0N)")
                            failure_reasons.append(f"非法接触力 ({max_contact:.2f}N)")
                    
                    # 2. 检查高度差终止
                    if hasattr(env, '_ref_root_pos') and hasattr(env.cfg, 'rewards'):
                        ref_height = env._ref_root_pos[0, 2].item()
                        height_diff = abs(base_height - ref_height)
                        threshold = getattr(env.cfg.rewards, 'root_height_diff_threshold', 0.5)
                        if height_diff > threshold:
                            console.print(f"  [red]✗ 高度跟踪偏差过大[/red]: 当前={base_height:.3f}m, 参考={ref_height:.3f}m, 偏差={height_diff:.3f}m (阈值 > {threshold}m)")
                            failure_reasons.append(f"高度偏差 ({height_diff:.3f}m)")
                    
                    # 3. 检查 Roll/Pitch 终止
                    if hasattr(env, 'roll') and hasattr(env, 'pitch'):
                        env_roll = env.roll[0].item() if hasattr(env.roll[0], 'item') else env.roll[0]
                        env_pitch = env.pitch[0].item() if hasattr(env.pitch[0], 'item') else env.pitch[0]
                        roll_threshold = getattr(env.cfg.rewards, 'termination_roll', 1.0)
                        pitch_threshold = getattr(env.cfg.rewards, 'termination_pitch', 1.0)
                        if abs(env_roll) > roll_threshold:
                            console.print(f"  [red]✗ Roll 超限[/red]: {np.degrees(env_roll):.1f}° (阈值 > {np.degrees(roll_threshold):.1f}°)")
                            failure_reasons.append(f"Roll超限 ({np.degrees(env_roll):.1f}°)")
                        if abs(env_pitch) > pitch_threshold:
                            console.print(f"  [red]✗ Pitch 超限[/red]: {np.degrees(env_pitch):.1f}° (阈值 > {np.degrees(pitch_threshold):.1f}°)")
                            failure_reasons.append(f"Pitch超限 ({np.degrees(env_pitch):.1f}°)")
                    
                    # 4. 检查速度终止
                    if vel_magnitude > 5.0:
                        console.print(f"  [red]✗ 速度过大[/red]: {vel_magnitude:.2f} m/s (阈值 > 5.0 m/s)")
                        failure_reasons.append(f"速度过大 ({vel_magnitude:.2f}m/s)")
                    
                    # 5. 检查动作结束
                    motion_time = env.episode_length_buf[0].item() * env.dt
                    motion_len = env._motion_lib.get_motion_length(env._motion_ids[0:1]).item()
                    if motion_time >= motion_len:
                        console.print(f"  [green]✓ 动作正常完成[/green]: 时间={motion_time:.2f}s >= 动作长度={motion_len:.2f}s")
                        failure_reasons.append("动作完成")
                    
                    # 6. 检查姿态跟踪终止 (pose_termination)
                    if hasattr(env, '_pose_termination') and env._pose_termination:
                        if hasattr(env, '_key_body_ids') and hasattr(env, '_ref_body_pos'):
                            from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos
                            
                            body_pos = env.rigid_body_states[0:1, env._key_body_ids, 0:3] - env.rigid_body_states[0:1, 0:1, 0:3]
                            tar_body_pos = env._ref_body_pos[0:1, env._key_body_ids] - env._ref_root_pos[0:1, None, :]
                            
                            # 转换到局部坐标系（与 humanoid_mimic.check_termination 保持一致）
                            if hasattr(env, 'global_obs') and not env.global_obs:
                                body_pos = convert_to_local_root_body_pos(env.root_states[0:1, 3:7], body_pos)
                                tar_body_pos = convert_to_local_root_body_pos(env._ref_root_rot[0:1], tar_body_pos)
                            
                            body_pos_diff = tar_body_pos - body_pos
                            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
                            max_dist = torch.max(body_pos_dist).item()
                            max_dist_sqrt = np.sqrt(max_dist)
                            
                            # 获取终止阈值
                            if hasattr(env, '_pose_termination_dist'):
                                term_dist = env._pose_termination_dist
                            else:
                                term_dist = 1.0
                            
                            if max_dist > term_dist ** 2:
                                console.print(f"  [red]✗ 姿态跟踪失败[/red]: 最大关键点偏差 = {max_dist_sqrt:.3f}m (阈值 > {term_dist:.2f}m)")
                                failure_reasons.append(f"姿态跟踪失败 ({max_dist_sqrt:.3f}m)")
                                
                                # 关键点中英文名称映射
                                body_name_cn = {
                                    'left_rubber_hand': '左手',
                                    'right_rubber_hand': '右手',
                                    'left_ankle_roll_link': '左踝',
                                    'right_ankle_roll_link': '右踝',
                                    'left_knee_link': '左膝',
                                    'right_knee_link': '右膝',
                                    'left_elbow_link': '左肘',
                                    'right_elbow_link': '右肘',
                                    'head_mocap': '头部',
                                    'pelvis': '骨盆',
                                    'torso_link': '躯干',
                                    'left_hip_pitch_link': '左髋',
                                    'right_hip_pitch_link': '右髋',
                                    'left_shoulder_pitch_link': '左肩',
                                    'right_shoulder_pitch_link': '右肩',
                                }
                                
                                # 显示各关键点的详细偏差
                                console.print(f"\n    [bold yellow]关键点偏差详情:[/bold yellow]")
                                key_body_ids = env._key_body_ids.tolist()
                                body_names = env.body_names if hasattr(env, 'body_names') else None
                                
                                # 获取每个关键点的偏差
                                for i, body_id in enumerate(key_body_ids):
                                    dist = np.sqrt(body_pos_dist[0, i].item())
                                    curr_pos = body_pos[0, i].cpu().numpy()
                                    tar_pos = tar_body_pos[0, i].cpu().numpy()
                                    
                                    # 获取 body 名称
                                    if body_names and body_id < len(body_names):
                                        name_en = body_names[body_id]
                                    else:
                                        name_en = f"body_{body_id}"
                                    
                                    # 获取中文名称
                                    name_cn = body_name_cn.get(name_en, '')
                                    if name_cn:
                                        name_display = f"{name_en} ({name_cn})"
                                    else:
                                        name_display = name_en
                                    
                                    # 根据偏差大小选择颜色
                                    if dist > term_dist:
                                        color = "red"
                                        marker = "✗"
                                    elif dist > term_dist * 0.5:
                                        color = "yellow"
                                        marker = "⚠"
                                    else:
                                        color = "green"
                                        marker = "✓"
                                    
                                    console.print(f"      [{color}]{marker} {name_display:30s}[/{color}]: 偏差={dist:.3f}m  当前=({curr_pos[0]:+.2f}, {curr_pos[1]:+.2f}, {curr_pos[2]:+.2f})  参考=({tar_pos[0]:+.2f}, {tar_pos[1]:+.2f}, {tar_pos[2]:+.2f})")
                    
                    # 7. 检查 root 跟踪终止
                    if hasattr(env, '_track_root') and env._track_root:
                        if hasattr(env, '_ref_root_pos') and hasattr(env, '_root_tracking_termination_dist'):
                            root_pos_diff = env._ref_root_pos[0, 0:2] - env.root_states[0, 0:2]
                            root_pos_dist = torch.norm(root_pos_diff).item()
                            term_dist = env._root_tracking_termination_dist
                            if root_pos_dist > term_dist:
                                console.print(f"  [red]✗ Root XY 跟踪失败[/red]: 偏差 = {root_pos_dist:.3f}m (阈值 > {term_dist:.2f}m)")
                                failure_reasons.append(f"Root跟踪失败 ({root_pos_dist:.3f}m)")
                    
                    # 检查关节误差（补充信息）
                    if history['joint_error'] and history['joint_error'][-1] > 0.3:
                        console.print(f"  [yellow]⚠ 关节跟踪误差较大[/yellow]: {history['joint_error'][-1]:.3f} rad")
                        if history['joint_error'][-1] > 0.5:
                            failure_reasons.append(f"关节误差={history['joint_error'][-1]:.3f}")
                    
                    # 如果没有明确原因
                    if not failure_reasons:
                        # 检查超时
                        if hasattr(env, 'time_out_buf') and env.time_out_buf[0]:
                            console.print(f"  [yellow]⚠ 超时终止[/yellow]")
                            failure_reasons.append("超时")
                        else:
                            console.print(f"  [yellow]⚠ 其他终止条件触发[/yellow]")
                            # 打印更多调试信息帮助分析
                            console.print(f"\n[bold yellow]调试信息:[/bold yellow]")
                            console.print(f"  • episode_length_buf: {env.episode_length_buf[0].item()}")
                            if hasattr(env, '_motion_times'):
                                console.print(f"  • motion_time: {env._motion_times[0].item():.4f}s")
                            if hasattr(env, '_motion_ids'):
                                console.print(f"  • motion_id: {env._motion_ids[0].item()}")
                            failure_reasons.append("其他终止条件")
                    
                    # 显示失败时刻状态对比
                    console.print(f"\n[bold cyan]失败时刻状态:[/bold cyan]")
                    console.print(f"  • 基座高度: {base_height:.3f}m", end="")
                    if hasattr(env, '_ref_root_pos'):
                        ref_h = env._ref_root_pos[0, 2].item()
                        console.print(f" (参考: {ref_h:.3f}m, 偏差: {abs(base_height - ref_h):.3f}m)")
                    else:
                        console.print("")
                    console.print(f"  • 基座姿态: Roll={np.degrees(roll):.1f}°, Pitch={np.degrees(pitch):.1f}°")
                    console.print(f"  • 基座速度: {vel_magnitude:.2f}m/s")
                    if history['joint_error']:
                        console.print(f"  • 平均关节误差: {np.mean(history['joint_error'][-10:]):.4f}rad")
                    
                    break
                
                # 实时显示进度
                if step % 50 == 0:
                    progress_pct = (current_time / motion_length) * 100
                    console.print(f"  Step {step}: Time={current_time:.2f}s/{motion_length:.2f}s ({progress_pct:.1f}%), Height={base_height:.3f}m")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]用户中断[/yellow]")
        
        # 成功完成
        if not done:
            completion_rate = 1.0
            console.print(f"\n[bold green]━━━ 动作成功完成! ━━━[/bold green]")
            console.print(f"[green]完成率: 100%[/green]")
        
        # 统计信息
        console.print(f"\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        console.print(f"[bold cyan]                     统计摘要                               [/bold cyan]")
        console.print(f"[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("最小值", justify="right")
        table.add_column("最大值", justify="right")
        table.add_column("平均值", justify="right")
        
        if history['base_height']:
            table.add_row(
                "基座高度 (m)",
                f"{min(history['base_height']):.3f}",
                f"{max(history['base_height']):.3f}",
                f"{np.mean(history['base_height']):.3f}"
            )
        
        if history['base_orientation']:
            rolls = [o[0] for o in history['base_orientation']]
            pitches = [o[1] for o in history['base_orientation']]
            table.add_row(
                "Roll (°)",
                f"{min(rolls):.1f}",
                f"{max(rolls):.1f}",
                f"{np.mean(rolls):.1f}"
            )
            table.add_row(
                "Pitch (°)",
                f"{min(pitches):.1f}",
                f"{max(pitches):.1f}",
                f"{np.mean(pitches):.1f}"
            )
        
        if history['joint_error']:
            table.add_row(
                "关节误差 (rad)",
                f"{min(history['joint_error']):.4f}",
                f"{max(history['joint_error']):.4f}",
                f"{np.mean(history['joint_error']):.4f}"
            )
        
        if history['velocity']:
            table.add_row(
                "基座速度 (m/s)",
                f"{min(history['velocity']):.2f}",
                f"{max(history['velocity']):.2f}",
                f"{np.mean(history['velocity']):.2f}"
            )
        
        console.print(table)
        
        # 结论
        console.print(f"\n[bold]结论:[/bold]")
        if failure_reasons:
            console.print(f"  [red]失败原因: {', '.join(failure_reasons)}[/red]")
        else:
            console.print(f"  [green]动作执行成功[/green]")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_yaml_path):
            os.unlink(temp_yaml_path)
        console.print("\n[dim]Debug 模式结束[/dim]")


# ============================================================================
# 主进程调度逻辑 (Coordinator Mode)
# ============================================================================

def coordinator_main(args):
    """主进程：解析配置，调度子进程"""
    console = Console()
    
    # 验证
    if not os.path.exists(args.policy):
        console.print(f"[red]Error: Policy file {args.policy} not found[/red]")
        return
    if not os.path.exists(args.motion_config):
        console.print(f"[red]Error: Motion config {args.motion_config} not found[/red]")
        return
        
    console.print(f"\n[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]      TWIST2 Motion Evaluation (Multi-Process + Resume)    [/bold cyan]")
    console.print(f"[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # 解析配置
    root_path, motions_by_folder = parse_motion_config(args.motion_config)
    total_motions = sum(len(m) for m in motions_by_folder.values())
    console.print(f"[green]Found {total_motions} motions in {len(motions_by_folder)} folders[/green]")
    
    # 准备输出文件名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.basename(args.policy).replace('.onnx', '')
        args.output = f"eval_results_{name}_{timestamp}.json"
    
    console.print(f"[cyan]Results will be saved to: {args.output}[/cyan]")
    
    # 恢复已有结果
    all_results = []
    folder_stats = {}
    completed_folders = set()
    
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                existing_data = json.load(f)
            
            completed_folders = set(existing_data.get('folder_stats', {}).keys())
            all_results = existing_data.get('motion_results', [])
            folder_stats = existing_data.get('folder_stats', {})
            
            console.print(f"[yellow]Resume mode: Found {len(completed_folders)} completed folders[/yellow]")
            for fn in sorted(completed_folders):
                console.print(f"  [dim]Skipping: {fn}[/dim]")
        except Exception as e:
            console.print(f"[red]Warning: Failed to load existing results: {e}[/red]")
    
    # 调度循环
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, style="cyan", complete_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=False
        ) as progress:
            
            total_task = progress.add_task("[yellow]Overall Progress", total=total_motions)
            
            # 跳过已完成的进度
            for fn in completed_folders:
                if fn in motions_by_folder:
                    progress.update(total_task, advance=len(motions_by_folder[fn]))
            
            for folder_idx, (folder_name, folder_motions) in enumerate(sorted(motions_by_folder.items())):
                # 跳过已完成的文件夹
                if folder_name in completed_folders:
                    continue
                
                folder_start = time.time()
                num_folder_motions = len(folder_motions)
                
                # 生成临时配置
                temp_yaml_path = os.path.join(temp_dir, f"config_{folder_idx}.yaml")
                temp_result_path = os.path.join(temp_dir, f"result_{folder_idx}.json")
                
                with open(temp_yaml_path, 'w') as f:
                    yaml.dump({'root_path': root_path, 'motions': folder_motions}, f)
                
                # 启动子进程
                cmd = [
                    sys.executable, __file__,
                    '--policy', args.policy,
                    '--motion_config', temp_yaml_path,
                    '--task', args.task,
                    '--device', args.device,
                    '--num_envs', str(args.num_envs),
                    '--max_steps', str(args.max_steps),
                    '--output', temp_result_path,
                    '--subprocess_mode'
                ]
                
                folder_task = progress.add_task(f"[cyan]Processing: {folder_name}", total=num_folder_motions)
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)
                    
                    if result.returncode != 0:
                        console.print(f"[red]Subprocess failed for {folder_name}:[/red]")
                        console.print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                        progress.update(total_task, advance=num_folder_motions)
                        progress.remove_task(folder_task)
                        continue
                        
                    # 读取结果
                    if os.path.exists(temp_result_path):
                        with open(temp_result_path, 'r') as f:
                            data = json.load(f)
                        
                        if 'error' in data:
                            console.print(f"[red]Error in subprocess {folder_name}: {data['error']}[/red]")
                        elif 'results' in data:
                            results = data['results']
                            folder_rates = []
                            for r in results:
                                r['folder'] = folder_name
                                all_results.append(r)
                                folder_rates.append(r['completion_rate'])
                            
                            if folder_rates:
                                folder_mean = np.mean(folder_rates)
                                folder_stats[folder_name] = {
                                    'count': len(folder_rates),
                                    'mean': folder_mean,
                                    'std': np.std(folder_rates),
                                    'min': np.min(folder_rates),
                                    'max': np.max(folder_rates)
                                }
                                
                                folder_elapsed = time.time() - folder_start
                                color = "green" if folder_mean >= 0.8 else ("yellow" if folder_mean >= 0.5 else "red")
                                progress.console.print(f"  ✓ [bold {color}]{folder_name}[/bold {color}]: Mean={folder_mean:.4f}, Time={folder_elapsed:.1f}s")
                    
                except subprocess.TimeoutExpired:
                    console.print(f"[red]Timeout for {folder_name}[/red]")
                except Exception as e:
                    console.print(f"[red]Exception for {folder_name}: {e}[/red]")
                
                # 更新进度
                progress.update(total_task, advance=num_folder_motions)
                progress.remove_task(folder_task)
                
                # 增量保存
                try:
                    current_overall_rates = [r['completion_rate'] for r in all_results]
                    current_stats = {
                        'count': len(current_overall_rates),
                        'mean': np.mean(current_overall_rates) if current_overall_rates else 0.0,
                        'std': np.std(current_overall_rates) if current_overall_rates else 0.0,
                        'min': np.min(current_overall_rates) if current_overall_rates else 0.0,
                        'max': np.max(current_overall_rates) if current_overall_rates else 0.0
                    }
                    
                    temp_output = args.output + ".tmp"
                    with open(temp_output, 'w') as f:
                        json.dump({
                            'partial_save': True,
                            'timestamp': datetime.now().isoformat(),
                            'overall_stats': current_stats,
                            'folder_stats': folder_stats,
                            'motion_results': all_results
                        }, f, indent=2)
                    os.replace(temp_output, args.output)
                except Exception as e:
                    console.print(f"[red]Warning: Failed to save incremental results: {e}[/red]")
    
    # 最终结果
    elapsed_time = time.time() - start_time
    
    all_rates = [r['completion_rate'] for r in all_results]
    overall_stats = {
        'count': len(all_rates),
        'mean': np.mean(all_rates) if all_rates else 0.0,
        'std': np.std(all_rates) if all_rates else 0.0,
        'min': np.min(all_rates) if all_rates else 0.0,
        'max': np.max(all_rates) if all_rates else 0.0
    }
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"Overall Mean: [green]{overall_stats['mean']:.4f}[/green] ({overall_stats['count']} motions)")
    console.print(f"Total Time: {elapsed_time:.1f}s")
    console.print(f"[green]Saved to: {args.output}[/green]")
    
    with open(args.output, 'w') as f:
        json.dump({
            'overall_stats': overall_stats,
            'folder_stats': folder_stats,
            'motion_results': all_results
        }, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TWIST2 Motion Evaluation Script")
    parser.add_argument('--policy', type=str, required=True, help='Path to ONNX policy file')
    parser.add_argument('--motion_config', type=str, default=None, help='Path to motion config YAML (for batch mode)')
    parser.add_argument('--motion_file', type=str, default=None, help='Path to single motion PKL file (for debug mode)')
    parser.add_argument('--task', type=str, default='g1_stu_future', help='Task name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_envs', type=int, default=256, help='Number of parallel environments')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--max_steps', type=int, default=5000, help='Maximum simulation steps')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    parser.add_argument('--debug', action='store_true', help='Debug mode: test single motion file with detailed failure analysis')
    parser.add_argument('--no_pose_term', action='store_true', help='Disable pose termination in debug mode')
    parser.add_argument('--subprocess_mode', action='store_true', help='Run in worker mode (internal use)')
    
    args = parser.parse_args()
    
    # Debug 模式
    if args.debug:
        if args.motion_file is None:
            print("Error: --motion_file is required in debug mode")
            print("Usage: python eval_motions.py --policy xxx.onnx --motion_file xxx.pkl --debug")
            sys.exit(1)
        debug_main(args)
    # 子进程模式
    elif args.subprocess_mode:
        worker_main(args)
    # 主进程模式
    else:
        if args.motion_config is None:
            print("Error: --motion_config is required in batch mode")
            print("Usage: python eval_motions.py --policy xxx.onnx --motion_config xxx.yaml")
            sys.exit(1)
        coordinator_main(args)
