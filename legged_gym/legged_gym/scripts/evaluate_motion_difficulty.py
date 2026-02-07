#!/usr/bin/env python3
"""
TWIST2 动作难度评估脚本

使用训练好的教师模型遍历所有动作数据，计算每个动作的难度分数。
难度分数定义：完成越差，分数越高（表示难度越高）。

评估指标：
1. completion_rate: 完成率 (0-1)，动作完成的百分比
2. tracking_error: 跟踪误差，关节位置/速度误差
3. pose_error: 姿态误差，关键点位置误差
4. stability_error: 稳定性误差，roll/pitch角度
5. termination_reason: 终止原因分类

难度分数 = 1.0 - completion_rate + 各类误差惩罚

使用方法：
    # 使用PyTorch checkpoint
    python evaluate_motion_difficulty.py \
        --task g1_priv_mimic \
        --checkpoint logs/g1_priv_mimic/<exptid>/model_10000.pt \
        --motion_config motion_data_configs/twist2_dataset.yaml \
        --output difficulty_scores.csv

    # 使用ONNX模型
    python evaluate_motion_difficulty.py \
        --task g1_priv_mimic \
        --policy assets/ckpts/teacher.onnx \
        --motion_config motion_data_configs/twist2_dataset.yaml \
        --output difficulty_scores.csv
"""

import argparse
import csv
import os
import sys
import math
from datetime import datetime
from collections import defaultdict
import tempfile
import yaml

# 添加项目路径（必须在isaacgym导入之前）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Isaac Gym必须在torch之前导入
from isaacgym import gymapi

# 导入所有环境模块（必须在模块级别，不能在函数内import *）
from legged_gym.envs import *

import numpy as np
import torch

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
)


def parse_motion_config(config_path: str) -> tuple:
    """解析 YAML 配置文件，获取所有动作"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    root_path = config.get('root_path', '')
    motions = config.get('motions', [])

    motions_by_folder = defaultdict(list)
    for motion in motions:
        file_path = motion.get('file', '')
        weight = motion.get('weight', 1.0)
        if file_path:
            parts = file_path.split('/')
            if len(parts) > 0:
                folder_name = parts[0]
                motions_by_folder[folder_name].append({
                    'file': file_path,
                    'weight': weight
                })

    return root_path, dict(motions_by_folder)


class DifficultyEvaluator:
    """动作难度评估器"""

    # 终止原因类型
    TERMINATION_REASONS = {
        'completed': '动作完成',
        'contact': '非法接触力',
        'height_diff': '高度偏差过大',
        'roll_pitch': 'Roll/Pitch超限',
        'pose_tracking': '姿态跟踪失败',
        'root_tracking': 'Root跟踪失败',
        'timeout': '超时',
        'unknown': '未知原因'
    }

    def __init__(self, env, policy, device, args):
        self.env = env
        self.policy = policy
        self.device = device
        self.args = args

        self.motion_lib = env._motion_lib
        self.num_motions = self.motion_lib.num_motions()
        self.motion_names = self.motion_lib.get_motion_names()
        self.motion_files = self.motion_lib._motion_files
        self.num_envs = env.num_envs
        self.dt = env.dt

    def evaluate_motion_batch(self, batch_motion_ids):
        """评估一批动作"""
        batch_size = len(batch_motion_ids)

        # 准备motion id tensor
        motion_ids_tensor = torch.tensor(batch_motion_ids, device=self.device, dtype=torch.int64)

        # 填充（使用最短动作填充剩余环境）
        if batch_size < self.num_envs:
            current_lens = self.motion_lib.get_motion_length(motion_ids_tensor)
            min_idx = torch.argmin(current_lens).item()
            padding_id = batch_motion_ids[min_idx]
            padding = torch.full((self.num_envs - batch_size,), padding_id, device=self.device, dtype=torch.int64)
            motion_ids_tensor = torch.cat([motion_ids_tensor, padding])

        # 重置环境
        from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        HumanoidMimic.reset_idx(self.env, all_env_ids, motion_ids=motion_ids_tensor)
        obs = self.env.get_observations()

        # 跟踪变量
        episode_lengths = torch.zeros(self.num_envs, device=self.device)
        motion_lengths = self.motion_lib.get_motion_length(motion_ids_tensor)
        done_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 用于累积统计的数据
        tracking_errors = torch.zeros(self.num_envs, device=self.device)
        pose_errors = torch.zeros(self.num_envs, device=self.device)
        joint_errors = torch.zeros(self.num_envs, device=self.device)
        max_roll = torch.zeros(self.num_envs, device=self.device)
        max_pitch = torch.zeros(self.num_envs, device=self.device)
        termination_reasons = ['unknown'] * self.num_envs

        # 动态max_steps
        max_motion_length = motion_lengths.max().item()
        actual_max_steps = min(self.args.max_steps, int(max_motion_length / self.dt * 1.5) + 50)

        # 运行仿真
        for step in range(actual_max_steps):
            with torch.no_grad():
                valid_obs = obs[:batch_size]
                valid_actions = self.policy(valid_obs.detach())

                if isinstance(valid_actions, torch.Tensor):
                    valid_actions = valid_actions.to(self.device)
                else:
                    valid_actions = torch.from_numpy(valid_actions).to(self.device)

                if batch_size < self.num_envs:
                    actions = torch.zeros((self.num_envs, valid_actions.shape[1]), device=self.device, dtype=valid_actions.dtype)
                    actions[:batch_size] = valid_actions
                else:
                    actions = valid_actions

            obs, _, rewards, dones, extras = self.env.step(actions)

            # 更新统计
            episode_lengths[~done_mask] += 1

            # 计算误差（只对未完成的环境）
            for i in range(batch_size):
                if not done_mask[i]:
                    # 关节位置误差
                    if hasattr(self.env, 'dof_pos') and hasattr(self.env, '_ref_dof_pos'):
                        joint_err = torch.mean(torch.abs(
                            self.env.dof_pos[i] - self.env._ref_dof_pos[i]
                        )).item()
                        joint_errors[i] += joint_err

                    # 关键点姿态误差
                    if hasattr(self.env, '_ref_body_pos') and hasattr(self.env, 'rigid_body_states'):
                        if hasattr(self.env, '_key_body_ids'):
                            key_body_ids = self.env._key_body_ids
                            ref_pos = self.env._ref_body_pos[i, key_body_ids]
                            curr_pos = self.env.rigid_body_states[i, key_body_ids, 0:3]
                            pose_err = torch.mean(torch.norm(ref_pos - curr_pos, dim=-1)).item()
                            pose_errors[i] += pose_err

                    # Roll/Pitch
                    if hasattr(self.env, 'roll') and hasattr(self.env, 'pitch'):
                        max_roll[i] = torch.maximum(max_roll[i], torch.abs(self.env.roll[i]))
                        max_pitch[i] = torch.maximum(max_pitch[i], torch.abs(self.env.pitch[i]))

            # 检查终止
            for i in range(batch_size):
                if dones[i] and not done_mask[i]:
                    done_mask[i] = True
                    termination_reasons[i] = self._detect_termination_reason(i)

            if done_mask[:batch_size].all():
                break

        # 计算完成率
        completion_rates = (episode_lengths * self.dt) / motion_lengths
        completion_rates = torch.clamp(completion_rates, 0.0, 1.0)

        # 汇总结果
        results = []
        for i, motion_idx in enumerate(batch_motion_ids):
            # 平均误差
            steps = max(episode_lengths[i].item(), 1)
            avg_joint_error = (joint_errors[i] / steps).item()
            avg_pose_error = (pose_errors[i] / steps).item()
            completion = completion_rates[i].item()

            # ==================== 基于跟踪误差的难度分数 ====================
            # 完全基于跟踪质量评分，不考虑是否提前终止
            #
            # 评分逻辑：
            # 1. 关节误差越大 → 跟踪越差 → 难度越高
            # 2. 姿态误差越大 → 跟踪越差 → 难度越高
            # 3. Roll/Pitch越大 → 稳定性越差 → 难度越高

            # 关节误差惩罚（线性，权重100）
            # avg_joint_error 通常在 0.0-1.0 之间
            joint_score = avg_joint_error * 100.0

            # 姿态误差惩罚（线性，权重100）
            # avg_pose_error 通常在 0.0-1.0 之间
            pose_score = avg_pose_error * 100.0

            # 稳定性惩罚：Roll/Pitch角度
            roll_deg = abs(math.degrees(max_roll[i].item()))
            pitch_deg = abs(math.degrees(max_pitch[i].item()))
            stability_score = (roll_deg + pitch_deg) * 2.0  # 每度2分

            # 综合难度分数（纯基于跟踪误差）
            difficulty_score = joint_score + pose_score + stability_score

            results.append({
                'motion_idx': motion_idx,
                'motion_name': self.motion_names[motion_idx],
                'motion_file': self.motion_files[motion_idx],
                'completion_rate': completion,
                'episode_length': episode_lengths[i].item(),
                'motion_length': motion_lengths[i].item(),
                'avg_joint_error': avg_joint_error,
                'avg_pose_error': avg_pose_error,
                'max_roll': max_roll[i].item(),
                'max_pitch': max_pitch[i].item(),
                'termination_reason': termination_reasons[i],
                'difficulty_score': difficulty_score,
                'joint_score': joint_score,
                'pose_score': pose_score,
                'stability_score': stability_score,
            })

        return results

    def _detect_termination_reason(self, env_idx):
        """检测终止原因"""
        # 1. 检查是否是动作完成
        motion_time = self.env.episode_length_buf[env_idx].item() * self.dt
        motion_len = self.env._motion_lib.get_motion_length(
            self.env._motion_ids[env_idx:env_idx+1]
        ).item()
        if motion_time >= motion_len * 0.95:
            return 'completed'

        # 2. 检查接触力
        if hasattr(self.env, 'termination_contact_indices'):
            contact_forces_norm = torch.norm(
                self.env.contact_forces[env_idx, self.env.termination_contact_indices, :],
                dim=-1
            )
            if torch.any(contact_forces_norm > 1.0):
                return 'contact'

        # 3. 检查高度差
        if hasattr(self.env, '_ref_root_pos'):
            ref_height = self.env._ref_root_pos[env_idx, 2].item()
            curr_height = self.env.root_states[env_idx, 2].item()
            threshold = getattr(self.env.cfg.rewards, 'root_height_diff_threshold', 0.5)
            if abs(curr_height - ref_height) > threshold:
                return 'height_diff'

        # 4. 检查Roll/Pitch
        if hasattr(self.env, 'roll') and hasattr(self.env, 'pitch'):
            roll_threshold = getattr(self.env.cfg.rewards, 'termination_roll', 1.0)
            pitch_threshold = getattr(self.env.cfg.rewards, 'termination_pitch', 1.0)
            if abs(self.env.roll[env_idx]) > roll_threshold or abs(self.env.pitch[env_idx]) > pitch_threshold:
                return 'roll_pitch'

        # 5. 检查姿态跟踪
        if hasattr(self.env, '_pose_termination') and self.env._pose_termination:
            return 'pose_tracking'

        # 6. 检查Root跟踪
        if hasattr(self.env, '_track_root') and self.env._track_root:
            if hasattr(self.env, '_ref_root_pos') and hasattr(self.env, '_root_tracking_termination_dist'):
                root_pos_diff = self.env._ref_root_pos[env_idx, 0:2] - self.env.root_states[env_idx, 0:2]
                root_pos_dist = torch.norm(root_pos_diff).item()
                if root_pos_dist > self.env._root_tracking_termination_dist:
                    return 'root_tracking'

        return 'unknown'


def load_pytorch_policy(checkpoint_path, env, device):
    """从PyTorch checkpoint加载策略"""
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取actor模型
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'actor_state_dict' in checkpoint:
        state_dict = checkpoint['actor_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 这里需要根据实际的环境结构来获取actor
    # 简化版：直接从环境的runner获取
    class PyTorchPolicyWrapper:
        def __init__(self, actor):
            self.actor = actor
            self.actor.eval()

        def __call__(self, obs):
            with torch.no_grad():
                if isinstance(obs, dict):
                    # 处理字典类型的obs
                    return self.actor(**obs)
                return self.actor(obs)

    # 尝试从环境中获取actor
    if hasattr(env, 'actor'):
        env.actor.load_state_dict(state_dict)
        return PyTorchPolicyWrapper(env.actor)
    else:
        # 需要创建actor
        print("Warning: Cannot extract actor from env, using dummy policy")
        return lambda obs: torch.zeros(obs.shape[0], env.num_actions, device=device)


def load_onnx_policy(policy_path, device):
    """加载ONNX策略"""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime-gpu")

    class OnnxPolicyWrapper:
        def __init__(self, session, input_name, device):
            self.session = session
            self.input_name = input_name
            self.device = device

        def __call__(self, obs_tensor):
            obs_np = obs_tensor.cpu().numpy().astype(np.float32)
            outputs = self.session.run(None, {self.input_name: obs_np})
            return torch.from_numpy(outputs[0]).to(self.device)

    providers = []
    if device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name

    return OnnxPolicyWrapper(session, input_name, device)


def save_results_to_csv(results, output_path):
    """保存结果到CSV文件"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if not results:
            return

        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TWIST2 动作难度评估脚本")
    parser.add_argument('--task', type=str, default='g1_priv_mimic', help='任务名称')
    parser.add_argument('--checkpoint', type=str, default=None, help='PyTorch checkpoint路径')
    parser.add_argument('--policy', type=str, default=None, help='ONNX策略路径')
    parser.add_argument('--motion_config', type=str, required=True, help='动作配置YAML路径')
    parser.add_argument('--output', type=str, default='difficulty_scores.csv', help='输出CSV文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--num_envs', type=int, default=256, help='并行环境数')
    parser.add_argument('--max_steps', type=int, default=5000, help='最大仿真步数')
    parser.add_argument('--save_video', action='store_true', help='保存可视化视频')

    args = parser.parse_args()

    console = Console()

    # 验证输入
    if args.checkpoint is None and args.policy is None:
        console.print("[red]Error: 必须指定 --checkpoint 或 --policy[/red]")
        return

    if args.checkpoint is not None and args.policy is not None:
        console.print("[yellow]Warning: 同时指定了checkpoint和policy，将使用checkpoint[/yellow]")

    if not os.path.exists(args.motion_config):
        console.print(f"[red]Error: Motion config not found: {args.motion_config}[/red]")
        return

    console.print(f"\n[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]              TWIST2 动作难度评估                        [/bold cyan]")
    console.print(f"[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]\n")

    # 导入必要模块（避免循环导入：先导入所有环境，再导入task_registry）
    from legged_gym.gym_utils import get_args, task_registry
    from legged_gym.envs.base.humanoid_mimic import HumanoidMimic

    # 解析动作配置
    root_path, motions_by_folder = parse_motion_config(args.motion_config)
    total_motions = sum(len(m) for m in motions_by_folder.values())

    console.print(f"[green]发现 {total_motions} 个动作，分布在 {len(motions_by_folder)} 个文件夹中[/green]\n")

    # 设置环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.motion.motion_file = args.motion_config
    env_cfg.env.num_envs = args.num_envs

    # 关键：禁用resume模式，避免make_alg_runner尝试加载checkpoint
    train_cfg.runner.resume = False

    # 评估模式配置
    env_cfg.env.debug_viz = args.save_video
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

    # 关键：评估时禁用pose_termination，否则会过早终止
    if hasattr(env_cfg.env, 'pose_termination'):
        env_cfg.env.pose_termination = False
    if hasattr(env_cfg.env, 'enable_early_termination'):
        env_cfg.env.enable_early_termination = False

    # 环境参数
    class Args:
        def __init__(self):
            self.task = args.task
            self.device = args.device
            self.headless = not args.save_video
            self.use_jit = False
            self.record_video = False
            self.record_log = False
            self.jit_path = None
            self.teleop_mode = False
            self.num_envs = args.num_envs
            self.seed = None
            self.rows = None
            self.cols = None
            self.no_rand = False
            self.max_iterations = None
            self.resume = False
            self.resumeid = None  # task_registry.make_alg_runner 需要
            self.proj_name = None  # task_registry.make_alg_runner 需要
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

    # 加载策略
    console.print("[cyan]加载策略...[/cyan]")
    if args.checkpoint:
        # 首先创建环境
        env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=env_cfg)

        # 关键：直接禁用环境的termination属性（评估不需要提前终止）
        if hasattr(env, '_pose_termination'):
            env._pose_termination = False
        if hasattr(env, '_track_root'):
            env._track_root = False

        # 替换check_termination方法，只保留超时终止
        original_check_termination = env.check_termination
        def eval_check_termination():
            # 只检查超时，不检查其他终止条件
            env.reset_buf = env.time_out_buf.clone()
        env.check_termination = eval_check_termination
        console.print("[yellow]评估模式：已替换check_termination方法，仅保留超时终止[/yellow]")

        # 然后加载策略
        if args.checkpoint.endswith('.onnx'):
            policy = load_onnx_policy(args.checkpoint, args.device)
            console.print(f"[green]加载ONNX策略: {args.checkpoint}[/green]")
        else:
            # 使用PPO runner加载策略（无需config.json）
            from rsl_rl.runners import OnPolicyRunner
            # 使用当前的train_cfg创建runner（resume=False）
            runner, _ = task_registry.make_alg_runner(
                env=env,
                name=args.task,
                args=env_args,
                train_cfg=train_cfg
            )
            # 手动加载checkpoint
            console.print(f"[cyan]加载checkpoint: {args.checkpoint}[/cyan]")
            runner.load(args.checkpoint)
            policy = runner.get_inference_policy(device=args.device)
            console.print(f"[green]加载PyTorch策略: {args.checkpoint}[/green]")
    else:
        # 创建环境后加载ONNX策略
        env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=env_cfg)
        policy = load_onnx_policy(args.policy, args.device)
        console.print(f"[green]加载ONNX策略: {args.policy}[/green]")

    # 创建评估器
    evaluator = DifficultyEvaluator(env, policy, args.device, args)

    # 评估所有动作
    all_results = []
    completed_folders = set()

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False
    ) as progress:

        total_task = progress.add_task("[yellow]评估进度", total=total_motions)

        # 按批次评估
        for batch_start in range(0, evaluator.num_motions, args.num_envs):
            batch_end = min(batch_start + args.num_envs, evaluator.num_motions)
            batch_motion_ids = list(range(batch_start, batch_end))

            # 评估这批动作
            results = evaluator.evaluate_motion_batch(batch_motion_ids)
            all_results.extend(results)

            # 更新进度
            progress.update(total_task, advance=len(batch_motion_ids))

    # 保存结果
    console.print(f"\n[cyan]保存结果到: {args.output}[/cyan]")
    save_results_to_csv(all_results, args.output)

    # 统计摘要
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]                        评估摘要                            [/bold cyan]")
    console.print(f"[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

    difficulty_scores = [r['difficulty_score'] for r in all_results]
    completion_rates = [r['completion_rate'] for r in all_results]

    # 按难度等级分类（基于跟踪误差的评分）
    # 难度分 = joint_error*100 + pose_error*100 + (roll+pitch)*2
    easy_motions = [r for r in all_results if r['difficulty_score'] < 30]
    medium_motions = [r for r in all_results if 30 <= r['difficulty_score'] < 60]
    hard_motions = [r for r in all_results if 60 <= r['difficulty_score'] < 100]
    very_hard_motions = [r for r in all_results if r['difficulty_score'] >= 100]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("难度等级", style="cyan")
    table.add_column("数量", justify="right")
    table.add_column("占比", justify="right")
    table.add_column("平均完成率", justify="right")
    table.add_column("平均难度分", justify="right")

    table.add_row(
        "[green]简单[/green]",
        str(len(easy_motions)),
        f"{len(easy_motions)/len(all_results)*100:.1f}%",
        f"{np.mean([r['completion_rate'] for r in easy_motions]):.3f}" if easy_motions else "N/A",
        f"{np.mean([r['difficulty_score'] for r in easy_motions]):.3f}" if easy_motions else "N/A",
    )
    table.add_row(
        "[yellow]中等[/yellow]",
        str(len(medium_motions)),
        f"{len(medium_motions)/len(all_results)*100:.1f}%",
        f"{np.mean([r['completion_rate'] for r in medium_motions]):.3f}" if medium_motions else "N/A",
        f"{np.mean([r['difficulty_score'] for r in medium_motions]):.3f}" if medium_motions else "N/A",
    )
    table.add_row(
        "[orange3]困难[/orange3]",
        str(len(hard_motions)),
        f"{len(hard_motions)/len(all_results)*100:.1f}%",
        f"{np.mean([r['completion_rate'] for r in hard_motions]):.3f}" if hard_motions else "N/A",
        f"{np.mean([r['difficulty_score'] for r in hard_motions]):.1f}" if hard_motions else "N/A",
    )
    table.add_row(
        "[red]极难[/red]",
        str(len(very_hard_motions)),
        f"{len(very_hard_motions)/len(all_results)*100:.1f}%",
        f"{np.mean([r['completion_rate'] for r in very_hard_motions]):.3f}" if very_hard_motions else "N/A",
        f"{np.mean([r['difficulty_score'] for r in very_hard_motions]):.1f}" if very_hard_motions else "N/A",
    )

    console.print(table)

    console.print(f"\n[bold]整体统计:[/bold]")
    console.print(f"  总动作数: {len(all_results)}")
    console.print(f"  平均完成率: {np.mean(completion_rates):.3f}")
    console.print(f"  平均难度分: {np.mean(difficulty_scores):.1f}")
    console.print(f"  难度分范围: [{np.min(difficulty_scores):.1f}, {np.max(difficulty_scores):.1f}]")

    # 显示最难和最简单的动作
    sorted_results = sorted(all_results, key=lambda x: x['difficulty_score'], reverse=True)
    console.print(f"\n[bold red]最难的5个动作:[/bold red]")
    for i, r in enumerate(sorted_results[:5]):
        console.print(f"  {i+1}. {r['motion_name'][:45]:45s} | 难度: [red]{r['difficulty_score']:.1f}[/red] | 完成率: {r['completion_rate']:.3f} | 原因: {r['termination_reason']}")

    console.print(f"\n[bold green]最简单的5个动作:[/bold green]")
    for i, r in enumerate(sorted_results[-5:]):
        console.print(f"  {i+1}. {r['motion_name'][:45]:45s} | 难度: [green]{r['difficulty_score']:.1f}[/green] | 完成率: {r['completion_rate']:.3f}")

    # 清理
    del env
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
