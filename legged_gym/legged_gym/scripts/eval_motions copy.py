#!/usr/bin/env python3
"""
TWIST2 动作评估脚本 (v4 - 热插拔 MotionLib 版本)

功能说明：
本脚本采用“热插拔”技术，在单进程中动态替换动作库，从而避免了：
1. IsaacGym 不允许重启仿真的限制 ("Foundation object exists already")
2. 一次性加载所有动作导致的 OOM
3. 多进程架构的复杂性和潜在死锁

工作原理：
1. 初始化一次仿真环境。
2. 遍历每个数据集文件夹。
3. 为每个文件夹创建一个新的 MotionLib 对象，直接替换环境中的旧对象。
4. 动态更新环境的内部缓冲区（如 motion_difficulty）以匹配新动作库。
5. 评估并释放内存。

使用方法：
    python eval_motions.py --policy <onnx路径> --motion_config <yaml路径> [其他选项]
"""
# 导入 IsaacGym (必须在 torch 之前，虽然这里已经有 torch 导入保护，但保持习惯)
from isaacgym import gymapi
import sys
import os
# 添加路径以导入项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from legged_gym.envs import *
from legged_gym.gym_utils import task_registry
from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
# 导入 MotionLib 类，用于手动创建实例
from pose.utils.motion_lib_pkl import MotionLib

import argparse
import json


import time
import tempfile
from collections import defaultdict
from datetime import datetime
import yaml
import gc
import torch
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

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# ============================================================================
# Core Functions
# ============================================================================

class OnnxPolicyWrapper:
    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor):
        # 局部导入确保安全
        import torch
        obs_np = obs_tensor.cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        actions = torch.from_numpy(outputs[self.output_index])
        return actions

def load_onnx_policy(policy_path, device):
    if ort is None:
        raise ImportError("onnxruntime is required")
    
    providers = []
    if device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return OnnxPolicyWrapper(session, input_name)

def parse_motion_config(config_path):
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

def hot_swap_motion_lib(env, motion_config_path, device):
    """
    热插拔动作库：替换环境中的 MotionLib 并更新相关依赖
    这是本脚本的核心黑科技。
    """
    # 1. 创建新的 MotionLib
    new_motion_lib = MotionLib(
        motion_file=motion_config_path,
        device=device,
        sample_ratio=env.cfg.motion.sample_ratio,
        motion_decompose=env.cfg.motion.motion_decompose,
        motion_smooth=env.cfg.motion.motion_smooth
    )
    
    # 2. 替换环境中的引用
    env._motion_lib = new_motion_lib
    
    # 3. 更新依赖于动作数量的缓冲区
    num_motions = new_motion_lib.num_motions()
    
    # 更新难度系数缓冲
    env.motion_difficulty = 100 * torch.ones((num_motions), device=device, dtype=torch.float)
    
    # 更新终止距离缓冲
    env.motion_termination_dist = torch.ones((num_motions), device=device, dtype=torch.float) * env._pose_termination_dist
    
    # 更新动作名称列表
    env.motion_names = new_motion_lib.get_motion_names()
    
    # 更新错误感知采样缓冲
    env.max_key_body_error = torch.zeros((num_motions), device=device, dtype=torch.float)

    # 4. 关键修复：更新关键体索引 (Key Body IDs)
    # 如果新数据集的骨骼结构有微小差异，或者 MotionLib 重新索引了，这个必须更新
    if hasattr(env, '_key_body_ids_motion') and hasattr(env.cfg.motion, 'key_bodies'):
        try:
            env._key_body_ids_motion = new_motion_lib.get_key_body_idx(key_body_names=env.cfg.motion.key_bodies)
        except Exception as e:
            print(f"[Warning] Failed to update key body ids: {e}")

    # 5. 更新最大动作长度 (Max Episode Length)
    # 如果新动作比旧动作长得多，旧的 max_episode_length 可能会导致意外的超时/重置
    try:
        max_len = 0
        # motion_lib.get_motion_length 可能返回 tensor
        # 我们使用 batch 方式获取所有长度，然后取最大值，效率更高
        all_motion_ids = torch.arange(num_motions, device=device, dtype=torch.int64)
        all_lengths = new_motion_lib.get_motion_length(all_motion_ids)
        max_len = all_lengths.max().item() # .item() 转换为 Python float
        
        env.max_episode_length_s = max_len
        env.max_episode_length = np.ceil(max_len / env.dt)
    except Exception as e:
        print(f"[Warning] Failed to update max episode length: {e}")
    
    # 强制进行垃圾回收，释放旧 MotionLib 的内存
    gc.collect()
    torch.cuda.empty_cache()
    
    return num_motions

def evaluate_current_motions(policy, env, device, max_steps, progress, folder_task_id):
    """评估当前加载的所有动作"""
    motion_lib = env._motion_lib
    num_motions = motion_lib.num_motions()
    motion_names = motion_lib.get_motion_names()
    motion_files = motion_lib._motion_files
    num_envs = env.num_envs
    
    results = []
    
    # 批量处理
    for batch_start in range(0, num_motions, num_envs):
        batch_end = min(batch_start + num_envs, num_motions)
        batch_motion_ids = list(range(batch_start, batch_end))
        batch_size = len(batch_motion_ids)
        
        # 准备 Motion ID
        motion_ids_tensor = torch.tensor(batch_motion_ids, device=device, dtype=torch.int64)
        
        # 填充策略优化：使用当前 batch 中最短的动作进行填充
        # 这样可以确保 padding 不会无谓地增加 max_steps 的等待时间
        # 虽然我们后面会根据有效 batch 提前退出，但这仍然是一个好的实践
        current_batch_lens = motion_lib.get_motion_length(torch.tensor(batch_motion_ids, device=device))
        min_len_idx = torch.argmin(current_batch_lens).item()
        padding_motion_idx = batch_motion_ids[min_len_idx]
        
        if batch_size < num_envs:
            padding = torch.full((num_envs - batch_size,), padding_motion_idx, device=device, dtype=torch.int64)
            motion_ids_tensor = torch.cat([motion_ids_tensor, padding])
            
        # 重置环境
        all_env_ids = torch.arange(num_envs, device=device)
        HumanoidMimic.reset_idx(env, all_env_ids, motion_ids=motion_ids_tensor)
        obs = env.get_observations()
        
        # 追踪
        episode_lengths = torch.zeros(num_envs, device=device)
        motion_lengths = motion_lib.get_motion_length(motion_ids_tensor)
        done_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        
        # 动态计算 max_steps
        # 注意：这里我们计算的是 motion_ids_tensor 中的最大长度
        # 如果 padding 使用的是最短动作，那么 max_len 将完全由有效 batch 中的动作决定
        max_motion_len = motion_lengths.max().item()
        actual_max_steps = min(max_steps, int(max_motion_len / env.dt * 1.2) + 10)
        
        # print("actual_max_steps", actual_max_steps)

        # 仿真
        for _ in range(actual_max_steps):
            with torch.no_grad():
                # 优化：只对有效 batch 运行 Policy，减少推理开销
                # 对于 padding 环境，直接给零动作
                valid_obs = obs[:batch_size]
                valid_actions = policy(valid_obs.detach())
                
                if isinstance(valid_actions, torch.Tensor):
                    valid_actions = valid_actions.to(device)
                else:
                    valid_actions = torch.from_numpy(valid_actions).to(device)
                
                # 构造完整动作张量
                # padding 环境的动作为 0
                if batch_size < num_envs:
                    actions = torch.zeros((num_envs, valid_actions.shape[1]), device=device, dtype=valid_actions.dtype)
                    actions[:batch_size] = valid_actions
                else:
                    actions = valid_actions
            
            obs, _, _, dones, _ = env.step(actions)
            episode_lengths[~done_mask] += 1
            done_mask = done_mask | dones
            
            # 关键优化：只检查有效 batch 是否全部完成
            # padding 环境是否完成无关紧要
            if done_mask[:batch_size].all():
                break
                
        # 结果
        completion_rates = (episode_lengths * env.dt) / motion_lengths
        completion_rates = torch.clamp(completion_rates, 0.0, 1.0)
        
        for i, idx in enumerate(batch_motion_ids):
            results.append({
                'motion_name': motion_names[idx],
                'motion_file': motion_files[idx],
                'completion_rate': completion_rates[i].item()
            })
            
        # 更新进度
        progress.update(folder_task_id, advance=batch_size)
        
    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate policy (Hot-Swap Mode)')
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--motion_config', type=str, required=True)
    parser.add_argument('--task', type=str, default='g1_stu_future')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_envs', type=int, default=256)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    args = parser.parse_args()
    
    console = Console()
    console.print(f"\n[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]             TWIST2 Motion Evaluation (Hot-Swap)            [/bold cyan]")
    console.print(f"[bold cyan]══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # 1. 解析配置
    root_path, motions_by_folder = parse_motion_config(args.motion_config)
    total_motions = sum(len(m) for m in motions_by_folder.values())
    console.print(f"[green]Found {total_motions} motions in {len(motions_by_folder)} folders[/green]")
    
    # 2. 加载 Policy
    policy = load_onnx_policy(args.policy, args.device)
    
    # 3. 初始化环境 (只创建一次!)
    # 我们先用第一个文件夹的配置来初始化环境，之后再替换
    first_folder = next(iter(motions_by_folder.values()))
    
    # 创建临时配置用于初始化
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'root_path': root_path, 'motions': first_folder}, f)
        init_yaml_path = f.name
        
    console.print(f"[cyan]Initializing environment...[/cyan]")
    
    # 环境配置
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.motion.motion_file = init_yaml_path
    env_cfg.env.num_envs = args.num_envs
    
    # 强制评估配置
    env_cfg.env.debug_viz = False
    env_cfg.env.episode_length_s = 60
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    if hasattr(env_cfg, 'motion'): env_cfg.motion.motion_curriculum = False
    env_cfg.env.rand_reset = False
    
    # 参数对象
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
            self.seed = None; self.rows = None; self.cols = None
            self.no_rand = True; self.max_iterations = None; self.resume = False
            self.experiment_name = None; self.run_name = None; self.load_run = None
            self.checkpoint = None; self.fix_action_std = False; self.teacher_exptid = None
            self.teacher_checkpoint = None; self.eval_student = False; self.config_overrides = {}
            from isaacgym import gymapi
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

    try:
        env, _ = task_registry.make_env(name=args.task, args=Args(), env_cfg=env_cfg)
    except Exception as e:
        console.print(f"[red]Env creation failed: {e}[/red]")
        return
        
    os.unlink(init_yaml_path) # 删除临时文件
    
    # 4. 评估循环
    # 准备输出文件名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.basename(args.policy).replace('.onnx', '')
        args.output = f"eval_results_{name}_{timestamp}.json"
    
    console.print(f"[cyan]Results will be saved to: {args.output}[/cyan]")

    all_results = []
    folder_stats = {}
    completed_folders = set()
    
    # 如果启用了恢复模式，读取已有结果
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                existing_data = json.load(f)
            
            # 恢复已完成的文件夹列表
            completed_folders = set(existing_data.get('folder_stats', {}).keys())
            
            # 恢复已有的结果数据
            all_results = existing_data.get('motion_results', [])
            folder_stats = existing_data.get('folder_stats', {})
            
            console.print(f"[yellow]Resume mode: Found {len(completed_folders)} completed folders[/yellow]")
            for fn in sorted(completed_folders):
                console.print(f"  [dim]Skipping: {fn}[/dim]")
        except Exception as e:
            console.print(f"[red]Warning: Failed to load existing results: {e}[/red]")
            console.print(f"[red]Starting from scratch...[/red]")
            completed_folders = set()
    
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
            
            total_task = progress.add_task("[yellow]Overall", total=total_motions)
            
            for folder_idx, (folder_name, folder_motions) in enumerate(sorted(motions_by_folder.items())):
                # 跳过已完成的文件夹
                if folder_name in completed_folders:
                    progress.update(total_task, advance=len(folder_motions))
                    continue
                
                folder_start = time.time()
                
                # ... (中间生成临时 YAML 和热插拔代码保持不变) ...
                # 4.1 生成临时 YAML
                temp_yaml = os.path.join(temp_dir, f"{folder_name}.yaml")
                with open(temp_yaml, 'w') as f:
                    yaml.dump({'root_path': root_path, 'motions': folder_motions}, f)
                
                # 4.2 热插拔动作库!
                hot_swap_motion_lib(env, temp_yaml, args.device)
                
                # 4.3 评估
                num_ms = len(folder_motions)
                folder_task = progress.add_task(f"[cyan]Processing: {folder_name}", total=num_ms)
                
                results = evaluate_current_motions(policy, env, args.device, args.max_steps, progress, folder_task)
                
                # 4.4 收集数据
                rates = []
                for r in results:
                    r['folder'] = folder_name
                    all_results.append(r)
                    rates.append(r['completion_rate'])
                
                # 统计
                mean_rate = np.mean(rates) if rates else 0.0
                folder_stats[folder_name] = {
                    'count': len(rates), 'mean': mean_rate,
                    'std': np.std(rates) if rates else 0.0,
                    'min': np.min(rates) if rates else 0.0,
                    'max': np.max(rates) if rates else 0.0
                }
                
                # 4.5 更新界面
                progress.update(total_task, advance=num_ms)
                progress.remove_task(folder_task)
                
                elapsed = time.time() - folder_start
                color = "green" if mean_rate >= 0.8 else "yellow" if mean_rate >= 0.5 else "red"
                console.print(f"  ✓ [bold {color}]{folder_name}[/bold {color}]: Mean={mean_rate:.4f}, Time={elapsed:.1f}s")
                
                # 4.6 💎 关键改进：每处理一个文件夹就保存一次（增量备份）
                try:
                    current_overall_rates = [r['completion_rate'] for r in all_results]
                    current_stats = {
                        'count': len(current_overall_rates),
                        'mean': np.mean(current_overall_rates) if current_overall_rates else 0.0,
                        'std': np.std(current_overall_rates) if current_overall_rates else 0.0,
                        'min': np.min(current_overall_rates) if current_overall_rates else 0.0,
                        'max': np.max(current_overall_rates) if current_overall_rates else 0.0
                    }
                    
                    # 写入临时文件再重命名，防止写入中断损坏文件
                    temp_output = args.output + ".tmp"
                    with open(temp_output, 'w') as f:
                        json.dump({
                            'partial_save': True, # 标记这是一个未完成的保存
                            'timestamp': datetime.now().isoformat(),
                            'overall_stats': current_stats,
                            'folder_stats': folder_stats,
                            'motion_results': all_results
                        }, f, indent=2)
                    os.replace(temp_output, args.output) # 原子替换
                    
                except Exception as e:
                    console.print(f"[red]Warning: Failed to save incremental results: {e}[/red]")

    # 5. 保存最终结果（覆盖 partial_save 标记）
        
    overall_rates = [r['completion_rate'] for r in all_results]
    overall_stats = {
        'count': len(overall_rates),
        'mean': np.mean(overall_rates) if overall_rates else 0.0,
        'std': np.std(overall_rates) if overall_rates else 0.0,
        'min': np.min(overall_rates) if overall_rates else 0.0,
        'max': np.max(overall_rates) if overall_rates else 0.0
    }
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"Overall Mean: [green]{overall_stats['mean']:.4f}[/green]")
    console.print(f"[green]Saved to: {args.output}[/green]")
    
    with open(args.output, 'w') as f:
        json.dump({
            'overall_stats': overall_stats,
            'folder_stats': folder_stats,
            'motion_results': all_results
        }, f, indent=2)

if __name__ == "__main__":
    main()
