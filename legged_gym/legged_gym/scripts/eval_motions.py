#!/usr/bin/env python3
"""
TWIST2 动作评估脚本 (v5 - 多进程 + 断点续评)

核心设计：
1. 主进程：解析配置，调度子进程，收集结果
2. 子进程：每个子进程只处理一个文件夹，完成后退出彻底释放资源
3. 断点续评：指定 --resume 和 --output 后，会跳过已完成的文件夹

使用方法：
    python eval_motions.py --policy <onnx路径> --motion_config <yaml路径> [其他选项]
    
断点续评：
    python eval_motions.py --policy xxx.onnx --motion_config xxx.yaml --output results.json --resume
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--motion_config', type=str, required=True)
    parser.add_argument('--task', type=str, default='g1_stu_future')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_envs', type=int, default=256)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    parser.add_argument('--subprocess_mode', action='store_true', help='Run in worker mode (internal use)')
    
    args = parser.parse_args()
    
    if args.subprocess_mode:
        worker_main(args)
    else:
        coordinator_main(args)
