#!/usr/bin/env python3
"""
TWIST2 Motion Player - 可视化动作播放脚本

功能：
1. 使用 ONNX 模型播放指定的 pkl 动作文件
2. 同时可视化参考动作和当前动作
3. 失败时输出警告但不暂停
4. 支持录制视频

用法:
    python play_motion.py --policy xxx.onnx --motion_file xxx.pkl --record_video
"""

import os
import sys
import argparse
import tempfile

# 必须在其他 import 之前导入 isaacgym
import isaacgym
import torch
import numpy as np
import yaml
import onnxruntime as ort
from tqdm import tqdm
from termcolor import cprint
from rich.console import Console
from rich.table import Table

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.gym_utils import task_registry


console = Console()


def create_motion_config(motion_file_path):
    """为单个动作文件创建临时 yaml 配置"""
    motion_dir = os.path.dirname(os.path.abspath(motion_file_path))
    motion_basename = os.path.basename(motion_file_path)
    
    temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_config = {
        'root_path': motion_dir,
        'motions': [{'file': motion_basename, 'weight': 1.0}]
    }
    yaml.dump(temp_config, temp_yaml)
    temp_yaml.close()
    return temp_yaml.name


def load_onnx_policy(policy_path, device):
    """加载 ONNX 策略模型"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def check_termination_conditions(env, step, console):
    """检查终止条件并输出警告"""
    warnings = []
    
    # 1. 检查高度
    base_height = env.root_states[0, 2].item()
    if hasattr(env, '_ref_root_pos'):
        ref_height = env._ref_root_pos[0, 2].item()
        height_diff = abs(base_height - ref_height)
        threshold = env.cfg.rewards.root_height_diff_threshold if hasattr(env.cfg.rewards, 'root_height_diff_threshold') else 0.2
        if height_diff > threshold:
            warnings.append(f"[yellow]⚠ 高度偏差: {height_diff:.3f}m > {threshold:.2f}m[/yellow]")
    
    # 2. 检查 Roll/Pitch
    roll = env.roll[0].item() if hasattr(env, 'roll') else 0
    pitch = env.pitch[0].item() if hasattr(env, 'pitch') else 0
    roll_thresh = env.cfg.rewards.termination_roll if hasattr(env.cfg.rewards, 'termination_roll') else 1.0
    pitch_thresh = env.cfg.rewards.termination_pitch if hasattr(env.cfg.rewards, 'termination_pitch') else 1.0
    
    if abs(roll) > roll_thresh * 0.7:  # 70% 阈值时警告
        warnings.append(f"[yellow]⚠ Roll 接近极限: {np.degrees(roll):.1f}°[/yellow]")
    if abs(pitch) > pitch_thresh * 0.7:
        warnings.append(f"[yellow]⚠ Pitch 接近极限: {np.degrees(pitch):.1f}°[/yellow]")
    
    # 3. 检查速度
    if hasattr(env, 'root_states'):
        vel = torch.norm(env.root_states[0, 7:10]).item()
        if vel > 4.0:  # 接近 5.0 阈值
            warnings.append(f"[yellow]⚠ 速度过大: {vel:.2f} m/s[/yellow]")
    
    # 4. 检查姿态跟踪（使用局部坐标系，与 humanoid_mimic.check_termination 保持一致）
    if hasattr(env, '_pose_termination') and env._pose_termination:
        if hasattr(env, '_key_body_ids') and hasattr(env, '_ref_body_pos'):
            from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos
            
            body_pos = env.rigid_body_states[0:1, env._key_body_ids, 0:3] - env.rigid_body_states[0:1, 0:1, 0:3]
            tar_body_pos = env._ref_body_pos[0:1, env._key_body_ids] - env._ref_root_pos[0:1, None, :]
            
            # 转换到局部坐标系
            if hasattr(env, 'global_obs') and not env.global_obs:
                body_pos = convert_to_local_root_body_pos(env.root_states[0:1, 3:7], body_pos)
                tar_body_pos = convert_to_local_root_body_pos(env._ref_root_rot[0:1], tar_body_pos)
            
            body_pos_diff = tar_body_pos - body_pos
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
            max_dist = torch.sqrt(torch.max(body_pos_dist)).item()
            
            term_dist = env._pose_termination_dist if hasattr(env, '_pose_termination_dist') else 1.0
            if max_dist > term_dist * 0.5:  # 50% 阈值时警告
                color = "red" if max_dist > term_dist else "yellow"
                warnings.append(f"[{color}]⚠ 姿态偏差: {max_dist:.3f}m (阈值 {term_dist:.2f}m)[/{color}]")
    
    # 5. 检查接触力
    if hasattr(env, 'termination_contact_indices') and hasattr(env, 'contact_forces'):
        contact_force = torch.norm(env.contact_forces[0, env.termination_contact_indices, :], dim=-1)
        if torch.any(contact_force > 0.8).item():  # 接近 1.0 阈值
            warnings.append(f"[red]⚠ 检测到非法接触力[/red]")
    
    return warnings


def main():
    parser = argparse.ArgumentParser(description="TWIST2 Motion Player - 可视化动作播放")
    parser.add_argument('--policy', type=str, required=True, help='Path to ONNX policy file')
    parser.add_argument('--motion_file', type=str, required=True, help='Path to motion PKL file')
    parser.add_argument('--task', type=str, default='g1_stu_future', help='Task name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--record_video', action='store_true', help='Record video')
    parser.add_argument('--output_video', type=str, default=None, help='Output video path')
    parser.add_argument('--headless', action='store_true', help='Run headless')
    parser.add_argument('--no_pose_term', action='store_true', help='Disable pose termination')
    parser.add_argument('--max_steps', type=int, default=5000, help='Maximum simulation steps')
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.policy):
        cprint(f"Error: Policy file not found: {args.policy}", "red")
        return
    if not os.path.exists(args.motion_file):
        cprint(f"Error: Motion file not found: {args.motion_file}", "red")
        return
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]        TWIST2 Motion Player - 可视化动作播放               [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    console.print(f"Policy: [green]{args.policy}[/green]")
    console.print(f"Motion: [green]{args.motion_file}[/green]")
    
    # 加载策略
    console.print("\n[yellow]Loading ONNX policy...[/yellow]")
    session, input_name, output_name = load_onnx_policy(args.policy, args.device)
    console.print("[green]✓ Policy loaded[/green]")
    
    # 创建临时配置
    temp_yaml_path = create_motion_config(args.motion_file)
    console.print(f"[dim]Temp config: {temp_yaml_path}[/dim]")
    
    # 设置环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.motion.motion_file = temp_yaml_path
    env_cfg.env.num_envs = 1
    env_cfg.env.debug_viz = True
    env_cfg.env.episode_length_s = 120
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.env.rand_reset = False
    
    if hasattr(env_cfg, 'motion'):
        env_cfg.motion.motion_curriculum = False
    
    if args.no_pose_term:
        env_cfg.env.pose_termination = False
        console.print("[yellow]⚠ 已禁用姿态终止检测[/yellow]")
    
    # 录制视频配置
    if args.record_video:
        env_cfg.env.record_video = True
        console.print("[green]✓ Video recording enabled[/green]")
    
    # 创建环境
    console.print("\n[yellow]Creating environment...[/yellow]")
    
    class EnvArgs:
        def __init__(self):
            self.sim_device = args.device
            self.rl_device = args.device
            self.compute_device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            self.graphics_device_id = self.compute_device_id
            self.headless = args.headless
            self.physics_engine = isaacgym.gymapi.SIM_PHYSX
            self.use_gpu = True
            self.use_gpu_pipeline = True
            self.subscenes = 0
            self.num_threads = 0
            # 额外必需属性
            self.teleop_mode = False
            self.record_video = args.record_video
            self.no_rand = True
            self.num_envs = 1
            self.seed = None
            self.rows = None
            self.cols = None
            self.config_overrides = {}
    
    env_args = EnvArgs()
    env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=env_cfg)
    console.print("[green]✓ Environment created[/green]")
    
    # 获取动作信息
    motion_name = os.path.basename(args.motion_file)
    if hasattr(env, '_motion_lib'):
        motion_length = env._motion_lib.get_motion_length(env._motion_ids[0]).item()
    else:
        motion_length = 10.0
    
    max_steps = min(args.max_steps, int(motion_length / env.dt) + 50)
    
    console.print(f"\n[bold]Motion Info:[/bold]")
    console.print(f"  Name: {motion_name}")
    console.print(f"  Duration: {motion_length:.2f}s")
    console.print(f"  Max steps: {max_steps}")
    
    # 录制设置
    mp4_writer = None
    if args.record_video:
        import imageio
        output_path = args.output_video if args.output_video else f"motion_playback_{motion_name.replace('.pkl', '')}.mp4"
        mp4_writer = imageio.get_writer(output_path, fps=int(1/env.dt))
        console.print(f"[green]Recording to: {output_path}[/green]")
    
    # 运行仿真
    console.print(f"\n[bold cyan]Running simulation...[/bold cyan]")
    console.print("Press Ctrl+C to stop early\n")
    
    obs = env.get_observations()
    
    # 统计数据
    history = {
        'base_height': [],
        'roll': [],
        'pitch': [],
        'velocity': [],
        'warnings': []
    }
    
    try:
        for step in tqdm(range(max_steps), desc="Simulating"):
            # ONNX 推理
            obs_np = obs.cpu().numpy().astype(np.float32)
            actions_np = session.run([output_name], {input_name: obs_np})[0]
            actions = torch.from_numpy(actions_np).to(env.device)
            
            # 环境步进
            obs, _, rews, dones, infos = env.step(actions)
            
            # 收集数据
            base_height = env.root_states[0, 2].item()
            roll = env.roll[0].item() if hasattr(env, 'roll') else 0
            pitch = env.pitch[0].item() if hasattr(env, 'pitch') else 0
            vel = torch.norm(env.root_states[0, 7:10]).item()
            
            history['base_height'].append(base_height)
            history['roll'].append(np.degrees(roll))
            history['pitch'].append(np.degrees(pitch))
            history['velocity'].append(vel)
            
            # 检查警告条件
            if step % 50 == 0:  # 每 50 步检查一次
                warnings = check_termination_conditions(env, step, console)
                if warnings:
                    current_time = step * env.dt
                    console.print(f"[dim]Step {step} ({current_time:.2f}s):[/dim]")
                    for w in warnings:
                        console.print(f"  {w}")
                    history['warnings'].extend(warnings)
            
            # 每 10 帧打印当前真实 root 角度和相对参考动作的 root 角度
            if step % 10 == 0:
                # 获取真实的 root 四元数 [x, y, z, w]
                real_quat = env.root_states[0, 3:7].cpu().numpy()
                
                # 获取参考的 root 四元数
                # 注意：env._ref_root_rot 用于内部计算，可能已经转换过，或者是 [x,y,z,w] 格式
                ref_quat = env._ref_root_rot[0].cpu().numpy()
                
                def get_yaw_deg(q):
                    # q: [x, y, z, w]
                    x, y, z, w = q
                    # yaw (z-axis rotation)
                    siny_cosp = 2.0 * (w * z + x * y)
                    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                    yaw = np.arctan2(siny_cosp, cosy_cosp)
                    return np.degrees(yaw)
                
                real_yaw = get_yaw_deg(real_quat)
                ref_yaw = get_yaw_deg(ref_quat)
                
                console.print(f"[dim]Frame {step}:[/dim] Real Yaw: {real_yaw:.2f}°, Ref Yaw: {ref_yaw:.2f}°, Diff: {abs(real_yaw - ref_yaw):.2f}°")
            
            # 录制视频帧
            if mp4_writer is not None:
                imgs = env.render_record(mode='rgb_array')
                if imgs is not None:
                    mp4_writer.append_data(imgs[0])
            
            # 检查是否结束
            if dones[0] and not args.no_pose_term:
                current_time = step * env.dt
                console.print(f"\n[yellow]Motion terminated at step {step} ({current_time:.2f}s)[/yellow]")
                break
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    
    # 关闭视频写入
    if mp4_writer is not None:
        mp4_writer.close()
        console.print(f"\n[green]Video saved![/green]")
    
    # 统计摘要
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]                     统计摘要                               [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    
    table = Table()
    table.add_column("指标", style="cyan")
    table.add_column("最小值", justify="right")
    table.add_column("最大值", justify="right")
    table.add_column("平均值", justify="right")
    
    if history['base_height']:
        table.add_row("基座高度 (m)", 
                     f"{min(history['base_height']):.3f}",
                     f"{max(history['base_height']):.3f}",
                     f"{np.mean(history['base_height']):.3f}")
    if history['roll']:
        table.add_row("Roll (°)", 
                     f"{min(history['roll']):.1f}",
                     f"{max(history['roll']):.1f}",
                     f"{np.mean(history['roll']):.1f}")
    if history['pitch']:
        table.add_row("Pitch (°)", 
                     f"{min(history['pitch']):.1f}",
                     f"{max(history['pitch']):.1f}",
                     f"{np.mean(history['pitch']):.1f}")
    if history['velocity']:
        table.add_row("基座速度 (m/s)", 
                     f"{min(history['velocity']):.2f}",
                     f"{max(history['velocity']):.2f}",
                     f"{np.mean(history['velocity']):.2f}")
    
    console.print(table)
    
    if history['warnings']:
        console.print(f"\n[yellow]总计 {len(history['warnings'])} 个警告[/yellow]")
    
    # 清理
    try:
        os.unlink(temp_yaml_path)
    except:
        pass
    
    console.print("\n[green]Done![/green]")


if __name__ == "__main__":
    main()
