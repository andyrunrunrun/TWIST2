#!/usr/bin/env python3
"""
可视化 retargeted pkl 文件中各关节位置随时间的变化

用法:
    python visualize_motion.py --input motion_001.pkl --output_dir photo
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_motion(pkl_path, output_dir):
    """可视化动作数据"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    motion_name = os.path.splitext(os.path.basename(pkl_path))[0]
    print(f"加载: {pkl_path}")
    print(f"输出目录: {output_dir}")
    
    # 提取数据
    fps = data.get('fps', 30.0)
    root_pos = data['root_pos']  # (N, 3)
    root_rot = data['root_rot']  # (N, 4)
    dof_pos = data['dof_pos']    # (N, n_dof)
    local_body_pos = data['local_body_pos']  # (N, n_bodies, 3)
    body_names = data.get('link_body_list', [f'body_{i}' for i in range(local_body_pos.shape[1])])
    
    n_frames = root_pos.shape[0]
    n_bodies = local_body_pos.shape[1]
    n_dof = dof_pos.shape[1]
    
    time = np.arange(n_frames) / fps
    
    print(f"帧数: {n_frames}, 时长: {time[-1]:.2f}s, FPS: {fps}")
    print(f"Bodies: {n_bodies}, DOFs: {n_dof}")
    print()
    
    # ==================== 1. Root Position ====================
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'{motion_name} - Root Position', fontsize=14)
    
    labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, root_pos[:, i], color=color, linewidth=1)
        ax.set_ylabel(f'{label} (m)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{motion_name}_root_pos.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")
    
    # ==================== 2. Root Rotation (Quaternion) ====================
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{motion_name} - Root Rotation (Quaternion xyzw)', fontsize=14)
    
    labels = ['X', 'Y', 'Z', 'W']
    colors = ['red', 'green', 'blue', 'purple']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, root_rot[:, i], color=color, linewidth=1)
        ax.set_ylabel(f'{label}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{motion_name}_root_rot.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")
    
    # ==================== 3. DOF Positions ====================
    # 分组绘制 DOF
    n_cols = 4
    n_rows = (n_dof + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2.5), sharex=True)
    fig.suptitle(f'{motion_name} - Joint Positions (DOF)', fontsize=14)
    axes = axes.flatten()
    
    for i in range(n_dof):
        ax = axes[i]
        ax.plot(time, dof_pos[:, i], linewidth=1)
        ax.set_title(f'DOF {i}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time[-1]])
    
    # 隐藏多余的子图
    for i in range(n_dof, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{motion_name}_dof_pos.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")
    
    # ==================== 4. Key Bodies Local Position ====================
    # 选择关键 body 来显示
    key_body_names = [
        'pelvis', 'head_mocap',
        'left_rubber_hand', 'right_rubber_hand',
        'left_ankle_roll_link', 'right_ankle_roll_link',
        'left_elbow_link', 'right_elbow_link',
        'left_knee_link', 'right_knee_link'
    ]
    
    # 找到对应的索引
    key_body_indices = []
    key_body_labels = []
    for name in key_body_names:
        if name in body_names:
            key_body_indices.append(body_names.index(name))
            key_body_labels.append(name)
    
    if len(key_body_indices) > 0:
        n_key = len(key_body_indices)
        n_cols = 2
        n_rows = (n_key + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        fig.suptitle(f'{motion_name} - Key Body Local Positions', fontsize=14)
        axes = axes.flatten()
        
        for idx, (body_idx, body_name) in enumerate(zip(key_body_indices, key_body_labels)):
            ax = axes[idx]
            pos = local_body_pos[:, body_idx, :]
            
            ax.plot(time, pos[:, 0], 'r-', label='X', linewidth=1, alpha=0.8)
            ax.plot(time, pos[:, 1], 'g-', label='Y', linewidth=1, alpha=0.8)
            ax.plot(time, pos[:, 2], 'b-', label='Z', linewidth=1, alpha=0.8)
            
            ax.set_title(body_name, fontsize=10)
            ax.set_ylabel('Position (m)', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, time[-1]])
        
        for i in range(n_key, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{motion_name}_key_bodies.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存: {save_path}")
    
    # ==================== 5. All Bodies Overview ====================
    # 绘制所有 body 的 Z 位置（高度）
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(f'{motion_name} - All Bodies Z Position (Height)', fontsize=14)
    
    # 使用 colormap
    cmap = plt.cm.get_cmap('tab20', n_bodies)
    
    for i in range(n_bodies):
        color = cmap(i % 20)
        alpha = 0.7 if i < 20 else 0.5
        ax.plot(time, local_body_pos[:, i, 2], color=color, linewidth=0.8, alpha=alpha, label=body_names[i])
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Z Position (m)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time[-1]])
    
    # 图例放在右侧
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=2)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{motion_name}_all_bodies_z.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")
    
    # ==================== 6. 3D Trajectory ====================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f'{motion_name} - Root 3D Trajectory', fontsize=14)
    
    # 绘制轨迹
    ax.plot(root_pos[:, 0], root_pos[:, 1], root_pos[:, 2], 'b-', linewidth=1, alpha=0.7)
    
    # 标记起点和终点
    ax.scatter(root_pos[0, 0], root_pos[0, 1], root_pos[0, 2], c='green', s=100, marker='o', label='Start')
    ax.scatter(root_pos[-1, 0], root_pos[-1, 1], root_pos[-1, 2], c='red', s=100, marker='s', label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    
    # 设置相等比例
    max_range = np.max([
        root_pos[:, 0].max() - root_pos[:, 0].min(),
        root_pos[:, 1].max() - root_pos[:, 1].min(),
        root_pos[:, 2].max() - root_pos[:, 2].min()
    ]) / 2
    
    mid_x = (root_pos[:, 0].max() + root_pos[:, 0].min()) / 2
    mid_y = (root_pos[:, 1].max() + root_pos[:, 1].min()) / 2
    mid_z = (root_pos[:, 2].max() + root_pos[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    save_path = os.path.join(output_dir, f'{motion_name}_3d_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {save_path}")
    
    print()
    print(f"完成! 所有图片已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="可视化 retargeted pkl 文件中的动作数据")
    parser.add_argument("--input", type=str, required=True, help="输入 pkl 文件路径")
    parser.add_argument("--output_dir", type=str, default="photo", help="输出目录 (默认: photo)")
    
    args = parser.parse_args()
    
    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在: {input_path}")
        return
    
    visualize_motion(input_path, output_dir)


if __name__ == "__main__":
    main()
