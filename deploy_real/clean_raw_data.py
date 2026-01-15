"""
Clean Raw VR Data
==================
清理原始 VR 数据：检测卡顿区域并分割片段，删除过短的碎片。

使用方法:
    conda activate gmr
    
    # 处理 raw 文件夹，输出到 raw_cleaned 文件夹
    python clean_raw_data.py --input_dir ~/twist2_pico/huanghao/raw --output_dir ~/twist2_pico/huanghao/raw_cleaned
    
    # 自定义参数
    python clean_raw_data.py --input_dir ~/twist2_pico/huanghao/raw --output_dir ~/twist2_pico/huanghao/raw_cleaned --max_gap 15 --min_frames 30

处理逻辑:
    1. 检测每帧与前一帧的差异（基于 Pelvis 位置）
    2. 如果两帧之间差了超过 max_gap 帧（默认 15），则分割为两个片段
    3. 删除所有帧数小于 min_frames（默认 30）的碎片片段
    4. 保存清理后的片段到输出目录
"""
import argparse
import os
import pickle
from glob import glob
from tqdm import tqdm
import numpy as np
from rich import print


def load_raw_pkl(filepath):
    """加载原始 VR 数据 PKL 文件"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_raw_pkl(filepath, data):
    """保存原始 VR 数据 PKL 文件"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def detect_gap_regions(frames, threshold=1e-10):
    """
    检测帧之间的卡顿区域
    
    Args:
        frames: 帧列表
        threshold: 位置变化阈值（判断是否为重复帧）
    
    Returns:
        gap_counts: 每个位置与前一个有效帧之间的重复帧数量
        valid_indices: 有效帧的索引列表
    """
    if len(frames) <= 1:
        return [0], [0]
    
    valid_indices = [0]
    gap_counts = [0]  # 第一帧没有 gap
    
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        prev_valid_frame = frames[valid_indices[-1]]
        
        if curr_frame is None or prev_valid_frame is None:
            continue
        
        # 获取 Pelvis 位置
        curr_pelvis = curr_frame.get('Pelvis', [[0,0,0], [0,0,0,1]])
        prev_pelvis = prev_valid_frame.get('Pelvis', [[0,0,0], [0,0,0,1]])
        
        if isinstance(curr_pelvis, list) and len(curr_pelvis) >= 1:
            curr_pos = np.array(curr_pelvis[0])
            prev_pos = np.array(prev_pelvis[0])
            diff = np.linalg.norm(curr_pos - prev_pos)
            
            if diff > threshold:
                # 当前帧是有效帧
                gap = i - valid_indices[-1] - 1  # 中间跳过的帧数
                valid_indices.append(i)
                gap_counts.append(gap)
    
    return gap_counts, valid_indices


def split_frames_by_gap(frames, max_gap=15, min_frames=30):
    """
    根据卡顿区域分割帧，并删除过短的片段
    
    Args:
        frames: 帧列表
        max_gap: 最大允许的连续重复帧数量，超过则分割
        min_frames: 最小片段长度，小于此值的片段将被删除
    
    Returns:
        segments: 分割后的片段列表，每个片段是帧列表
    """
    gap_counts, valid_indices = detect_gap_regions(frames)
    
    if len(valid_indices) == 0:
        return []
    
    # 找出需要分割的位置
    split_points = [0]  # 起始位置
    for i, gap in enumerate(gap_counts):
        if gap > max_gap:
            # 在这个位置分割
            split_points.append(i)
    split_points.append(len(valid_indices))  # 结束位置
    
    # 分割片段
    segments = []
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]
        
        if end_idx - start_idx < 1:
            continue
        
        # 提取这个片段的帧
        segment_valid_indices = valid_indices[start_idx:end_idx]
        segment_frames = [frames[idx] for idx in segment_valid_indices]
        
        # 检查片段长度
        if len(segment_frames) >= min_frames:
            segments.append(segment_frames)
    
    return segments


def process_file(input_path, output_dir, base_name, max_gap=15, min_frames=30, target_fps=30):
    """
    处理单个文件
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        base_name: 基础文件名（不含扩展名）
        max_gap: 最大允许的连续重复帧数量
        min_frames: 最小片段长度
        target_fps: 目标帧率，所有输出文件统一使用此帧率
    
    Returns:
        n_segments: 生成的片段数量
        total_frames: 保留的总帧数
    """
    # 加载数据
    raw_data = load_raw_pkl(input_path)
    frames = raw_data.get('frames', [])
    fps = raw_data.get('fps', 30.0)
    
    if len(frames) == 0:
        return 0, 0
    
    # 分割片段
    segments = split_frames_by_gap(frames, max_gap=max_gap, min_frames=min_frames)
    
    if len(segments) == 0:
        return 0, 0
    
    # 保存片段
    total_frames = 0
    for i, segment in enumerate(segments):
        # 生成文件名
        if len(segments) == 1:
            output_name = f"{base_name}.pkl"
        else:
            output_name = f"{base_name}_part{i+1:02d}.pkl"
        
        output_path = os.path.join(output_dir, output_name)
        
        # 构建输出数据 (强制使用 target_fps)
        output_data = {
            'fps': float(target_fps),
            'frames': segment,
            'n_frames': len(segment)
        }
        
        save_raw_pkl(output_path, output_data)
        total_frames += len(segment)
    
    return len(segments), total_frames


def process_directory(input_dir, output_dir, max_gap=15, min_frames=30, target_fps=30):
    """
    处理整个目录
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        max_gap: 最大允许的连续重复帧数量
        min_frames: 最小片段长度
        target_fps: 目标帧率
    """
    # 展开路径
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"[red]错误: 输入目录不存在: {input_dir}[/red]")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 pkl 文件
    pkl_files = sorted(glob(os.path.join(input_dir, "*.pkl")))
    
    if len(pkl_files) == 0:
        print(f"[yellow]警告: 在 {input_dir} 中没有找到 .pkl 文件[/yellow]")
        return
    
    print(f"[bold]清理原始 VR 数据[/bold]")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  最大卡顿帧数: {max_gap}")
    print(f"  最小片段长度: {min_frames}")
    print(f"  目标帧率: {target_fps} fps")
    print(f"  找到 {len(pkl_files)} 个文件")
    print()
    
    # 统计
    total_input_files = len(pkl_files)
    total_output_segments = 0
    total_output_frames = 0
    skipped_files = 0
    
    for pkl_path in tqdm(pkl_files, desc="处理中"):
        filename = os.path.basename(pkl_path)
        base_name = os.path.splitext(filename)[0]
        
        try:
            n_segments, n_frames = process_file(
                pkl_path, output_dir, base_name,
                max_gap=max_gap, min_frames=min_frames, target_fps=target_fps
            )
            
            if n_segments == 0:
                skipped_files += 1
                print(f"[dim]跳过 (无有效片段): {filename}[/dim]")
            else:
                total_output_segments += n_segments
                total_output_frames += n_frames
                if n_segments > 1:
                    print(f"[cyan]分割: {filename} -> {n_segments} 个片段 ({n_frames} 帧)[/cyan]")
                    
        except Exception as e:
            print(f"[red]✗ {filename}: {e}[/red]")
            skipped_files += 1
    
    # 完成报告
    print()
    print("[bold]处理完成![/bold]")
    print(f"  输入文件: {total_input_files}")
    print(f"  输出片段: {total_output_segments}")
    print(f"  输出帧数: {total_output_frames}")
    print(f"  跳过文件: {skipped_files}")
    print(f"  输出目录: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Clean Raw VR Data")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="原始 VR 数据目录 (raw/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="清理后的输出目录 (raw_cleaned/)"
    )
    parser.add_argument(
        "--max_gap",
        type=int,
        default=15,
        help="最大允许的连续重复帧数量，超过则分割 (默认 15)"
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=30,
        help="最小片段长度，小于此值的片段将被删除 (默认 30)"
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="目标帧率，所有输出文件统一使用此帧率 (默认 30)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_gap=args.max_gap,
        min_frames=args.min_frames,
        target_fps=args.target_fps
    )
