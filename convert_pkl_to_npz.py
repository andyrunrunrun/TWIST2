#!/usr/bin/env python3
"""
将 NumPy 2.x 生成的 .pkl 动作文件转换为跨版本兼容的 .npz 格式。

使用方法：
    # 在拥有 NumPy 2.x 的环境中运行
    python convert_pkl_to_npz.py /path/to/motion/folder

说明：
    - 递归扫描指定目录下的所有 .pkl 文件
    - 将每个 .pkl 转换为同名的 .npz 文件
    - 如果 .npz 已存在则跳过
    - 转换后可在 NumPy 1.x 环境中加载
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path


def convert_pkl_to_npz(pkl_path: str, overwrite: bool = False) -> bool:
    """
    将单个 .pkl 文件转换为 .npz 格式。
    
    Args:
        pkl_path: .pkl 文件路径
        overwrite: 是否覆盖已存在的 .npz 文件
    
    Returns:
        True 如果转换成功，False 如果跳过或失败
    """
    npz_path = pkl_path[:-4] + ".npz"
    
    # 检查是否已存在
    if os.path.exists(npz_path) and not overwrite:
        return False  # 跳过
    
    try:
        # 加载 pkl 文件
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        # 验证必要的字段
        required_keys = ["fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"]
        for key in required_keys:
            if key not in data:
                print(f"  跳过 {pkl_path}: 缺少必要字段 '{key}'")
                return False
        
        # 保存为 npz 格式
        np.savez(
            npz_path,
            fps=np.array(data["fps"]),
            root_pos=np.array(data["root_pos"]),
            root_rot=np.array(data["root_rot"]),
            dof_pos=np.array(data["dof_pos"]),
            local_body_pos=np.array(data["local_body_pos"]),
            link_body_list=np.array(data["link_body_list"]),
        )
        return True
        
    except Exception as e:
        print(f"  转换失败 {pkl_path}: {e}")
        return False


def find_pkl_files(root_dir: str) -> list:
    """递归查找所有 .pkl 文件"""
    pkl_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_pkl_to_npz.py <目录路径> [--overwrite]")
        print("示例: python convert_pkl_to_npz.py /home/huanghao/source/datasets/gmr_retarget_x")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    overwrite = "--overwrite" in sys.argv
    
    if not os.path.isdir(root_dir):
        print(f"错误: 目录不存在 - {root_dir}")
        sys.exit(1)
    
    print(f"扫描目录: {root_dir}")
    pkl_files = find_pkl_files(root_dir)
    print(f"找到 {len(pkl_files)} 个 .pkl 文件")
    
    if not pkl_files:
        print("没有找到需要转换的文件")
        return
    
    converted = 0
    skipped = 0
    failed = 0
    
    for pkl_path in tqdm(pkl_files, desc="转换中"):
        npz_path = pkl_path[:-4] + ".npz"
        if os.path.exists(npz_path) and not overwrite:
            skipped += 1
            continue
        
        if convert_pkl_to_npz(pkl_path, overwrite):
            converted += 1
        else:
            failed += 1
    
    print(f"\n转换完成:")
    print(f"  成功: {converted}")
    print(f"  跳过 (已存在): {skipped}")
    print(f"  失败: {failed}")


if __name__ == "__main__":
    main()
