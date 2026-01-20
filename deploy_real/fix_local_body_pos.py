#!/usr/bin/env python3
"""
修复 retargeted pkl 文件中的 local_body_pos

问题：旧版本的 batch_retarget_raw.py 保存的是全局坐标，
      应该保存的是相对于 root_pos 的局部坐标。

用法:
    conda activate gmr  # 或任何有 numpy 的环境
    python fix_local_body_pos.py --input motion_001.pkl --output motion_001_fixed.pkl
    
    # 批量修复目录下所有 pkl 文件
    python fix_local_body_pos.py --input_dir ~/path/to/retargeted --output_dir ~/path/to/fixed
"""

import argparse
import os
import pickle
import numpy as np
from glob import glob
import sys
from types import ModuleType

# Patch sys.modules to fake missing modules from numpy 2.x
# allowing pickles created with numpy 2.x to be loaded in 1.x environments
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

# Patch potentially missing modules
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))


def fix_pkl_file(input_path, output_path):
    """修复单个 pkl 文件"""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查是否是 retargeted 格式
    if 'root_pos' not in data or 'local_body_pos' not in data:
        print(f"  跳过 (非 retargeted 格式): {input_path}")
        return False
    
    root_pos = data['root_pos']  # (N, 3)
    local_body_pos = data['local_body_pos']  # (N, n_bodies, 3)
    
    # 检查是否需要修复
    # 如果第一帧的第一个 body (pelvis) 位置接近 root_pos，说明是全局坐标
    first_pelvis = local_body_pos[0, 0, :]  # pelvis 位置
    first_root = root_pos[0, :]
    
    
    # 修复：减去 root_pos
    fixed_local_body_pos = local_body_pos - root_pos[:, None, :]
    data['local_body_pos'] = fixed_local_body_pos.astype(np.float32)
    
    # 保存
    with open(output_path, 'wb') as f:      
        pickle.dump(data, f)
    
    # 验证
    new_pelvis = fixed_local_body_pos[0, 0, :]
    print(f"  修复后 pelvis 第一帧: {new_pelvis}")
    print(f"  ✓ 已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="修复 retargeted pkl 文件中的 local_body_pos")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="单个 pkl 文件路径")
    group.add_argument("--input_dir", type=str, help="包含多个 pkl 文件的目录")
    
    parser.add_argument("--output", type=str, help="输出文件路径 (单文件模式)")
    parser.add_argument("--output_dir", type=str, help="输出目录 (批量模式)")
    
    args = parser.parse_args()
    
    if args.input:
        # 单文件模式
        input_path = os.path.expanduser(args.input)
        output_path = os.path.expanduser(args.output) if args.output else input_path.replace('.pkl', '_fixed.pkl')
        
        print(f"修复: {input_path}")
        fix_pkl_file(input_path, output_path)
        
    else:
        # 批量模式 (支持递归)
        input_dir = os.path.expanduser(args.input_dir)
        output_dir = os.path.expanduser(args.output_dir) if args.output_dir else input_dir + "_fixed"
        
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print()
        
        pkl_files = []
        # 使用 os.walk 递归查找
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".pkl"):
                    # 获取相对于 input_dir 的路径
                    rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                    pkl_files.append(rel_path)
        
        pkl_files.sort()
        print(f"找到 {len(pkl_files)} 个 pkl 文件")
        
        success = 0
        for rel_path in pkl_files:
            input_path = os.path.join(input_dir, rel_path)
            output_path = os.path.join(output_dir, rel_path)
            
            # 确保输出子目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"处理: {rel_path}")
            if fix_pkl_file(input_path, output_path):
                success += 1
            # print() # 减少输出行数
        
        print(f"\n完成! 成功处理 {success}/{len(pkl_files)} 个文件")


if __name__ == "__main__":
    main()
