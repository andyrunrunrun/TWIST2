#!/usr/bin/env python3
"""
PKL 动作文件对比分析脚本

用法:
    python compare_motions.py
"""

import pickle
import numpy as np

# 要对比的两个文件
file1 = "/home/huanghao/source/datasets/gmr_retarget_x/AMASS/ACCAD/Female1General_c3d/A1_-_Stand_stageii.pkl"  # 正常数据
file2 = "/home/huanghao/source/datasets/gmr_retarget_x/twist2_pico_clean/huanghao/retargeted_clean/motion_001.pkl"  # 问题数据

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def analyze_motion(name, data):
    print(f"\n{'='*60}")
    print(f"分析: {name}")
    print(f"{'='*60}")
    
    if not isinstance(data, dict):
        print(f"数据类型: {type(data)}")
        return
    
    print(f"Keys: {list(data.keys())}")
    
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"\n[{k}]")
            print(f"  shape: {v.shape}")
            print(f"  dtype: {v.dtype}")
            print(f"  min: {v.min():.4f}, max: {v.max():.4f}")
            
            # 显示第一帧数据
            if len(v.shape) >= 1:
                first_frame = v[0]
                if isinstance(first_frame, np.ndarray):
                    if first_frame.size <= 10:
                        print(f"  第一帧: {first_frame}")
                    else:
                        print(f"  第一帧前10个值: {first_frame.flatten()[:10]}")
                else:
                    print(f"  第一帧: {first_frame}")
        else:
            print(f"\n[{k}]: {type(v).__name__} = {v}")

def compare_first_frame(name1, data1, name2, data2):
    print(f"\n{'='*60}")
    print(f"第一帧对比")
    print(f"{'='*60}")
    
    common_keys = set(data1.keys()) & set(data2.keys())
    
    for k in sorted(common_keys):
        v1 = data1[k]
        v2 = data2[k]
        
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if v1.shape == v2.shape:
                # 对比第一帧
                if len(v1.shape) >= 1:
                    frame1 = v1[0] if isinstance(v1[0], np.ndarray) else v1[0]
                    frame2 = v2[0] if isinstance(v2[0], np.ndarray) else v2[0]
                    
                    if isinstance(frame1, np.ndarray) and isinstance(frame2, np.ndarray):
                        diff = np.abs(frame1 - frame2)
                        max_diff = diff.max()
                        mean_diff = diff.mean()
                        
                        if max_diff > 0.1:  # 只显示差异大的
                            print(f"\n[{k}] 差异较大!")
                            print(f"  正常数据 第一帧: {frame1[:5] if len(frame1) > 5 else frame1}")
                            print(f"  问题数据 第一帧: {frame2[:5] if len(frame2) > 5 else frame2}")
                            print(f"  最大差异: {max_diff:.4f}, 平均差异: {mean_diff:.4f}")
            else:
                print(f"\n[{k}] shape 不同: {v1.shape} vs {v2.shape}")

# 特别关注: root position 和 rotation
def analyze_root_state(name, data):
    print(f"\n{'='*60}")
    print(f"Root 状态分析: {name}")
    print(f"{'='*60}")
    
    possible_root_keys = ['root_pos', 'root_rot', 'root_position', 'root_rotation', 
                          'root_states', 'global_translation', 'root_translation']
    
    for k in data.keys():
        if 'root' in k.lower() or 'translation' in k.lower() or 'position' in k.lower():
            v = data[k]
            if isinstance(v, np.ndarray):
                print(f"\n[{k}]")
                print(f"  shape: {v.shape}")
                print(f"  第一帧: {v[0]}")
                print(f"  最后一帧: {v[-1]}")
                
                # 如果是位置数据，检查范围
                if v.shape[-1] == 3:  # XYZ
                    print(f"  X 范围: [{v[..., 0].min():.3f}, {v[..., 0].max():.3f}]")
                    print(f"  Y 范围: [{v[..., 1].min():.3f}, {v[..., 1].max():.3f}]")
                    print(f"  Z 范围: [{v[..., 2].min():.3f}, {v[..., 2].max():.3f}]")

def main():
    print("加载数据...")
    data1 = load_pkl(file1)
    data2 = load_pkl(file2)
    
    print(f"\n正常数据路径: {file1}")
    print(f"问题数据路径: {file2}")
    
    # 分析每个文件
    analyze_motion("正常数据 (A1_-_Stand)", data1)
    analyze_motion("问题数据 (motion_001)", data2)
    
    # 对比第一帧
    compare_first_frame("正常数据", data1, "问题数据", data2)
    
    # 分析 root 状态
    analyze_root_state("正常数据", data1)
    analyze_root_state("问题数据", data2)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)

if __name__ == "__main__":
    main()
