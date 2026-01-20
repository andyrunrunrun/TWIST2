#!/usr/bin/env python3
"""
将 retargeted pkl 文件从 30fps 插值为 60fps

用法:
    python upsample_motion.py --input motion_001.pkl --output motion_001_60fps.pkl
    python upsample_motion.py --input motion_001.pkl --output motion_001_60fps.pkl --target_fps 60
"""

import argparse
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp


def upsample_motion(pkl_path, output_path, target_fps=60):
    """将动作数据插值到更高帧率"""
    
    # 加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    src_fps = data.get('fps', 30.0)
    root_pos = data['root_pos']  # (N, 3)
    root_rot = data['root_rot']  # (N, 4) xyzw
    dof_pos = data['dof_pos']    # (N, n_dof)
    local_body_pos = data['local_body_pos']  # (N, n_bodies, 3)
    
    n_frames = root_pos.shape[0]
    duration = (n_frames - 1) / src_fps
    
    print(f"输入: {pkl_path}")
    print(f"  源帧率: {src_fps} fps")
    print(f"  帧数: {n_frames}")
    print(f"  时长: {duration:.2f}s")
    print()
    
    # 计算目标帧数
    n_frames_new = int(duration * target_fps) + 1
    
    print(f"目标帧率: {target_fps} fps")
    print(f"目标帧数: {n_frames_new}")
    print()
    
    # 创建时间轴
    t_src = np.linspace(0, duration, n_frames)
    t_tgt = np.linspace(0, duration, n_frames_new)
    
    # 1. 插值 root_pos (线性插值)
    print("插值 root_pos...")
    interp_func = interp1d(t_src, root_pos, axis=0, kind='cubic', fill_value='extrapolate')
    root_pos_new = interp_func(t_tgt)
    
    # 2. 插值 root_rot (球面线性插值 SLERP)
    print("插值 root_rot (SLERP)...")
    # 将 xyzw 转换为 scipy 需要的格式
    rotations = R.from_quat(root_rot)  # scipy 使用 xyzw 格式
    slerp = Slerp(t_src, rotations)
    root_rot_new = slerp(t_tgt).as_quat()  # 返回 xyzw 格式
    
    # 3. 插值 dof_pos (三次样条)
    print("插值 dof_pos...")
    interp_func = interp1d(t_src, dof_pos, axis=0, kind='cubic', fill_value='extrapolate')
    dof_pos_new = interp_func(t_tgt)
    
    # 4. 插值 local_body_pos (三次样条)
    print("插值 local_body_pos...")
    n_bodies = local_body_pos.shape[1]
    local_body_pos_new = np.zeros((n_frames_new, n_bodies, 3), dtype=np.float32)
    
    for body_idx in range(n_bodies):
        interp_func = interp1d(t_src, local_body_pos[:, body_idx, :], axis=0, kind='cubic', fill_value='extrapolate')
        local_body_pos_new[:, body_idx, :] = interp_func(t_tgt)
    
    # 创建新数据
    new_data = {
        'fps': np.float64(target_fps),
        'root_pos': root_pos_new.astype(np.float64),
        'root_rot': root_rot_new.astype(np.float64),
        'dof_pos': dof_pos_new.astype(np.float64),
        'local_body_pos': local_body_pos_new.astype(np.float32),
        'link_body_list': data.get('link_body_list', [])
    }
    
    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)
    
    print()
    print(f"✓ 保存: {output_path}")
    print(f"  帧数: {n_frames} -> {n_frames_new}")
    print(f"  帧率: {src_fps} -> {target_fps} fps")


def main():
    parser = argparse.ArgumentParser(description="将 retargeted pkl 文件插值到更高帧率")
    parser.add_argument("--input", type=str, required=True, help="输入 pkl 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 pkl 文件路径 (默认: 添加 _60fps 后缀)")
    parser.add_argument("--target_fps", type=int, default=60, help="目标帧率 (默认: 60)")
    
    args = parser.parse_args()
    
    input_path = os.path.expanduser(args.input)
    
    if args.output:
        output_path = os.path.expanduser(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_{args.target_fps}fps{ext}"
    
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在: {input_path}")
        return
    
    upsample_motion(input_path, output_path, args.target_fps)


if __name__ == "__main__":
    main()
