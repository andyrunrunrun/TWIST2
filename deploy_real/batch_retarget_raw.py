"""
Batch Retarget Raw VR Data
===========================
批量处理已录制的原始 VR 数据，使用 GMR 进行重定向。

使用方法:
    conda activate gmr
    
    # 处理指定目录下的所有 raw 数据
    python batch_retarget_raw.py --input_dir ~/twist2_pico/huanghao/raw --output_dir ~/twist2_pico/huanghao/retargeted
    
    # 启用网络延迟优化 (去除重复帧 + 平滑滤波)
    python batch_retarget_raw.py --input_dir ~/twist2_pico/huanghao/raw --output_dir ~/twist2_pico/huanghao/retargeted --optimize_latency
    
    # 单文件预览模式 (可视化，不保存)
    python batch_retarget_raw.py --preview ~/twist2_pico/huanghao/raw/motion_001.pkl
    
    # 单文件预览 + 网络优化
    python batch_retarget_raw.py --preview ~/twist2_pico/huanghao/raw/motion_001.pkl --optimize_latency

    # 预览单个文件，启用延迟优化
    python batch_retarget_raw.py --preview ~/twist2_pico/huanghao/raw/motion_001.pkl --optimize_latency

    # 批量处理，启用延迟优化
    python batch_retarget_raw.py --input_dir ~/twist2_pico/huanghao/raw --output_dir ~/twist2_pico/huanghao/retargeted --optimize_latency
输入格式 (raw/*.pkl):
    {
        'fps': float64,
        'frames': list[dict],  # 每帧是 smplx_data 字典
        'n_frames': int
    }

输出格式 (retargeted/*.pkl):
    {
        'fps': float64,
        'root_pos': (N, 3) float64,
        'root_rot': (N, 4) float64,  # (x, y, z, w)
        'dof_pos': (N, 29) float64,
        'local_body_pos': (N, 38, 3) float32,
        'link_body_list': list[str]
    }
"""
import argparse
import os
import pickle
import time
from glob import glob
from tqdm import tqdm

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, RobotMotionViewer
from rich import print


def load_raw_pkl(filepath):
    """加载原始 VR 数据 PKL 文件"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def save_retargeted_pkl(filepath, data):
    """保存重定向后的数据"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def detect_duplicate_regions(frames, threshold=1e-10):
    """
    检测重复帧区域，返回区域信息用于后续插值
    
    Args:
        frames: 帧列表
        threshold: 位置变化阈值 (米)
    
    Returns:
        duplicate_regions: list of (start_idx, end_idx) - 重复帧区间 (不包括起始和结束的有效帧)
        valid_indices: 有效帧的索引列表
    """
    if len(frames) <= 1:
        return [], list(range(len(frames)))
    
    valid_indices = [0]  # 第一帧总是有效的
    duplicate_regions = []
    
    i = 1
    while i < len(frames):
        curr_frame = frames[i]
        prev_valid_frame = frames[valid_indices[-1]]
        
        if curr_frame is None or prev_valid_frame is None:
            i += 1
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
                # 检查是否有重复区域需要记录
                if i > valid_indices[-1] + 1:
                    # 存在重复帧区域: 从上一个有效帧+1 到 当前帧-1
                    duplicate_regions.append((valid_indices[-1], i))
                valid_indices.append(i)
        
        i += 1
    
    return duplicate_regions, valid_indices


def interpolate_duplicate_regions(data_array, duplicate_regions):
    """
    对重复帧区域进行线性插值
    
    Args:
        data_array: numpy 数组 (N, ...) - 可以是任意维度
        duplicate_regions: list of (start_valid_idx, end_valid_idx)
    
    Returns:
        插值后的数组
    """
    if len(duplicate_regions) == 0:
        return data_array
    
    result = np.copy(data_array)
    
    for start_idx, end_idx in duplicate_regions:
        if end_idx >= len(data_array):
            continue
            
        start_val = data_array[start_idx]
        end_val = data_array[end_idx]
        n_interp = end_idx - start_idx  # 需要插值的帧数
        
        # 线性插值
        for j in range(1, n_interp):
            t = j / n_interp  # 插值系数 [0, 1)
            result[start_idx + j] = start_val * (1 - t) + end_val * t
    
    return result


def smooth_motion_data(data_array, window_size=3):
    """
    平滑运动数据 (滑动窗口平均) - 备用方法
    
    Args:
        data_array: numpy 数组 (N, D)
        window_size: 窗口大小
    
    Returns:
        平滑后的数组
    """
    if len(data_array) <= window_size:
        return data_array
    
    smoothed = np.copy(data_array)
    half_window = window_size // 2
    
    for i in range(half_window, len(data_array) - half_window):
        smoothed[i] = np.mean(data_array[i-half_window:i+half_window+1], axis=0)
    
    return smoothed


def retarget_raw_data(raw_data, retarget, model, data, target_fps, optimize_latency=False, smooth_window=3):
    """
    对原始 VR 数据进行重定向
    
    Args:
        raw_data: 原始 VR pkl 数据
        retarget: GMR 实例
        model: MuJoCo 模型
        data: MuJoCo 数据
        target_fps: 目标帧率
        optimize_latency: 是否优化网络延迟 (使用插值而非去除重复帧)
        smooth_window: 平滑窗口大小 (备用)
    
    Returns:
        dict: 重定向后的数据
    """
    frames = raw_data.get('frames', [])
    fps = raw_data.get('fps', target_fps)
    
    if len(frames) == 0:
        print("[yellow]警告: 没有帧数据[/yellow]")
        return None
    
    # 检测重复帧区域 (用于后续插值)
    duplicate_regions = []
    if optimize_latency:
        duplicate_regions, valid_indices = detect_duplicate_regions(frames)
        if len(duplicate_regions) > 0:
            total_dup_frames = sum(end - start - 1 for start, end in duplicate_regions)
            print(f"  [检测] 发现 {len(duplicate_regions)} 个卡顿区域，共 {total_dup_frames} 个重复帧需要插值")
    
    # 存储结果
    root_pos_list = []
    root_rot_list = []
    dof_pos_list = []
    local_body_pos_list = []
    
    for frame_data in frames:
        if frame_data is None or len(frame_data) == 0:
            continue
        
        try:
            # 进行重定向
            qpos = retarget.retarget(frame_data, offset_to_ground=True)
            
            # 更新 MuJoCo
            data.qpos[:] = qpos.copy()
            mj.mj_forward(model, data)
            
            # 提取数据
            root_pos = qpos[0:3]
            
            # MuJoCo 格式 [w,x,y,z] -> 标准格式 [x,y,z,w]
            root_rot_wxyz = qpos[3:7]
            root_rot = root_rot_wxyz[[1, 2, 3, 0]]
            
            dof_pos = qpos[7:]
            
            # 计算局部 body 位置 (相对于根节点，在根节点局部坐标系中)
            # 参考 mujoco_exec_eval.py 中的 _Kinematics.bodies_rel_pelvis 方法
            # 1. 先计算全局坐标系中的位置差
            global_body_pos = data.xpos.copy()
            delta = global_body_pos - root_pos[None, :]  # (n_bodies, 3) - (1, 3)
            # 2. 将位置差旋转到根关节的局部坐标系
            # root_rot_wxyz 是 MuJoCo 格式 [w,x,y,z]，需要转换为 scipy 格式 [x,y,z,w]
            root_rot_scipy = root_rot_wxyz[[1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
            root_rot_mat = R.from_quat(root_rot_scipy).as_matrix()  # (3, 3)
            # 应用逆旋转：local = R^T @ delta^T -> (3, n_bodies) -> 转置得到 (n_bodies, 3)
            local_body_pos = (root_rot_mat.T @ delta.T).T
            
            root_pos_list.append(root_pos)
            root_rot_list.append(root_rot)
            dof_pos_list.append(dof_pos)
            local_body_pos_list.append(local_body_pos)
            
        except Exception as e:
            print(f"[red]帧处理错误: {e}[/red]")
            continue
    
    if len(root_pos_list) == 0:
        print("[yellow]警告: 没有成功处理的帧[/yellow]")
        return None
    
    # 获取刚体名称列表，并过滤掉 'world' body 以保持与其他数据集一致 (38 bodies)
    all_body_names = [model.body(i).name for i in range(model.nbody)]
    
    # 找到需要保留的 body 索引（排除 'world'）
    valid_body_indices = [i for i, name in enumerate(all_body_names) if name != 'world']
    link_body_list = [all_body_names[i] for i in valid_body_indices]
    
    # 将结果转换为 numpy 数组
    root_pos_arr = np.array(root_pos_list, dtype=np.float64)
    root_rot_arr = np.array(root_rot_list, dtype=np.float64)
    dof_pos_arr = np.array(dof_pos_list, dtype=np.float64)
    local_body_pos_arr = np.array(local_body_pos_list, dtype=np.float32)
    
    # 过滤掉 'world' body 的位置数据
    local_body_pos_arr = local_body_pos_arr[:, valid_body_indices, :]
    
    # 网络延迟优化: 对重复帧区域进行线性插值
    if optimize_latency and len(duplicate_regions) > 0:
        root_pos_arr = interpolate_duplicate_regions(root_pos_arr, duplicate_regions)
        root_rot_arr = interpolate_duplicate_regions(root_rot_arr, duplicate_regions)
        dof_pos_arr = interpolate_duplicate_regions(dof_pos_arr, duplicate_regions)
        local_body_pos_arr = interpolate_duplicate_regions(local_body_pos_arr, duplicate_regions)
        print(f"  [插值] 已对 {len(duplicate_regions)} 个卡顿区域进行线性插值")
    
    # 整理输出数据
    retargeted_data = {
        'fps': np.float64(fps),
        'root_pos': root_pos_arr,
        'root_rot': root_rot_arr,
        'dof_pos': dof_pos_arr,
        'local_body_pos': local_body_pos_arr,
        'link_body_list': link_body_list
    }
    
    return retargeted_data


def process_directory(input_dir, output_dir, robot_name, actual_human_height, target_fps, 
                      visualize=False, optimize_latency=False, smooth_window=3):
    """
    处理整个目录中的原始 VR 数据
    
    Args:
        input_dir: 原始数据目录 (raw/)
        output_dir: 输出目录 (retargeted/)
        robot_name: 机器人名称
        actual_human_height: 操作者身高
        target_fps: 目标帧率
        visualize: 是否可视化
        optimize_latency: 是否优化网络延迟
        smooth_window: 平滑窗口大小
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
    
    print(f"[bold]找到 {len(pkl_files)} 个 raw 文件待处理[/bold]")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    if optimize_latency:
        print(f"  [启用] 网络延迟优化 (平滑窗口={smooth_window})")
    print()
    
    # 初始化 GMR
    print("[bold]初始化 GMR 重定向系统...[/bold]")
    retarget = GMR(
        src_human="xrobot",
        tgt_robot=robot_name,
        actual_human_height=actual_human_height,
    )
    print("  ✓ GMR 已初始化")
    
    # 初始化 MuJoCo
    xml_file = ROBOT_XML_DICT[robot_name]
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    print("  ✓ MuJoCo 模型已加载")
    print()
    
    # 处理每个文件
    success_count = 0
    error_count = 0
    
    for pkl_path in tqdm(pkl_files, desc="处理中"):
        filename = os.path.basename(pkl_path)
        output_path = os.path.join(output_dir, filename)
        
        # 跳过已存在的文件
        if os.path.exists(output_path):
            print(f"[dim]跳过 (已存在): {filename}[/dim]")
            success_count += 1
            continue
        
        try:
            # 加载原始数据
            raw_data = load_raw_pkl(pkl_path)
            
            # 重定向
            retargeted_data = retarget_raw_data(
                raw_data, retarget, model, data, target_fps,
                optimize_latency=optimize_latency, smooth_window=smooth_window
            )
            
            if retargeted_data is not None:
                # 保存
                save_retargeted_pkl(output_path, retargeted_data)
                n_frames = retargeted_data['root_pos'].shape[0]
                print(f"[green]✓ {filename}: {n_frames} 帧[/green]")
                success_count += 1
            else:
                print(f"[yellow]跳过 (无有效帧): {filename}[/yellow]")
                error_count += 1
                
        except Exception as e:
            print(f"[red]✗ {filename}: {e}[/red]")
            error_count += 1
    
    # 完成报告
    print()
    print("[bold]处理完成![/bold]")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {error_count} 个文件")
    print(f"  输出目录: {output_dir}")


def preview_single_file(pkl_path, robot_name, actual_human_height, target_fps, 
                        optimize_latency=False, smooth_window=3, record_video=None):
    """
    预览单个 raw 文件，进行可视化，不保存结果
    
    Args:
        pkl_path: raw pkl 文件路径
        robot_name: 机器人名称
        actual_human_height: 操作者身高
        target_fps: 目标帧率
        optimize_latency: 是否优化网络延迟
        smooth_window: 平滑窗口大小
        record_video: 视频保存路径 (如果指定则录制视频)
    """
    # 展开路径
    pkl_path = os.path.expanduser(pkl_path)
    
    if not os.path.exists(pkl_path):
        print(f"[red]错误: 文件不存在: {pkl_path}[/red]")
        return
    
    print(f"[bold]预览模式: {pkl_path}[/bold]")
    if optimize_latency:
        print(f"  [启用] 网络延迟优化 (平滑窗口={smooth_window})")
    print()
    
    # 初始化 GMR
    print("[bold]初始化 GMR 重定向系统...[/bold]")
    retarget = GMR(
        src_human="xrobot",
        tgt_robot=robot_name,
        actual_human_height=actual_human_height,
    )
    print("  ✓ GMR 已初始化")
    
    # 初始化 MuJoCo
    xml_file = ROBOT_XML_DICT[robot_name]
    robot_base = ROBOT_BASE_DICT[robot_name]
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    print("  ✓ MuJoCo 模型已加载")
    
    # 加载原始数据
    print(f"  加载: {pkl_path}")
    raw_data = load_raw_pkl(pkl_path)
    
    # 重定向
    retargeted_data = retarget_raw_data(
        raw_data, retarget, model, data, target_fps,
        optimize_latency=optimize_latency, smooth_window=smooth_window
    )
    
    if retargeted_data is None:
        print("[red]错误: 无法处理数据[/red]")
        return
    
    n_frames = retargeted_data['root_pos'].shape[0]
    fps = retargeted_data['fps']
    print(f"  ✓ 重定向完成: {n_frames} 帧")
    
    # 保存重定向后的数据到 temp 目录
    temp_dir = "/home/huanghao/source/temp"
    os.makedirs(temp_dir, exist_ok=True)
    pkl_basename = os.path.splitext(os.path.basename(pkl_path))[0]
    output_pkl_path = os.path.join(temp_dir, f"{pkl_basename}_retargeted.pkl")
    save_retargeted_pkl(output_pkl_path, retargeted_data)
    print(f"  ✓ 已保存到: {output_pkl_path}")
    print()
    
    # 可视化播放 / 录制视频
    if record_video:
        # 录制模式: 使用 RobotMotionViewer (自动处理 OpenGL 上下文)
        video_dir = os.path.expanduser("~/video")
        os.makedirs(video_dir, exist_ok=True)
        
        pkl_basename = os.path.splitext(os.path.basename(pkl_path))[0]
        video_path = os.path.join(video_dir, f"{pkl_basename}.mp4")
        
        print(f"[bold green]开始录制视频: {video_path}[/bold green]")
        
        # 使用 RobotMotionViewer 录制视频 (解决 headless OpenGL 问题)
        robot_motion_viewer = RobotMotionViewer(
            robot_type=robot_name,
            motion_fps=fps,
            transparent_robot=0,
            record_video=True,
            video_path=video_path,
            video_width=1280,
            video_height=720
        )
        
        for frame_idx in tqdm(range(n_frames), desc="录制中"):
            # 获取当前帧数据
            root_pos = retargeted_data['root_pos'][frame_idx]
            root_rot = retargeted_data['root_rot'][frame_idx]  # [x,y,z,w]
            dof_pos = retargeted_data['dof_pos'][frame_idx]
            
            # 转换四元数: [x,y,z,w] -> MuJoCo 格式 [w,x,y,z]
            root_rot_wxyz = root_rot[[3, 0, 1, 2]]
            
            robot_motion_viewer.step(
                root_pos=root_pos,
                root_rot=root_rot_wxyz,
                dof_pos=dof_pos,
                rate_limit=False,  # 录制时不限速
                follow_camera=True
            )
        
        robot_motion_viewer.close()
        print(f"[bold green]✓ 视频已保存: {video_path}[/bold green]")
        
    else:
        # 预览模式: 显示窗口
        print("[bold green]开始可视化播放...[/bold green]")
        print("  按 ESC 或关闭窗口退出")
        print()
        
        rate = RateLimiter(frequency=fps, warn=False)
        
        with mjv.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False
        ) as viewer:
            frame_idx = 0
            
            while viewer.is_running():
                # 获取当前帧数据
                root_pos = retargeted_data['root_pos'][frame_idx]
                root_rot = retargeted_data['root_rot'][frame_idx]  # [x,y,z,w]
                dof_pos = retargeted_data['dof_pos'][frame_idx]
                
                # 转换四元数回 MuJoCo 格式 [w,x,y,z]
                root_rot_wxyz = root_rot[[3, 0, 1, 2]]
                
                # 更新 qpos
                data.qpos[0:3] = root_pos
                data.qpos[3:7] = root_rot_wxyz
                data.qpos[7:] = dof_pos
                
                mj.mj_forward(model, data)
                
                # 相机跟随
                robot_base_pos = data.xpos[model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                viewer.cam.distance = 3.0
                
                viewer.sync()
                rate.sleep()
                
                # 循环播放
                frame_idx = (frame_idx + 1) % n_frames
        
        print("[bold]预览结束[/bold]")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch Retarget Raw VR Data")
    
    # 互斥组: 批量处理 vs 单文件预览
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir",
        type=str,
        help="原始 VR 数据目录 (raw/) - 批量处理模式"
    )
    group.add_argument(
        "--preview",
        type=str,
        help="单个 raw pkl 文件路径 - 预览模式 (可视化，不保存)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="重定向后的输出目录 (retargeted/) - 批量处理模式必须"
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
        help="目标机器人类型"
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.80,
        help="操作者实际身高 (米)"
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="目标帧率"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="是否显示可视化窗口"
    )
    parser.add_argument(
        "--optimize_latency",
        action="store_true",
        help="启用网络延迟优化 (去除重复帧 + 平滑滤波)"
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=8,
        help="平滑窗口大小 (默认 3)"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="录制视频并保存到 ~/video/ 目录 (视频名与 pkl 文件名相同) - 仅预览模式可用"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.preview:
        # 单文件预览模式
        preview_single_file(
            pkl_path=args.preview,
            robot_name=args.robot,
            actual_human_height=args.actual_human_height,
            target_fps=args.target_fps,
            optimize_latency=args.optimize_latency,
            smooth_window=args.smooth_window,
            record_video=args.record
        )
    else:
        # 批量处理模式
        if args.output_dir is None:
            print("[red]错误: 批量处理模式需要指定 --output_dir[/red]")
            exit(1)
        
        process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            robot_name=args.robot,
            actual_human_height=args.actual_human_height,
            target_fps=args.target_fps,
            visualize=args.visualize,
            optimize_latency=args.optimize_latency,
            smooth_window=args.smooth_window
        )
