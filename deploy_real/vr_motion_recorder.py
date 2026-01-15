"""
VR Motion Capture Data Recorder
================================
基于 VR 遥操作进行动捕数据收集，将 VR 数据实时重定向为机器人格式并保存为 PKL 文件。
支持无限循环录制，一个进程可以录制多个项目。

使用方法:
    conda activate gmr
    
    # 单人模式 (停止录制后自动保存，无需键盘确认)
    python vr_motion_recorder.py --robot unitree_g1 --output_dir  ~/twist2_pico/huanghao --solo_mode
    
    # 双人模式 (停止录制后先保存，再询问是否保留，输入 n 则删除文件)
    python vr_motion_recorder.py --robot unitree_g1 --output_dir  ~/twist2_pico/huanghao
    

控制按键:
    - 右手 B键 (key_two): 开始/停止录制 (停止后自动保存)
    - 左手 A键 (key_one): 退出程序

输出格式 (PKL):
    {
        'fps': float64,              # 录制帧率
        'root_pos': (N, 3) float64,  # 根节点位置 XYZ
        'root_rot': (N, 4) float64,  # 根节点旋转四元数 (x,y,z,w)
        'dof_pos': (N, 29) float64,  # 关节位置 (G1 有 29 个自由度)
        'local_body_pos': (N, 38, 3) float32,  # 局部刚体位置
        'link_body_list': list[str]  # 刚体名称列表 (38 个)
    }
"""
import argparse
import os
import pickle
import time
import subprocess
import threading
from datetime import datetime

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import draw_frame
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from general_motion_retargeting import XRobotStreamer
from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()


def play_beep(frequency=800, duration_ms=200, count=1):
    """
    播放蜂鸣声提示 (用于通知 VR 操作者)
    异步播放，不阻塞主线程
    """
    def _play():
        try:
            for i in range(count):
                # 使用 Linux 的 beep 命令 (需要安装 beep 包)
                # 如果没有 beep，尝试使用 aplay + 生成的音频
                result = subprocess.run(
                    ['paplay', '/usr/share/sounds/freedesktop/stereo/bell.oga'],
                    capture_output=True, timeout=1
                )
                if result.returncode != 0:
                    # 备用：终端响铃
                    print('\a', end='', flush=True)
                if count > 1 and i < count - 1:
                    time.sleep(0.15)
        except Exception:
            # 失败时使用终端响铃
            print('\a', end='', flush=True)
    
    # 异步播放，不阻塞主循环
    threading.Thread(target=_play, daemon=True).start()


class MotionRecorder:
    """动捕数据录制器"""
    
    def __init__(self, target_fps=30.0):
        self.target_fps = target_fps
        self.is_recording = False
        self.recorded_frames = []  # 重定向后的数据
        self.recorded_raw_frames = []  # 原始 VR 数据 (重定向前)
        self.recording_start_time = None
        
        # 按键状态追踪
        self.right_key_two_was_pressed = False  # B键 - 开始/停止录制
        self.left_key_one_was_pressed = False   # A键 - 退出
        
        # 统计信息
        self.total_saved_files = 0
        self.last_saved_retargeted_path = None  # 最后保存的重定向文件路径
        self.last_saved_raw_path = None  # 最后保存的原始数据文件路径
        
    def update_controls(self, controller_data):
        """更新控制器状态，返回触发的动作"""
        actions = []
        
        if controller_data is None:
            return actions
        
        # 获取当前按键状态
        right_key_two = controller_data.get('RightController', {}).get('key_two', False)
        left_key_one = controller_data.get('LeftController', {}).get('key_one', False)
        
        # 检测按键按下事件 (Rising Edge)
        # B键: 开始/停止录制
        if right_key_two and not self.right_key_two_was_pressed:
            if self.is_recording:
                actions.append('stop_recording')
            else:
                actions.append('start_recording')
            
        # 左手 A键: 退出
        # if left_key_one and not self.left_key_one_was_pressed:
            # actions.append('exit')
        
        # 更新按键历史状态
        self.right_key_two_was_pressed = right_key_two
        self.left_key_one_was_pressed = left_key_one
        
        return actions
    
    def start_recording(self):
        """开始录制"""
        self.is_recording = True
        self.recorded_frames = []  # 清空重定向数据
        self.recorded_raw_frames = []  # 清空原始数据
        self.recording_start_time = time.time()
        play_beep(count=1)  # 单次蜂鸣 = 开始录制
        print("[bold green]▶ 开始录制...[/bold green]\n")
        
    def stop_recording(self):
        """停止录制"""
        self.is_recording = False
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        play_beep(count=2)  # 双次蜂鸣 = 停止录制
        print(f"[bold yellow]⏹ 停止录制 | 总帧数: {len(self.recorded_frames)} | 时长: {duration:.1f}s[/bold yellow]\n")
        
    def add_frame(self, root_pos, root_rot, dof_pos, local_body_pos, smplx_data=None):
        """添加一帧数据 (重定向后 + 原始 VR 数据)"""
        if not self.is_recording:
            return
        
        # 保存重定向后的数据
        self.recorded_frames.append({
            'root_pos': root_pos.copy(),
            'root_rot': root_rot.copy(),
            'dof_pos': dof_pos.copy(),
            'local_body_pos': local_body_pos.copy()
        })
        
        # 保存原始 VR 数据 (深拷贝)
        if smplx_data is not None:
            raw_frame = {}
            for key, value in smplx_data.items():
                if isinstance(value, list) and len(value) == 2:
                    # value 是 [position, rotation] 格式
                    raw_frame[key] = [list(value[0]), list(value[1])]
                else:
                    raw_frame[key] = value
            self.recorded_raw_frames.append(raw_frame)
        
    def _get_next_file_number(self, retargeted_dir):
        """获取下一个文件编号 (从1开始，基于 retargeted 目录中已有文件数量)"""
        if not os.path.exists(retargeted_dir):
            return 1
        
        existing_files = [f for f in os.listdir(retargeted_dir) if f.endswith('.pkl')]
        return len(existing_files) + 1
    
    def save_to_pkl(self, output_dir, link_body_list):
        """保存录制数据到 PKL 文件，返回重定向文件路径
        
        文件保存结构:
            output_dir/
                retargeted/  # 重定向后的机器人数据
                    motion_001.pkl
                    motion_002.pkl
                raw/  # 原始 VR 动捕数据
                    motion_001.pkl
                    motion_002.pkl
        """
        if len(self.recorded_frames) == 0:
            print("[bold red]✗ 没有录制数据可保存[/bold red]\n")
            return None
        
        # 确保输出目录存在 (展开 ~ 符号)
        expanded_dir = os.path.expanduser(output_dir)
        retargeted_dir = os.path.join(expanded_dir, "retargeted")
        raw_dir = os.path.join(expanded_dir, "raw")
        os.makedirs(retargeted_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)
        
        # 获取下一个文件编号
        file_number = self._get_next_file_number(retargeted_dir)
        filename = f"motion_{file_number:03d}.pkl"
        
        retargeted_filepath = os.path.join(retargeted_dir, filename)
        raw_filepath = os.path.join(raw_dir, filename)
        
        # 整理重定向后的数据
        n_frames = len(self.recorded_frames)
        retargeted_data = {
            'fps': np.float64(self.target_fps),
            'root_pos': np.array([f['root_pos'] for f in self.recorded_frames], dtype=np.float64),
            'root_rot': np.array([f['root_rot'] for f in self.recorded_frames], dtype=np.float64),
            'dof_pos': np.array([f['dof_pos'] for f in self.recorded_frames], dtype=np.float64),
            'local_body_pos': np.array([f['local_body_pos'] for f in self.recorded_frames], dtype=np.float32),
            'link_body_list': link_body_list
        }
        
        # 整理原始 VR 数据
        raw_data = {
            'fps': np.float64(self.target_fps),
            'frames': self.recorded_raw_frames,  # 原始 smplx_data 列表
            'n_frames': len(self.recorded_raw_frames)
        }
        
        # 保存重定向后的数据
        with open(retargeted_filepath, 'wb') as f:
            pickle.dump(retargeted_data, f)
        
        # 保存原始 VR 数据
        with open(raw_filepath, 'wb') as f:
            pickle.dump(raw_data, f)
            
        self.total_saved_files += 1
        self.last_saved_retargeted_path = retargeted_filepath
        self.last_saved_raw_path = raw_filepath
        
        # 打印信息
        print(f"[bold green]✓ 已保存 #{file_number:03d}:[/bold green]\n")
        print(f"  重定向数据: {retargeted_filepath}\n")
        print(f"  原始VR数据: {raw_filepath}\n")
        print(f"  帧数: {n_frames} | FPS: {self.target_fps}\n")
        
        # 清空录制帧，准备下一次录制
        self.recorded_frames = []
        self.recorded_raw_frames = []
        
        return retargeted_filepath
    
    def delete_last_saved_file(self):
        """删除最后保存的文件对 (用于双人模式)"""
        deleted = False
        
        # 删除重定向数据文件
        if self.last_saved_retargeted_path and os.path.exists(self.last_saved_retargeted_path):
            os.remove(self.last_saved_retargeted_path)
            print(f"[yellow]✗ 已删除: {self.last_saved_retargeted_path}[/yellow]\n")
            self.last_saved_retargeted_path = None
            deleted = True
        
        # 删除原始数据文件
        if self.last_saved_raw_path and os.path.exists(self.last_saved_raw_path):
            os.remove(self.last_saved_raw_path)
            print(f"[yellow]✗ 已删除: {self.last_saved_raw_path}[/yellow]\n")
            self.last_saved_raw_path = None
            deleted = True
        
        if deleted:
            self.total_saved_files -= 1
        
        return deleted
        
    def get_status_string(self):
        """获取当前状态字符串 (用于终端显示)"""
        if self.is_recording:
            duration = time.time() - self.recording_start_time
            n_frames = len(self.recorded_frames)
            actual_fps = n_frames / duration if duration > 0 else 0
            return f"[red]● 录制中[/red] | 帧数: {n_frames:5d} | 时长: {duration:6.1f}s | FPS: {actual_fps:5.1f}\n"
        else:
            return f"[dim]○ 待命[/dim] | 已保存: {self.total_saved_files} 个文件\n"


class VRMotionRecorderApp:
    """VR 动捕数据录制应用"""
    
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        self.target_fps = args.target_fps
        self.output_dir = args.output_dir
        self.solo_mode = args.solo_mode  # 单人模式：自动保存，无需确认
        
        # 初始化组件
        self.teleop_streamer = None
        self.retarget = None
        self.model = None
        self.data = None
        self.rate = None
        self.recorder = MotionRecorder(target_fps=self.target_fps)
        
        # 状态追踪
        self.last_qpos = None
        self.link_body_list = None
        self.should_exit = False
        
    def setup(self):
        """初始化所有系统"""
        print("[bold]初始化 VR 动捕录制系统...[/bold]")
        
        # 初始化 VR 数据流
        self.teleop_streamer = XRobotStreamer()
        print("  ✓ VR 数据流已连接\n")
        
        # 初始化运动重定向
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=self.args.actual_human_height,
        )
        print("  ✓ GMR 重定向系统已初始化\n")
        
        # 初始化 MuJoCo
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        print("  ✓ MuJoCo 模型已加载\n")
        
        # 获取刚体名称列表
        self.link_body_list = [self.model.body(i).name for i in range(self.model.nbody)]
        print(f"  ✓ 刚体列表: {len(self.link_body_list)} 个\n")
        
        # 初始化帧率限制器
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        print(f"  ✓ 帧率限制器: {self.target_fps} FPS\n")
        
        print()
        print("[bold green]系统就绪! 支持无限循环录制[/bold green]\n")
        print("[dim]控制说明:[/dim]\n")
        print("  右手 B键: 开始/停止录制 (停止后自动保存)\n")
        print("  左手 A键: 退出程序\n")
        if self.solo_mode:
            print("  [green]单人模式: 停止录制后自动保存[/green]\n")
        else:
            print("  [cyan]双人模式: 停止录制后保存，再询问是否保留 (y=保留, n=删除)[/cyan]\n")
        print()
        
    def get_teleop_data(self):
        """获取 VR 数据"""
        if self.teleop_streamer is not None:
            return self.teleop_streamer.get_current_frame()
        return None, None, None, None, None
        
    def process_retargeting(self, smplx_data):
        """处理运动重定向"""
        if smplx_data is None or self.retarget is None:
            return None
            
        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        self.last_qpos = qpos.copy()
        return qpos
        
    def extract_frame_data(self, qpos):
        """从 qpos 和 MuJoCo 数据中提取帧数据"""
        if qpos is None:
            return None, None, None, None
            
        # 更新 MuJoCo 状态
        self.data.qpos[:] = qpos.copy()
        mj.mj_forward(self.model, self.data)
        
        # 提取数据
        root_pos = qpos[0:3]  # (3,)
        
        # MuJoCo 的四元数格式是 [w, x, y, z] (scalar-first)
        # 转换为标准格式 [x, y, z, w] (scalar-last)，与其他数据集一致
        root_rot_wxyz = qpos[3:7]  # MuJoCo 格式: [w, x, y, z]
        root_rot = root_rot_wxyz[[1, 2, 3, 0]]  # 转换为: [x, y, z, w]
        
        dof_pos = qpos[7:]    # (29,) - 关节位置
        
        # 获取局部刚体位置 (从 MuJoCo)
        # xpos 是全局位置，我们需要转换为局部位置
        local_body_pos = self.data.xpos.copy()  # (nbody, 3)
        
        return root_pos, root_rot, dof_pos, local_body_pos
        
    def update_visualization(self, qpos, smplx_data, viewer):
        """更新 MuJoCo 可视化"""
        if qpos is None:
            return
            
        # 清除自定义几何体
        if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
            
        # 绘制人体骨骼参考
        if smplx_data is not None and self.retarget is not None:
            for robot_link, ik_data in self.retarget.ik_match_table1.items():
                body_name = ik_data[0]
                if body_name not in smplx_data:
                    continue
                draw_frame(
                    self.retarget.scaled_human_data[body_name][0] - self.retarget.ground,
                    R.from_quat(smplx_data[body_name][1]).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1]),
                )
                
        # 更新仿真状态
        self.data.qpos[:] = qpos.copy()
        mj.mj_forward(self.model, self.data)
        
        # 相机跟随
        robot_base_pos = self.data.xpos[self.model.body(self.robot_base).id]
        viewer.cam.lookat = robot_base_pos
        viewer.cam.distance = 3.0
        
    def run(self):
        """主循环"""
        self.setup()
        
        with mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            
            last_print_time = time.time()
            
            while viewer.is_running() and not self.should_exit:
                # 获取 VR 数据
                smplx_data, _, _, controller_data, _ = self.get_teleop_data()
                
                # 处理控制器输入
                actions = self.recorder.update_controls(controller_data)
                for action in actions:
                    if action == 'start_recording':
                        self.recorder.start_recording()
                    elif action == 'stop_recording':
                        self.recorder.stop_recording()
                        
                        # 停止后立即保存
                        saved_path = self.recorder.save_to_pkl(self.output_dir, self.link_body_list)
                        
                        if not self.solo_mode and saved_path:
                            # 双人模式：先保存，再询问是否保留
                            console.print("[bold cyan]是否保留此录制? (y=保留, n=删除): [/bold cyan]\n", end="")
                            import sys
                            sys.stdout.flush()
                            
                            try:
                                user_input = input().strip().lower()
                                if user_input == 'n':
                                    self.recorder.delete_last_saved_file()
                                else:
                                    console.print("[green]✓ 已保留[/green]\n")
                            except EOFError:
                                pass
                        
                        # 准备下一次录制
                        print("[dim]准备下一次录制，按 B键 开始...[/dim]\n")
                        
                    elif action == 'exit':
                        self.should_exit = True
                        print("[bold]退出程序...[/bold]\n")
                        break
                
                # 处理运动重定向
                qpos = None
                if smplx_data is not None:
                    qpos = self.process_retargeting(smplx_data)
                    
                    # 提取帧数据并录制
                    if qpos is not None:
                        root_pos, root_rot, dof_pos, local_body_pos = self.extract_frame_data(qpos)
                        self.recorder.add_frame(root_pos, root_rot, dof_pos, local_body_pos, smplx_data)
                    
                    # 更新可视化
                    self.update_visualization(qpos, smplx_data, viewer)
                
                # 定期打印状态 (每 0.5 秒)
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    status = self.recorder.get_status_string()
                    # 使用 \r 覆盖同一行
                    console.print(f"\r{status}", end="")
                    last_print_time = current_time
                
                viewer.sync()
                self.rate.sleep()
            
            print()  # 换行
            print("[bold]VR 动捕录制系统已关闭[/bold]\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="VR Motion Capture Data Recorder")
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
        help="目标机器人类型"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vr_recordings",
        help="PKL 文件保存目录"
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.75,
        help="操作者实际身高 (米)"
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=60,
        help="目标录制帧率"
    )
    parser.add_argument(
        "--solo_mode",
        action="store_true",
        help="单人模式：停止录制后自动保存，无需键盘确认"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    app = VRMotionRecorderApp(args)
    app.run()
