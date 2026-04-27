#!/usr/bin/env python3
# 指定该脚本使用 Python 3 解释器执行
import argparse
# 导入 argparse 模块，用于解析命令行参数
import random
# 导入 random 模块，虽然在当前代码中未直接使用，但保留以备后续扩展
import time
# 导入 time 模块，用于获取时间戳和实现循环中的定时休眠
import json
# 导入 json 模块，用于将 NumPy 数组等数据结构序列化为 JSON 字符串后写入 Redis
import numpy as np
# 导入 NumPy 库，用于高效的数值计算和数组操作
import torch
# 导入 PyTorch 库，用于将观测张量送入 ONNXRuntime 策略网络前进行格式转换
import redis
# 导入 Redis 客户端库，用于与本地 Redis 服务器通信，收发高层/低层控制信号
from collections import deque
# 从 collections 模块导入 deque（双端队列），用于实现观测历史缓冲区的滑动窗口
# from robot_control.common.remote_controller import KeyMap
# 注释掉的导入：原本计划引入遥控器按键映射类，当前未使用

from robot_control.g1_wrapper import G1RealWorldEnv
# 从 robot_control.g1_wrapper 模块导入 G1RealWorldEnv 类，负责与真实 G1 机器人的底层通信
from robot_control.config import Config
# 从 robot_control.config 模块导入 Config 类，用于解析 YAML 格式的机器人配置文件
import os
# 导入 os 模块，用于检查文件是否存在等操作系统相关操作
from data_utils.rot_utils import quatToEuler
# 从 data_utils.rot_utils 模块导入 quatToEuler 函数，将 IMU 四元数转换为欧拉角 (roll, pitch, yaw)

from robot_control.dex_hand_wrapper import Dex3_1_Controller
# 从 robot_control.dex_hand_wrapper 模块导入 Dex3_1_Controller 类，用于控制灵巧手（可选）

try:
# 尝试执行以下导入语句
    import onnxruntime as ort
# 导入 ONNXRuntime 库，用于加载并推理导出的 ONNX 格式策略模型
except ImportError:
# 如果 above 导入失败（库未安装），捕获 ImportError 异常
    ort = None
# 将 ort 置为 None，后续通过检查该变量给出更友好的错误提示


class OnnxPolicyWrapper:
# 定义一个包装类，使 ONNXRuntime 的推理接口与 PyTorch 的调用方式保持一致
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""
# 类文档字符串：说明该类的作用是让 ONNXRuntime 策略的调用签名与 TorchScript 类似

    def __init__(self, session, input_name, output_index=0):
# 构造函数，初始化 ONNX 推理会话、输入节点名称和输出节点索引
        self.session = session
# 将 ONNXRuntime 的 InferenceSession 对象保存为实例属性
        self.input_name = input_name
# 保存 ONNX 模型的输入节点名称，后续推理时通过该名称传入数据
        self.output_index = output_index
# 保存需要返回的输出节点索引，默认取第一个输出

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
# 重载调用运算符，使对象可以像函数一样直接传入 obs_tensor 进行前向推理
        if isinstance(obs_tensor, torch.Tensor):
# 判断输入是否为 PyTorch 张量
            obs_np = obs_tensor.detach().cpu().numpy()
# 若是张量，先断开计算图、移到 CPU，再转为 NumPy 数组
        else:
# 若输入不是 PyTorch 张量
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
# 直接将其转换为 float32 类型的 NumPy 数组
        outputs = self.session.run(None, {self.input_name: obs_np})
# 调用 ONNXRuntime 的 run 方法进行推理：None 表示返回所有输出，字典传入输入数据
        result = outputs[self.output_index]
# 从输出列表中取出指定索引的结果
        if not isinstance(result, np.ndarray):
# 如果结果不是 NumPy 数组（某些 provider 可能返回列表）
            result = np.asarray(result, dtype=np.float32)
# 将其强制转换为 float32 的 NumPy 数组
        return torch.from_numpy(result.astype(np.float32))
# 最后将 NumPy 数组转回 PyTorch 张量并返回


class EMASmoother:
# 定义指数移动平均（EMA）平滑器类，用于对策略输出的动作做时域平滑
    """Exponential Moving Average smoother for body actions."""
# 类文档字符串：说明该类是用于身体动作的 EMA 平滑器
    
    def __init__(self, alpha=0.1, initial_value=None):
# 构造函数，传入平滑系数 alpha 和可选的初始值
        """
# 以下三行为该函数的文档字符串，说明参数含义
        Args:
            alpha: Smoothing factor (0.0=no smoothing, 1.0=maximum smoothing)
            initial_value: Initial value for smoothing (if None, will use first input)
        """
        self.alpha = alpha
# 保存平滑系数，alpha 越大对新值的权重越高
        self.initialized = False
# 初始化标志位，False 表示尚未收到第一个输入
        self.smoothed_value = initial_value
# 保存可选的初始值；若为 None，则在收到第一个输入时自动赋值
        
    def smooth(self, new_value):
# 定义平滑方法，传入新的动作值，返回平滑后的动作值
        """Apply EMA smoothing to new value."""
# 方法文档字符串：对 new_value 应用 EMA 平滑
        if not self.initialized:
# 如果是第一次调用 smooth 方法
            self.smoothed_value = new_value.copy() if hasattr(new_value, 'copy') else new_value
# 将平滑值初始化为 new_value 的副本（若支持 copy），否则直接引用
            self.initialized = True
# 将初始化标志置为 True，后续调用进入 EMA 计算分支
            return self.smoothed_value
# 直接返回初始化的平滑值
        
        # EMA formula: smoothed = alpha * new + (1 - alpha) * previous
# 注释说明 EMA 公式：新平滑值 = alpha × 新输入 + (1 - alpha) × 上一次平滑值
        self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
# 按 EMA 公式更新 smoothed_value
        return self.smoothed_value
# 返回更新后的平滑值
    
    def reset(self):
# 定义重置方法，用于清空平滑器状态
        """Reset the smoother to uninitialized state."""
# 方法文档字符串：将平滑器重置为未初始化状态
        self.initialized = False
# 将初始化标志恢复为 False
        self.smoothed_value = None
# 清空平滑值，等待下一次 smooth 调用时重新初始化


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
# 定义加载 ONNX 策略模型的函数，返回包装好的 OnnxPolicyWrapper 对象
    if ort is None:
# 检查之前 try 导入的 onnxruntime 是否成功
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
# 若未安装，抛出导入错误并给出明确提示
    providers = []
# 初始化空的执行 provider 列表，用于指定 ONNXRuntime 在 CPU 还是 GPU 上推理
    available = ort.get_available_providers()
# 获取当前系统支持的 provider 列表（如 CUDAExecutionProvider、CPUExecutionProvider）
    if device.startswith('cuda'):
# 如果用户指定设备以 cuda 开头（如 cuda:0）
        if 'CUDAExecutionProvider' in available:
# 检查 CUDA provider 是否可用
            providers.append('CUDAExecutionProvider')
# 将 GPU 推理 provider 加入列表
        else:
# 若 CUDA provider 不可用
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
# 打印降级提示信息
    providers.append('CPUExecutionProvider')
# 无论是否使用 GPU，都将 CPUExecutionProvider 作为兜底选项加入列表
    session = ort.InferenceSession(policy_path, providers=providers)
# 使用指定的 provider 列表创建 ONNXRuntime 推理会话，加载指定路径的 ONNX 模型
    input_name = session.get_inputs()[0].name
# 获取模型第一个输入节点的名称，后续用于构造输入字典
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
# 打印加载成功的日志，显示模型路径和实际使用的 providers
    return OnnxPolicyWrapper(session, input_name)
# 返回包装好的策略对象，供主控循环调用


class RealTimePolicyController(object):
# 定义实时策略控制器类，封装 TWIST2 真机部署的全部逻辑
    """
# 类文档字符串开始
    Real robot controller for TWIST2 policy.
# 说明这是 TWIST2 策略的真实机器人控制器
    Based on server_low_level_g1_real.py but adapted for TWIST2 architecture.
# 基于早期的 server_low_level_g1_real.py，但针对 TWIST2 架构做了适配
    """
# 类文档字符串结束
    def __init__(self, 
# 构造函数开始，定义控制器初始化所需的全部参数
                 policy_path,
# ONNX 策略模型文件的路径
                 config_path,
# 机器人配置文件（YAML）的路径
                 device='cuda',
# 策略推理使用的计算设备，默认为 CUDA
                 net='eno1',
# 与机器人通信使用的网卡名称，默认为 eno1
                 use_hand=False,
# 是否启用灵巧手控制，默认关闭
                 record_proprio=False,
# 是否记录本体感知数据，默认关闭
                 smooth_body=0.0,
                 sonic_pd=False):
# 身体动作的 EMA 平滑系数，默认 0.0 表示不平滑
        self.redis_client = None
# 初始化 Redis 客户端变量为 None，后续用于存储实际连接对象
        try:
# 尝试连接本地 Redis 服务器
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
# 创建 Redis 连接对象，地址为本地 6379 端口，数据库编号 0
            self.redis_pipeline = self.redis_client.pipeline()
# 创建 Redis 管道对象，用于批量执行多条 Redis 命令以减少网络往返
        except Exception as e:
# 如果连接 Redis 失败
            print(f"Error connecting to Redis: {e}")
# 打印连接错误信息
            exit()
# 直接退出程序，因为 Redis 是 Sim2Real 通信的核心依赖
       
        self.config = Config(config_path, use_sonic_pd=sonic_pd)
# 使用 Config 类解析 YAML 配置文件，获取机器人的关节映射、默认角度、PD 增益等参数
        if sonic_pd:
            print("[PD] Using SONIC-derived G1 PD gains for real robot deployment.")
        self.env = G1RealWorldEnv(net=net, config=self.config)
# 实例化真实机器人环境，建立与 G1 机器人的底层网络连接
        self.use_hand = use_hand
# 保存是否启用手部控制的标志
        if use_hand:
# 如果启用了手部控制
            self.hand_ctrl = Dex3_1_Controller(net, re_init=False)
# 实例化灵巧手控制器，re_init=False 表示不重新初始化硬件

        self.device = device
# 保存策略推理设备参数
        self.policy = load_onnx_policy(policy_path, device)
# 调用上方定义的函数加载 ONNX 策略模型

        self.num_actions = 29
# 定义机器人动作维度为 29，对应 G1 的 29 个自由度
        self.default_dof_pos = self.config.default_angles
# 从配置中读取默认关节角度，后续用于计算观测中的 dof_pos 偏差和动作到目标位置的转换
        
        # scaling factors
# 以下为本体观测各项的缩放系数，需与训练时的配置保持一致
        self.ang_vel_scale = 0.25
# 角速度观测缩放系数：3 rad/s 映射到观测空间的 0.75
        self.dof_vel_scale = 0.05
# 关节速度观测缩放系数
        self.dof_pos_scale = 1.0
# 关节位置偏差观测缩放系数
        self.ankle_idx = [4, 5, 10, 11]
# 脚踝关节在 29 维动作/观测中的索引：左踝 pitch、左踝 roll、右踝 pitch、右踝 roll

        # TWIST2 observation structure
# 以下定义 TWIST2 的观测空间维度结构
        self.n_mimic_obs = 35        # 6 + 29 (modified: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos)
# mimic_obs 维度为 35：根速度 xy（2）+ 根高度 z（1）+ roll/pitch（2）+ 偏航角速度（1）+ 29 维关节位置（29）
        self.n_proprio = 92          # from config analysis  
# 本体感知观测维度为 92：角速度（3）+ roll/pitch（2）+ 关节位置偏差（29）+ 关节速度（29）+ 上一帧动作（29）
        self.n_obs_single = 127      # n_mimic_obs + n_proprio = 35 + 92 = 127
# 单帧完整观测维度为 mimic_obs 与 proprio 之和，即 127
        self.history_len = 10
# 历史观测缓冲区长度为 10 帧
        
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 127*11 + 35 = 1402
# 最终送入策略网络的观测总维度 = 当前帧（127）+ 历史 10 帧（127*10）+ 未来 mimic_obs（35）= 1402
        
        print(f"TWIST2 Real Controller Configuration:")
# 打印控制器的配置信息标题
        print(f"  n_mimic_obs: {self.n_mimic_obs}")
# 打印 mimic_obs 维度
        print(f"  n_proprio: {self.n_proprio}")
# 打印 proprio 维度
        print(f"  n_obs_single: {self.n_obs_single}")
# 打印单帧观测维度
        print(f"  history_len: {self.history_len}")
# 打印历史长度
        print(f"  total_obs_size: {self.total_obs_size}")
# 打印总观测维度

        self.proprio_history_buf = deque(maxlen=self.history_len)
# 创建双端队列作为历史观测缓冲区，最大长度固定为 history_len（10）
        for _ in range(self.history_len):
# 循环 history_len 次，预填充零向量，避免首次推理时历史维度不足
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))
# 向队列追加一个长度为 n_obs_single 的 float32 零向量

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
# 初始化上一帧动作向量为 29 维零向量，用于拼接进 proprio 观测

        self.control_dt = self.config.control_dt
# 从配置中读取控制周期，通常为 0.02s（50Hz）
        self.action_scale = self.config.action_scale
# 从配置中读取动作缩放系数，通常为 0.5
        
        self.record_proprio = record_proprio
# 保存是否记录本体感知数据的标志
        self.proprio_recordings = [] if record_proprio else None
# 若需要记录，则初始化空列表；否则置为 None
        self.sonic_pd = sonic_pd
# 保存是否启用 SONIC PD 的标志，同时用于控制是否屏蔽脚踝观测
        
        # Smoothing processing
# 平滑处理相关初始化
        self.smooth_body = smooth_body
# 保存传入的身体平滑系数
        if smooth_body > 0.0:
# 如果平滑系数大于 0，说明用户希望开启 EMA 平滑
            self.body_smoother = EMASmoother(alpha=smooth_body)
# 实例化 EMA 平滑器
            print(f"Body action smoothing enabled with alpha={smooth_body}")
# 打印平滑已开启的提示
        else:
# 如果平滑系数为 0
            self.body_smoother = None
# 不创建平滑器，保持为 None

        
    def reset_robot(self):
# 定义机器人复位方法，在策略主循环开始前将机器人移动到安全默认姿态
        print("Press START on remote to move to default position ...")
# 提示用户按下遥控器 START 键，使机器人进入默认位置
        self.env.move_to_default_pos()
# 调用 G1RealWorldEnv 的方法，以插值方式在 2 秒内将机器人关节移动到 default_angles

        print("Now in default position, press A to continue ...")
# 提示用户按下遥控器 A 键，确认机器人已就位并继续
        self.env.default_pos_state()
# 进入默认位置保持状态，持续发送默认角度指令，直到 A 键被按下

        print("Robot will hold default pos. If needed, do other checks here.")
# 打印提示：机器人在开始策略循环前会保持默认姿态，可在此阶段做最后检查

    def run(self):
# 定义策略主循环方法，这是 TWIST2 Sim2Real 部署的核心运行函数
        self.reset_robot()
# 先执行机器人复位流程（等待 START 和 A 键）
        print("Begin main TWIST2 policy loop. Press [Select] on remote to exit.")
# 提示用户策略主循环已开始，按 Select 键可退出

        try:
# 使用 try 块包裹主循环，便于捕获异常并执行 finally 中的安全退出逻辑
            while True:
# 无限循环，持续执行感知-决策-控制循环
                t_start = time.time()
# 记录当前循环迭代的开始时间，用于后续控制频率对齐

                # Send remote control signals to Redis for motion server
# 以下代码读取遥控器状态，并通过 Redis 发送给高层的 Motion Server
                if self.redis_client:
# 确保 Redis 客户端已成功连接
                    # Send B button status (for motion start)
# 读取并发送 B 键状态，用于触发/暂停高层参考运动的播放
                    b_pressed = self.env.read_controller_input().keys == self.env.controller_mapping["B"]
# 检查遥控器当前按下的键是否为 B 键
                    self.redis_client.set("motion_start_signal", "1" if b_pressed else "0")
# 将 motion_start_signal 写入 Redis：按下为 "1"，否则为 "0"
                    
                    # Send Select button status (for motion exit)
# 读取并发送 Select 键状态，用于通知 Motion Server 用户请求退出
                    select_pressed = self.env.read_controller_input().keys == self.env.controller_mapping["select"]
# 检查遥控器当前按下的键是否为 Select 键
                    self.redis_client.set("motion_exit_signal", "1" if select_pressed else "0")
# 将 motion_exit_signal 写入 Redis：按下为 "1"，否则为 "0"
                    
                if self.env.read_controller_input().keys == self.env.controller_mapping["select"]:
# 再次检查 Select 键：如果本轮回合中被按下，则直接退出低层控制循环
                    print("Select pressed, exiting main loop.")
# 打印退出提示
                    break
# 跳出 while 循环，结束主控逻辑
                
                dof_pos, dof_vel, quat, ang_vel, dof_temp, dof_tau, dof_vol = self.env.get_robot_state()
# 从 G1 机器人底层接口读取当前状态：关节位置、速度、姿态四元数、角速度、温度、力矩、电压
                
                rpy = quatToEuler(quat)
# 将 IMU 四元数转换为欧拉角 (roll, pitch, yaw)

                obs_dof_vel = dof_vel.copy()
# 复制一份关节速度，用于构造观测向量（避免修改原始读取值）
                if not self.sonic_pd:
                    obs_dof_vel[self.ankle_idx] = 0.0
# 默认将脚踝关节的速度观测置零；启用 SONIC PD 时保留脚踝信号

                obs_proprio = np.concatenate([
# 拼接本体感知观测的各个组成部分
                    ang_vel * self.ang_vel_scale,
# 角速度（3 维）乘以缩放系数
                    rpy[:2], # 只使用 roll 和 pitch
# 欧拉角的前两维：roll 和 pitch（不使用 yaw，避免全局方向信息泄漏）
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
# 当前关节位置与默认位置的偏差（29 维）
                    obs_dof_vel * self.dof_vel_scale,
# 处理后的关节速度（29 维，其中脚踝为 0）乘以缩放系数
                    self.last_action
# 上一帧策略输出的动作（29 维），提供动作时序连续性
                ])
# proprio 观测拼接完成
                
                state_body = np.concatenate([
# 拼接一个额外的身体状态向量，用于通过 Redis 上报给上层系统（如数据记录、可视化）
                    ang_vel,
# 原始角速度（3 维）
                    rpy[:2],
# roll 和 pitch（2 维）
                    dof_pos]) # 3+2+29 = 34 dims
# 当前关节位置（29 维），共 34 维

                self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))
# 将身体状态序列化为 JSON 字符串后，通过 Redis Pipeline 设置到指定 key
                
                if self.use_hand:
# 如果启用了手部控制
                    left_hand_state, right_hand_state = self.hand_ctrl.get_hand_state()
# 获取左右手的当前状态
                    lh_pos, rh_pos, lh_temp, rh_temp, lh_tau, rh_tau = self.hand_ctrl.get_hand_all_state()
# 获取左右手的详细状态：位置、温度、力矩
                    hand_left_json = json.dumps(left_hand_state.tolist())
# 将左手状态转为 JSON 字符串
                    hand_right_json = json.dumps(right_hand_state.tolist())
# 将右手状态转为 JSON 字符串
                    self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", hand_left_json)
# 通过 Pipeline 设置左手状态到 Redis
                    self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", hand_right_json)
# 通过 Pipeline 设置右手状态到 Redis
                
                # execute the pipeline once here for setting the keys
# 注释说明：在此处一次性执行 Pipeline 中的所有写命令，减少网络往返
                self.redis_pipeline.execute()
# 执行 Pipeline，将上述 state_body 和手部状态批量写入 Redis

                # 5. 从 Redis 接收模仿观察
# 第 5 步：从 Redis 读取高层 Motion Server 下发的模仿观测（mimic observation）
                keys = ["action_body_unitree_g1_with_hands", "action_hand_left_unitree_g1_with_hands", "action_hand_right_unitree_g1_with_hands", "action_neck_unitree_g1_with_hands"]
# 定义需要从 Redis 读取的四个 key：身体动作、左右手动作、颈部动作
                for key in keys:
# 遍历上述 key 列表
                    self.redis_pipeline.get(key)
# 将每个 get 命令加入 Pipeline
                redis_results = self.redis_pipeline.execute()
# 批量执行 get 命令，返回结果列表
                action_mimic = json.loads(redis_results[0])
# 解析第一个结果（身体 mimic observation）从 JSON 字符串转为 Python 列表/数组
                action_hand_left = json.loads(redis_results[1])
# 解析左手动作
                action_hand_right = json.loads(redis_results[2])
# 解析右手动作
                action_neck = json.loads(redis_results[3])
# 解析颈部动作
                
                # Apply smoothing to body actions if enabled
# 如果启用了身体动作平滑，则对读取到的 mimic observation 做 EMA 滤波
                if self.body_smoother is not None:
# 检查平滑器是否存在
                    action_mimic = self.body_smoother.smooth(np.array(action_mimic, dtype=np.float32))
# 将 mimic observation 转为 float32 NumPy 数组后输入 EMA 平滑器
                    action_mimic = action_mimic.tolist()
# 平滑后再转回 Python 列表，方便后续拼接
            
                
                if self.use_hand:
# 如果启用了手部控制
                    action_hand_left = np.array(action_hand_left, dtype=np.float32)
# 将左手动作转为 float32 数组
                    action_hand_right = np.array(action_hand_right, dtype=np.float32)
# 将右手动作转为 float32 数组
                else:
# 如果未启用手部控制
                    action_hand_left = np.zeros(7, dtype=np.float32)
# 左手动作置为 7 维零向量
                    action_hand_right = np.zeros(7, dtype=np.float32)
# 右手动作置为 7 维零向量

                obs_full = np.concatenate([action_mimic, obs_proprio])
# 将 mimic observation（35 维）与本体感知观测（92 维）拼接，得到当前帧完整观测（127 维）
                
                obs_hist = np.array(self.proprio_history_buf).flatten()
# 将历史缓冲区中的 10 帧观测取出并展平为一维数组（127 * 10 = 1270 维）
                self.proprio_history_buf.append(obs_full)
# 将当前帧观测加入历史缓冲区，最旧的一帧会自动被 deque 移除
                
                future_obs = action_mimic.copy()
# 构造未来观测：当前代码简化为直接使用当前 mimic observation 作为 future obs（与训练时不完全一致）
                
                obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
# 拼接最终观测向量：当前帧（127）+ 历史（1270）+ 未来 mimic（35）= 1402 维
                
                assert obs_buf.shape[0] == self.total_obs_size, f"Expected {self.total_obs_size} obs, got {obs_buf.shape[0]}"
# 断言观测维度是否正确，若不一致则抛出错误提示维度不匹配
                
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
# 将 NumPy 观测数组转为 PyTorch float32 张量，增加 batch 维度 (1, 1402)，并移到指定设备（GPU/CPU）
                with torch.no_grad():
# 关闭梯度计算，节省显存并加速推理
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
# 调用 ONNX 策略网络进行推理，将结果移回 CPU、转为 NumPy，并去掉 batch 维度，得到 29 维动作
                
                self.last_action = raw_action.copy()
# 保存本帧原始动作，用于下一帧的 proprio 观测拼接

                raw_action = np.clip(raw_action, -10.0, 10.0)
# 对策略输出做裁剪，限制在 [-10, 10] 范围内，与训练时的 clip_actions 保持一致
                target_dof_pos = self.default_dof_pos + raw_action * self.action_scale
# 将裁剪后的动作乘以 action_scale（0.5）并叠加到默认关节角度上，得到目标关节位置

                # self.redis_client.set("action_low_level_unitree_g1", json.dumps(raw_action.tolist()))
# 注释掉的代码：原本计划将低层动作也写入 Redis，当前未使用

                kp_scale = 1.0
# PD 控制器比例增益缩放系数，默认 1.0
                kd_scale = 1.0
# PD 控制器微分增益缩放系数，默认 1.0
                self.env.send_robot_action(target_dof_pos, kp_scale, kd_scale)
# 将计算好的目标关节位置及 PD 增益缩放发送给真实机器人底层执行
                
                if self.use_hand:
# 如果启用了手部控制
                    self.hand_ctrl.ctrl_dual_hand(action_hand_left, action_hand_right)
# 将左右手目标动作发送给灵巧手控制器执行
                
                elapsed = time.time() - t_start
# 计算本轮循环实际耗时
                if elapsed < self.control_dt:
# 如果实际耗时小于控制周期（如 0.02s）
                    time.sleep(self.control_dt - elapsed)
# 通过 sleep 进行忙等待，使整体控制频率尽可能稳定在 50Hz

                if self.record_proprio:
# 如果开启了本体感知数据记录
                    proprio_data = {
# 构造一个字典，存储当前帧的各类感知数据
                        'timestamp': time.time(),
# 当前时间戳
                        'body_dof_pos': dof_pos.tolist(),
# 身体关节位置
                        'target_dof_pos': action_mimic.tolist()[-29:],
# 高层下发的目标关节位置（取 mimic observation 的最后 29 维，即参考 dof_pos）
                        'temperature': dof_temp.tolist(),
# 关节温度
                        'tau': dof_tau.tolist(),
# 关节力矩
                        'voltage': dof_vol.tolist(),
# 关节电压
                    }
                    
                    if self.use_hand:
# 若启用手部控制，补充手部数据
                        proprio_data['lh_pos'] = lh_pos.tolist()
# 左手位置
                        proprio_data['rh_pos'] = rh_pos.tolist()
# 右手位置
                        proprio_data['lh_temp'] = lh_temp.tolist()
# 左手温度
                        proprio_data['rh_temp'] = rh_temp.tolist()
# 右手温度
                        proprio_data['lh_tau'] = lh_tau.tolist()
# 左手力矩
                        proprio_data['rh_tau'] = rh_tau.tolist()
# 右手力矩
                    self.proprio_recordings.append(proprio_data)
# 将当前帧数据追加到记录列表中
                

        except Exception as e:
# 捕获主循环中发生的任何异常
            print(f"Error in main loop: {e}")
# 打印异常简要信息
            import traceback
# 动态导入 traceback 模块
            traceback.print_exc()
# 打印完整的异常堆栈，便于调试
        finally:
# finally 块：无论主循环正常结束还是异常退出，都会执行安全清理
            if self.record_proprio and self.proprio_recordings:
# 如果开启了数据记录且列表非空
                timestamp = time.strftime("%Y%m%d_%H%M%S")
# 生成当前时间的字符串格式时间戳
                filename = f'logs/twist2_real_recordings_{timestamp}.json'
# 构造记录文件的保存路径
                with open(filename, 'w') as f:
# 以写入模式打开文件
                    json.dump(self.proprio_recordings, f)
# 将记录列表序列化为 JSON 并写入文件
                print(f"Proprioceptive recordings saved as {filename}")
# 打印保存成功的提示

            self.env.close()
# 关闭与真实机器人底层通信的接口，使机器人进入安全状态
            if self.use_hand:
# 如果启用了手部控制
                self.hand_ctrl.close()
# 关闭灵巧手控制器连接
            print("TWIST2 real controller finished.")
# 打印控制器已结束的提示信息


def main():
# 定义程序入口函数
    parser = argparse.ArgumentParser(description='Run TWIST2 policy on real G1 robot')
# 创建命令行参数解析器，设置程序描述
    parser.add_argument('--policy', type=str, required=True,
# 添加 --policy 参数：ONNX 策略文件路径，必填
                        help='Path to TWIST2 ONNX policy file')
# 该参数的帮助说明
    parser.add_argument('--config', type=str, default="robot_control/configs/g1.yaml",
# 添加 --config 参数：机器人配置文件路径，默认值为 g1.yaml
                        help='Path to robot configuration file')
# 帮助说明
    parser.add_argument('--device', type=str, default='cuda',
# 添加 --device 参数：策略推理设备，默认 cuda
                        help='Device to run policy on (cuda/cpu)')
# 帮助说明
    parser.add_argument('--net', type=str, default='wlp0s20f3',
# 添加 --net 参数：机器人通信网卡名，默认 wlp0s20f3
                        help='Network interface for robot communication')
# 帮助说明
    parser.add_argument('--use_hand', action='store_true',
# 添加 --use_hand 开关参数：若命令行中出现该参数，则值为 True
                        help='Enable hand control')
# 帮助说明
    parser.add_argument('--record_proprio', action='store_true',
# 添加 --record_proprio 开关参数：用于开启本体感知数据记录
                        help='Record proprioceptive data')
# 帮助说明
    parser.add_argument('--smooth_body', type=float, default=0.0,
# 添加 --smooth_body 参数：身体动作 EMA 平滑系数，默认 0.0
                        help='Smoothing factor for body actions (0.0=no smoothing, 1.0=maximum smoothing)')
# 帮助说明
    parser.add_argument('--sonic_pd', action='store_true',
                        help='Use SONIC-derived G1 PD gains instead of the YAML kps/kds values')
    
    args = parser.parse_args()
# 解析命令行参数，结果存入 args 对象

    
    # 验证文件存在
# 以下代码验证用户传入的文件路径是否真实存在
    if not os.path.exists(args.policy):
# 检查 ONNX 策略文件是否存在
        print(f"Error: Policy file {args.policy} does not exist")
# 若不存在，打印错误并返回
        return
    
    if not os.path.exists(args.config):
# 检查 YAML 配置文件是否存在
        print(f"Error: Config file {args.config} does not exist")
# 若不存在，打印错误并返回
        return
    
    print(f"Starting TWIST2 real robot controller...")
# 打印启动信息
    print(f"  Policy file: {args.policy}")
# 打印策略文件路径
    print(f"  Config file: {args.config}")
# 打印配置文件路径
    print(f"  Device: {args.device}")
# 打印推理设备
    print(f"  Network interface: {args.net}")
# 打印网卡名称
    print(f"  Use hand: {args.use_hand}")
# 打印是否启用手部控制
    print(f"  Record proprio: {args.record_proprio}")
# 打印是否记录感知数据
    print(f"  Smooth body: {args.smooth_body}")
# 打印平滑系数
    print(f"  SONIC PD: {args.sonic_pd}")
# 打印是否启用 SONIC PD
    
    # 安全提示
# 以下打印安全警告信息，提醒用户确保真机环境安全
    print("\n" + "="*50)
# 打印换行和 50 个等号组成的分隔线
    print("SAFETY WARNING:")
# 打印安全警告标题
    print("You are about to run a policy on a real robot.")
# 提醒用户即将在真实机器人上运行策略
    print("Make sure the robot is in a safe environment.")
# 提醒确保环境安全
    print("Press Ctrl+C to stop at any time.")
# 提示可随时按 Ctrl+C 中断
    print("Use the remote controller [Select] button to exit.")
# 提示也可通过遥控器 Select 键安全退出
    print("="*50 + "\n")
# 打印分隔线和换行
    
    controller = RealTimePolicyController(
# 实例化实时策略控制器
        policy_path=args.policy,
# 传入 ONNX 策略路径
        config_path=args.config,
# 传入配置文件路径
        device=args.device,
# 传入推理设备
        net=args.net,
# 传入网卡名称
        use_hand=args.use_hand,
# 传入是否启用手部控制
        record_proprio=args.record_proprio,
# 传入是否记录感知数据
        smooth_body=args.smooth_body,
# 传入平滑系数
        sonic_pd=args.sonic_pd,
# 传入是否启用 SONIC PD
    )
# 实例化完成
    
    controller.run()
# 调用控制器的 run 方法，进入主循环
    


if __name__ == "__main__":
# Python 的惯用入口判断：当该脚本被直接运行时，以下代码才会执行
    main()
# 调用 main 函数，启动整个 TWIST2 真机控制器
