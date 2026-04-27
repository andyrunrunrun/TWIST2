Sim2Sim（Simulation-to-Simulation）是指在MuJoCo物理仿真环境中验证TWIST2策略控制器的性能，而不涉及真实的机器人硬件。这一阶段是Sim2Real（实物部署）前的重要验证环节，用于在安全可控的仿真环境中检验策略的正确性、稳定性以及运动跟踪效果。

本页面面向初级开发者，详细介绍Sim2Sim仿真验证的完整流程、架构设计、配置参数以及常见问题的解决方案。

---

## Sim2Sim架构概述

Sim2Sim验证系统采用**多进程架构**，通过Redis消息队列实现运动参考轨迹与策略控制器之间的异步通信。这种设计使得系统各组件可以独立运行、解耦测试，同时也便于调试和性能分析。

### 系统组件关系

Sim2Sim系统由三个核心组件构成，它们通过Redis进行数据交换：

```mermaid
flowchart TB
    subgraph Motion_Server["运动服务器 (server_motion_lib.py)"]
        A1[".pkl运动文件"] --> A2[MotionLib运动库]
        A2 --> A3[构建Mimic观测]
        A3 --> A4[Redis发布 action_body"]
    end
    
    subgraph Policy_Controller["策略控制器 (server_low_level_g1_sim.py)"]
        B1[ONNX策略模型] --> B2[策略推理]
        B3[MuJoCo物理仿真] --> B4[PD关节控制]
        B2 --> B3
    end
    
    subgraph Redis["Redis消息队列"]
        C1["action_body_unitree_g1_with_hands"]
        C2["state_body_unitree_g1_with_hands"]
    end
    
    A4 --> C1
    C1 --> B2
    B3 --> C2
    
    Motion_Server --> Redis
    Policy_Controller --> Redis
```

**数据流向说明**：运动服务器从`.pkl`文件中读取参考运动数据，计算并发布Mimic观测到Redis；策略控制器订阅这些观测，运行ONNX策略推理后输出控制动作到MuJoCo仿真器；仿真器执行物理仿真后反馈机器人状态回Redis供后续处理。

Sources: [server_motion_lib.py](deploy_real/server_motion_lib.py#L1-L50)
Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L1-L150)

---

## 快速启动流程

Sim2Sim验证需要依次启动三个服务：Redis服务器、运动服务器和策略控制器。以下是完整启动命令：

### 第一步：启动Redis服务器

Redis作为消息中间件，负责运动数据与策略控制之间的数据传输：

```bash
# 在终端1中启动Redis服务器
redis-server
```

如果Redis未安装，可以使用Docker快速部署：

```bash
docker run -d -p 6379:6379 redis:latest
```

### 第二步：启动运动服务器

运动服务器负责加载`.pkl`运动文件并发布Mimic观测数据：

```bash
# 在终端2中执行
cd /home/huanghao/source/code/TWIST2
bash run_motion_server.sh
```

`run_motion_server.sh`脚本内容如下：

```bash
#!/bin/bash
script_dir=$(dirname $(realpath $0))
motion_file="${script_dir}/assets/example_motions/0807_yanjie_walk_001.pkl"

cd deploy_real

python server_motion_lib.py \
    --motion_file ${motion_file} \
    --robot unitree_g1_with_hands \
    --vis \
    --redis_ip localhost
```

**关键参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--motion_file` | 运动文件路径 | 必填 |
| `--robot` | 机器人类型 | `unitree_g1_with_hands` |
| `--vis` | 是否可视化 | 启用 |
| `--redis_ip` | Redis服务器地址 | `localhost` |

Sources: [run_motion_server.sh](run_motion_server.sh#L1-L25)
Sources: [server_motion_lib.py](deploy_real/server_motion_lib.py#L80-L120)

### 第三步：启动策略控制器

策略控制器加载ONNX策略模型，在MuJoCo仿真环境中运行推理：

```bash
# 在终端3中执行
cd /home/huanghao/source/code/TWIST2
bash sim2sim.sh
```

`sim2sim.sh`脚本内容如下：

```bash
#!/bin/bash
SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

cd deploy_real

python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --measure_fps 1 \
    --policy_frequency 100 \
    --viewer_decimation 100 \
    --limit_fps 1
```

Sources: [sim2sim.sh](sim2sim.sh#L1-L15)

---

## 观察空间与状态结构

理解TWIST2策略的输入观察空间是进行Sim2Sim验证的基础。策略接收的观察数据包含三个主要部分：Mimic观测、本体感知观测以及历史缓冲。

### Mimic观测（动作参考）

Mimic观测是从运动库获取的参考动作目标，包含根节点运动信息和关节角度：

```
mimic_obs = [root_vel_xy, root_pos_z, roll_pitch, yaw_ang_vel, dof_pos]
         = [2, 1, 2, 1, 29] = 35维度
```

| 维度范围 | 内容 | 物理含义 |
|----------|------|----------|
| 0-1 | root_vel_xy | 骨盆线速度（局部坐标系） |
| 2 | root_pos_z | 骨盆高度 |
| 3-4 | roll_pitch | 翻滚角和俯仰角 |
| 5 | yaw_ang_vel | 偏航角速度 |
| 6-34 | dof_pos (29) | 29个关节角度 |

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L160-L175)
Sources: [params.py](deploy_real/data_utils/params.py#L1-L35)

### 本体感知观测

本体感知观测是机器人当前状态的实时反馈：

```
proprio_obs = [ang_vel, rpy, dof_pos_delta, dof_vel, last_action]
           = [3, 2, 29, 29, 29] = 92维度
```

| 组成部分 | 维度 | 描述 |
|----------|------|------|
| ang_vel | 3 | 角速度（乘以0.25缩放） |
| rpy | 2 | 翻滚角和俯仰角 |
| dof_pos_delta | 29 | 关节位置与默认位置之差 |
| dof_vel | 29 | 关节速度（踝关节置零，乘以0.05缩放） |
| last_action | 29 | 上一步策略输出 |

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L280-L300)

### 完整观察向量

策略接收的完整观察向量结构如下：

```python
total_obs_size = n_obs_single * (history_len + 1) + n_mimic_obs
               = 127 * 11 + 35
               = 1402维度
```

| 参数 | 值 | 说明 |
|------|-----|------|
| n_mimic_obs | 35 | 当前时刻Mimic观测 |
| n_proprio | 92 | 单帧本体感知观测 |
| n_obs_single | 127 | n_mimic_obs + n_proprio |
| history_len | 10 | 历史帧数 |
| total_obs_size | 1402 | 完整观察向量维度 |

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L168-L176)

---

## MuJoCo机器人模型配置

Sim2Sim使用专用的MuJoCo XML模型文件定义G1机器人的物理结构，包括刚体层级、关节约束、惯性参数和碰撞几何体。

### 模型文件选择

项目提供了多个G1机器人模型变体，Sim2Sim验证默认使用`g1_sim2sim_29dof.xml`：

```bash
# 可用的G1模型文件
assets/g1/g1_sim2sim_29dof.xml          # Sim2Sim标准模型 [推荐]
assets/g1/g1_sim2sim_29dof_modified.xml  # 修改版
assets/g1/g1_sim2sim_29dof_with_hands.xml # 含手部
assets/g1/g1_mocap_29dof.xml             # 动捕版本
```

Sources: [assets/g1/](assets/g1/)

### 关节配置

G1机器人具有29个自由度，配置如下：

| 部位 | 自由度 | 关节名称 |
|------|--------|----------|
| 左腿 | 6 | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll |
| 右腿 | 6 | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll |
| 躯干 | 3 | waist_yaw, waist_roll, waist_pitch |
| 左臂 | 7 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw |
| 右臂 | 7 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw |

Sources: [g1_sim2sim_29dof.xml](assets/g1/g1_sim2sim_29dof.xml#L1-L100)

### PD控制器参数

仿真环境使用PD（比例-微分）控制器跟踪目标关节角度：

```python
# 刚度系数 [N*m/rad]
stiffness = [
    100, 100, 100, 150, 40, 40,  # 左腿
    100, 100, 100, 150, 40, 40,  # 右腿
    150, 150, 150,               # 躯干
    40, 40, 40, 40, 4.0, 4.0, 4.0,  # 左臂
    40, 40, 40, 40, 4.0, 4.0, 4.0,  # 右臂
]

# 阻尼系数 [N*m*s/rad]
damping = [
    2, 2, 2, 4, 2, 2,  # 左腿
    2, 2, 2, 4, 2, 2,  # 右腿
    4, 4, 4,           # 躯干
    5, 5, 5, 5, 0.2, 0.2, 0.2,  # 左臂
    5, 5, 5, 5, 0.2, 0.2, 0.2,  # 右臂
]
```

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L125-L150)

---

## 关键运行参数

`server_low_level_g1_sim.py`提供了丰富的命令行参数用于配置仿真行为：

### 常用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--xml` | str | 必填 | MuJoCo模型XML文件路径 |
| `--policy` | str | 必填 | ONNX策略模型路径 |
| `--device` | str | `cuda` | 推理设备（cuda/cpu） |
| `--policy_frequency` | int | `100` | 策略推理频率（Hz） |
| `--viewer_decimation` | int | `0` | 渲染降采样（0=自动） |
| `--smooth_body` | float | `0.0` | 动作平滑系数（0.05-0.2推荐） |

### 性能测量参数

| 参数 | 说明 |
|------|------|
| `--measure_fps 1` | 启用FPS测量（每1000步输出统计） |
| `--limit_fps 1` | 启用FPS限制 |
| `--record_video` | 录制仿真视频 |
| `--record_proprio` | 记录本体感知数据 |

### 推荐配置组合

**实时可视化（开发调试）**：
```bash
python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --device cuda \
    --policy_frequency 100 \
    --viewer_decimation 10
```

**性能基准测试**：
```bash
python server_low_level_g1_sim.py \
    --xml ../assets/g1/g1_sim2sim_29dof.xml \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --device cuda \
    --measure_fps 1 \
    --limit_fps 1 \
    --policy_frequency 100
```

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L430-L480)

---

## 批量评估工具

除了实时仿真，项目还提供了`mujoco_exec_eval.py`批量评估工具，用于对多个运动文件进行系统性的性能测试。

### 基本用法

```bash
python tools/mujoco_exec_eval.py \
    --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --out_csv ./outputs/twist2_exec_metrics.csv \
    --policy_path assets/ckpts/twist2_1017_20k.onnx \
    --xml_path assets/g1/g1_sim2sim_29dof.xml \
    --disable_termination \
    --body_set joint_bodies29 \
    --workers 16
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--motion_yaml` | 运动配置文件（YAML格式） |
| `--out_csv` | 输出结果CSV文件路径 |
| `--policy_path` | 策略模型路径（支持ONNX和.pt） |
| `--xml_path` | MuJoCo模型XML文件 |
| `--workers` | 并行评估进程数 |
| `--body_set` | 身体关键点集合 |

### 评估指标

批量评估工具会计算以下性能指标：

- **关节角度跟踪误差**：策略输出与目标关节角度的均方根误差
- **根节点跟踪误差**：骨盆位置和姿态的跟踪精度
- **脚步滑移距离**：脚部与地面的相对位移
- **成功完成率**：运动片段完成的比例

Sources: [mujoco_exec_eval.py](tools/mujoco_exec_eval.py#L1-L100)

---

## 数据流与通信协议

Sim2Sim系统通过Redis实现进程间通信，定义了标准的数据键名和数据格式。

### Redis键名定义

| 键名 | 方向 | 数据类型 | 说明 |
|------|------|----------|------|
| `action_body_unitree_g1_with_hands` | 服务器→控制器 | JSON数组[35] | Mimic观测（动作参考） |
| `action_hand_left_unitree_g1_with_hands` | 服务器→控制器 | JSON数组[7] | 左手动作 |
| `action_hand_right_unitree_g1_with_hands` | 服务器→控制器 | JSON数组[7] | 右手动作 |
| `action_neck_unitree_g1_with_hands` | 服务器→控制器 | JSON数组[2] | 颈部动作 |
| `state_body_unitree_g1_with_hands` | 控制器→服务器 | JSON数组[34] | 机器人状态反馈 |

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L300-L340)

### 数据更新频率

系统的数据更新遵循固定的时序关系：

```
仿真步长:  0.001s (1ms)
策略频率:  100Hz (每10ms推理一次)
降采样比:  sim_decimation = 1 / (policy_frequency * sim_dt) = 10
```

这意味着MuJoCo每10步执行一次策略推理，与100Hz的控制频率相匹配。

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L95-L105)

---

## 常见问题与解决方案

### 1. Redis连接失败

**症状**：启动时报错 `Error connecting to Redis`

**解决方案**：
```bash
# 检查Redis是否运行
redis-cli ping

# 如果未运行，启动Redis
redis-server

# 或使用Docker
docker run -d -p 6379:6379 redis:latest
```

### 2. 策略推理FPS过低

**症状**：策略执行FPS远低于100Hz

**可能原因**：
- GPU内存不足
- ONNX模型过大
- CPU推理（未使用CUDA）

**解决方案**：
```bash
# 确认使用CUDA设备
python server_low_level_g1_sim.py \
    --device cuda \
    --measure_fps 1

# 限制仿真FPS以便观察
--limit_fps 1 --policy_frequency 50
```

### 3. 仿真不稳定/机器人摔倒

**症状**：仿真过程中机器人姿态失控

**可能原因**：
- 策略模型与仿真环境不匹配
- PD控制器参数不当
- 运动参考过于剧烈

**解决方案**：
```bash
# 启用动作平滑
python server_low_level_g1_sim.py \
    --smooth_body 0.1

# 更换为更稳定的运动文件
# 例如：从 walk_001.pkl 开始而非复杂动作
```

### 4. 观察空间维度不匹配

**症状**：运行时断言失败 `Expected {total_obs_size} obs, got {obs_buf.shape[0]}`

**可能原因**：
- 策略模型训练配置与仿真配置不一致
- 历史缓冲长度不匹配

**解决方案**：
- 确保训练时使用的`history_len`与运行时一致
- 检查策略模型的输入维度定义

Sources: [server_low_level_g1_sim.py](deploy_real/server_low_level_g1_sim.py#L320-L330)

---

## 下一步学习路径

完成Sim2Sim验证后，建议继续学习以下内容：

| 顺序 | 主题 | 说明 |
|------|------|------|
| 1 | [Sim2Real实物部署](15-sim2realshi-wu-bu-shu) | 将验证通过的控制策略部署到真实G1机器人 |
| 2 | [低层控制器](17-di-ceng-kong-zhi-qi) | 深入理解PD控制器和关节控制实现 |
| 3 | [ONNX模型导出](23-onnxmo-xing-dao-chu) | 学习如何导出训练好的策略为ONNX格式 |
| 4 | [评估与可视化](24-ping-gu-yu-ke-shi-hua) | 掌握更全面的策略评估方法 |

---

## 总结

Sim2Sim仿真验证是TWIST2开发流程中的关键环节，它允许开发者在安全的仿真环境中验证策略性能。通过本页面介绍的多进程架构、Redis通信机制和观察空间设计，开发者应该能够顺利启动Sim2Sim验证流程并进行策略评估。

如果在实验过程中遇到其他问题，建议查阅项目文档或参考[评估工具指南](../tools/TEACHER_EVAL_GUIDE.md)获取更多帮助。