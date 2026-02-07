# Teacher Policy Evaluation Guide

本文档介绍如何使用 `gym_exec_eval_teacher.py` 和 `mujoco_exec_eval_teacher.py` 评估 Teacher (privileged) 策略。

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [输出目录结构](#输出目录结构)
- [gym_exec_eval_teacher.py - IsaacGym评估](#gym_exec_eval_teacherpy---isaacgym评估)
- [mujoco_exec_eval_teacher.py - MuJoCo评估](#mujoco_exec_eval_teacherpy---mujoco评估)
- [参数说明](#参数说明)
- [输出文件格式](#输出文件格式)
- [常见问题](#常见问题)

---

## 概述

TWIST2 项目包含两种策略：

| 策略类型 | 任务名称 | 观测维度 | 说明 |
|---------|---------|---------|------|
| **Teacher** | `g1_priv_mimic` | 1734 | 特权观测，包含21步motion预测 + priv_info |
| **Student** | `g1_stu_future` | 1107 | 限制观测，仅当前帧 + 历史编码 |

本指南专门针对 **Teacher 策略** 的评估。

## 参数说明

| 参数 | 说明 |
|-----|------|
| `--exptid` | **测试ID**，用于组织输出目录。同一模型可以用不同exptid测试多次。 |
| `--out_csv` | 输出CSV文件名，自动保存到 `outputs/{exptid}/` 目录。 |
| `--policy_path` | **模型文件路径**，直接指定 `.pt` / `.pth` / `.onnx` 文件路径 |

---

## 环境准备

### Conda 环境
```bash
conda activate twist2  # Python 3.8，用于 IsaacGym
```

### 必需依赖
- IsaacGym (for gym_exec_eval_teacher.py)
- MuJoCo (for mujoco_exec_eval_teacher.py)
- PyTorch
- ONNXRuntime (可选)

---

## 输出目录结构

使用 `--exptid` 参数组织输出，所有结果保存在 `outputs/{exptid}/` 目录：

```
outputs/
├── test_001/                    # 测试ID: test_001
│   ├── teacher_eval.csv         # 评估结果
│   └── teacher_eval.csv.summary.json  # 统计摘要
├── test_002/                    # 测试ID: test_002 (同一模型，不同配置)
│   ├── teacher_eval.csv
│   └── teacher_eval.csv.summary.json
└── test_wbc_dataset/            # 测试ID: test_wbc_dataset (同一模型，不同数据集)
    ├── teacher_eval.csv
    └── teacher_eval.csv.summary.json
```

---

## gym_exec_eval_teacher.py - IsaacGym评估

基于 IsaacGym 的 Teacher 策略评估器，支持大规模并行评估。

### 基本用法

```bash
python tools/gym_exec_eval_teacher.py \
    --exptid <测试ID> \
    --policy_path <模型文件路径> \
    --motion_yaml <运动配置.yaml> \
    --device cuda:0 \
    --num_envs 4096
```

### 运行模式

#### 1. 单GPU单进程 (适合调试)
```bash
python tools/gym_exec_eval_teacher.py \
    --exptid test_debug \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml \
    --device cuda:0 \
    --headless \
    --num_envs 1 \
    --episode_length_s 300
```

#### 2. 单GPU Queue模式 (推荐，最快)
```bash
python tools/gym_exec_eval_teacher.py \
    --queue_eval \
    --exptid test_wbc_230k \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml \
    --device cuda:0 \
    --headless \
    --num_envs 4096 \
    --episode_length_s 300 \
    --queue_metrics fast
```

#### 3. 多进程并行 (多GPU或CPU)
```bash
python tools/gym_exec_eval_teacher.py \
    --exptid test_multi_worker \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml \
    --device cuda:0 \
    --headless \
    --num_envs 1 \
    --workers 4
```

#### 4. 使用ONNX策略 (最快推理)
```bash
python tools/gym_exec_eval_teacher.py \
    --exptid test_onnx \
    --policy_path assets/ckpts/teacher_0106.onnx \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml \
    --device cuda:0 \
    --num_envs 4096
```

#### 5. 断点续传
```bash
# 使用 --append 继续之前的评估
python tools/gym_exec_eval_teacher.py \
    --queue_eval \
    --exptid test_resume \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml \
    --append
```

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--exptid` | str | **必需** | 测试ID，创建 `outputs/{exptid}/` 目录 |
| `--policy_path` | str | **必需** | 模型文件路径 (.pt/.pth/.onnx) |
| `--motion_yaml` | str | **必需** | 运动数据YAML配置 |
| `--out_csv` | str | `teacher_eval.csv` | 输出CSV文件名 |
| `--device` | str | `cuda:0` | 设备 (cuda:0, cpu) |
| `--headless` | flag | - | 无头模式 |
| `--num_envs` | int | `1` | 并行环境数 |
| `--episode_length_s` | float | `120.0` | 单回合最大时长(秒) |
| `--queue_eval` | flag | - | 启用高吞吐量queue模式 |
| `--queue_metrics` | str | `fast` | Queue模式指标 (fast/final/mean) |
| `--runner_backend` | str | `onnx` | 策略后端，仅.pt模型有效 (onnx/torch) |
| `--workers` | int | `1` | Worker进程数 |
| `--append` | flag | - | 追加模式(断点续传) |
| `--motion_ids` | str | `""` | 运动子集索引 (如 "0,3,10-20") |
| `--max_motions` | int | `0` | 最大运动数量 |
| `--shard_idx` | int | `0` | 分片索引 |
| `--num_shards` | int | `1` | 分片总数 |

---

## mujoco_exec_eval_teacher.py - MuJoCo评估

基于 MuJoCo 的 Teacher 策略评估器，支持多进程并行。

### 基本用法

```bash
python tools/mujoco_exec_eval_teacher.py \
    --exptid <测试ID> \
    --motion_yaml <运动配置.yaml> \
    --policy_path <模型路径.pt/.onnx> \
    --xml_path <MuJoCo XML> \
    --workers <并行数>
```

### 运行示例

#### 1. CPU多进程评估
```bash
python tools/mujoco_exec_eval_teacher.py \
    --exptid test_cpu_eval \
    --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --xml_path assets/g1/g1_sim2sim_29dof.xml \
    --device cpu \
    --workers 128 \
    --disable_termination
```

#### 2. GPU加速 (需要 .pt 模型)
```bash
python tools/mujoco_exec_eval_teacher.py \
    --exptid test_gpu_eval \
    --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --xml_path assets/g1/g1_sim2sim_29dof.xml \
    --device cuda:0 \
    --workers 32
```

#### 3. 使用ONNX模型
```bash
python tools/mujoco_exec_eval_teacher.py \
    --exptid test_onnx_eval \
    --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --policy_path assets/ckpts/teacher_0106.onnx \
    --xml_path assets/g1/g1_sim2sim_29dof.xml \
    --workers 64
```

#### 4. 测试运动子集
```bash
# 只测试前100个运动
python tools/mujoco_exec_eval_teacher.py \
    --exptid test_subset \
    --motion_yaml legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --xml_path assets/g1/g1_sim2sim_29dof.xml \
    --max_motions 100 \
    --workers 64
```

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--exptid` | str | **必需** | 测试ID，创建 `outputs/{exptid}/` 目录 |
| `--motion_yaml` | str | **必需** | 运动数据YAML配置 |
| `--policy_path` | str | **必需** | 模型路径 (.pt/.pth/.onnx) |
| `--xml_path` | str | **必需** | MuJoCo XML文件路径 |
| `--out_csv` | str | `teacher_eval.csv` | 输出CSV文件名 |
| `--device` | str | `cpu` | 设备 (cpu, cuda:0) |
| `--workers` | int | `1` | 并行进程数 |
| `--disable_termination` | flag | - | 禁用终止条件 |
| `--loop` | flag | - | 循环运动 |
| `--motion_ids` | str | `""` | 运动子集索引 |
| `--max_motions` | int | `0` | 最大运动数量 |
| `--stiffness` | float | `100.0` | PD刚度 |
| `--damping` | float | `2.0` | PD阻尼 |
| `--torque_limits` | float | `50.0` | 力矩限制 |
| `--action_scale` | float | `0.5` | 动作缩放 |
| `--policy_freq` | float | `50.0` | 策略频率(Hz) |
| `--sim_freq` | float | `500.0` | 仿真频率(Hz) |
| `--idle_s` | float | `0.5` | 起始空闲时间(秒) |
| `--tail_s` | float | `0.5` | 结尾空闲时间(秒) |
| `--transition_s` | float | `0.4` | 过渡时间(秒) |
| `--future_step` | int | `1` | 未来预测步数 |

---

## 参数说明

### 通用参数

| 参数 | 说明 |
|-----|------|
| `--exptid` | **测试ID**，用于组织输出目录。同一模型可以用不同exptid测试多次。 |
| `--out_csv` | 输出CSV文件名，自动保存到 `outputs/{exptid}/` 目录。 |

### 模型路径参数

| 脚本 | 参数 | 说明 |
|-----|------|------|
| gym_exec_eval_teacher.py | `--resumeid` | 模型实验ID，自动解析checkpoint路径 |
| gym_exec_eval_teacher.py | `--proj_name` | 项目名称，默认 `g1_priv_mimic` |
| mujoco_exec_eval_teacher.py | `--policy_path` | 直接指定模型文件路径 |

### 数据集参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| `--motion_yaml` | 运动数据YAML配置 | `legged_gym/motion_data_configs/wbc_0117_230k.yaml` |
| `--motion_ids` | 子集索引 | `"0,3,10-20"` 测试第0,3,10-20号运动 |
| `--max_motions` | 最大运动数量 | `100` 只测试前100个 |

### 性能参数

| 参数 | gym版本 | mujoco版本 |
|-----|---------|-----------|
| 设备 | `--device cuda:0` | `--device cuda:0` |
| 并行 | `--num_envs 4096` | `--workers 128` |
| 加速 | `--runner_backend onnx` | (使用.onnx模型) |

---

## 输出文件格式

### CSV 文件 (`teacher_eval.csv`)

| 字段 | 说明 |
|-----|------|
| `motion_idx_original` | 运动在YAML中的原始索引 |
| `motion_relpath` | 运动文件相对路径 |
| `status` | 状态: `ok`(成功) / `fail`(失败) / `error`(错误) |
| `done_reason` | 终止原因 |
| `done_time_s` | 运行时长(秒) |
| `motion_len_s` | 运动总长度(秒) |
| `progress` | 完成进度 (0-1) |
| `wall_time_s` | 实际评估耗时(秒) |
| `steps_exec` | 执行步数 |
| `err_root_pos_l2_mean` | 根位置L2误差均值(米) |
| `err_root_rot_deg_mean` | 根旋转误差均值(度) |
| `err_dof_pos_l2_mean` | 关节位置L2误差均值 |
| `err_dof_vel_l2_mean` | 关节速度L2误差均值 |
| `err_keybody_pos_l1_mean` | 关键body位置L1误差 |

### JSON摘要 (`teacher_eval.csv.summary.json`)

```json
{
  "generated_at": "2025-02-06 12:00:00",
  "total_rows": 1000,
  "ok_rows": 950,
  "ok_rate": 0.95,
  "status_counts": {"ok": 950, "fail": 50},
  "done_reason_counts": {"motion_end": 950, "contact_force": 30, ...},
  "metrics_all": { ... },   // 所有动作的统计量
  "metrics_ok": { ... }      // 成功动作的统计量
}
```

---

## 常见问题

### Q1: Teacher和Student策略有什么区别？

| 属性 | Teacher (g1_priv_mimic) | Student (g1_stu_future) |
|-----|------------------------|------------------------|
| 观测维度 | 1734 | 1107 |
| motion时间步 | 21步多预测 | 1步当前帧 |
| 历史编码 | 无 | 有 (history_len=10) |
| priv_info | 包含 | 不包含 |

### Q2: 如何选择评估脚本？

- **gym_exec_eval_teacher.py**: 大规模快速评估，推荐用于批量测试
- **mujoco_exec_eval_teacher.py**: 独立MuJoCo仿真，适合不依赖IsaacGym的场景

### Q3: 如何断点续传？

```bash
# gym版本
python tools/gym_exec_eval_teacher.py ... --append

# mujoco版本
# 自动跳过已完成的运动，使用相同的 --exptid 即可
```

### Q4: 同一模型多次测试？

使用不同的 `--exptid`，相同 `--policy_path`：

```bash
# 测试1: wbc数据集
python tools/gym_exec_eval_teacher.py --exptid test_wbc \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml configs/wbc.yaml ...

# 测试2: gmr数据集
python tools/gym_exec_eval_teacher.py --exptid test_gmr \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml configs/gmr.yaml ...

# 输出分别在 outputs/test_wbc/ 和 outputs/test_gmr/
```

### Q5: 检查点路径格式？

Teacher模型检查点位于：
```
legged_gym/logs/g1_priv_mimic/<模型ID>/model_<迭代次数>.pt
```

现在直接使用 `--policy_path` 指定完整路径。

### Q6: 如何快速测试？

```bash
# 少量运动，单环境
python tools/gym_exec_eval_teacher.py \
    --exptid quick_test \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml configs/test.yaml \
    --max_motions 10 \
    --num_envs 1
```

---

## 快速参考

### 典型评估命令

```bash
# gym - 完整数据集评估 (queue模式)
python tools/gym_exec_eval_teacher.py --queue_eval \
    --exptid eval_full_dataset \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml \
    --device cuda:0 --num_envs 4096

# mujoco - 完整数据集评估 (CPU多进程)
python tools/mujoco_exec_eval_teacher.py \
    --exptid eval_full_dataset \
    --policy_path legged_gym/logs/g1_priv_mimic/0106_teacher/model_85000.pt \
    --motion_yaml legged_gym/motion_data_configs/wbc_0117_230k.yaml \
    --xml_path assets/g1/g1_sim2sim_29dof.xml \
    --device cpu --workers 128 --disable_termination
```

### 查看结果

```bash
# 查看CSV
cat outputs/<exptid>/teacher_eval.csv

# 查看摘要
cat outputs/<exptid>/teacher_eval.csv.summary.json

# Python分析
python -c "import pandas as pd; df=pd.read_csv('outputs/<exptid>/teacher_eval.csv'); print(df.describe())"
```
