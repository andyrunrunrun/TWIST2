# TWIST2 通用模型测评脚本

## 简介

`evaluate_model.py` 是一个兼容多种模型架构（MLP/MoE/Transformer）和格式（PT/ONNX）的统一测评脚本。

## 测评指标

### 核心指标：动作完成度评分

**评分公式**: `得分 = (实际执行时间 / 动作总时长) × 100`

- **满分100分**: 动作完整执行完毕
- **失败情况**: 训练时触发的失败条件会提前终止
  - 非法接触力（如膝盖触地）
  - 高度偏差过大
  - Roll/Pitch角度超限（身体姿态失控）
  - 速度过大
  - 姿态跟踪失败

**示例**:
- 10秒动作执行10秒 → 100分
- 10秒动作5秒时摔倒 → 50分
- 10秒动作2秒时失败 → 20分

### 附加指标

- `mjpe` / `mpjpe`: 关节位置误差
- `tracking_error`: 跟踪误差
- `keypoint_error`: 关键点误差

## 使用方法

### 基本用法

```bash
# 使用Python脚本直接运行
python evaluate_model.py \
    --model_path /path/to/your/model.pt \
    --motion_config /path/to/motion_config.yaml \
    --task g1_stu_future

# 或使用ONNX模型
python evaluate_model.py \
    --model_path /path/to/your/model.onnx \
    --motion_config /path/to/motion_config.yaml \
    --task g1_stu_future
```

### 使用Shell包装脚本

```bash
# 基本用法
bash eval_model.sh /path/to/model.pt

# 指定GPU
bash eval_model.sh /path/to/model.pt 1
```

### 完整参数

```bash
python evaluate_model.py \
    --model_path /path/to/model.pt \
    --motion_config /path/to/motion_config.yaml \
    --task g1_stu_future \
    --device cuda:0 \
    --num_envs 256 \
    --max_steps 5000 \
    --output_dir ./eval_results \
    --headless
```

参数说明:
- `--model_path`: 模型文件路径（支持 .pt 和 .onnx）
- `--motion_config`: 动作配置文件路径
- `--task`: 任务名称（默认: g1_stu_future）
- `--device`: 计算设备（默认: cuda:0）
- `--num_envs`: 并行环境数量（默认: 256）
- `--max_steps`: 最大模拟步数（默认: 5000）
- `--output_dir`: 结果输出目录（默认: ./eval_results）
- `--headless`: 无头模式运行

## 输出结果

### 输出文件命名

```
{model_name}_{timestamp}_eval.json

示例:
- model_15000_20250408_143022_eval.json
- g1_moe_best_20250408_143022_eval.json
```

### 输出内容结构

```json
{
  "model_info": {
    "path": "...",
    "name": "model_15000",
    "type": "pt",
    "steps": 15000
  },
  "overall": {
    "completion_score": {
      "mean": 85.3,
      "std": 12.1,
      "min": 20.0,
      "max": 100.0,
      "count": 49706
    },
    "mjpe": {
      "mean": 0.052,
      "std": 0.015
    },
    "total_motions": 49706
  },
  "motion_groups": {
    "AMASS_numpy123": {
      "count": 12345,
      "completion_score": {
        "mean": 87.2,
        "std": 10.5,
        "min": 50.0,
        "max": 100.0
      }
    },
    "twist1_to_twist2_numpy123": {
      "count": 9876,
      "completion_score": {...}
    }
  },
  "motion_results": [
    {
      "motion_idx": 0,
      "motion_name": "...",
      "motion_file": "AMASS_numpy123/...",
      "completion_rate": 0.95,
      "completion_score": 95.0,
      "actual_time": 9.5,
      "motion_length": 10.0,
      "metrics": {...}
    }
  ]
}
```

## 动作库分组

脚本会自动按动作文件的第一级目录分组：

1. **AMASS_numpy123** - AMASS数据集
2. **EgoBody_numpy123** - EgoBody数据集
3. **OMOMO_numpy123** - OMOMO数据集
4. **interhuman_numpy123** - InterHuman数据集
5. **lafan1_numpy123** - LaFAN1数据集
6. **pico_numpy123** - PICO数据集
7. **twist1_to_twist2_numpy123** - TWIST迁移数据
8. **v1_v2_v3_g1_numpy123** - 版本化G1数据

## 支持的模型架构

- **标准MLP**: `ActorCriticMimic`, `ActorCriticFuture`
- **MoE**: `ActorCriticFuture` (use_moe=True)
- **Transformer**: `ActorCriticFuture` (use_transformer=True)

## 示例输出

```
============================================================
模型测评结果汇总
============================================================

整体性能:
  完成度得分: 85.32 ± 12.15
  范围: [20.00, 100.00]
  测试动作数: 49706

按动作库分组统计:
------------------------------------------------------------
动作库                             数量   完成度均值       标准差
------------------------------------------------------------
AMASS_numpy123                    12345        87.25      10.52
EgoBody_numpy123                   2341        82.15      14.23
OMOMO_numpy123                     1876        88.42       9.87
interhuman_numpy123                2156        84.33      11.45
lafan1_numpy123                    1567        86.78      10.12
pico_numpy123                      1234        83.45      13.21
twist1_to_twist2_numpy123          9876        81.23      15.67
v1_v2_v3_g1_numpy123              18311        87.92       9.34
------------------------------------------------------------

结果已保存到: ./eval_results/model_15000_20250408_143022_eval.json
```

## 注意事项

1. **显存不足**: 如果显存不足，可以减少 `--num_envs`（如改为128或64）
2. **模型加载**: PT模型需要对应的任务配置（--task参数），ONNX模型可直接加载
3. **环境依赖**: 需要安装 `isaacgym`, `torch`, `onnxruntime-gpu`, `rich`
