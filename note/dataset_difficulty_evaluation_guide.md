# 动作难度评估与数据集筛选指南

本文档介绍如何使用训练好的教师模型评估动作数据集的难度，并据此筛选清理数据集。

---

## 背景

在动作数据收集过程中，可能存在一些机器人难以完成的动作。这些"困难"动作加入训练会：
- 扰乱策略学习
- 降低整体训练效果
- 增加训练不稳定性

**解决方案**：用训练好的教师模型遍历所有动作，计算难度分数，然后筛选掉难度过高的数据。

---

## 工作流程

```
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────┐
│  训练好的教师模型  │ ───> │  遍历评估所有动作      │ ───> │  生成难度评分CSV  │
│ (g1_priv_mimic)  │      │  evaluate_motion...  │      │  difficulty_*   │
└─────────────────┘      └──────────────────────┘      └────────┬────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────┐
│  筛选后的数据集   │ <─── │  根据难度筛选数据      │ <─── │  分析评估结果    │
│  filtered_*.yaml │      │  filter_dataset...   │      │  决定筛选阈值    │
└─────────────────┘      └──────────────────────┘      └─────────────────┘
```

---

## 脚本一：动作难度评估

### 脚本位置

```
legged_gym/legged_gym/scripts/evaluate_motion_difficulty.py
```

### 功能

- 使用训练好的教师模型（PyTorch checkpoint 或 ONNX）遍历所有动作
- 对每个动作计算多个评估指标
- 生成包含难度分数的CSV文件

### 评估指标

| 指标 | 说明 | 范围 |
|------|------|------|
| `difficulty_score` | **难度分数**（越高越难，无硬上限） | 0-150+ |
| `completion_rate` | 完成率（动作完成百分比） | 0-1 |
| `avg_joint_error` | 平均关节位置误差（弧度） | ≥0 |
| `avg_pose_error` | 平均关键点姿态误差（米） | ≥0 |
| `max_roll` | 最大Roll角度（弧度） | ≥0 |
| `max_pitch` | 最大Pitch角度（弧度） | ≥0 |
| `termination_reason` | 终止原因分类 | 见下表 |
| `base_score` | 基础分数（来自完成率） | 0-100 |
| `early_failure_penalty` | 早终惩罚 | 0-20 |
| `joint_penalty` | 关节误差惩罚 | 0-15+ |
| `pose_penalty` | 姿态误差惩罚 | 0-10+ |
| `stability_penalty` | 稳定性惩罚 | 0-20+ |
| `termination_penalty` | 终止原因惩罚 | 0-5 |

### 终止原因类型

| 原因代码 | 说明 | 建议 |
|----------|------|------|
| `completed` | 动作成功完成 | 保留 |
| `contact` | 非法接触力（如手脚触地） | 可能排除 |
| `height_diff` | 高度偏差过大 | 可能排除 |
| `roll_pitch` | Roll/Pitch超限（跌倒） | 建议排除 |
| `pose_tracking` | 姿态跟踪失败 | 可能排除 |
| `root_tracking` | Root位置跟踪失败 | 可能排除 |
| `timeout` | 超时 | 可能排除 |
| `unknown` | 未知原因 | 需人工检查 |

### 难度分数计算公式

**设计目标**：让分数有更好的区分度，避免很多困难动作堆积在上限值。

```
# 1. 基础分数：完成率倒数 (0完成=100分, 100%完成=0分)
base_score = (1 - completion_rate) × 100

# 2. 失败速度惩罚：越早失败，惩罚越大（二次函数）
early_failure_penalty = 0
if completion_rate < 0.5:
    early_failure_penalty = ((0.5 - completion_rate) / 0.5)² × 20

# 3. 误差惩罚：使用平方根变换，让大误差之间有区分度
joint_penalty = √(avg_joint_error) × 15
pose_penalty = √(avg_pose_error) × 10

# 4. 稳定性惩罚：Roll/Pitch超过10度后指数增长
stability_penalty = 0
if roll > 10°:
    stability_penalty += (roll - 10)^1.5 × 0.5
if pitch > 10°:
    stability_penalty += (pitch - 10)^1.5 × 0.5

# 5. 终止原因惩罚
termination_penalty:
  - roll_pitch（跌倒）: 5.0
  - contact（非法接触）: 3.0
  - height_diff（高度偏差）: 2.0
  - 其他: 0

# 综合难度分数（无硬上限）
difficulty_score = base_score + early_failure_penalty +
                   joint_penalty + pose_penalty +
                   stability_penalty + termination_penalty
```

**分数范围说明**：

| 分数范围 | 难度等级 | 说明 |
|----------|----------|------|
| 0-20 | 简单 | 完成率高，误差小 |
| 20-50 | 中等 | 有一定难度，但基本可完成 |
| 50-80 | 困难 | 完成率低或误差大 |
| 80-100 | 极难 | 大部分无法完成 |
| 100+ | 极端 | 几乎完全失败，且失败原因严重 |

**关键改进**：
- **无硬上限**：极端困难的动作可以超过100分
- **非线性区分**：使用平方根和指数函数，让大数值之间仍有区分度
- **失败速度因子**：越早失败，额外惩罚越大

### 使用方法

#### 方法1：使用 PyTorch Checkpoint

```bash
cd legged_gym/legged_gym/scripts

python evaluate_motion_difficulty.py \
    --task g1_priv_mimic \
    --checkpoint /home/huanghao/source/code/TWIST2/legged_gym/logs/g1_priv_mimic/dataset_mix_8203b425_total328739/model_35000.pt \
    --motion_config /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/test_pico.yaml \
    --output /home/huanghao/source/code/TWIST2/difficulty/difficulty_scores_for_pico.csv \
    --device cuda:0 \
    --num_envs 4096
```

#### 方法2：使用 ONNX 模型

```bash
python evaluate_motion_difficulty.py \
    --task g1_priv_mimic \
    --policy ../../assets/ckpts/teacher.onnx \
    --motion_config ../../motion_data_configs/your_dataset.yaml \
    --output ../../difficulty_scores.csv \
    --device cuda:0
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | `g1_priv_mimic` | 任务名称（教师环境） |
| `--checkpoint` | str | `None` | PyTorch checkpoint 路径 |
| `--policy` | str | `None` | ONNX 模型路径 |
| `--motion_config` | str | **必需** | 动作数据集 YAML 配置文件 |
| `--output` | str | `difficulty_scores.csv` | 输出 CSV 文件路径 |
| `--device` | str | `cuda:0` | 运行设备 |
| `--num_envs` | int | `256` | 并行环境数量（内存不足可调小） |
| `--max_steps` | int | `5000` | 单个动作最大仿真步数 |
| `--save_video` | flag | `False` | 是否保存可视化视频 |

**注意**：`--checkpoint` 和 `--policy` 必须指定其中一个。

### 输出示例

```csv
motion_idx,motion_name,motion_file,completion_rate,episode_length,...,difficulty_score,base_score,early_failure_penalty,joint_penalty,pose_penalty,stability_penalty,termination_penalty
0,walk.pkl,data/walk.pkl,0.95,475,500,...,5.2,5.0,0.0,0.2,0.0,0.0,0.0
1,kick.pkl,data/kick.pkl,0.60,300,500,...,42.5,40.0,0.0,1.5,0.5,0.5,0.0
2,dance.pkl,data/dance.pkl,0.20,100,500,...,95.8,80.0,18.0,3.2,1.5,2.1,0.0
3,flip.pkl,data/flip.pkl,0.05,25,500,...,128.3,95.0,18.0,5.5,2.0,4.3,3.5
```

### 控制台输出示例

```
═══════════════════════════════════════════════════════════
                        评估摘要
═══════════════════════════════════════════════════════════

┏━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ 难度等级 ┃ 数量  ┃ 占比  ┃ 平均完成率  ┃ 平均难度分 ┃
┡━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 简单    │ 1250  │ 62.5% │ 0.95       │ 8.2        │
│ 中等    │ 500   │ 25.0% │ 0.75       │ 35.5       │
│ 困难    │ 200   │ 10.0% │ 0.40       │ 68.3       │
│ 极难    │ 40    │ 2.0%  │ 0.15       │ 92.1       │
│ 极端    │ 10    │ 0.5%  │ 0.03       │ 125.7      │
└────────┴───────┴───────┴────────────┴────────────┘

整体统计:
  总动作数: 2000
  平均完成率: 0.82
  平均难度分: 22.5
  难度分范围: [2.1, 138.5]

最难的5个动作:
  1. backflip_motion.pkl          | 难度: 138.5 | 完成率: 0.02 | 原因: roll_pitch
  2. high_kick_motion.pkl         | 难度: 125.3 | 完成率: 0.05 | 原因: contact
  3. cartwheel_motion.pkl         | 难度: 112.8 | 完成率: 0.08 | 原因: roll_pitch
  4. split_jump_motion.pkl        | 难度: 105.2 | 完成率: 0.10 | 原因: pose_tracking
  5. 720_spin_motion.pkl          | 难度: 98.7  | 完成率: 0.12 | 原因: timeout
```

---

## 脚本二：数据集筛选

### 脚本位置

```
legged_gym/legged_gym/scripts/filter_dataset_by_difficulty.py
```

### 功能

- 根据难度评估结果筛选动作
- 生成清理后的 YAML 配置文件
- 支持多种筛选条件组合

### 使用方法

#### 1. 按难度分数筛选

```bash
# 保留难度分数 < 1.5 的动作
python filter_dataset_by_difficulty.py \
    --difficulty_csv ../../difficulty_scores.csv \
    --original_config ../../motion_data_configs/original_dataset.yaml \
    --output_config ../../motion_data_configs/filtered_dataset.yaml \
    --max_difficulty 1.5
```

#### 2. 按完成率筛选

```bash
# 只保留完成率 > 0.8 的动作
python filter_dataset_by_difficulty.py \
    --difficulty_csv ../../difficulty_scores.csv \
    --original_config ../../motion_data_configs/original_dataset.yaml \
    --output_config ../../motion_data_configs/filtered_dataset.yaml \
    --min_completion 0.8
```

#### 3. 排除特定终止原因的动作

```bash
# 排除因跌倒或非法接触终止的动作
python filter_dataset_by_difficulty.py \
    --difficulty_csv ../../difficulty_scores.csv \
    --original_config ../../motion_data_configs/original_dataset.yaml \
    --output_config ../../motion_data_configs/filtered_dataset.yaml \
    --exclude_reasons roll_pitch contact
```

#### 4. 只保留成功完成的动作

```bash
python filter_dataset_by_difficulty.py \
    --difficulty_csv ../../difficulty_scores.csv \
    --original_config ../../motion_data_configs/original_dataset.yaml \
    --output_config ../../motion_data_configs/filtered_dataset.yaml \
    --only_reasons completed
```

#### 5. 组合多个条件

```bash
# 保留完成率 > 0.5 且难度 < 2.0 的动作
python filter_dataset_by_difficulty.py \
    --difficulty_csv ../../difficulty_scores.csv \
    --original_config ../../motion_data_configs/original_dataset.yaml \
    --output_config ../../motion_data_configs/filtered_dataset.yaml \
    --min_completion 0.5 \
    --max_difficulty 2.0 \
    --exclude_reasons roll_pitch
```

#### 6. 选择最简单的 K 个动作

```bash
# 只保留难度最低的 1000 个动作
python filter_dataset_by_difficulty.py \
    --difficulty_csv ../../difficulty_scores.csv \
    --original_config ../../motion_data_configs/original_dataset.yaml \
    --output_config ../../motion_data_configs/filtered_dataset.yaml \
    --top_k 1000
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--difficulty_csv` | str | **必需** | 难度评估结果 CSV 文件 |
| `--original_config` | str | **必需** | 原始数据集 YAML 配置文件 |
| `--output_config` | str | **必需** | 输出的筛选后配置文件 |
| `--max_difficulty` | float | `None` | 保留难度小于此值的动作 |
| `--min_completion` | float | `None` | 保留完成率大于此值的动作 |
| `--exclude_reasons` | list | `[]` | 排除指定终止原因的动作 |
| `--only_reasons` | list | `[]` | 只保留指定终止原因的动作 |
| `--skip_unevaluated` | flag | `False` | 跳过没有评估结果的动作 |
| `--top_k` | int | `None` | 只保留难度最低的K个动作 |
| `--bottom_k` | int | `None` | 只保留难度最高的K个动作（用于分析） |

### 终止原因选项

- `completed` - 动作完成
- `contact` - 非法接触力
- `height_diff` - 高度偏差过大
- `roll_pitch` - Roll/Pitch超限（跌倒）
- `pose_tracking` - 姿态跟踪失败
- `root_tracking` - Root跟踪失败
- `timeout` - 超时
- `unknown` - 未知原因

### 输出示例

```
筛选摘要:
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ 类别     ┃ 数量  ┃ 占比  ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 原始动作数 │ 2000  │ 100% │
│ 保留动作数 │ 1750  │ 87.5%│
│ 排除动作数 │ 250   │ 12.5%│
└─────────┴───────┴───────┘

排除原因统计:
  high_difficulty: 150
  low_completion: 80
  excluded_roll_pitch: 20

筛选条件:
  最大难度分数: 1.5
  最小完成率: 0.5
  排除终止原因: roll_pitch

Filtered config saved to: ../../motion_data_configs/filtered_dataset.yaml
```

---

## 完整工作流程示例

```bash
# 1. 进入脚本目录
cd legged_gym/legged_gym/scripts

# 2. 评估所有动作的难度（假设教师模型已训练）
python evaluate_motion_difficulty.py \
    --task g1_priv_mimic \
    --checkpoint ../../logs/g1_priv_mimic/teacher_exp/model_10000.pt \
    --motion_config ../../motion_data_configs/full_dataset.yaml \
    --output ../../difficulty_scores.csv \
    --num_envs 512

# 3. 查看评估结果
cat ../../difficulty_scores.csv | head -20

# 4. 根据结果筛选数据集（保留难度 < 50 且完成率 > 0.5 的动作）
python filter_dataset_by_difficulty.py \
    --difficulty_csv ../../difficulty_scores.csv \
    --original_config ../../motion_data_configs/full_dataset.yaml \
    --output_config ../../motion_data_configs/clean_dataset.yaml \
    --max_difficulty 50 \
    --min_completion 0.5 \
    --exclude_reasons roll_pitch contact

# 5. 使用筛选后的数据集重新训练
cd ../../..
python legged_gym/scripts/train.py \
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid clean_dataset_train \
    --motion.motion_file legged_gym/motion_data_configs/clean_dataset.yaml
```

---

## 筛选策略建议

根据训练目标不同，建议采用不同的筛选策略（**注意**：分数范围已更新为0-150+）：

### 策略1：保守筛选（适合初期训练）

```bash
--min_completion 0.8 --exclude_reasons roll_pitch contact
```

只保留完成率高、没有跌倒或非法接触的动作。

### 策略2：中等筛选（适合进阶训练）

```bash
--min_completion 0.5 --max_difficulty 50
```

保留中等难度以下的动作，提供一定挑战性。

### 策略3：只保留简单动作

```bash
--max_difficulty 20 --only_reasons completed
```

只保留简单且成功完成的动作，用于基础能力训练。

### 策略4：课程学习（渐进式筛选）

```bash
# 第一阶段：只保留简单动作（难度<20）
--max_difficulty 20

# 第二阶段：加入中等难度（难度<50）
--max_difficulty 50

# 第三阶段：加入困难动作（难度<80）
--max_difficulty 80

# 第四阶段：使用全部数据（包括极难动作）
# 不使用--max_difficulty参数
```

---

## 常见问题

### Q: 如何选择筛选阈值？

A: 建议先运行评估，查看难度分数的分布：

```python
import pandas as pd
df = pd.read_csv('difficulty_scores.csv')
print(df['difficulty_score'].describe())
```

根据百分位数选择阈值，例如（分数范围0-150+）：
- `difficulty_score < 20`：简单动作（约下25%）
- `difficulty_score < 50`：中等动作（约下50%）
- `difficulty_score < 80`：困难动作（约下80%）
- `difficulty_score >= 100`：极端困难动作（可能需要排除）

### Q: GPU 内存不足怎么办？

A: 减少 `--num_envs` 参数：

```bash
--num_envs 128  # 默认256
```

### Q: 如何验证筛选效果？

A: 对比筛选前后的训练曲线：

```bash
# 原始数据集训练
python train.py --task g1_stu_future --motion.motion_file original.yaml --exptid original

# 筛选后数据集训练
python train.py --task g1_stu_future --motion.motion_file filtered.yaml --exptid filtered

# 对比日志
tensorboard --logdir logs/g1_stu_future/
```

### Q: 评估时间太长怎么办？

A: 可以先抽样评估一部分动作：

1. 创建一个小规模测试配置
2. 用小配置快速评估
3. 根据结果决定是否进行全量评估

---

## 总结

1. **评估阶段**：用教师模型遍历所有动作，生成难度评分 CSV
2. **分析阶段**：查看统计摘要，决定筛选阈值
3. **筛选阶段**：使用筛选脚本生成清理后的配置文件
4. **验证阶段**：用筛选后的数据集训练，对比效果

通过这种方式，可以逐步优化数据集质量，提升训练效果。
