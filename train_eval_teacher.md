## Train

```bash
CUDA_VISIBLE_DEVICES=4 python legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0106_teacher \
    --device cuda:0 \
    --num_envs 4096 --max_iterations 30000 \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml

# 多卡 DDP（torchrun）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid dataset_mix_8203b425_total328739 \
    --num_envs 4096 --max_iterations 300000 \
    --motion.motion_file /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/dataset_mix_8203b425_total328739.yaml

# 续训

CUDA_VISIBLE_DEVICES=5 python legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic --proj_name g1_priv_mimic --exptid 0106_teacher \
    --resumeid 0106_teacher --checkpoint 27500 \
    --device cuda:0 \
    --num_envs 4096 --max_iterations 80000 \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml


# 加上delta_local

CUDA_VISIBLE_DEVICES=4 python legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic --proj_name g1_priv_mimic --exptid 0106_teacher_deltalocal \
    --resumeid 0106_teacher --checkpoint 77500 \
    --rewards.scales.tracking_root_pose_delta_local 1.5 \
    --device cuda:0 \
    --num_envs 4096 --max_iterations 50000 \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml

CUDA_VISIBLE_DEVICES=2,3,4,6 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic --proj_name g1_priv_mimic --exptid 0106_teacher_deltalocal \
    --resumeid 0106_teacher --checkpoint 95000 \
    --rewards.scales.tracking_root_pose_delta_local 1.5 \
    --num_envs 4096 --max_iterations 50000 \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml
```

防抖续训

```bash
CUDA_VISIBLE_DEVICES=4 python legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic --proj_name g1_priv_mimic --exptid 0106_teacher_smooth_strict \
    --resumeid 0106_teacher --checkpoint 27500 \
    --device cuda:0 --num_envs 4096 --max_iterations 30000 \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --train.algorithm.learning_rate 5e-5 --train.algorithm.schedule fixed --train.algorithm.entropy_coef 0.01 \
    --rewards.scales.action_rate -0.1 \
    --rewards.scales.dof_vel -1e-3 \
    --rewards.scales.dof_acc -5e-7
```

## Play

```bash
cd legged_gym/legged_gym/scripts
python play.py \
  --task g1_priv_mimic --proj_name g1_priv_mimic --exptid 0106_teacher \
  --device cuda:2 --num_envs 1 \
  --headless --record_video \
  --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
  --motion.max_motions 8 --record_num_motions 8 --random
```

## 训练精度优化（减少显存使用）

当数据集过大导致内存不足时，可以使用混合精度训练来减少显存占用。

### 参数说明

| 参数                  | 可选值       | 说明                                                       |
| --------------------- | ------------ | ---------------------------------------------------------- |
| `--train.precision` | `float32`  | 默认精度，完全向后兼容                                     |
|                       | `float16`  | 半精度，显存节省约 50%，适用于大多数 GPU（Turing+）        |
|                       | `bfloat16` | Brain Float16，动态范围更大，适用于 Ampere+ GPU（如 A100） |

### 使用示例

```bash
# 使用 float16 精度训练（推荐，显存节省约 50%）
CUDA_VISIBLE_DEVICES=4 python legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0106_teacher_fp16 \
    --device cuda:0 \
    --num_envs 4096 --max_iterations 30000 \
    --train.precision float16 \
    --motion.motion_file /path/to/motion_config.yaml

# 使用 bfloat16 精度训练（需要 Ampere+ GPU，如 A100/3090）
CUDA_VISIBLE_DEVICES=4 python legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0106_teacher_bf16 \
    --device cuda:0 \
    --num_envs 4096 --max_iterations 30000 \
    --train.precision bfloat16 \
    --motion.motion_file /path/to/motion_config.yaml
```

### 多卡 DDP 混合精度训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0123_teacher_ddp_fp16 \
    --num_envs 4096 --max_iterations 300000 \
    --train.precision bfloat16 \
    --motion.motion_file /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/dataset_mix_9dad215a_total758900.yaml
```

### 注意事项

1. **GPU 兼容性**

   - `float16`：需要 Turing 架构及以上（如 RTX 2080, 3090, 4090）
   - `bfloat16`：需要 Ampere 架构及以上（如 A100, RTX 3090, 4090）
2. **训练稳定性**

   - 混合精度训练可能在某些场景下出现数值不稳定
   - 如果训练出现 NaN，建议回退到 `float32` 或降低学习率
3. **显存估算**

   - `float32` → `float16`：显存减少约 40-50%
   - 例如：原本需要 24GB 显存的任务，使用 float16 可能只需 12-15GB

## 内存优化（大型数据集）

当动作数据集过大导致 CPU 内存不足时，可以使用以下参数优化内存使用。

### 参数说明

| 参数 | 默认值 | 说明 | 对应配置项 |
|------|--------|------|------|
| `--motion.storage_dtype` | `float32` | 数据存储精度，设为 `float16` 可减少约 50% CPU 内存 | `--motion.storage_dtype` |
| `--gpu_cache` | `4.0` | GPU 缓存预算（GiB），用于缓存活跃动作 | `--motion.gpu_cache_gib` |
| `--cpu_cache` | `50.0` | CPU LRU 缓存预算（GiB），配合 lazy_load 使用 | `--motion.cpu_cache_gib` |
| `--lazy_load` | `False` | 延迟加载模式，启动时只加载元数据，按需加载数据 | `--motion.lazy_load` |

### 使用示例

```bash
# 使用 float16 存储减少内存（最简单的优化，内存减半）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0123_teacher_lowmem \
    --num_envs 4096 --max_iterations 300000 \
    --motion.storage_dtype float16 \
    --gpu_cache 8.0 \
    --motion.motion_file /path/to/large_dataset.yaml

# 延迟加载模式（超大数据集，按需从磁盘加载）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0123_teacher_lazyload \
    --num_envs 4096 --max_iterations 300000 \
    --lazy_load \
    --cpu_cache 100.0 \
    --motion.storage_dtype float16 \
    --motion.motion_file /path/to/huge_dataset.yaml
```

### 内存优化组合建议

| 场景 | 推荐配置 |
|------|---------|
| 中等数据集（~100GB） | `--motion.storage_dtype float16` |
| 大型数据集（~200GB） | `--motion.storage_dtype float16 --gpu_cache 8.0` |
| 超大数据集（>200GB） | `--lazy_load --cpu_cache 100.0 --motion.storage_dtype float16` |

---

## 定期重采样模式（Periodic Resample）

定期重采样模式是一种新的训练策略：**每次只加载部分数据到 GPU，训练一定步数后清空并重新采样新数据**。

### 工作原理

```
启动 → 采样 N 条 motion → 训练 X 次迭代 → 清空 GPU 缓存
              ↓                              ↓
         采样新 motion ← ← ← ← ← ← ← ← ← ← ← ←
              ↓
         继续训练 X 次迭代
              ↓
         循环...
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--motion_resample_interval` | `0` | 重采样间隔（迭代次数），`0` 表示禁用该模式 |
| `--motion_resample_per_gpu` | `15000` | 每张卡加载的 motion 条数 |

### 使用示例

```bash
# 每 50 次迭代重采样，每卡 15000 条数据（4 卡 = 60000 条）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0203_teacher_resample \
    --num_envs 4096 --max_iterations 300000 \
    --motion_resample_interval 100 \
    --motion_resample_per_gpu 15000 \
    --motion.motion_file /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/dataset_mix_8203b425_total328739.yaml

# 更激进的重采样：每 20 次迭代重采样，每卡 10000 条
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0203_teacher_resample_aggressive \
    --num_envs 4096 --max_iterations 300000 \
    --motion_resample_interval 20 \
    --motion_resample_per_gpu 10000 \
    --motion.motion_file /path/to/dataset.yaml
```

### 与原模式的对比

| 特性 | 原缓存模式 | 重采样模式 |
|------|-----------|-----------|
| GPU 内存 | 动态 LRU 缓存 | 固定 N 条数据 |
| CPU 内存 | 大量缓存 | 不需要（lazy_load） |
| 数据多样性 | 逐步访问全部数据 | 定期重新采样 |
| 适用场景 | 数据集适中 | 超大数据集 / 想增加数据多样性 |

### 多卡自动计算

重采样模式下，实际加载的 motion 总数 = `motion_resample_per_gpu × GPU 数量`：

| GPU 数 | per_gpu=15000 | per_gpu=10000 |
|--------|---------------|---------------|
| 4 卡 | 60000 条 | 40000 条 |
| 8 卡 | 120000 条 | 80000 条 |
