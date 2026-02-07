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
  --task g1_priv_mimic --proj_name g1_priv_mimic --exptid weijin \
  --device cuda:2 --num_envs 1 \
  --headless --record_video \
  --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
  --motion.max_motions 8 --record_num_motions 8 --random
```

## 内存优化（大型数据集）

当动作数据集过大导致 CPU 内存不足时，可以使用以下参数优化内存使用。

### 参数说明

| 命令行参数 | 默认值 | 说明 | 对应配置项 |
|-----------|--------|------|----------|
| `--motion.storage_dtype` | `float32` | 数据存储精度，设为 `float16` 可减少约 50% CPU 内存 | `cfg.motion.storage_dtype` |
| `--gpu_cache` | `4.0` | GPU 缓存预算（GiB），用于缓存活跃动作 | `cfg.motion.gpu_cache_gib` |
| `--cpu_cache` | `50.0` | CPU LRU 缓存预算（GiB），**仅 lazy_load 模式生效** | `cfg.motion.cpu_cache_gib` |
| `--lazy_load` | `False` | 延迟加载模式，启动时只加载元数据，按需加载数据 | `cfg.motion.lazy_load` |

### 使用示例

```bash
# 方式 1：使用 float16 存储减少内存（最简单的优化，内存减半）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0123_teacher_lowmem \
    --num_envs 4096 --max_iterations 300000 \
    --motion.storage_dtype float16 \
    --gpu_cache 8.0 \
    --motion.motion_file /path/to/large_dataset.yaml

# 方式 2：延迟加载模式（超大数据集，按需从磁盘加载）
# 注意：需要同时设置 --lazy_load 和 --cpu_cache
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
启动 → lazy_load（仅元数据）→ 采样 N 条 motion → 加载到 GPU → 训练 X 次迭代 → 清空
                                                                         ↓
                              采样新 motion ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
                                    ↓
                              继续训练 X 次迭代
                                    ↓
                              循环...
```

> **⚠️ 重要提示**：虽然代码会自动启用 `lazy_load`，但**强烈建议显式添加 `--lazy_load` 参数**。
>
> 原因：自动启用机制在某些情况下可能不可靠（配置覆盖、深拷贝等），导致初始化时仍然加载全部数据，启动变慢。手动指定可以确保始终使用懒加载模式。

### 参数说明

| 命令行参数 | 默认值 | 说明 |
|-----------|--------|------|
| `--motion_resample_interval` | `0` | 重采样间隔（迭代次数），`0` 表示禁用该模式，`>0` 时自动启用 `lazy_load` |
| `--motion_resample_per_gpu` | `15000` | 每张卡加载的 motion 条数 |

### 使用示例

```bash
# 每 100 次迭代重采样，每卡 15000 条数据（6 卡 = 90000 条）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0203_teacher_resample \
    --num_envs 4096 --max_iterations 300000 \
    --motion_resample_interval 100 \
    --motion_resample_per_gpu 15000 \
    --lazy_load \
    --motion.motion_file /path/to/dataset.yaml

# 更激进的重采样：每 20 次迭代重采样，每卡 10000 条
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0203_teacher_resample_aggressive \
    --num_envs 4096 --max_iterations 300000 \
    --motion_resample_interval 20 \
    --motion_resample_per_gpu 10000 \
    --lazy_load \
    --motion.motion_file /path/to/dataset.yaml
```

### 与原模式的对比

| 特性 | 原缓存模式 | 重采样模式 |
|------|-----------|-----------|
| lazy_load | 可选（手动 `--lazy_load`） | **自动启用** |
| GPU 内存 | 动态 LRU 缓存 | 固定 N 条数据（合并张量） |
| CPU 内存 | 可选 LRU 缓存（需 `--cpu_cache`） | 仅元数据，无需 `--cpu_cache` |
| 数据多样性 | 逐步访问全部数据 | 定期重新采样 subset |
| 适用场景 | 数据集适中 | 超大数据集 / 想增加数据多样性 |

### 多卡自动计算

重采样模式下，实际加载的 motion 总数 = `motion_resample_per_gpu × GPU 数量`：

| GPU 数 | per_gpu=15000 | per_gpu=10000 | per_gpu=5000 |
|--------|---------------|---------------|--------------|
| 1 卡 | 15000 条 | 10000 条 | 5000 条 |
| 4 卡 | 60000 条 | 40000 条 | 20000 条 |
| 6 卡 | 90000 条 | 60000 条 | 30000 条 |
| 8 卡 | 120000 条 | 80000 条 | 40000 条 |

---

## 常见问题（FAQ）

### Q1: resample 模式下为什么启动还是很慢？

**A:** 确保显式添加 `--lazy_load` 参数。虽然代码会自动启用，但配置覆盖可能导致失效：

```bash
# ❌ 不推荐（可能启动慢）
--motion_resample_interval 100

# ✅ 推荐（启动快）
--motion_resample_interval 100 --lazy_load
```

### Q2: 如何选择 `motion_resample_interval`？

**A:** 根据数据集大小和训练需求：
- 小数据集（<5万条）：`interval=50-100`
- 大数据集（>10万条）：`interval=20-50`
- 想要更多数据多样性：更小的 interval

### Q3: 如何选择 `motion_resample_per_gpu`？

**A:** 根据 GPU 内存：
- 24GB 显存：`per_gpu=5000-10000`
- 40GB 显存：`per_gpu=10000-15000`
- 80GB 显存：`per_gpu=15000-30000`

### Q4: Teacher 训练和 Student 训练有什么区别？

**A:**
| 特性 | Teacher (g1_priv_mimic) | Student (g1_stu_future) |
|------|------------------------|------------------------|
| 观测空间 | 包含特权信息（如目标位置） | 仅包含机器人自身状态 |
| 是否需要 teacher | 不需要 | 需要（用于蒸馏） |
| 部署用途 | 无（仅用于训练 student） | 可部署到真实机器人 |

### Q5: CPU 内存不足怎么办？

**A:** 使用 lazy_load 模式，并设置适当的 cpu_cache：

```bash
--lazy_load --cpu_cache 50.0  # 50GB CPU 缓存
```

如果仍然不足，可以降低 `--cpu_cache` 值，但可能会影响性能（更频繁的磁盘读取）。
