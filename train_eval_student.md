## Train student (g1_stu_future)

本仓库里 student 的训练任务是 `g1_stu_future`（对应 `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`），默认使用 `twist2_dataset.yaml`。

> Skills：未使用（现有 skills 都不匹配“编写 student 训练说明文档”这个需求）。

### 0) 约定：teacher ckpt 路径

蒸馏时，代码会从下面路径加载 teacher policy：

`legged_gym/logs/g1_priv_mimic/<teacher_exptid>/model_<teacher_checkpoint>.pt`

其中：
- `<teacher_exptid>` 对应你训练 teacher 时传入的 `--exptid`
- `<teacher_checkpoint>` 对应保存的迭代号（`-1` 表示自动取最新的 `model_*.pt`）


### Anti-shuffle 参数（可选，默认关闭）

用于抑制站立/慢速段的小碎步。默认不启用，旧训练行为不变。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_anti_shuffle_reward` | `False` | 启用 anti-shuffle 奖励（`step_switch_rate` + `stance_foot_speed`） |
| `--anti_shuffle_ref_vel_th` | `0.12` | 仅在参考速度低于该阈值时施加 anti-shuffle（m/s） |
| `--anti_shuffle_tilt_th` | `0.25` | 仅在机体倾斜低于该阈值时施加 anti-shuffle（投影重力 XY 范数） |
| `--anti_shuffle_contact_force_th` | `5.0` | 判定足接触的竖直力阈值（N） |
| `--anti_shuffle_step_switch_scale` | `-0.20` | 频繁换脚惩罚权重（更负=更强抑制小碎步） |
| `--anti_shuffle_stance_foot_speed_scale` | `-0.05` | 支撑脚平面速度惩罚权重（更负=更强抑制脚底小抖） |

示例（启用 anti-shuffle）：

```bash
CUDA_VISIBLE_DEVICES=0 python legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid 0213_student_antishuffle \
  --device cuda:0 \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1 \
  --enable_anti_shuffle_reward \
  --anti_shuffle_step_switch_scale -0.20 \
  --anti_shuffle_stance_foot_speed_scale -0.05
```


### 1) 单卡训练（推荐：带 teacher 蒸馏）

```bash
CUDA_VISIBLE_DEVICES=0 python legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid 0116_student \
  --device cuda:0 \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1
```

常用可选项：
- 覆盖 motion yaml（一般不需要；默认就是 TWIST2 数据集）：
  - `--motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/twist2_dataset.yaml`
- 覆盖训练步数/并行环境数：
  - `--num_envs 4096 --max_iterations 30000`
- 不想上 wandb：
  - `--no_wandb`


### 2) 单卡训练（不蒸馏：纯 student）

如果你只是想按 README 的默认方式快速开跑（不加载 teacher），也可以直接用封装脚本：

```bash
bash train.sh 0116_student cuda:0
# 开启 anti-shuffle：
# bash train.sh 0116_student cuda:0 true -0.20 -0.05
```

```bash
CUDA_VISIBLE_DEVICES=0 python legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid 0116_student_nodistill \
  --device cuda:0 \
  --teacher_exptid None
```

`--teacher_exptid` 为 `None/dummy` 时不会加载 teacher，KL 蒸馏项会自动关闭。


### 3) 多卡 DDP（torchrun）

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid 0116_student_ddp \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1 \
  --num_envs 4096 --max_iterations 100000 \
  --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml
```

说明：
- 每个 rank 会自动绑定到 `cuda:$LOCAL_RANK`（脚本内部做了绑定），一般不需要再传 `--device`。
- 只会由 rank0 写 checkpoints/logs。


### 4) 续训 / 微调（从旧 student ckpt 载入，但写到新 exptid）

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid 0116_student_ddp \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1 \
  --resumeid 0116_student_ddp \
  --checkpoint -1 \
  --num_envs 4096 --max_iterations 100000 \
  --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml

CUDA_VISIBLE_DEVICES=0,7 torchrun --standalone --nproc_per_node=2 legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid weijin_65000 \
  --resume \
  --resumeid weijin_65000 \
  --checkpoint -1 \
  --teacher_exptid "None" \
  --num_envs 4096 --max_iterations 100000 \
  --motion.motion_file  /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/pico_numpy123_w1_total563.yaml
```

其中：
- `--resumeid` 指向“要加载的旧 student 目录”（`legged_gym/logs/g1_stu_future/<resumeid>/`）
- `--exptid` 是“新目录名”，新的 ckpt 会保存到 `legged_gym/logs/g1_stu_future/<exptid>/`


## Play / Eval student

`play.py` 内部用的是相对路径找日志目录，建议按下面方式从脚本目录运行：

```bash
cd legged_gym/legged_gym/scripts

python play.py \
  --task g1_stu_future --proj_name g1_stu_future --exptid weiji_65000backup \
  --device cuda:0 --num_envs 1 \
  --headless --record_video \
  --checkpoint -1 \
  --eval_student \
  --motion.motion_file /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/test.yaml \
  --motion.max_motions 8 --record_num_motions 8 --random
```

- `--checkpoint -1`：自动选最新的 `model_*.pt`
- `--record_video`：会在日志目录下保存视频（若你的环境没有图形上下文，可能需要 `xvfb-run` 或正确设置 `--graphics_device_id`）

仓库根目录下的 `eval.sh` 也是调用 `play.py`，但其中 `motion_file` 默认写死成了别人的路径；你需要先把 `motion_file=...` 改成自己机器上的 `.pkl` 或 `.yaml` 路径再用。


## Export student to ONNX

```bash
bash to_onnx.sh legged_gym/logs/g1_stu_future/0116_student/model_20000.pt
```

输出的 onnx 默认会在脚本 `legged_gym/legged_gym/scripts/save_onnx.py` 里指定的目录/文件名规则下生成。

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
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid 0116_student_lowmem \
    --teacher_exptid 0106_teacher \
    --num_envs 4096 --max_iterations 100000 \
    --motion.storage_dtype float16 \
    --gpu_cache 8.0 \
    --motion.motion_file /path/to/large_dataset.yaml

# 方式 2：延迟加载模式（超大数据集，按需从磁盘加载）
# 注意：需要同时设置 --lazy_load 和 --cpu_cache
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid 0116_student_lazyload \
    --teacher_exptid 0106_teacher \
    --num_envs 4096 --max_iterations 100000 \
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
# 每 100 次迭代重采样，每卡 10000 条数据（2 卡 = 20000 条）
CUDA_VISIBLE_DEVICES=0,7 torchrun --standalone --nproc_per_node=2 legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid dataset_mix_8203b425_total328739_stu \
    --teacher_exptid dataset_mix_8203b425_total328739 \
    --teacher_checkpoint -1 \
    --num_envs 4096 --max_iterations 1000000 \
    --motion_resample_interval 100 \
    --motion_resample_per_gpu 10000 \
    --lazy_load \
    --motion.motion_file /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/dataset_mix_8203b425_total328739.yaml

# 更激进的重采样：每 20 次迭代重采样，每卡 10000 条
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid 0203_student_resample_aggressive \
    --teacher_exptid 0106_teacher \
    --teacher_checkpoint -1 \
    --num_envs 4096 --max_iterations 100000 \
    --motion_resample_interval 20 \
    --motion_resample_per_gpu 10000 \
    --lazy_load
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
| 2 卡 | 30000 条 | 20000 条 | 10000 条 |
| 4 卡 | 60000 条 | 40000 条 | 20000 条 |
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

### Q4: CPU 内存不足怎么办？

**A:** 使用 lazy_load 模式，并设置适当的 cpu_cache：

```bash
--lazy_load --cpu_cache 50.0  # 50GB CPU 缓存
```

如果仍然不足，可以降低 `--cpu_cache` 值，但可能会影响性能（更频繁的磁盘读取）。

### Q5: 训练时如何确认 resample 模式已启用？

**A:** 查看日志输出，启动时会打印：

```
[Motion Resample] Enabled: interval=100, per_gpu=10000
[MotionLib] Loading 20000 motions to GPU (merged tensors)...
```

如果没有看到这些信息，说明 resample 模式未正确启用。
