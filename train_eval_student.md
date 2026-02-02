## Train student (g1_stu_future)

本仓库里 student 的训练任务是 `g1_stu_future`（对应 `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`），默认使用 `twist2_dataset.yaml`。

> Skills：未使用（现有 skills 都不匹配“编写 student 训练说明文档”这个需求）。

### 0) 约定：teacher ckpt 路径

蒸馏时，代码会从下面路径加载 teacher policy：

`legged_gym/logs/g1_priv_mimic/<teacher_exptid>/model_<teacher_checkpoint>.pt`

其中：
- `<teacher_exptid>` 对应你训练 teacher 时传入的 `--exptid`
- `<teacher_checkpoint>` 对应保存的迭代号（`-1` 表示自动取最新的 `model_*.pt`）


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
```

其中：
- `--resumeid` 指向“要加载的旧 student 目录”（`legged_gym/logs/g1_stu_future/<resumeid>/`）
- `--exptid` 是“新目录名”，新的 ckpt 会保存到 `legged_gym/logs/g1_stu_future/<exptid>/`


## Play / Eval student

`play.py` 内部用的是相对路径找日志目录，建议按下面方式从脚本目录运行：

```bash
cd legged_gym/legged_gym/scripts

python play.py \
  --task g1_stu_future --proj_name g1_stu_future --exptid 0116_student_ddp \
  --device cuda:0 --num_envs 1 \
  --headless --record_video \
  --checkpoint -1 \
  --eval_student \
  --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
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
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid 0116_student_lowmem \
    --teacher_exptid 0106_teacher \
    --num_envs 4096 --max_iterations 100000 \
    --motion.storage_dtype float16 \
    --gpu_cache 8.0 \
    --motion.motion_file /path/to/large_dataset.yaml

# 延迟加载模式（超大数据集，按需从磁盘加载）
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
