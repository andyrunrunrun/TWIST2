## Train student（g1_stu_hymotion100k_hyfeat）

本仓库包含 HY feature 的 student 训练任务：`g1_stu_hymotion100k_hyfeat`
（配置：`legged_gym/legged_gym/envs/g1/g1_mimic_hyfeat_config.py`）。

### 0) 必要环境变量

需要设置 HY-Motion 根目录，YAML 才能找到特征文件：

```bash
export HY_HUMANOID_ROOT=/home/weijin/source/Humanoid/HY-Humanoid
```

默认 motion YAML：
`legged_gym/motion_data_configs/hymotion100k_g1_gmr_30fps.yaml`

teacher ckpt 路径（DAgger 会用到）：
`legged_gym/logs/g1_priv_hymotion100k/<teacher_exptid>/model_<teacher_checkpoint>.pt`


### 1) 训练

```bash
export HY_HUMANOID_ROOT=/home/weijin/source/Humanoid/HY-Humanoid
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_hymotion100k_hyfeat \
  --proj_name g1_stu_hymotion100k_hyfeat \
  --exptid 0118_hyfeat_ddp \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1 \
  --num_envs 4096 --max_iterations 100000
```

说明：
- 每个 rank 会自动绑定到 `cuda:$LOCAL_RANK`。
- 只会由 rank0 写 checkpoints/logs。


### 2) 读 student ckpt 继续训练 / 微调

从已有 student run 加载权重，但写到新的 `exptid` 目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_hymotion100k_hyfeat \
  --proj_name g1_stu_hymotion100k_hyfeat \
  --exptid 0116_hyfeat_ddp_ft \
  --resumeid 0116_hyfeat_ddp \
  --checkpoint -1 \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1 \
  --num_envs 4096 --max_iterations 100000
```

其中：
- `--resumeid` 指向 `legged_gym/logs/g1_stu_hymotion100k_hyfeat/<resumeid>/`
- `--checkpoint -1` 表示自动选最新的 `model_*.pt`


### 3) Play / Eval student

`play.py` 内部用相对路径找日志目录，建议从脚本目录运行：

```bash
cd legged_gym/legged_gym/scripts

HY_HUMANOID_ROOT=/path/to/HY-Humanoid python play.py \
  --task g1_stu_hymotion100k_hyfeat --proj_name g1_stu_hymotion100k_hyfeat --exptid 0116_hyfeat_ddp \
  --device cuda:0 --num_envs 1 \
  --headless --record_video \
  --checkpoint -1
```

可选覆盖项：
- `--motion.motion_file /abs/path/to/hymotion100k_g1_gmr_30fps.yaml`
