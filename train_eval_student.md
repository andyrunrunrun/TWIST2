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
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid 0116_student_ddp \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1
```

说明：
- 每个 rank 会自动绑定到 `cuda:$LOCAL_RANK`（脚本内部做了绑定），一般不需要再传 `--device`。
- 只会由 rank0 写 checkpoints/logs。


### 4) 续训 / 微调（从旧 student ckpt 载入，但写到新 exptid）

```bash
CUDA_VISIBLE_DEVICES=0 python legged_gym/legged_gym/scripts/train.py \
  --task g1_stu_future \
  --proj_name g1_stu_future \
  --exptid 0116_student_finetune \
  --resumeid 0116_student \
  --checkpoint 20000 \
  --device cuda:0 \
  --teacher_exptid 0106_teacher \
  --teacher_checkpoint -1
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
