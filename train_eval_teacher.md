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
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --standalone --nproc_per_node=5 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic \
    --proj_name g1_priv_mimic \
    --exptid 0121_teacher_ddp_AMASS_w1_EgoBody_numpy123_w1_inter_x_w1_interhuman_numpy123_w1_lafan1_w1_MotionMillion_numpy123_w0.1_OMOMO_w1_twist1_to_twist2_w1_pico_numpy123_w20_v1_v2_v3_g1_w10_CORE4D_Real_numpy123_w1_total1443478 \
    --num_envs 4096 --max_iterations 300000 \
    --motion.motion_file /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_w1_EgoBody_numpy123_w1_inter_x_w1_interhuman_numpy123_w1_lafan1_w1_MotionMillion_numpy123_w0.1_OMOMO_w1_twist1_to_twist2_w1_pico_numpy123_w20_v1_v2_v3_g1_w10_CORE4D_Real_numpy123_w1_total1443478.yaml

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
