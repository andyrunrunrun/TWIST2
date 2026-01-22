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
