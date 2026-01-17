## Train

```bash
# 多卡 DDP（torchrun）
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic_limbw \
    --proj_name g1_priv_mimic_limbw \
    --exptid 0116_teacher_limbw \
    --num_envs 4096 --max_iterations 100000 \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml

# 续训

CUDA_VISIBLE_DEVICES=5 python legged_gym/legged_gym/scripts/train.py \
    --task g1_priv_mimic --proj_name g1_priv_mimic --exptid 0106_teacher \
    --resumeid 0106_teacher --checkpoint 27500 \
    --device cuda:0 \
    --num_envs 4096 --max_iterations 80000 \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml
```

## Play

### Limb-weights 对比可视化（同一 motion，不同 limb_weights）

脚本：`legged_gym/legged_gym/scripts/play_limbw_compare.py`

- 选取 N 个 motion 样本（`--record_num_motions` 或 `--record_motion_ids`）
- 对每个样本循环 N 组 `limb_weights`（JSON 传参，顺序固定为 `[L_arm, R_arm, L_leg, R_leg]`）
- 每组都会在 `reset_idx` 前 reseed，保证初始随机项一致，只改变 limb_weights
- 录制完整 motion 长度，并把所有 clips 拼到一个 mp4；overlay 会标注 sample/motion_id/weights

```bash
# teacher policy：5 个样本 × 4 组 weights => 20 段 clip 拼成一个视频
cd legged_gym/legged_gym/scripts
python play_limbw_compare.py \
    --task g1_priv_mimic_limbw \
    --proj_name g1_priv_mimic_limbw \
    --exptid 0116_teacher_limbw \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --random \
    --motion.max_motions 8 \
    --record_num_motions 8 \
    --seed 123 \
    --limbw_cases_json '[[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]'

# student policy（同样用法；替换 task/proj/exptid 为 student 的）
cd legged_gym/legged_gym/scripts
python play_limbw_compare.py \
    --task g1_stu_mimic_limbw \
    --proj_name g1_stu_mimic_limbw \
    --exptid <stu_exptid> \
    --motion.motion_file /home/weijin/source/Humanoid/TWIST2/legged_gym/motion_data_configs/humanoid_wbc_gmr_30fps_mix.yaml \
    --random \
    --motion.max_motions 8 \
    --record_num_motions 8 \
    --seed 123 \
    --limbw_cases_json '[[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]'
```

输出视频默认在：
`legged_gym/logs/videos_retarget/<exptid>/<proj_name>-<exptid>-limbw_compare.mp4`（可用 `--record_video_name xxx.mp4` 改名）

注：该脚本会强制开启录视频，需要可用的图形上下文（X/Wayland 或 `xvfb-run`），否则会报 “camera sensors unavailable”。


