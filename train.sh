#!/bin/bash
# bash /home/huanghao/source/code/TWIST2/train.sh \
#     0213_student_single cuda:7 true -0.20 -0.05 \
#     /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_OMOMO_numpy123_w1_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total35772.yaml \
#     backup_for_sota_teacher -1
# Usage:
#   bash train.sh <experiment_id> <device> [enable_anti_shuffle] [step_switch_scale] [stance_foot_speed_scale] [motion_yaml] [teacher_exptid] [teacher_checkpoint]
# Example:
#   bash train.sh 1103_twist2 cuda:0 true -0.20 -0.05 /path/to/dataset.yaml 0106_teacher -1

# ============================================================
# 环境配置
# ============================================================
# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2

# 设置 LD_LIBRARY_PATH（解决 libpython3.8.so.1.0 找不到的问题）
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# ============================================================
# 训练参数配置
# ============================================================
robot_name="g1"
exptid=$1
device=$2
enable_anti_shuffle=${3:-false}
step_switch_scale=${4:--0.20}
stance_foot_speed_scale=${5:--0.05}
motion_yaml=${6:-}
teacher_exptid=${7:-None}
teacher_checkpoint=${8:--1}

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

# ============================================================
# 说明：
# 1. 如果只训练学生（纯RL），设置 teacher_exptid 为 "None"
# 2. 如果要进行蒸馏（Teacher -> Student），设置为教师实验的 ID
#    例如 "1103_teacher_run"，此时需确保：
#    - task_name 为 "g1_stu_future" 或其他学生环境
#    - 教师模型已训练好
# ============================================================

cd legged_gym/legged_gym/scripts

extra_args=(
  --anti_shuffle_step_switch_scale "${step_switch_scale}"
  --anti_shuffle_stance_foot_speed_scale "${stance_foot_speed_scale}"
)

if [[ "${enable_anti_shuffle}" == "1" || "${enable_anti_shuffle}" == "true" || "${enable_anti_shuffle}" == "True" ]]; then
  extra_args+=(--enable_anti_shuffle_reward)
fi

if [[ -n "${motion_yaml}" ]]; then
  extra_args+=(--motion.motion_file "${motion_yaml}")
fi

# Run the training script (student, no distillation by default)
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "${teacher_exptid}" \
                --teacher_checkpoint "${teacher_checkpoint}" \
                --max_iterations 150000 \
                "${extra_args[@]}"
