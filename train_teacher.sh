#!/bin/bash

# Usage:
#   bash train_teacher.sh <experiment_id> <device> [enable_anti_shuffle] [step_switch_scale] [stance_foot_speed_scale]
# Example:
#   bash train_teacher.sh 0201_teacher cuda:0 true -0.20 -0.05

# ============================================================
# 环境配置
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
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

task_name="${robot_name}_priv_mimic"
proj_name="${robot_name}_priv_mimic"

cd legged_gym/legged_gym/scripts

extra_args=(
  --anti_shuffle_step_switch_scale "${step_switch_scale}"
  --anti_shuffle_stance_foot_speed_scale "${stance_foot_speed_scale}"
)

if [[ "${enable_anti_shuffle}" == "1" || "${enable_anti_shuffle}" == "true" || "${enable_anti_shuffle}" == "True" ]]; then
  extra_args+=(--enable_anti_shuffle_reward)
fi

# Run the teacher training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --max_iterations 150000 \
                "${extra_args[@]}"
