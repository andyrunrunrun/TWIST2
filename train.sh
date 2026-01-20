#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>
# bash train.sh 1103_twist2 cuda:0

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

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                --max_iterations 150000 \
                --resume \
                # --debug