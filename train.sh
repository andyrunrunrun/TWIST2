#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0


cd legged_gym/legged_gym/scripts

robot_name="g1"
exptid=$1
device=$2

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                # 1. 如果只训练学生（纯RL），设置 teacher_exptid 为 "None"
                # 2. 如果要进行蒸馏（Teacher -> Student），设置为教师实验的 ID（例如 "1103_teacher_run"），此时需确保 task_name 为 "g1_stu_future" 或其他学生环境，且教师模型已训练好
                --teacher_exptid "None" \
                # --resume \
                # --debug \
