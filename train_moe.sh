#!/bin/bash
# MoE (Mixture of Experts) 训练脚本
# 
# 使用方法:
#   bash train_moe.sh <exptid> <dataset_type> <device>
#
# 参数:
#   exptid: 实验名称 (如: AMASS_17k_moe)
#   dataset_type: 数据集类型 (17k 或 35k)
#   device: GPU 设备 (如: cuda:0)
#
# 示例:
#   bash train_moe.sh AMASS_17k_moe 17k cuda:0
#   bash train_moe.sh TWIST2_35k_moe 35k cuda:1

set -e

# 参数解析
EXPTID=${1:-"test_moe"}
DATASET_TYPE=${2:-"17k"}
DEVICE=${3:-"cuda:0"}

# 环境设置
source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# 数据集路径
if [ "$DATASET_TYPE" = "17k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_total17029.yaml"
    echo "使用 17k 数据集 (AMASS_numpy123_w1_total17029)"
elif [ "$DATASET_TYPE" = "35k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_OMOMO_numpy123_w1_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total35772.yaml"
    echo "使用 35k 数据集 (TWIST2 完整小规模)"
else
    echo "错误: 未知数据集类型 '$DATASET_TYPE'，请使用 '17k' 或 '35k'"
    exit 1
fi

# 提取 GPU 编号
GPU_ID=$(echo $DEVICE | grep -o '[0-9]*$')

echo "============================================"
echo "MoE 训练配置"
echo "============================================"
echo "实验名称: $EXPTID"
echo "任务: g1_stu_future_moe"
echo "数据集: $DATASET_TYPE"
echo "设备: $DEVICE (GPU $GPU_ID)"
echo "Motion YAML: $MOTION_YAML"
echo "============================================"

# 运行训练
CUDA_VISIBLE_DEVICES=$GPU_ID NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future_moe \
    --proj_name g1_stu_future_moe \
    --exptid "$EXPTID" \
    --teacher_exptid "None" \
    --device "$DEVICE" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0

echo "训练完成: $EXPTID"
