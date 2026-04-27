#!/bin/bash
# Diffusion Student 训练脚本
#
# 使用方法:
#   bash train_diffusion.sh <exptid> <dataset_type> <scale> <device>
#
# 参数:
#   exptid: 实验名称
#   dataset_type: 数据集类型 (17k 或 35k)
#   scale: 参数量 (2x 或 4x)
#   device: GPU 设备 (如: cuda:0)

set -e

EXPTID=${1:-"test_diffusion"}
DATASET_TYPE=${2:-"17k"}
SCALE=${3:-"2x"}
DEVICE=${4:-"cuda:0"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

if [ "$DATASET_TYPE" = "17k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_total17029.yaml"
    echo "使用 17k 数据集"
elif [ "$DATASET_TYPE" = "35k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_OMOMO_numpy123_w1_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total35772.yaml"
    echo "使用 35k 数据集"
else
    echo "错误: 未知数据集类型 '$DATASET_TYPE'，请使用 '17k' 或 '35k'"
    exit 1
fi

GPU_ID=$(echo $DEVICE | grep -o '[0-9]*$')

if [ "$SCALE" = "2x" ]; then
    TASK="g1_stu_future_diff2x"
    echo "使用 Diffusion-2x (~5.24M params, ~2.25x of MLP actor)"
elif [ "$SCALE" = "4x" ]; then
    TASK="g1_stu_future_diff4x"
    echo "使用 Diffusion-4x (~9.44M params, ~4.05x of MLP actor)"
else
    echo "错误: 未知参数量 '$SCALE'，请使用 '2x' 或 '4x'"
    exit 1
fi

echo "============================================"
echo "Diffusion 训练配置"
echo "============================================"
echo "实验名称: $EXPTID"
echo "任务: $TASK"
echo "数据集: $DATASET_TYPE"
echo "设备: $DEVICE (GPU $GPU_ID)"
echo "Motion YAML: $MOTION_YAML"
echo "============================================"

CUDA_VISIBLE_DEVICES=$GPU_ID NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task $TASK \
    --proj_name $TASK \
    --exptid "$EXPTID" \
    --device "$DEVICE" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0

echo "训练完成: $EXPTID"
