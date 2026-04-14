#!/bin/bash
# Transformer Backbone 训练脚本
#
# 使用方法:
#   bash train_transformer.sh <exptid> <dataset_type> <scale> <device>
#
# 参数:
#   exptid: 实验名称 (如: AMASS_17k_trans2x)
#   dataset_type: 数据集类型 (17k 或 35k)
#   scale: 参数量 (2x 或 4x)
#   device: GPU 设备 (如: cuda:0)
#
# 示例:
#   bash train_transformer.sh AMASS_17k_trans2x 17k 2x cuda:0
#   bash train_transformer.sh TWIST2_35k_trans4x 35k 4x cuda:1

set -e

# 参数解析
EXPTID=${1:-"test_transformer"}
DATASET_TYPE=${2:-"17k"}
SCALE=${3:-"2x"}
DEVICE=${4:-"cuda:0"}

# 环境设置
source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# 数据集路径
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

# 选择 task
if [ "$SCALE" = "2x" ]; then
    TASK="g1_stu_future_trans2x"
    echo "使用 Transformer-2x (d_model=232, nhead=4, layers=2, ~1.44M params)"
elif [ "$SCALE" = "4x" ]; then
    TASK="g1_stu_future_trans4x"
    echo "使用 Transformer-4x (d_model=280, nhead=4, layers=3, ~3.01M params)"
else
    echo "错误: 未知参数量 '$SCALE'，请使用 '2x' 或 '4x'"
    exit 1
fi

# 提取 GPU 编号
GPU_ID=$(echo $DEVICE | grep -o '[0-9]*$')

echo "============================================"
echo "Transformer 训练配置"
echo "============================================"
echo "实验名称: $EXPTID"
echo "任务: $TASK"
echo "数据集: $DATASET_TYPE"
echo "设备: $DEVICE (GPU $GPU_ID)"
echo "============================================"

# 运行训练
CUDA_VISIBLE_DEVICES=$GPU_ID NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task $TASK \
    --proj_name $TASK \
    --exptid "$EXPTID" \
    --teacher_exptid "None" \
    --device "$DEVICE" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0

echo "训练完成: $EXPTID"
