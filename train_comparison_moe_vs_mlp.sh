#!/bin/bash
# MLP vs MoE 对比实验脚本
#
# 同时运行原版 MLP 和 MoE 进行对比
# 建议在不同 GPU 上同时运行
#
# 使用方法:
#   bash train_comparison_moe_vs_mlp.sh [17k|35k]

set -e

DATASET_TYPE=${1:-"17k"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# 设置数据集
if [ "$DATASET_TYPE" = "17k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_total17029.yaml"
    DATASET_NAME="AMASS_17k"
    GPU_MLP=0
    GPU_MOE=1
elif [ "$DATASET_TYPE" = "35k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_OMOMO_numpy123_w1_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total35772.yaml"
    DATASET_NAME="TWIST2_35k"
    GPU_MLP=2
    GPU_MOE=3
else
    echo "错误: 未知数据集类型 '$DATASET_TYPE'"
    exit 1
fi

echo "============================================"
echo "对比实验: MLP vs MoE"
echo "数据集: $DATASET_NAME"
echo "============================================"

# 实验 1: 原版 MLP (后台运行)
echo "启动 MLP 训练 (GPU $GPU_MLP)..."
CUDA_VISIBLE_DEVICES=$GPU_MLP NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid "${DATASET_NAME}_mlp_baseline" \
    --teacher_exptid "None" \
    --device "cuda:$GPU_MLP" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0 &

PID_MLP=$!
echo "MLP PID: $PID_MLP"

# 实验 2: MoE (后台运行)
echo "启动 MoE 训练 (GPU $GPU_MOE)..."
CUDA_VISIBLE_DEVICES=$GPU_MOE NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future_moe \
    --proj_name g1_stu_future_moe \
    --exptid "${DATASET_NAME}_moe" \
    --teacher_exptid "None" \
    --device "cuda:$GPU_MOE" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0 &

PID_MOE=$!
echo "MoE PID: $PID_MOE"

echo ""
echo "============================================"
echo "两个实验已在后台启动"
echo "MLP PID: $PID_MLP"
echo "MoE PID: $PID_MOE"
echo ""
echo "查看日志:"
echo "  tail -f legged_gym/logs/g1_stu_future/${DATASET_NAME}_mlp_baseline/events*"
echo "  tail -f legged_gym/logs/g1_stu_future_moe/${DATASET_NAME}_moe/events*"
echo "============================================"

# 等待两个进程完成
wait $PID_MLP
wait $PID_MOE

echo "对比实验完成！"
