#!/bin/bash
# 全架构对比实验: MLP vs MoE vs Transformer
#
# 对比实验设计:
# - MLP Baseline: [512, 512, 256, 128] (~0.74M)
# - MoE: 4 experts, top-2 (~2.4M)
# - Transformer-2x: d_model=232, layers=2 (~1.44M)
# - Transformer-4x: d_model=280, layers=3 (~3.01M)
#
# 使用方法:
#   bash train_all_comparison.sh [17k|35k]

set -e

DATASET_TYPE=${1:-"17k"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# 设置数据集
if [ "$DATASET_TYPE" = "17k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_total17029.yaml"
    DATASET_NAME="AMASS_17k"
    # GPU 分配
    GPU_MLP=0
    GPU_MOE=1
    GPU_TRANS2X=2
    GPU_TRANS4X=3
elif [ "$DATASET_TYPE" = "35k" ]; then
    MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_OMOMO_numpy123_w1_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total35772.yaml"
    DATASET_NAME="TWIST2_35k"
    # GPU 分配
    GPU_MLP=4
    GPU_MOE=5
    GPU_TRANS2X=6
    GPU_TRANS4X=7
else
    echo "错误: 未知数据集类型 '$DATASET_TYPE'"
    exit 1
fi

echo "============================================"
echo "全架构对比实验"
echo "数据集: $DATASET_NAME"
echo "============================================"
echo ""

# 实验 1: MLP 基线
echo "[1/4] 启动 MLP 基线 (GPU $GPU_MLP)..."
CUDA_VISIBLE_DEVICES=$GPU_MLP NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future \
    --proj_name g1_stu_future \
    --exptid "${DATASET_NAME}_mlp_baseline" \
    --teacher_exptid "None" \
    --device "cuda:0" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0 &
PID_MLP=$!

# 实验 2: MoE
echo "[2/4] 启动 MoE (GPU $GPU_MOE)..."
CUDA_VISIBLE_DEVICES=$GPU_MOE NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future_moe \
    --proj_name g1_stu_future_moe \
    --exptid "${DATASET_NAME}_moe" \
    --teacher_exptid "None" \
    --device "cuda:0" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0 &
PID_MOE=$!

# 实验 3: Transformer-2x
echo "[3/4] 启动 Transformer-2x (GPU $GPU_TRANS2X)..."
CUDA_VISIBLE_DEVICES=$GPU_TRANS2X NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future_trans2x \
    --proj_name g1_stu_future_trans2x \
    --exptid "${DATASET_NAME}_trans2x" \
    --teacher_exptid "None" \
    --device "cuda:0" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0 &
PID_TRANS2X=$!

# 实验 4: Transformer-4x
echo "[4/4] 启动 Transformer-4x (GPU $GPU_TRANS4X)..."
CUDA_VISIBLE_DEVICES=$GPU_TRANS4X NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future_trans4x \
    --proj_name g1_stu_future_trans4x \
    --exptid "${DATASET_NAME}_trans4x" \
    --teacher_exptid "None" \
    --device "cuda:0" \
    --num_envs 4096 \
    --max_iterations 50000 \
    --motion.motion_file "$MOTION_YAML" \
    --gpu_cache 6.0 &
PID_TRANS4X=$!

echo ""
echo "============================================"
echo "所有实验已在后台启动"
echo "MLP PID: $PID_MLP"
echo "MoE PID: $PID_MOE"
echo "Transformer-2x PID: $PID_TRANS2X"
echo "Transformer-4x PID: $PID_TRANS4X"
echo ""
echo "查看日志:"
echo "  tail -f legged_gym/logs/g1_stu_future/${DATASET_NAME}_mlp_baseline/events*"
echo "  tail -f legged_gym/logs/g1_stu_future_moe/${DATASET_NAME}_moe/events*"
echo "  tail -f legged_gym/logs/g1_stu_future_trans2x/${DATASET_NAME}_trans2x/events*"
echo "  tail -f legged_gym/logs/g1_stu_future_trans4x/${DATASET_NAME}_trans4x/events*"
echo "============================================"

# 等待所有进程完成
wait $PID_MLP
wait $PID_MOE
wait $PID_TRANS2X
wait $PID_TRANS4X

echo "对比实验完成！"
