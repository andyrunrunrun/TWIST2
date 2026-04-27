#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_ID="${1:-0}"
MOTION_CONFIG="${2:-$ROOT_DIR/legged_gym/motion_data_configs/AMASS_numpy123_w1_EgoBody_numpy123_w1_OMOMO_numpy123_w1_interhuman_numpy123_w1_lafan1_numpy123_w1_pico_numpy123_w30_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total49706.yaml}"
NUM_ENVS="${3:-256}"
OUTPUT_DIR="${4:-$ROOT_DIR/eval_results/twist2_35k_models_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${5:-5000}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_LABELS=(
    "sonic_35k_without_teacher"
    "TWIST2_35k_0_2_mlp_baseline"
    "TWIST2_35k_0_3_mlp_baseline"
    "TWIST2_35k_0_4_mlp_baseline"
)

SELECTED_MODELS=(
    "$ROOT_DIR/legged_gym/logs/g1_stu_future/sonic_35k_without_teacher/model_49999.pt"
    "$ROOT_DIR/legged_gym/logs/g1_stu_future/TWIST2_35k_0_2_mlp_baseline/model_49999.pt"
    "$ROOT_DIR/legged_gym/logs/g1_stu_future/TWIST2_35k_0_3_mlp_baseline/model_49999.pt"
    "$ROOT_DIR/legged_gym/logs/g1_stu_future/TWIST2_35k_0_4_mlp_baseline/model_49999.pt"
)

if [[ ! -f "$MOTION_CONFIG" ]]; then
    echo "错误: motion config 不存在: $MOTION_CONFIG"
    exit 1
fi

for model_path in "${SELECTED_MODELS[@]}"; do
    if [[ ! -f "$model_path" ]]; then
        echo "错误: 模型不存在: $model_path"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "批量评测指定 35k 模型"
echo "设备: cuda:${DEVICE_ID}"
echo "动作配置: $MOTION_CONFIG"
echo "并行环境数: $NUM_ENVS"
echo "最大步数: $MAX_STEPS"
echo "输出目录: $OUTPUT_DIR"
echo "实验数: ${#SELECTED_MODELS[@]}"
echo "DRY_RUN: $DRY_RUN"
echo "========================================"

for idx in "${!SELECTED_MODELS[@]}"; do
    printf "[%02d/%02d] %-30s %s\n" \
        "$((idx + 1))" "${#SELECTED_MODELS[@]}" "${MODEL_LABELS[$idx]}" "${SELECTED_MODELS[$idx]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1，仅打印待评测模型，不执行评测。"
    exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate twist2
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib"

for idx in "${!SELECTED_MODELS[@]}"; do
    model_path="${SELECTED_MODELS[$idx]}"
    model_label="${MODEL_LABELS[$idx]}"

    echo
    echo "========================================"
    echo "[$((idx + 1))/${#SELECTED_MODELS[@]}] 开始评测"
    echo "实验标签: $model_label"
    echo "模型路径: $model_path"
    echo "========================================"

    python "$ROOT_DIR/evaluate_model.py" \
        --model_path "$model_path" \
        --motion_config "$MOTION_CONFIG" \
        --task g1_stu_future \
        --device "cuda:${DEVICE_ID}" \
        --num_envs "$NUM_ENVS" \
        --max_steps "$MAX_STEPS" \
        --output_dir "$OUTPUT_DIR" \
        --headless
done

echo
echo "全部评测完成，结果保存在: $OUTPUT_DIR"
