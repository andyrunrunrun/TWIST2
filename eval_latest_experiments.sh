#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_ID="${1:-0}"
MOTION_CONFIG="${2:-$ROOT_DIR/legged_gym/motion_data_configs/AMASS_numpy123_w1_EgoBody_numpy123_w1_OMOMO_numpy123_w1_interhuman_numpy123_w1_lafan1_numpy123_w1_pico_numpy123_w30_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total49706.yaml}"
NUM_ENVS="${3:-256}"
OUTPUT_DIR="${4:-$ROOT_DIR/eval_results/latest_batch_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${5:-5000}"
DRY_RUN="${DRY_RUN:-0}"

LOG_ROOTS=(
    "$ROOT_DIR/legged_gym/logs/g1_stu_future_moe"
    "$ROOT_DIR/legged_gym/logs/g1_stu_future_trans2x"
    "$ROOT_DIR/legged_gym/logs/g1_stu_future_trans4x"
    "$ROOT_DIR/legged_gym/logs/g1_stu_future"
)

SKIP_DIRS=(
    "$ROOT_DIR/legged_gym/logs/g1_stu_future/student汇总"
    "$ROOT_DIR/legged_gym/logs/g1_stu_future/dataset_335003_hard_big_stu"
)

find_latest_model() {
    python - "$1" <<'PY'
from pathlib import Path
import re
import sys

exp_dir = Path(sys.argv[1])
candidates = []

for path in exp_dir.iterdir():
    if not path.is_file():
        continue
    if path.suffix.lower() not in {".pt", ".onnx"}:
        continue
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if not match:
        continue
    step = int(match.group(1))
    ext_priority = 1 if path.suffix.lower() == ".pt" else 0
    candidates.append((step, ext_priority, path.name))

if not candidates:
    sys.exit(1)

candidates.sort(key=lambda item: (item[0], item[1], item[2]))
print((exp_dir / candidates[-1][2]).resolve())
PY
}

should_skip_dir() {
    local exp_dir="$1"
    local skip_dir
    for skip_dir in "${SKIP_DIRS[@]}"; do
        if [[ "$exp_dir" == "$skip_dir" ]]; then
            return 0
        fi
    done
    return 1
}

if [[ ! -f "$MOTION_CONFIG" ]]; then
    echo "错误: motion config 不存在: $MOTION_CONFIG"
    exit 1
fi

for log_root in "${LOG_ROOTS[@]}"; do
    if [[ ! -d "$log_root" ]]; then
        echo "错误: 日志目录不存在: $log_root"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

SELECTED_EXPERIMENTS=()
SELECTED_MODELS=()

for log_root in "${LOG_ROOTS[@]}"; do
    while IFS= read -r exp_dir; do
        [[ -z "$exp_dir" ]] && continue

        if should_skip_dir "$exp_dir"; then
            echo "跳过实验目录: $exp_dir"
            continue
        fi

        if latest_model="$(find_latest_model "$exp_dir" 2>/dev/null)"; then
            SELECTED_EXPERIMENTS+=("$exp_dir")
            SELECTED_MODELS+=("$latest_model")
        else
            echo "警告: 未找到可评测 checkpoint，跳过: $exp_dir"
        fi
    done < <(find "$log_root" -mindepth 1 -maxdepth 1 -type d | sort)
done

if [[ "${#SELECTED_MODELS[@]}" -eq 0 ]]; then
    echo "错误: 没有找到任何可评测模型"
    exit 1
fi

echo "========================================"
echo "批量评测最新模型"
echo "设备: cuda:${DEVICE_ID}"
echo "动作配置: $MOTION_CONFIG"
echo "并行环境数: $NUM_ENVS"
echo "最大步数: $MAX_STEPS"
echo "输出目录: $OUTPUT_DIR"
echo "实验数: ${#SELECTED_MODELS[@]}"
echo "DRY_RUN: $DRY_RUN"
echo "========================================"

for idx in "${!SELECTED_MODELS[@]}"; do
    printf "[%02d/%02d] %s\n" \
        "$((idx + 1))" "${#SELECTED_MODELS[@]}" "${SELECTED_MODELS[$idx]}"
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
    exp_dir="${SELECTED_EXPERIMENTS[$idx]}"

    echo
    echo "========================================"
    echo "[$((idx + 1))/${#SELECTED_MODELS[@]}] 开始评测"
    echo "实验目录: $exp_dir"
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
