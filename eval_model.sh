#!/bin/bash
# TWIST2 模型测评脚本包装器
# 
# 使用方法:
#   bash eval_model.sh <model_path> [cuda_device_id]
#
# 示例:
#   bash eval_model.sh /path/to/model.pt 0
#   bash eval_model.sh /path/to/model.onnx 1

set -e

# 默认参数
MODEL_PATH="${1:-}"
DEVICE_ID="${2:-0}"

# 检查参数
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 请提供模型路径"
    echo "用法: bash eval_model.sh <model_path> [cuda_device_id]"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 设置conda环境和路径
CONDA_ENV="twist2"
MOTION_CONFIG="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_EgoBody_numpy123_w1_OMOMO_numpy123_w1_interhuman_numpy123_w1_lafan1_numpy123_w1_pico_numpy123_w30_twist1_to_twist2_numpy123_w1_v1_v2_v3_g1_numpy123_w20_total49706.yaml"
DEVICE="cuda:${DEVICE_ID}"
NUM_ENVS=256

# 默认从模型路径推断 task，兼容 g1_stu_future_moe / transformer 等变体
TASK=$(python - "$MODEL_PATH" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve()
parts = path.parts
task = "g1_stu_future"
if "logs" in parts:
    idx = parts.index("logs")
    if len(parts) > idx + 1:
        task = parts[idx + 1]
print(task)
PY
)

# 激活conda环境
echo "激活conda环境: $CONDA_ENV"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 运行测评
echo "========================================"
echo "开始模型测评"
echo "模型: $MODEL_PATH"
echo "设备: $DEVICE"
echo "任务: $TASK"
echo "========================================"

python /home/huanghao/source/code/TWIST2/evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --motion_config "$MOTION_CONFIG" \
    --task "$TASK" \
    --device "$DEVICE" \
    --num_envs $NUM_ENVS \
    --headless

echo "测评完成!"
