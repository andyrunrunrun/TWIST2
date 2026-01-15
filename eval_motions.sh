#!/bin/bash
# Motion Evaluation Script for TWIST2
# 
# Usage: bash eval_motions.sh <onnx_policy_path> [motion_config] [num_envs] [device]
#
# Examples:
#   bash eval_motions.sh /home/huanghao/source/code/TWIST2/assets/ckpts/twist2_1017_25k.onnx
#   bash eval_motions.sh assets/ckpts/twist2_policy.onnx ./legged_gym/motion_data_configs/custom.yaml 512 cuda:1
export PYTHONPATH=/home/huanghao/source/env/isaacgym/python:$PYTHONPATH
export PYTHONPATH=/home/huanghao/source/code/TWIST2:$PYTHONPATH
export LD_LIBRARY_PATH=/home/huanghao/miniconda3/envs/twist2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/huanghao/source/code/TWIST2:$LD_LIBRARY_PATH

SCRIPT_DIR=$(dirname $(realpath $0))

# Default values
DEFAULT_MOTION_CONFIG="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/test.yaml"
DEFAULT_NUM_ENVS=4096
DEFAULT_DEVICE="cuda:0"

# Parse arguments
POLICY_PATH=$1
MOTION_CONFIG=${2:-$DEFAULT_MOTION_CONFIG}
NUM_ENVS=${3:-$DEFAULT_NUM_ENVS}
DEVICE=${4:-$DEFAULT_DEVICE}

# Validate policy path
if [ -z "$POLICY_PATH" ]; then
    echo "Error: Please provide path to ONNX policy file"
    echo "Usage: bash eval_motions.sh <onnx_policy_path> [motion_config] [num_envs] [device]"
    exit 1
fi

if [ ! -f "$POLICY_PATH" ]; then
    echo "Error: Policy file not found: $POLICY_PATH"
    exit 1
fi

# Set up environment
export PYTHONPATH=${SCRIPT_DIR}/legged_gym:${SCRIPT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${SCRIPT_DIR}:$LD_LIBRARY_PATH

# Run evaluation
echo "=================================================="
echo "TWIST2 Motion Evaluation"
echo "=================================================="
echo "Policy:        $POLICY_PATH"
echo "Motion Config: $MOTION_CONFIG"
echo "Num Envs:      $NUM_ENVS"
echo "Device:        $DEVICE"
echo "=================================================="

cd ${SCRIPT_DIR}/legged_gym/legged_gym/scripts

python eval_motions.py \
    --policy "$POLICY_PATH" \
    --motion_config "$MOTION_CONFIG" \
    --num_envs $NUM_ENVS \
    --device "$DEVICE" \
    --task g1_stu_future 
