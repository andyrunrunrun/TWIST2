#!/bin/bash
# Batch evaluate teacher checkpoints at different iterations
# Usage: bash tools/batch_eval_teacher.sh

# Common paths
MOTION_YAML="/home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/test_pico.yaml"
DEVICE="cuda:0"
NUM_ENVS=4096
# CONTINUE_ON_FAIL_FLAG="--continue_on_fail"
CONTINUE_ON_FAIL_FLAG=""


echo "======================================"
echo "Batch Teacher Evaluation (with --continue_on_fail)"
echo "======================================"

# ============================================================================
# Part 1: Evaluate weijin's teacher checkpoints (30k - 100k)
# ============================================================================
echo ""
echo "========== Part 1: weijin 0106 Teacher =========="

WEIJIN_EXPTID_BASE="weijin_0106"
WEIJIN_POLICY_DIR="/home/weijin/source/Humanoid/TWIST2/legged_gym/logs/g1_priv_mimic/0106_teacher"
WEIJIN_CHECKPOINTS=(30000 40000 50000 60000 70000 80000 100000)

for ITER in "${WEIJIN_CHECKPOINTS[@]}"; do
    echo ""
    echo "--------------------------------------"
    echo "[weijin] Evaluating checkpoint: model_${ITER}.pt"
    echo "--------------------------------------"

    EXPTID="teacher_${WEIJIN_EXPTID_BASE}_${ITER}_gym_pico"
    POLICY_PATH="${WEIJIN_POLICY_DIR}/model_${ITER}.pt"

    if [ ! -f "$POLICY_PATH" ]; then
        echo "[ERROR] Policy not found: $POLICY_PATH"
        echo "Skipping..."
        continue
    fi

    echo "EXPTID: $EXPTID"
    echo "Policy: $POLICY_PATH"

    python tools/gym_exec_eval_teacher.py \
        --exptid "$EXPTID" \
        --policy_path "$POLICY_PATH" \
        --motion_yaml "$MOTION_YAML" \
        --device "$DEVICE" \
        --num_envs "$NUM_ENVS" \
        --headless \
        $CONTINUE_ON_FAIL_FLAG

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Completed model_${ITER}.pt"
    else
        echo "[FAILED] model_${ITER}.pt"
    fi
done

# ============================================================================
# Part 2: Evaluate dataset_mix_8203b425_total328739 checkpoints (30k, 35k)
# ============================================================================
echo ""
echo "========== Part 2: dataset_mix_8203b425_total328739 =========="

DATASET_EXPTID_BASE="dataset_mix_8203b425_total328739"
DATASET_POLICY_DIR="/home/huanghao/source/code/TWIST2/legged_gym/logs/g1_priv_mimic/dataset_mix_8203b425_total328739"
DATASET_CHECKPOINTS=(30000 35000)

for ITER in "${DATASET_CHECKPOINTS[@]}"; do
    echo ""
    echo "--------------------------------------"
    echo "[dataset] Evaluating checkpoint: model_${ITER}.pt"
    echo "--------------------------------------"

    EXPTID="teacher_${DATASET_EXPTID_BASE}_${ITER}_gym_pico"
    POLICY_PATH="${DATASET_POLICY_DIR}/model_${ITER}.pt"

    if [ ! -f "$POLICY_PATH" ]; then
        echo "[ERROR] Policy not found: $POLICY_PATH"
        echo "Skipping..."
        continue
    fi

    echo "EXPTID: $EXPTID"
    echo "Policy: $POLICY_PATH"

    python tools/gym_exec_eval_teacher.py \
        --exptid "$EXPTID" \
        --policy_path "$POLICY_PATH" \
        --motion_yaml "$MOTION_YAML" \
        --device "$DEVICE" \
        --num_envs "$NUM_ENVS" \
        --headless \
        $CONTINUE_ON_FAIL_FLAG

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Completed model_${ITER}.pt"
    else
        echo "[FAILED] model_${ITER}.pt"
    fi
done

echo ""
echo "======================================"
echo "All evaluations completed!"
echo "Results saved in outputs/ directory"
echo "======================================"
