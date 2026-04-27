#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="/home/huanghao/source/code/TWIST2"
MODEL_LIST="${REPO_DIR}/model_for_eval.txt"
MOTION_CONFIG="${REPO_DIR}/legged_gym/motion_data_configs/dataset_mix_17e89ca9_total78669.yaml"
OUTPUT_DIR="${REPO_DIR}/eval_results/model_for_eval_easy_motion"
LOG_DIR="${OUTPUT_DIR}/logs"
GPU_ID=0
START_LINE=1
END_LINE=5

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${REPO_DIR}" || exit 1

echo "Running models ${START_LINE}-${END_LINE} on cuda:${GPU_ID}"

sed -n "${START_LINE},${END_LINE}p" "${MODEL_LIST}" | while IFS= read -r MODEL_PATH; do
    [[ -z "${MODEL_PATH}" ]] && continue

    RUN_NAME="$(basename "$(dirname "${MODEL_PATH}")")_$(basename "${MODEL_PATH}" .pt)"
    LOG_FILE="${LOG_DIR}/part1_gpu${GPU_ID}_${RUN_NAME}.log"

    echo "[$(date '+%F %T')] START ${MODEL_PATH}"
    python evaluate_model.py \
        --model_path "${MODEL_PATH}" \
        --motion_config "${MOTION_CONFIG}" \
        --device "cuda:${GPU_ID}" \
        --num_envs 256 \
        --output_dir "${OUTPUT_DIR}" \
        --headless 2>&1 | tee "${LOG_FILE}"

    STATUS=${PIPESTATUS[0]}
    if [[ ${STATUS} -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED status=${STATUS}: ${MODEL_PATH}" | tee -a "${LOG_FILE}"
    else
        echo "[$(date '+%F %T')] DONE ${MODEL_PATH}" | tee -a "${LOG_FILE}"
    fi
done
