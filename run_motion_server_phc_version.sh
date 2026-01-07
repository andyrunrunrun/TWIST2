#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# PHC dataset path (29DoF G1 rev_1_0)
DATASET_PATH="/home/weijin/source/Humanoid/human2humanoid/PHC/data/g1_29dof_rev_1_0/postprocessed/amass_vb2_filered_plus_hhi_vb2_grounded_with_axes.pkl"

# Optional: pick a fixed key for reproducibility (leave empty to stream sequentially/randomly).
DATASET_KEY=""

cd "${SCRIPT_DIR}/deploy_real" || exit 1

python server_motion_phc.py \
  --dataset_path "${DATASET_PATH}" \
  --redis_ip "localhost" \
  --robot "unitree_g1_with_hands" \
  --rate_hz 50 \
  --quat_order "xyzw" \
  --wait_for_space \
  --dataset_key "${DATASET_KEY}" \
  --sample_mode "sequential" \
  --start 0 \
  --clip_len 0 \
  --start_interp_seconds 2 \
  --exit_interp_seconds 2 \
  --loop
