#!/bin/bash
# MoE 快速测试脚本 - 验证代码是否能正常运行
# 
# 运行短时间的训练来验证 MoE 实现是否正确

set -e

echo "============================================"
echo "MoE 快速测试 (100 iterations)"
echo "============================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# 使用 17k 数据集进行快速测试
CUDA_VISIBLE_DEVICES=0 NCCL_TIMEOUT=2400 \
python legged_gym/legged_gym/scripts/train.py \
    --task g1_stu_future_moe \
    --proj_name g1_stu_future_moe \
    --exptid test_moe_quick \
    --teacher_exptid "None" \
    --device cuda:0 \
    --num_envs 256 \
    --max_iterations 100 \
    --motion.motion_file /home/huanghao/source/code/TWIST2/legged_gym/motion_data_configs/AMASS_numpy123_w1_total17029.yaml \
    --gpu_cache 6.0 \
    --no_wandb

echo "============================================"
echo "快速测试完成！"
echo "检查输出中是否包含 'ActorCriticFuture: Using MoE actor'"
echo "============================================"
