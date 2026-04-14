#!/bin/bash
# 快速测试所有架构是否能正常初始化

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

echo "============================================"
echo "测试所有架构"
echo "============================================"

# 测试 MLP
echo "[1/4] 测试 MLP..."
CUDA_VISIBLE_DEVICES=0 python -c "
from rsl_rl.modules import ActorCriticFuture
import torch
model = ActorCriticFuture(
    num_observations=800, num_critic_observations=1200,
    num_motion_observations=35, num_motion_steps=1,
    num_priop_observations=92, num_history_steps=10,
    num_actions=29, use_moe=False, use_transformer=False
)
obs = torch.randn(4, 800)
out = model.actor(obs)
print(f'✓ MLP 输出形状: {out.shape}')
"

# 测试 MoE
echo "[2/4] 测试 MoE..."
CUDA_VISIBLE_DEVICES=0 python -c "
from rsl_rl.modules import ActorCriticFuture
import torch
model = ActorCriticFuture(
    num_observations=800, num_critic_observations=1200,
    num_motion_observations=35, num_motion_steps=1,
    num_priop_observations=92, num_history_steps=10,
    num_actions=29, use_moe=True, use_transformer=False,
    num_experts=4, expert_hidden_dims=[512, 384, 192]
)
obs = torch.randn(4, 800)
out = model.actor(obs)
print(f'✓ MoE 输出形状: {out.shape}')
print(f'  Gating weights: {model.get_gating_weights().shape}')
"

# 测试 Transformer-2x
echo "[3/4] 测试 Transformer-2x..."
CUDA_VISIBLE_DEVICES=0 python -c "
from rsl_rl.modules import ActorCriticFuture
import torch
model = ActorCriticFuture(
    num_observations=800, num_critic_observations=1200,
    num_motion_observations=35, num_motion_steps=1,
    num_priop_observations=92, num_history_steps=10,
    num_actions=29, use_moe=False, use_transformer=True,
    d_model=232, nhead=4, num_transformer_layers=2
)
obs = torch.randn(4, 800)
out = model.actor(obs)
print(f'✓ Transformer-2x 输出形状: {out.shape}')
params = sum(p.numel() for p in model.actor.parameters())
print(f'  参数量: {params/1e6:.2f}M')
"

# 测试 Transformer-4x
echo "[4/4] 测试 Transformer-4x..."
CUDA_VISIBLE_DEVICES=0 python -c "
from rsl_rl.modules import ActorCriticFuture
import torch
model = ActorCriticFuture(
    num_observations=800, num_critic_observations=1200,
    num_motion_observations=35, num_motion_steps=1,
    num_priop_observations=92, num_history_steps=10,
    num_actions=29, use_moe=False, use_transformer=True,
    d_model=280, nhead=4, num_transformer_layers=3
)
obs = torch.randn(4, 800)
out = model.actor(obs)
print(f'✓ Transformer-4x 输出形状: {out.shape}')
params = sum(p.numel() for p in model.actor.parameters())
print(f'  参数量: {params/1e6:.2f}M')
"

echo ""
echo "============================================"
echo "✓ 所有架构测试通过！"
echo "============================================"
