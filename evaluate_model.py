#!/usr/bin/env python3
"""
TWIST2 通用模型测评脚本

支持多种模型架构（MLP/MoE/Transformer）和格式（PT/ONNX）的统一评估。

使用方法:
    # 评估PT模型
    python evaluate_model.py --model_path /path/to/model.pt --motion_config /path/to/motion_config.yaml --task g1_stu_future
    
    # 评估ONNX模型  
    python evaluate_model.py --model_path /path/to/model.onnx --motion_config /path/to/motion_config.yaml --task g1_stu_future
    
    # 指定输出目录和设备
    python evaluate_model.py --model_path model.pt --motion_config motion_config.yaml --device cuda:0 --num_envs 256 --output_dir ./eval_results
"""

# 注意：torch 必须在 isaacgym 之后导入
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


# ============================================================================
# 配置和工具函数
# ============================================================================

def parse_motion_config(config_path: str) -> tuple:
    """解析 motion config YAML，按第一级目录分组动作"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    root_path = config.get('root_path', '')
    motions = config.get('motions', [])
    
    motions_by_group = defaultdict(list)
    for i, motion in enumerate(motions):
        file_path = motion.get('file', '')
        if file_path:
            # 按第一级目录分组，如 "AMASS_numpy123/..." -> group: AMASS_numpy123
            parts = file_path.split('/')
            group_name = parts[0] if len(parts) > 0 else 'unknown'
            motion['idx'] = i  # 保存原始索引
            motions_by_group[group_name].append(motion)
    
    return root_path, dict(motions_by_group)


def extract_model_info(model_path: str) -> dict:
    """从模型路径提取模型信息"""
    path = Path(model_path)
    model_name = path.stem  # 去掉扩展名
    model_type = path.suffix.lower()  # .pt 或 .onnx
    
    # 尝试提取训练步数，如 model_15000.pt -> 15000
    steps = None
    if '_'.join(model_name.split('_')[-1:]).isdigit():
        steps = int(model_name.split('_')[-1])

    task_name = None
    run_name = None
    checkpoint_name = model_name

    path_parts = path.parts
    if 'logs' in path_parts:
        logs_idx = path_parts.index('logs')
        # 兼容 .../logs/<task>/<run>/<checkpoint>.pt
        if len(path_parts) > logs_idx + 3:
            task_name = path_parts[logs_idx + 1]
            run_name = path_parts[logs_idx + 2]

    return {
        'path': str(path.absolute()),
        'name': model_name,
        'type': 'pt' if model_type == '.pt' else 'onnx' if model_type == '.onnx' else 'unknown',
        'steps': steps,
        'task_name': task_name,
        'run_name': run_name,
        'checkpoint_name': checkpoint_name,
    }


def generate_output_filename(model_info: dict, output_dir: str) -> str:
    """生成规范命名的输出文件"""
    task_name = model_info.get('task_name')
    run_name = model_info.get('run_name')
    checkpoint_name = model_info.get('checkpoint_name') or model_info['name']

    if task_name and run_name:
        filename = f"{task_name}_{run_name}_{checkpoint_name}.json"
    else:
        filename = f"{checkpoint_name}.json"
    
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def resolve_task_name(cli_task: str, model_info: dict) -> str:
    """解析最终使用的 task 名称。

    优先使用用户显式传入的 task；但如果仍是默认的 g1_stu_future，
    且模型路径中能解析出 logs/<task>/...，则自动切换到路径中的 task，
    以兼容 moe/transformer 等变体模型。
    """
    inferred_task = model_info.get('task_name')
    if inferred_task and (not cli_task or cli_task == 'g1_stu_future'):
        if cli_task and cli_task != inferred_task:
            print(f"  Auto-detected task from model path: {inferred_task} (override {cli_task})")
        return inferred_task
    return cli_task


def detect_sonic_model(model_path: str) -> bool:
    """根据模型路径判断是否为 sonic_pd 训练的模型。"""
    return 'sonic' in str(model_path).lower()


def resolve_sonic_pd_mode(model_path: str, cli_override):
    """解析评估时是否启用 sonic_pd。

    优先级:
    1. --sonic_pd / --no_sonic_pd 显式覆盖
    2. 模型路径自动检测
    """
    is_sonic_model = detect_sonic_model(model_path)

    if cli_override is True:
        enable_sonic_pd = True
        source = 'cli_force_on'
    elif cli_override is False:
        enable_sonic_pd = False
        source = 'cli_force_off'
    elif is_sonic_model:
        enable_sonic_pd = True
        source = 'path_auto'
    else:
        enable_sonic_pd = False
        source = 'default_off'

    return {
        'is_sonic_model': is_sonic_model,
        'enable_sonic_pd': enable_sonic_pd,
        'preserve_ankle_obs': enable_sonic_pd,
        'source': source,
    }


def print_sonic_pd_summary(mode: dict):
    """打印 sonic_pd 检测与生效状态。"""
    source_text = {
        'path_auto': 'auto-detected from model path',
        'cli_force_on': 'forced by --sonic_pd',
        'cli_force_off': 'forced by --no_sonic_pd',
        'default_off': 'default disabled',
    }.get(mode.get('source'), 'unknown')

    detected_text = 'YES' if mode['is_sonic_model'] else 'NO'
    effective_text = 'ENABLED' if mode['enable_sonic_pd'] else 'DISABLED'
    ankle_text = 'PRESERVED' if mode['preserve_ankle_obs'] else 'MASKED'
    pd_text = 'SONIC-derived G1 gains' if mode['enable_sonic_pd'] else 'default env gains'

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        table = Table.grid(padding=(0, 2))
        table.add_row('Detected SONIC model', f"[bold]{detected_text}[/bold]")
        table.add_row('Effective SONIC PD', f"[bold]{effective_text}[/bold]")
        table.add_row('PD gains', pd_text)
        table.add_row('Ankle observations', ankle_text)
        table.add_row('Decision source', source_text)

        border_style = 'green' if mode['enable_sonic_pd'] else 'yellow'
        title = 'SONIC PD Evaluation Mode'
        console.print(Panel(table, title=title, border_style=border_style))
    except ImportError:
        lines = [
            'SONIC PD Evaluation Mode',
            f'Detected SONIC model : {detected_text}',
            f'Effective SONIC PD   : {effective_text}',
            f'PD gains             : {pd_text}',
            f'Ankle observations   : {ankle_text}',
            f'Decision source      : {source_text}',
        ]
        width = max(len(line) for line in lines)
        border = '+' + '-' * (width + 2) + '+'
        print(border)
        for line in lines:
            print(f"| {line.ljust(width)} |")
        print(border)


# ============================================================================
# 模型加载器
# ============================================================================

class OnnxPolicyWrapper:
    """ONNX模型包装器"""
    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor):
        import torch
        import numpy as np
        obs_np = obs_tensor.cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        return torch.from_numpy(outputs[self.output_index])


def load_onnx_model(model_path: str, device: str):
    """加载ONNX模型"""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Please install: pip install onnxruntime-gpu")
    
    providers = []
    if device.startswith('cuda') and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    return OnnxPolicyWrapper(session, input_name)


class PolicyModelWrapper:
    """包装器，确保模型输入输出与环境的obs兼容"""
    def __init__(self, actor_critic, model_obs_dim, env_obs_dim, device):
        self.actor_critic = actor_critic
        self.model_obs_dim = int(model_obs_dim)
        self.env_obs_dim = env_obs_dim
        self.device = device
        
    def _prepare_obs(self, obs):
        """调整obs维度以匹配模型输入"""
        import torch
        current_dim = obs.shape[1]
        
        if current_dim == self.model_obs_dim:
            return obs
        elif current_dim > self.model_obs_dim:
            # 截断多余的维度
            return obs[:, :self.model_obs_dim]
        else:
            # 对不足的维度补零，兼容相同接口但不同训练配置的模型
            padding = torch.zeros(obs.shape[0], self.model_obs_dim - current_dim, device=obs.device, dtype=obs.dtype)
            return torch.cat([obs, padding], dim=1)
    
    def __call__(self, obs):
        prepared_obs = self._prepare_obs(obs)
        return self.actor_critic.act_inference(prepared_obs)


def _infer_temporal_steps(state_dict: dict, prefix: str, default: int = 1) -> int:
    """根据 encoder 的卷积结构推断时间步数。"""
    conv0_key = f"{prefix}.conv_layers.0.weight"
    conv2_key = f"{prefix}.conv_layers.2.weight"
    conv4_key = f"{prefix}.conv_layers.4.weight"
    if conv0_key not in state_dict:
        return 1

    kernel0 = state_dict[conv0_key].shape[-1]
    kernel2 = state_dict[conv2_key].shape[-1] if conv2_key in state_dict else None
    has_third_conv = conv4_key in state_dict

    if kernel0 == 4 and kernel2 == 2:
        return 10
    if kernel0 == 6 and kernel2 == 4:
        return 20
    if kernel0 == 8 and kernel2 == 5 and has_third_conv:
        return 50
    return default


def _infer_mlp_hidden_dims(state_dict: dict, prefix: str) -> list:
    """从顺序 MLP 的线性层权重中推断隐藏层维度。"""
    linear_layers = []
    prefix_with_dot = f"{prefix}."

    for key, value in state_dict.items():
        if not key.startswith(prefix_with_dot) or not key.endswith(".weight"):
            continue
        suffix = key[len(prefix_with_dot):]
        module_idx = suffix.split(".")[0]
        if not module_idx.isdigit():
            continue
        # Linear weight is 2D; LayerNorm/bias-like weights are 1D.
        if value.ndim != 2:
            continue
        linear_layers.append((int(module_idx), value.shape[0]))

    linear_layers.sort(key=lambda item: item[0])
    if len(linear_layers) <= 1:
        return []

    return [out_dim for _, out_dim in linear_layers[:-1]]


def _infer_has_layer_norm(state_dict: dict, prefix: str) -> bool:
    prefix_with_dot = f"{prefix}."
    for key, value in state_dict.items():
        if not key.startswith(prefix_with_dot) or not key.endswith(".weight"):
            continue
        suffix = key[len(prefix_with_dot):]
        module_idx = suffix.split(".")[0]
        if not module_idx.isdigit():
            continue
        if value.ndim == 1:
            return True
    return False


def _infer_num_actions(state_dict: dict, fallback: int) -> int:
    if 'std' in state_dict:
        return int(state_dict['std'].numel())
    for key in ('actor.actor_backbone.6.bias', 'actor.actor_backbone.8.bias', 'actor.actor_backbone.4.bias'):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return int(fallback)


def _detect_policy_class_name(state_dict: dict, fallback: str) -> str:
    has_future = any(k.startswith('actor.history_encoder.') or k.startswith('actor.future_encoder.') for k in state_dict)
    has_mimic = any(k.startswith('actor.motion_encoder.') for k in state_dict)
    if has_future:
        return 'ActorCriticFuture'
    if has_mimic:
        return 'ActorCriticMimic'
    return fallback


def _filter_matching_state_dict(model, state_dict: dict):
    """过滤 shape 不匹配的参数，避免 strict=False 仍因 size mismatch 抛错。"""
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    unexpected = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value
    return filtered, skipped, unexpected


def _infer_future_steps(num_future_observations: int, future_encoder_input: int, fallback: int = 1) -> int:
    """根据 future obs 总维度和 future encoder 首层输入维度反推 future steps。"""
    if num_future_observations <= 0:
        return 1

    candidates = []
    for steps in range(1, num_future_observations + 1):
        if num_future_observations % steps != 0:
            continue
        single_future_obs = num_future_observations // steps
        if (single_future_obs - 1) * steps == future_encoder_input:
            candidates.append(steps)

    if not candidates:
        return fallback

    if fallback in candidates:
        return fallback
    return min(candidates)


def load_pt_model(model_path: str, env, device: str, task_name: str):
    """加载PyTorch PT模型 - 从state dict和训练配置推断所有参数"""
    import torch
    from rsl_rl.modules import ActorCritic, ActorCriticMimic, ActorCriticFuture
    from legged_gym.gym_utils.helpers import class_to_dict
    from legged_gym.gym_utils import task_registry
    
    # 获取训练配置
    _, train_cfg = task_registry.get_cfgs(name=task_name)
    policy_cfg = class_to_dict(train_cfg.policy)
    policy_class_name = getattr(train_cfg.runner, 'policy_class_name', 'ActorCritic')
    
    print(f"Loading model: {model_path}")
    print(f"  Config policy class: {policy_class_name}")
    
    # 从state dict推断参数
    loaded_dict = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = loaded_dict.get('model_state_dict', loaded_dict)
    
    # 移除'module.'前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    policy_class_name = _detect_policy_class_name(new_state_dict, policy_class_name)
    print(f"  Checkpoint policy class: {policy_class_name}")

    # 检测架构类型
    has_transformer = any('transformer' in k for k in new_state_dict.keys())
    has_moe = any('experts' in k or 'gating' in k for k in new_state_dict.keys())
    is_mlp = not has_transformer and not has_moe and any('actor_backbone.0.weight' in k for k in new_state_dict.keys())
    
    print(f"  Architecture: {'Transformer' if has_transformer else 'MoE' if has_moe else 'MLP'}")
    
    # 从 encoder 推断基本维度
    me_weight = new_state_dict.get('actor.motion_encoder.encoder.0.weight')
    single_motion_obs = me_weight.shape[1] if me_weight is not None else 35
    print(f"  Single motion obs: {single_motion_obs}")
    
    he_weight = new_state_dict.get('actor.history_encoder.encoder.0.weight')
    single_history_obs = he_weight.shape[1] if he_weight is not None else 127
    print(f"  Single history obs: {single_history_obs}")
    
    me_linear = new_state_dict.get('actor.motion_encoder.linear_output.weight')
    motion_latent_dim = me_linear.shape[0] if me_linear is not None else 64
    print(f"  Motion latent dim: {motion_latent_dim}")
    
    he_linear = new_state_dict.get('actor.history_encoder.linear_output.weight')
    history_latent_dim = he_linear.shape[0] if he_linear is not None else 64
    print(f"  History latent dim: {history_latent_dim}")
    
    # 推断 motion/history/future steps
    motion_steps = _infer_temporal_steps(new_state_dict, 'actor.motion_encoder', default=1)
    print(f"  Motion steps: {motion_steps}")
    num_motion_obs = single_motion_obs * motion_steps
    
    history_steps = _infer_temporal_steps(new_state_dict, 'actor.history_encoder', default=getattr(env.cfg.env, 'history_len', 0))
    print(f"  History steps: {history_steps}")
    
    # 推断 hidden dims / layer norm（MLP）
    actor_hidden_dims = _infer_mlp_hidden_dims(new_state_dict, 'actor.actor_backbone') if is_mlp else []
    actor_has_layer_norm = _infer_has_layer_norm(new_state_dict, 'actor.actor_backbone')
    critic_has_layer_norm = _infer_has_layer_norm(new_state_dict, 'critic')
    if is_mlp:
        print(f"  Actor hidden dims: {actor_hidden_dims}")
        print(f"  Actor layer norm: {actor_has_layer_norm}")
    
    # 推断 Transformer 参数
    if has_transformer:
        input_emb_key = 'actor.actor_backbone.input_embedding.weight'
        if input_emb_key in new_state_dict:
            d_model = new_state_dict[input_emb_key].shape[0]
            backbone_input_dim = new_state_dict[input_emb_key].shape[1]
            print(f"  Transformer d_model: {d_model}, backbone input: {backbone_input_dim}")
        else:
            d_model = 256
            backbone_input_dim = 511
        
        num_layers = max([int(k.split('.')[4]) for k in new_state_dict.keys() 
                         if 'transformer.layers.' in k and k.split('.')[4].isdigit()], default=0) + 1
        print(f"  Transformer layers: {num_layers}")
        
        nhead = policy_cfg.get('nhead', 8)
        print(f"  Transformer nhead: {nhead}")
    
    # 推断 critic hidden dims
    critic_hidden_dims = _infer_mlp_hidden_dims(new_state_dict, 'critic')
    print(f"  Critic hidden dims: {critic_hidden_dims}")
    print(f"  Critic layer norm: {critic_has_layer_norm}")

    actor_input_dim = new_state_dict.get('actor.actor_backbone.0.weight')
    actor_input_dim = actor_input_dim.shape[1] if actor_input_dim is not None else env.num_obs
    if has_transformer:
        actor_input_dim = locals().get('backbone_input_dim', actor_input_dim)
    critic_input_dim = new_state_dict.get('critic.0.weight')
    critic_input_dim = critic_input_dim.shape[1] if critic_input_dim is not None else (env.num_privileged_obs or env.num_obs)
    num_actions = _infer_num_actions(new_state_dict, env.num_actions)
    print(f"  Num actions: {num_actions}")
    
    # 使用推断的参数创建模型
    PolicyClass = eval(policy_class_name)
    
    if "Future" in policy_class_name:
        fe_linear = new_state_dict.get('actor.future_encoder.encoder.6.weight')
        future_latent_dim = fe_linear.shape[0] if fe_linear is not None else policy_cfg.get('future_latent_dim', 64)

        num_observations = env.num_obs
        num_critic_observations = critic_input_dim - motion_latent_dim - single_motion_obs + num_motion_obs

        # Future/MoE/Transformer 的 actor_backbone 输入不是原始 obs 维度，不能用它反推 n_proprio。
        # checkpoint 中 history_encoder.encoder.0.weight 的输入维度就是
        # 单帧历史观测维度 = num_motion_observations + n_proprio。
        history_input_dim = single_history_obs if he_weight is not None else None
        if history_input_dim is not None:
            n_proprio = history_input_dim - num_motion_obs
        else:
            n_proprio = getattr(env.cfg.env, 'n_proprio', None)
            if n_proprio is None:
                raise RuntimeError("Failed to infer n_proprio: history encoder weights not found and env cfg has no n_proprio")
        if n_proprio < 0:
            raise RuntimeError(
                f"Inferred invalid n_proprio={n_proprio} from checkpoint. "
                f"history_input_dim={history_input_dim}, num_motion_obs={num_motion_obs}"
            )

        current_obs_dim = num_motion_obs + n_proprio
        history_obs_dim = current_obs_dim * history_steps
        num_future_observations = max(0, num_observations - current_obs_dim - history_obs_dim)

        future_encoder_input = new_state_dict.get('actor.future_encoder.encoder.0.weight')
        future_encoder_input = future_encoder_input.shape[1] if future_encoder_input is not None else 0
        cfg_future_steps = len(getattr(env.cfg.env, 'tar_motion_steps_future', [])) or 1
        num_future_steps = _infer_future_steps(num_future_observations, future_encoder_input, fallback=cfg_future_steps)
        print(f"  Inferred n_proprio: {n_proprio}")
        print(f"  Inferred num_observations: {num_observations}")
        print(f"  Inferred num_critic_observations: {num_critic_observations}")
        print(f"  Inferred num_future_observations: {num_future_observations}")
        print(f"  Inferred num_future_steps: {num_future_steps}")
        
        actor_kwargs = {
            'num_observations': num_observations,
            'num_critic_observations': num_critic_observations,
            'num_motion_observations': num_motion_obs,
            'num_motion_steps': motion_steps,
            'num_priop_observations': n_proprio,
            'num_history_steps': history_steps,
            'num_actions': num_actions,
            'num_future_observations': num_future_observations,
            'num_future_steps': num_future_steps,
            'critic_hidden_dims': critic_hidden_dims,
            'activation': policy_cfg.get('activation', 'silu'),
            'motion_latent_dim': motion_latent_dim,
            'history_latent_dim': history_latent_dim,
            'future_latent_dim': future_latent_dim,
            'layer_norm': actor_has_layer_norm or critic_has_layer_norm,
        }
        
        if has_transformer:
            actor_kwargs['use_transformer'] = True
            actor_kwargs['d_model'] = d_model
            actor_kwargs['nhead'] = nhead
            actor_kwargs['num_transformer_layers'] = num_layers
            actor_kwargs['transformer_dropout'] = policy_cfg.get('transformer_dropout', 0.1)
        elif has_moe:
            actor_kwargs['use_moe'] = True
            actor_kwargs['num_experts'] = policy_cfg.get('num_experts', 4)
            actor_kwargs['expert_hidden_dims'] = policy_cfg.get('expert_hidden_dims', [256, 256])
            actor_kwargs['moe_topk'] = policy_cfg.get('moe_topk', 2)
            actor_kwargs['actor_hidden_dims'] = actor_hidden_dims or policy_cfg.get('actor_hidden_dims', [512, 512, 256])
        else:
            actor_kwargs['actor_hidden_dims'] = actor_hidden_dims or policy_cfg.get('actor_hidden_dims', [512, 512, 256])
        
        actor_critic = PolicyClass(**actor_kwargs).to(device)
        
    elif "Mimic" in policy_class_name:
        num_observations = actor_input_dim - motion_latent_dim - single_motion_obs + num_motion_obs
        num_critic_observations = critic_input_dim - motion_latent_dim - single_motion_obs + num_motion_obs
        print(f"  Inferred num_observations: {num_observations}")
        print(f"  Inferred num_critic_observations: {num_critic_observations}")
        actor_critic = PolicyClass(
            num_observations=num_observations,
            num_critic_observations=num_critic_observations,
            num_motion_observations=num_motion_obs,
            num_motion_steps=motion_steps,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims or policy_cfg.get('actor_hidden_dims', [256, 256, 256]),
            critic_hidden_dims=critic_hidden_dims or policy_cfg.get('critic_hidden_dims', [256, 256, 256]),
            activation=policy_cfg.get('activation', 'elu'),
            motion_latent_dim=motion_latent_dim,
            init_noise_std=policy_cfg.get('init_noise_std', 1.0),
            layer_norm=actor_has_layer_norm or critic_has_layer_norm,
        ).to(device)
    else:
        n_proprio = getattr(env.cfg.env, 'n_proprio', 92)
        actor_critic = PolicyClass(
            num_prop=n_proprio,
            num_critic_obs=critic_input_dim,
            num_priv_latent=getattr(env.cfg.env, 'n_priv_latent', 0),
            num_hist=history_steps,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims or policy_cfg.get('actor_hidden_dims', [256, 256, 256]),
            critic_hidden_dims=critic_hidden_dims or policy_cfg.get('critic_hidden_dims', [256, 256, 256]),
            activation=policy_cfg.get('activation', 'elu'),
            priv_encoder_dims=policy_cfg.get('priv_encoder_dims', [64, 64]),
        ).to(device)
    
    # 只加载 shape 完全匹配的权重，避免 strict=False 仍因 size mismatch 抛错
    filtered_state_dict, skipped_mismatch, filtered_unexpected = _filter_matching_state_dict(actor_critic, new_state_dict)
    if skipped_mismatch:
        mismatch_lines = [
            f"    - {key}: checkpoint {src_shape} != model {dst_shape}"
            for key, src_shape, dst_shape in skipped_mismatch[:20]
        ]
        remaining = len(skipped_mismatch) - len(mismatch_lines)
        if remaining > 0:
            mismatch_lines.append(f"    ... and {remaining} more")
        raise RuntimeError(
            "Checkpoint weights do not fully match the reconstructed model. "
            "Aborting evaluation to avoid testing a partially loaded model.\n"
            + "\n".join(mismatch_lines)
        )
    missing_keys, unexpected_keys = actor_critic.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print(f"  Warning: {len(missing_keys)} missing keys")
        for k in missing_keys[:5]:
            print(f"    - {k}")
    all_unexpected_keys = list(unexpected_keys) + filtered_unexpected
    if all_unexpected_keys:
        print(f"  Warning: {len(all_unexpected_keys)} unexpected keys")
        for k in all_unexpected_keys[:5]:
            print(f"    - {k}")
    actor_critic.eval()
    print(f"Model loaded: {sum(p.numel() for p in actor_critic.parameters())/1e6:.2f}M parameters")
    
    # 返回推理函数
    model_obs_dim = getattr(actor_critic, 'num_observations', env.num_obs)
    return PolicyModelWrapper(actor_critic, model_obs_dim=model_obs_dim, env_obs_dim=env.num_obs, device=device)


def load_model(model_path: str, env, device: str, task_name: str):
    """自动检测并加载模型"""
    model_type = Path(model_path).suffix.lower()
    
    if model_type == '.onnx':
        return load_onnx_model(model_path, device), None
    elif model_type == '.pt':
        policy = load_pt_model(model_path, env, device, task_name)
        return policy, None  # normalizer处理在eval中
    else:
        raise ValueError(f"Unsupported model format: {model_type}. Supported: .pt, .onnx")


# ============================================================================
# 环境配置
# ============================================================================

def setup_eval_env_cfg(env_cfg, motion_config_path: str, num_envs: int):
    """配置评估模式的环境参数"""
    env_cfg.motion.motion_file = motion_config_path
    env_cfg.env.num_envs = num_envs
    
    # 评估模式：关闭噪声和随机化
    env_cfg.env.debug_viz = False
    env_cfg.env.episode_length_s = 60
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    env_cfg.domain_rand.randomize_motor = False
    
    if hasattr(env_cfg, 'motion'):
        env_cfg.motion.motion_curriculum = False
    
    env_cfg.env.rand_reset = False
    
    return env_cfg


def reload_env_motion_group(env, motion_config_path: str):
    """在不重建 sim 的情况下切换动作库。"""
    import torch

    env.cfg.motion.motion_file = motion_config_path
    env._load_motions()
    env._init_motion_buffers()

    num_motions = env._motion_lib.num_motions()
    env.motion_difficulty = torch.ones((num_motions), device=env.device, dtype=torch.float32, requires_grad=False) * 100.0
    env.mean_motion_difficulty = 100.0
    env.motion_termination_dist = torch.ones((num_motions), device=env.device, dtype=torch.float32, requires_grad=False) * env._pose_termination_dist
    env.motion_names = env._motion_lib.get_motion_names()
    if hasattr(env, "max_key_body_error"):
        env.max_key_body_error = torch.zeros((num_motions), device=env.device, dtype=torch.float32, requires_grad=False)

    env.max_episode_length_s = env._get_max_motion_len().item()
    env.max_episode_length = np.ceil(env.max_episode_length_s / env.dt)

    all_env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_env_ids)


# ============================================================================
# 评估核心逻辑 - 分批加载动作到GPU
# ============================================================================

def evaluate_single_motion_group(
    env,
    policy,
    motion_config_path: str,
    group_motions: list,
    device: str,
    max_steps: int = 5000,
    collect_metrics: bool = True
):
    """评估单个动作库的所有动作，动作数据按需加载"""
    import torch
    from isaacgym.torch_utils import quat_rotate_inverse
    from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
    from legged_gym.envs.base.legged_robot import euler_from_quaternion

    def _refresh_metric_state():
        """首次 reset 后尚未经历 post_physics_step，需要手动补齐误差函数依赖的姿态状态。"""
        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
        env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
        env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    
    motion_lib = env._motion_lib
    num_motions = motion_lib.num_motions()
    motion_names = motion_lib.get_motion_names()
    motion_files = motion_lib._motion_files
    
    num_envs = env.num_envs
    dt = env.dt
    quality_metric_fns = {
        metric_name: getattr(env, f"_{metric_name}")
        for metric_name in QUALITY_METRIC_SPECS
        if hasattr(env, f"_{metric_name}")
    }
    
    results = []
    all_motion_ids = list(range(num_motions))
    all_env_ids = torch.arange(num_envs, device=device)
    
    # 获取所有动作长度
    motion_ids_tensor = torch.tensor(all_motion_ids, device=device, dtype=torch.int64)
    motion_lengths = motion_lib.get_motion_length(motion_ids_tensor)
    
    # 分批处理（每批最多num_envs个）
    for batch_start in range(0, num_motions, num_envs):
        batch_end = min(batch_start + num_envs, num_motions)
        batch_motion_ids = list(range(batch_start, batch_end))
        batch_size = len(batch_motion_ids)
        
        # 准备motion ids tensor
        batch_motion_ids_tensor = [all_motion_ids[i] for i in batch_motion_ids]
        motion_ids_tensor = torch.tensor(batch_motion_ids_tensor, device=device, dtype=torch.int64)
        
        # 填充（如果不足num_envs）
        if batch_size < num_envs:
            current_lens = motion_lengths[motion_ids_tensor]
            min_idx = torch.argmin(current_lens).item()
            padding_id = batch_motion_ids_tensor[min_idx]
            padding = torch.full((num_envs - batch_size,), padding_id, device=device, dtype=torch.int64)
            motion_ids_tensor = torch.cat([motion_ids_tensor, padding])
        
        # 重置环境
        HumanoidMimic.reset_idx(env, all_env_ids, motion_ids=motion_ids_tensor)
        obs = env.get_observations()
        
        # 追踪变量
        episode_lengths = torch.zeros(num_envs, device=device)
        batch_motion_lengths = motion_lengths[motion_ids_tensor[:batch_size]]
        done_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        
        # 动态max_steps
        max_motion_length = batch_motion_lengths.max().item()
        actual_max_steps = min(max_steps, int(max_motion_length / dt * 1.2) + 10)
        
        # 收集指标（可选）
        if collect_metrics:
            episode_metrics = defaultdict(lambda: torch.zeros(num_envs, device=device))
        quality_metric_sums = {
            key: torch.zeros(num_envs, device=device)
            for key in quality_metric_fns
        }
        quality_metric_counts = {
            key: torch.zeros(num_envs, device=device)
            for key in quality_metric_fns
        }
        skipped_quality_metrics = set()
        
        # 运行模拟
        for _ in range(actual_max_steps):
            active_batch_mask = ~done_mask[:batch_size]
            active_idx = torch.nonzero(active_batch_mask, as_tuple=False).squeeze(-1)
            if active_idx.numel() == 0:
                break

            with torch.no_grad():
                if quality_metric_fns:
                    if not hasattr(env, 'yaw'):
                        _refresh_metric_state()
                    for key, metric_fn in quality_metric_fns.items():
                        metric_values = metric_fn()
                        if isinstance(metric_values, tuple):
                            metric_values = metric_values[0]
                        if not isinstance(metric_values, torch.Tensor):
                            metric_values = torch.as_tensor(metric_values, device=device)
                        if metric_values.ndim == 0 or metric_values.shape[0] != num_envs:
                            if key not in skipped_quality_metrics:
                                print(
                                    f"  Warning: skip quality metric '{key}' because it is not per-env "
                                    f"(shape={tuple(metric_values.shape)})"
                                )
                                skipped_quality_metrics.add(key)
                            continue
                        quality_metric_counts[key][active_idx] += 1
                        quality_metric_sums[key][active_idx] += metric_values[active_idx]

                # 只对仍在评估中的env执行策略推理，done后保持零动作。
                active_obs = obs.index_select(0, active_idx)
                valid_actions = policy(active_obs.detach())

                if isinstance(valid_actions, torch.Tensor):
                    valid_actions = valid_actions.to(device)
                else:
                    valid_actions = torch.from_numpy(valid_actions).to(device)

                actions = torch.zeros((num_envs, env.num_actions), device=device, dtype=valid_actions.dtype)
                actions.index_copy_(0, active_idx, valid_actions)
            
            # 环境步进
            obs, _, _, dones, infos = env.step(actions)
            
            # 更新episode长度（只对未完成的）
            episode_lengths[~done_mask] += 1
            done_mask = done_mask | dones
            
            # 收集指标
            if collect_metrics and 'episode' in infos:
                for key, value in infos['episode'].items():
                    if isinstance(value, torch.Tensor):
                        if value.ndim == 0:
                            # 标量是reset env的均值，不能安全映射回单个动作结果。
                            continue
                        if value.shape[0] == num_envs:
                            value_batch = value[:batch_size]
                        elif value.shape[0] == batch_size:
                            value_batch = value
                        else:
                            continue

                        episode_metrics[key][active_idx] += value_batch[active_idx]
            
            # 提前退出条件
            if done_mask[:batch_size].all():
                break
        
        # 计算结果
        actual_times = episode_lengths[:batch_size] * dt
        completion_rates = actual_times / batch_motion_lengths
        completion_rates = torch.clamp(completion_rates, 0.0, 1.0)
        completion_scores = completion_rates * 100.0
        quality_metrics = {}
        if quality_metric_fns:
            for key, values in quality_metric_sums.items():
                counts = quality_metric_counts[key][:batch_size]
                if not torch.any(counts > 0):
                    continue
                valid_counts = torch.clamp(counts, min=1.0)
                quality_metrics[key] = values[:batch_size] / valid_counts
        
        # 保存结果
        for i, local_idx in enumerate(batch_motion_ids):
            result = {
                'motion_idx': batch_motion_ids_tensor[i],
                'motion_name': motion_names[batch_motion_ids_tensor[i]],
                'motion_file': motion_files[batch_motion_ids_tensor[i]],
                'completion_rate': completion_rates[i].item(),
                'completion_score': completion_scores[i].item(),
                'actual_time': actual_times[i].item(),
                'motion_length': batch_motion_lengths[i].item(),
            }
            
            # 添加指标
            result_metrics = {}
            if collect_metrics:
                for key, values in episode_metrics.items():
                    result_metrics[key] = values[i].item()
            for key, values in quality_metrics.items():
                result_metrics[key] = values[i].item()
            if result_metrics:
                result['metrics'] = result_metrics

            result['quality_score'] = compute_quality_score(result_metrics, fallback=result['completion_score'])
            result['ranking_score'] = compute_ranking_score(
                result['completion_score'],
                result['quality_score'],
            )
            
            results.append(result)
        
    return results


# ============================================================================
# 结果统计和输出
# ============================================================================

QUALITY_METRIC_SPECS = {
    'error_tracking_keybody_pos': {'threshold': 0.35, 'weight': 0.30},
    'error_tracking_root_translation': {'threshold': 0.30, 'weight': 0.20},
    'error_tracking_root_rotation': {'threshold': 1.00, 'weight': 0.15},
    'error_tracking_joint_dof': {'threshold': 0.35, 'weight': 0.15},
    'error_tracking_root_vel': {'threshold': 1.50, 'weight': 0.10},
    'error_tracking_joint_vel': {'threshold': 2.50, 'weight': 0.05},
    'error_tracking_root_ang_vel': {'threshold': 3.00, 'weight': 0.05},
}

RANKING_SCORE_WEIGHTS = {
    'completion_score': 0.70,
    'quality_score': 0.30,
}


def normalize_error_score(value: float, threshold: float) -> float:
    """Map an absolute tracking error to a 0-100 score using a fixed threshold."""
    if threshold <= 0:
        return 0.0
    return float(np.clip(1.0 - value / threshold, 0.0, 1.0) * 100.0)


def compute_quality_score(metrics: dict, fallback: float = 0.0) -> float:
    """Aggregate available tracking errors into a stable 0-100 quality score."""
    weighted_score = 0.0
    total_weight = 0.0

    for key, spec in QUALITY_METRIC_SPECS.items():
        if key not in metrics:
            continue
        weighted_score += normalize_error_score(metrics[key], spec['threshold']) * spec['weight']
        total_weight += spec['weight']

    if total_weight == 0.0:
        return float(fallback)

    return float(weighted_score / total_weight)


def compute_ranking_score(completion_score: float, quality_score: float) -> float:
    """Combine completion and motion quality into a ranking-friendly score."""
    return float(
        completion_score * RANKING_SCORE_WEIGHTS['completion_score']
        + quality_score * RANKING_SCORE_WEIGHTS['quality_score']
    )


def compute_statistics(values: list) -> dict:
    """计算统计信息"""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(values)
    }


def aggregate_results(results: list, motions_by_group: dict) -> dict:
    """汇总结果，按动作库分组统计"""
    # 整体统计
    overall_stats = {
        'total_motions': len(results)
    }
    for score_key in ('completion_score', 'quality_score', 'ranking_score'):
        overall_stats[score_key] = compute_statistics([r[score_key] for r in results])
    
    # 收集所有指标
    all_metrics = defaultdict(list)
    for r in results:
        if 'metrics' in r:
            for key, value in r['metrics'].items():
                all_metrics[key].append(value)
    
    for key, values in all_metrics.items():
        overall_stats[key] = compute_statistics(values)
    
    # 按组统计（结果中已有 motion_group 字段）
    group_stats = {}
    
    # 按组分组结果
    group_results = defaultdict(list)
    for r in results:
        group_name = r.get('motion_group', 'unknown')
        group_results[group_name].append(r)
    
    # 计算每组的统计
    for group_name, group_res in group_results.items():
        group_stat = {
            'count': len(group_res),
        }
        for score_key in ('completion_score', 'quality_score', 'ranking_score'):
            group_stat[score_key] = compute_statistics([r[score_key] for r in group_res])
        
        # 组内指标统计
        group_metrics = defaultdict(list)
        for r in group_res:
            if 'metrics' in r:
                for key, value in r['metrics'].items():
                    group_metrics[key].append(value)
        
        for key, values in group_metrics.items():
            group_stat[key] = compute_statistics(values)
        
        group_stats[group_name] = group_stat
    
    return overall_stats, group_stats


def print_results_table(overall_stats: dict, group_stats: dict):
    """打印结果表格到控制台"""
    try:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]                  模型测评结果汇总                          [/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
        
        # 整体统计
        console.print("[bold green]整体性能:[/bold green]")
        completion_stats = overall_stats['completion_score']
        quality_stats = overall_stats['quality_score']
        ranking_stats = overall_stats['ranking_score']
        console.print(f"  完成度得分: {completion_stats['mean']:.2f} ± {completion_stats['std']:.2f}")
        console.print(f"  质量得分: {quality_stats['mean']:.2f} ± {quality_stats['std']:.2f}")
        console.print(f"  排序得分: {ranking_stats['mean']:.2f} ± {ranking_stats['std']:.2f}")
        console.print(f"  完成度范围: [{completion_stats['min']:.2f}, {completion_stats['max']:.2f}]")
        console.print(f"  测试动作数: {overall_stats['total_motions']}\n")
        
        # 分组统计表
        table = Table(title="按动作库分组统计")
        table.add_column("动作库", style="cyan")
        table.add_column("数量", justify="right")
        table.add_column("完成度均值", justify="right")
        table.add_column("质量均值", justify="right")
        table.add_column("排序分均值", justify="right")
        table.add_column("完成度标准差", justify="right")
        
        for group_name in sorted(group_stats.keys()):
            stats = group_stats[group_name]
            completion = stats['completion_score']
            quality = stats['quality_score']
            ranking = stats['ranking_score']
            table.add_row(
                group_name,
                str(stats['count']),
                f"{completion['mean']:.2f}",
                f"{quality['mean']:.2f}",
                f"{ranking['mean']:.2f}",
                f"{completion['std']:.2f}",
            )
        
        console.print(table)
        
    except ImportError:
        # 如果rich不可用，使用简单打印
        print("\n" + "=" * 60)
        print("模型测评结果汇总")
        print("=" * 60)
        
        completion_stats = overall_stats['completion_score']
        quality_stats = overall_stats['quality_score']
        ranking_stats = overall_stats['ranking_score']
        print(f"\n整体完成度得分: {completion_stats['mean']:.2f} ± {completion_stats['std']:.2f}")
        print(f"整体质量得分: {quality_stats['mean']:.2f} ± {quality_stats['std']:.2f}")
        print(f"整体排序得分: {ranking_stats['mean']:.2f} ± {ranking_stats['std']:.2f}")
        print(f"完成度范围: [{completion_stats['min']:.2f}, {completion_stats['max']:.2f}]")
        print(f"测试动作数: {overall_stats['total_motions']}\n")
        
        print("按动作库分组统计:")
        print("-" * 60)
        print(f"{'动作库':<30} {'数量':>8} {'完成度均值':>12} {'质量均值':>10} {'排序均值':>10}")
        print("-" * 60)
        for group_name in sorted(group_stats.keys()):
            stats = group_stats[group_name]
            completion = stats['completion_score']
            quality = stats['quality_score']
            ranking = stats['ranking_score']
            print(
                f"{group_name:<30} {stats['count']:>8} "
                f"{completion['mean']:>12.2f} {quality['mean']:>10.2f} {ranking['mean']:>10.2f}"
            )


def save_results(output_path: str, model_info: dict, overall_stats: dict, 
                 group_stats: dict, motion_results: list):
    """保存结果到JSON文件"""
    result_data = {
        'model_info': model_info,
        'timestamp': datetime.now().isoformat(),
        'overall': overall_stats,
        'motion_groups': group_stats,
        'motion_results': motion_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


# ============================================================================
# 主函数 - 分批加载动作库
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TWIST2 通用模型测评脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径 (.pt 或 .onnx)')
    parser.add_argument('--motion_config', type=str, required=True, help='动作配置文件路径 (.yaml)')
    parser.add_argument('--task', type=str, default='g1_stu_future', help='任务名称')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--num_envs', type=int, default=256, help='并行环境数量')
    parser.add_argument('--max_steps', type=int, default=5000, help='最大模拟步数')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='输出目录')
    parser.add_argument('--headless', action='store_true', help='无头模式运行')
    parser.set_defaults(sonic_pd=None)
    sonic_pd_group = parser.add_mutually_exclusive_group()
    sonic_pd_group.add_argument(
        '--sonic_pd',
        dest='sonic_pd',
        action='store_true',
        help='显式启用 SONIC-derived G1 PD 参数，并保留脚踝观测输入',
    )
    sonic_pd_group.add_argument(
        '--no_sonic_pd',
        dest='sonic_pd',
        action='store_false',
        help='显式关闭 SONIC PD 自动识别，回退到默认 PD 参数和默认脚踝观测逻辑',
    )
    args = parser.parse_args()
    
    # 导入isaacgym（必须在torch之前！）
    from isaacgym import gymapi
    
    # 现在导入torch
    import torch
    import gc
    
    # 添加项目路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'legged_gym'))
    import legged_gym.envs
    from legged_gym.gym_utils import task_registry
    
    # 提取模型信息
    model_info = extract_model_info(args.model_path)
    args.task = resolve_task_name(args.task, model_info)
    sonic_pd_mode = resolve_sonic_pd_mode(model_info['path'], cli_override=args.sonic_pd)
    model_info['is_sonic_model'] = sonic_pd_mode['is_sonic_model']
    model_info['enable_sonic_pd'] = sonic_pd_mode['enable_sonic_pd']
    model_info['sonic_pd_source'] = sonic_pd_mode['source']
    print(f"\n{'='*60}")
    print(f"模型测评开始")
    print(f"{'='*60}")
    print(f"模型路径: {model_info['path']}")
    print(f"模型类型: {model_info['type']}")
    print(f"模型名称: {model_info['name']}")
    if model_info['steps']:
        print(f"训练步数: {model_info['steps']}")
    if sonic_pd_mode['is_sonic_model'] or args.sonic_pd is not None:
        print_sonic_pd_summary(sonic_pd_mode)
    
    # 解析动作配置
    print(f"\n解析动作配置...")
    root_path, motions_by_group = parse_motion_config(args.motion_config)
    total_motions = sum(len(m) for m in motions_by_group.values())
    print(f"发现 {total_motions} 个动作，分布在 {len(motions_by_group)} 个动作库中:")
    for group_name in sorted(motions_by_group.keys()):
        print(f"  - {group_name}: {len(motions_by_group[group_name])} 个动作")
    
    # 为每个动作库创建临时yaml配置
    import tempfile
    temp_dir = tempfile.mkdtemp()
    group_configs = {}
    for group_name, group_motions in motions_by_group.items():
        config = {
            'root_path': root_path,
            'motions': group_motions
        }
        config_path = os.path.join(temp_dir, f"{group_name}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        group_configs[group_name] = config_path
    
    # 创建环境参数类
    class Args:
        def __init__(self):
            self.task = args.task
            self.device = args.device
            self.headless = args.headless
            self.sonic_pd = sonic_pd_mode['enable_sonic_pd']
            self.use_jit = False
            self.record_video = False
            self.record_log = False
            self.jit_path = None
            self.teleop_mode = False
            self.num_envs = args.num_envs
            self.seed = None
            self.rows = None
            self.cols = None
            self.no_rand = False
            self.max_iterations = None
            self.resume = False
            self.experiment_name = None
            self.run_name = None
            self.load_run = None
            self.checkpoint = None
            self.fix_action_std = False
            self.teacher_exptid = None
            self.teacher_checkpoint = None
            self.eval_student = False
            self.config_overrides = {}
            self.physics_engine = gymapi.SIM_PHYSX
            self.use_gpu = True
            self.subscenes = 0
            self.use_gpu_pipeline = True
            self.num_threads = 0
            self.sim_device = args.device
            self.sim_device_type = 'cuda'
            self.compute_device_id = int(args.device.split(':')[-1]) if ':' in args.device else 0
            self.graphics_device_id = self.compute_device_id
            self.pipeline = 'gpu'
            self.rl_device = args.device
    
    # 逐个动作库进行评估，但整个流程只创建一个环境，后续原位切换动作库
    all_results = []
    start_time = time.time()
    
    # 按动作数量排序（少的优先）
    sorted_groups = sorted(group_configs.items(), key=lambda x: len(motions_by_group[x[0]]))

    # 先用第一个动作库创建环境
    first_group_name, first_config_path = sorted_groups[0]
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = setup_eval_env_cfg(env_cfg, first_config_path, args.num_envs)
    print(f"创建环境...")
    env_args = Args()
    env, _ = task_registry.make_env(name=args.task, args=env_args, env_cfg=env_cfg)

    print(f"加载模型...")
    policy, _ = load_model(args.model_path, env, args.device, args.task)
    print(f"模型加载成功")
    
    for group_idx, (group_name, config_path) in enumerate(sorted_groups):
        group_start_time = time.time()
        group_motions = motions_by_group[group_name]
        num_group_motions = len(group_motions)
        
        print(f"\n{'='*60}")
        print(f"[{group_idx+1}/{len(group_configs)}] 评估动作库: {group_name}")
        print(f"动作数量: {num_group_motions}")
        print(f"{'='*60}")

        if group_idx > 0:
            print(f"切换动作库...")
            reload_env_motion_group(env, config_path)
        
        # 执行评估
        print(f"开始评估 {num_group_motions} 个动作...")
        group_results = evaluate_single_motion_group(
            env=env,
            policy=policy,
            motion_config_path=config_path,
            group_motions=group_motions,
            device=args.device,
            max_steps=args.max_steps,
            collect_metrics=True
        )
        
        # 添加动作库信息
        for r in group_results:
            r['motion_group'] = group_name
        
        all_results.extend(group_results)
        
        group_elapsed = time.time() - group_start_time
        print(f"动作库 {group_name} 评估完成，耗时: {group_elapsed:.1f} 秒")

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)

    del env
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"所有动作库评估完成，总耗时: {elapsed_time:.1f} 秒")
    print(f"{'='*60}")
    
    # 汇总结果
    print(f"\n汇总统计结果...")
    overall_stats, group_stats = aggregate_results(all_results, motions_by_group)
    
    # 打印结果
    print_results_table(overall_stats, group_stats)
    
    # 保存结果
    output_path = generate_output_filename(model_info, args.output_dir)
    save_results(output_path, model_info, overall_stats, group_stats, all_results)
    
    print(f"\n{'='*60}")
    print(f"测评完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
