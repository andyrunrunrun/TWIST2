# Mixture of Experts (MoE) configuration for G1 student policy
# This config extends the standard future config with MoE-specific parameters

from legged_gym.envs.g1.g1_mimic_future_config import (
    G1MimicStuFutureCfg, 
    G1MimicStuFutureCfgDAgger
)
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicStuFutureMoECfg(G1MimicStuFutureCfg):
    """Student policy config with MoE (Mixture of Experts) backbone.
    
    This config uses the same encoders (Motion, History, Future) as the standard
    config, but replaces the MLP backbone with a MoE layer.
    
    Design principle: Gating + 1 Expert ≈ Original MLP parameters
    - Original MLP [512, 512, 256, 128]: ~738K params
    - MoE (4 experts, top-2): ~2.4M params total
      * Gating: ~77K
      * 1 Expert [512, 384, 192]: ~600K
    """
    
    class env(G1MimicStuFutureCfg.env):
        # Same observation structure as standard config
        pass
    
    class motion(G1MimicStuFutureCfg.motion):
        # Same motion configuration as standard config
        pass
    
    class rewards(G1MimicStuFutureCfg.rewards):
        # Same reward configuration as standard config
        pass


class G1MimicStuFutureMoECfgDAgger(G1MimicStuFutureCfgDAgger):
    """DAgger training config for MoE student policy."""
    
    seed = 1
    
    class teachercfg(G1MimicStuFutureCfgDAgger.teachercfg):
        pass
    
    class runner(G1MimicStuFutureCfgDAgger.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 150000
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'g1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False
        
        # Wandb model saving option
        save_to_wandb = False

    class algorithm(G1MimicStuFutureCfgDAgger.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        
        # MoE specific: load balancing loss weight
        # Set to 0.0 to disable load balancing (not recommended)
        moe_load_balancing_weight = 0.01

    class policy(G1MimicStuFutureCfgDAgger.policy):
        # Standard actor parameters (used when use_moe=False)
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        
        # When using MoE, actor_hidden_dims is not used for the backbone
        # but kept for API compatibility and critic network
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        
        # Future motion encoder parameters
        future_encoder_dims = [256, 256, 128]
        future_attention_heads = 4
        future_dropout = 0.1
        temporal_embedding_dim = 64
        future_latent_dim = 128
        num_future_steps = 1  # Should match tar_motion_steps_future length
        num_future_observations = 35  # Should match env.n_future_obs
        
        # ========================================
        # MoE specific parameters
        # ========================================
        use_moe = True  # Enable MoE backbone
        
        # Number of expert networks
        num_experts = 4
        
        # Hidden layer dimensions for each expert
        # Design: [512, 384, 192] gives ~600K params per expert
        # Gating + 1 Expert ≈ Original MLP (~738K)
        expert_hidden_dims = [512, 384, 192]
        
        # Gating network hidden dimension (lightweight)
        gating_hidden_dim = 128
        
        # Top-K selection: how many experts to activate per forward pass
        # Set to None or num_experts to use all experts
        moe_topk = 2
        
        # Temperature for gating softmax (lower = more discrete selection)
        moe_temperature = 1.0


# Alias for backward compatibility
G1MimicStuMoECfg = G1MimicStuFutureMoECfg
G1MimicStuMoECfgDAgger = G1MimicStuFutureMoECfgDAgger
