# Transformer-based Student Policy Configuration for G1
# This config replaces the MLP backbone with a Transformer encoder

from legged_gym.envs.g1.g1_mimic_future_config import (
    G1MimicStuFutureCfg, 
    G1MimicStuFutureCfgDAgger
)
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicStuFutureTrans2xCfg(G1MimicStuFutureCfg):
    """Student policy config with Transformer backbone (2x parameters).
    
    Transformer configuration:
    - d_model=232, nhead=4, num_layers=2
    - ~1.44M parameters (~1.96x of MLP baseline)
    """
    
    class env(G1MimicStuFutureCfg.env):
        pass
    
    class motion(G1MimicStuFutureCfg.motion):
        pass
    
    class rewards(G1MimicStuFutureCfg.rewards):
        pass


class G1MimicStuFutureTrans2xCfgDAgger(G1MimicStuFutureCfgDAgger):
    """DAgger training config for Transformer-2x student policy."""
    
    seed = 1
    
    class teachercfg(G1MimicStuFutureCfgDAgger.teachercfg):
        pass
    
    class runner(G1MimicStuFutureCfgDAgger.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 150000
        warm_iters = 100
        
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
        
        save_to_wandb = False

    class algorithm(G1MimicStuFutureCfgDAgger.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1

    class policy(G1MimicStuFutureCfgDAgger.policy):
        # Standard parameters (critic still uses these)
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
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
        num_future_steps = 1
        num_future_observations = 35
        
        # ========================================
        # Transformer specific parameters (2x)
        # ========================================
        use_transformer = True
        use_moe = False  # Disable MoE
        
        # 2x configuration: ~1.44M params (1.96x of MLP)
        d_model = 232
        nhead = 4
        num_transformer_layers = 2
        transformer_dropout = 0.1


class G1MimicStuFutureTrans4xCfg(G1MimicStuFutureCfg):
    """Student policy config with Transformer backbone (4x parameters).
    
    Transformer configuration:
    - d_model=280, nhead=4, num_layers=3
    - ~3.01M parameters (~4.08x of MLP baseline)
    """
    
    class env(G1MimicStuFutureCfg.env):
        pass
    
    class motion(G1MimicStuFutureCfg.motion):
        pass
    
    class rewards(G1MimicStuFutureCfg.rewards):
        pass


class G1MimicStuFutureTrans4xCfgDAgger(G1MimicStuFutureCfgDAgger):
    """DAgger training config for Transformer-4x student policy."""
    
    seed = 1
    
    class teachercfg(G1MimicStuFutureCfgDAgger.teachercfg):
        pass
    
    class runner(G1MimicStuFutureCfgDAgger.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 150000
        warm_iters = 100
        
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
        
        save_to_wandb = False

    class algorithm(G1MimicStuFutureCfgDAgger.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000
        dagger_coef = 0.2
        dagger_coef_min = 0.1

    class policy(G1MimicStuFutureCfgDAgger.policy):
        # Standard parameters
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
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
        num_future_steps = 1
        num_future_observations = 35
        
        # ========================================
        # Transformer specific parameters (4x)
        # ========================================
        use_transformer = True
        use_moe = False  # Disable MoE
        
        # 4x configuration: ~3.01M params (4.08x of MLP)
        d_model = 280
        nhead = 4
        num_transformer_layers = 3
        transformer_dropout = 0.1


# Aliases for convenience
G1MimicStuTrans2xCfg = G1MimicStuFutureTrans2xCfg
G1MimicStuTrans2xCfgDAgger = G1MimicStuFutureTrans2xCfgDAgger
G1MimicStuTrans4xCfg = G1MimicStuFutureTrans4xCfg
G1MimicStuTrans4xCfgDAgger = G1MimicStuFutureTrans4xCfgDAgger
