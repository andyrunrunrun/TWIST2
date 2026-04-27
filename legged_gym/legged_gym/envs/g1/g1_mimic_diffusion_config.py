from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym.envs.g1.g1_mimic_future_config import (
    G1MimicStuFutureCfg,
    G1MimicStuFutureCfgDAgger,
    TAR_MOTION_STEPS_FUTURE,
)


class G1MimicStuFutureDiffusionCfg(G1MimicStuFutureCfg):
    """Environment config for the diffusion student policy."""


class G1MimicStuFutureDiffusionCfgTrain(G1MimicStuFutureCfg):
    seed = 1

    class runner(G1MimicStuFutureCfgDAgger.runner):
        policy_class_name = "DiffusionPolicyFuture"
        algorithm_class_name = "DiffusionDagger"
        runner_class_name = "OnPolicyDiffusionRunner"
        max_iterations = 30_001
        save_interval = 500
        experiment_name = "g1_stu_future_diffusion"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        save_to_wandb = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-4
        max_grad_norm = 1.0
        normalizer_update_iterations = 1000

    class policy(G1MimicStuFutureCfgDAgger.policy):
        diffusion_hidden_dims = [1024, 1024, 512, 256]
        diffusion_timestep_embed_dim = 128
        diffusion_train_timesteps = 32
        diffusion_inference_steps = 4
        diffusion_beta_schedule = "cosine"
        diffusion_recon_loss_weight = 0.05
        diffusion_action_clip = 10.0
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)
        num_future_observations = G1MimicStuFutureCfg.env.n_future_obs


class G1MimicStuFutureDiffusion2xCfg(G1MimicStuFutureDiffusionCfg):
    """Diffusion student config around 2.25x the MLP actor baseline."""


class G1MimicStuFutureDiffusion2xCfgTrain(G1MimicStuFutureDiffusionCfgTrain):
    class runner(G1MimicStuFutureDiffusionCfgTrain.runner):
        experiment_name = "g1_stu_future_diff2x"

    class policy(G1MimicStuFutureDiffusionCfgTrain.policy):
        # ~5.24M params total (~2.25x of the 2.33M MLP actor baseline)
        diffusion_hidden_dims = [1536, 1280, 1024, 768]
        actor_hidden_dims = diffusion_hidden_dims


class G1MimicStuFutureDiffusion4xCfg(G1MimicStuFutureDiffusionCfg):
    """Diffusion student config around 4.05x the MLP actor baseline."""


class G1MimicStuFutureDiffusion4xCfgTrain(G1MimicStuFutureDiffusionCfgTrain):
    class runner(G1MimicStuFutureDiffusionCfgTrain.runner):
        experiment_name = "g1_stu_future_diff4x"

    class policy(G1MimicStuFutureDiffusionCfgTrain.policy):
        # ~9.44M params total (~4.05x of the 2.33M MLP actor baseline)
        diffusion_hidden_dims = [2304, 1792, 1280, 1024]
        actor_hidden_dims = diffusion_hidden_dims


G1MimicStuFutureDiffusionCfg2x = G1MimicStuFutureDiffusion2xCfg
G1MimicStuFutureDiffusionCfg2xTrain = G1MimicStuFutureDiffusion2xCfgTrain
G1MimicStuFutureDiffusionCfg4x = G1MimicStuFutureDiffusion4xCfg
G1MimicStuFutureDiffusionCfg4xTrain = G1MimicStuFutureDiffusion4xCfgTrain
