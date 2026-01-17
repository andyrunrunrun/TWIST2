from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1HyMotion100kPrivCfg(G1MimicPrivCfg):
    """Teacher (privileged) config for HYMotion100k."""

    class motion(G1MimicPrivCfg.motion):
        # HYMotion100k GMR motions (no HY features needed for teacher).
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/hymotion100k_g1_gmr_30fps_no_feat.yaml"
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        motion_decompose = False


class G1HyMotion100kPrivCfgPPO(G1MimicPrivCfgPPO):
    """PPO training config for HYMotion100k privileged teacher."""

    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = "ActorCriticMimicHyMotion"
        algorithm_class_name = "PPO"
        runner_class_name = "OnPolicyRunnerMimic"
        max_iterations = 30_001

        # logging
        experiment_name = "g1_priv_hymotion100k"
        run_name = ""

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        pass

    class policy(G1MimicPrivCfgPPO.policy):
        pass


class G1HyMotion100kStuHyFeatCfg(G1HyMotion100kPrivCfg):
    """Student config: mimic_obs (masked) + proprio + history + feature_history."""

    class env(G1HyMotion100kPrivCfg.env):
        obs_type = "student_hyfeat"

        # Match the default student target timing (1 step ahead).
        tar_motion_steps = [1]
        n_mimic_obs_single = 6 + 29
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = G1MimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio

        # HY single-stream feature settings (t=1.0 only).
        hy_feat_dim = 1280
        hy_feat_history_steps = G1MimicPrivCfg.env.history_len + 1
        # Align feature timestamps with mimic targets: feature(t + 1*dt).
        hy_feat_time_offset_steps = 1

        # Total student obs: (current + history) + feature_history
        num_observations = n_obs_single * (G1MimicPrivCfg.env.history_len + 1) + hy_feat_dim * hy_feat_history_steps

        # Curriculum masking schedule (mask mimic_obs only).
        mask_max_iterations = 30_001
        mask_reach_ratio = 0.8
        mask_shape = "cosine"  # cosine over [0, reach_ratio*max_iterations]
        mask_steps_per_iter = 24  # must match runner.num_steps_per_env

    class motion(G1HyMotion100kPrivCfg.motion):
        # Student needs HY feature paths enabled.
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/hymotion100k_g1_gmr_30fps.yaml"

        # MotionLib caches HY features per-motion on CPU (optional).
        hy_feat_cache_motions = 1024


class G1HyMotion100kStuHyFeatCfgDAgger(G1HyMotion100kStuHyFeatCfg):
    """DAgger/PPO distillation config for HYMotion100k student with HY features."""

    seed = 1

    class teachercfg(G1HyMotion100kPrivCfgPPO):
        pass

    class runner(G1HyMotion100kPrivCfgPPO.runner):
        policy_class_name = "ActorCriticHyFeat"
        algorithm_class_name = "DaggerPPO"
        runner_class_name = "OnPolicyDaggerRunner"
        max_iterations = 30_001
        warm_iters = 100

        # logging
        save_interval = 500
        experiment_name = "g1_stu_hymotion100k_hyfeat"
        run_name = ""

        teacher_experiment_name = "g1_priv_hymotion100k"
        teacher_proj_name = "g1_priv_hymotion100k"
        teacher_checkpoint = -1
        eval_student = False

        save_to_wandb = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005

        dagger_update_freq = 20
        dagger_coef_anneal_steps = 60_000
        dagger_coef = 0.2
        dagger_coef_min = 0.1

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0

        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = "silu"
        layer_norm = True

        motion_latent_dim = 128
        history_latent_dim = 128
        hy_feat_latent_dim = 160

        # Feature encoder MLP size (per-frame 1280 -> hy_feat_latent_dim) and temporal conv settings.
        hy_feat_proj_hidden = 512
        hy_feat_conv_layers = 2
