import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic_future import (
    FutureMotionEncoder,
    HistoryEncoder,
    MotionEncoder,
    get_activation,
)


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, target_shape):
    out = buffer.gather(0, timesteps)
    while len(out.shape) < len(target_shape):
        out = out.unsqueeze(-1)
    return out


def _build_mlp(input_dim, hidden_dims, output_dim, activation_name, layer_norm=False):
    layers = []
    prev_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(get_activation(activation_name))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _cosine_beta_schedule(num_timesteps, s=0.008):
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def _linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor):
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)

        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FutureConditionEncoder(nn.Module):
    def __init__(
        self,
        num_observations,
        num_motion_observations,
        num_priop_observations,
        num_motion_steps,
        num_future_observations,
        num_future_steps,
        motion_latent_dim,
        future_latent_dim,
        history_latent_dim,
        num_history_steps,
        activation="silu",
        future_attention_heads=4,
        future_dropout=0.1,
        temporal_embedding_dim=64,
        use_history_encoder=True,
        use_motion_encoder=True,
        **kwargs,
    ):
        super().__init__()
        activation_fn = get_activation(activation)

        self.num_observations = num_observations
        self.num_motion_observations = num_motion_observations
        self.num_priop_observations = num_priop_observations
        self.num_motion_steps = num_motion_steps
        self.num_history_steps = num_history_steps
        self.num_future_observations = num_future_observations
        self.num_future_steps = max(int(num_future_steps), 1)
        self.num_single_motion_observations = int(num_motion_observations / num_motion_steps)
        self.num_single_priop_observations = num_priop_observations
        self.num_single_history_observations = num_motion_observations + num_priop_observations
        self.num_single_future_observations = (
            int(num_future_observations / self.num_future_steps) if num_future_observations > 0 else 0
        )
        self.future_latent_dim = future_latent_dim

        if use_motion_encoder:
            self.motion_encoder = MotionEncoder(
                activation_fn,
                self.num_single_motion_observations,
                self.num_motion_steps,
                motion_latent_dim,
            )
        else:
            self.motion_encoder = nn.Identity()
            motion_latent_dim = self.num_single_motion_observations

        if use_history_encoder:
            self.history_encoder = HistoryEncoder(
                activation_fn,
                self.num_single_history_observations,
                self.num_history_steps,
                history_latent_dim,
            )
        else:
            self.history_encoder = nn.Identity()
            history_latent_dim = self.num_single_history_observations * self.num_history_steps

        if self.num_single_future_observations > 0 and num_future_observations > 0:
            # Keep the same interface as the existing future actor implementation.
            self.future_encoder = FutureMotionEncoder(
                activation_fn,
                self.num_single_future_observations - 1,
                self.num_future_steps,
                future_latent_dim,
                attention_heads=future_attention_heads,
                dropout=future_dropout,
                temporal_embedding_dim=temporal_embedding_dim,
            )
        else:
            self.future_encoder = None

        self.output_dim = (
            self.num_single_motion_observations
            + self.num_single_priop_observations
            + motion_latent_dim
            + history_latent_dim
            + future_latent_dim
        )

    def forward(self, obs):
        current_size = self.num_motion_observations + self.num_priop_observations
        history_size = self.num_history_steps * current_size

        motion_obs = obs[:, : self.num_motion_observations]
        single_motion_obs = obs[:, : self.num_single_motion_observations]
        priop_obs = obs[:, self.num_motion_observations : current_size]

        history_start = current_size
        history_end = history_start + history_size
        history_obs = obs[:, history_start:history_end]

        future_start = history_end
        future_end = future_start + self.num_future_observations
        future_obs = obs[:, future_start:future_end]

        motion_latent = self.motion_encoder(motion_obs)
        history_latent = self.history_encoder(history_obs)

        if self.future_encoder is not None and future_obs.shape[1] > 0:
            future_obs = future_obs.reshape(-1, self.num_future_steps, self.num_single_future_observations)
            future_latent = self.future_encoder(future_obs)
        else:
            future_latent = torch.zeros(obs.shape[0], self.future_latent_dim, device=obs.device)

        return torch.cat(
            [single_motion_obs, priop_obs, motion_latent, history_latent, future_latent], dim=-1
        )


class DiffusionPolicyFuture(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_observations,
        num_critic_observations=None,
        num_motion_observations=0,
        num_motion_steps=1,
        num_priop_observations=0,
        num_history_steps=0,
        num_actions=0,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        motion_latent_dim=128,
        history_latent_dim=128,
        future_latent_dim=128,
        activation="silu",
        layer_norm=False,
        future_encoder_dims=None,
        future_attention_heads=4,
        future_dropout=0.1,
        temporal_embedding_dim=64,
        num_future_observations=0,
        num_future_steps=1,
        diffusion_hidden_dims=None,
        diffusion_timestep_embed_dim=128,
        diffusion_train_timesteps=32,
        diffusion_inference_steps=4,
        diffusion_beta_schedule="cosine",
        diffusion_recon_loss_weight=0.05,
        diffusion_action_clip=10.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            known_kwargs = {
                "init_noise_std",
                "fix_action_std",
                "action_std",
                "tanh_encoder_output",
                "use_moe",
                "num_experts",
                "expert_hidden_dims",
                "gating_hidden_dim",
                "moe_topk",
                "moe_temperature",
                "use_transformer",
                "d_model",
                "nhead",
                "num_transformer_layers",
                "transformer_dropout",
            }
            unknown_kwargs = [key for key in kwargs.keys() if key not in known_kwargs]
            if unknown_kwargs:
                print(
                    "DiffusionPolicyFuture.__init__ got unexpected arguments, which will be ignored: "
                    + str(unknown_kwargs)
                )

        if actor_hidden_dims is None:
            actor_hidden_dims = [1024, 1024, 512, 256]
        if diffusion_hidden_dims is None:
            diffusion_hidden_dims = actor_hidden_dims

        self.num_actions = num_actions
        self.num_train_timesteps = int(diffusion_train_timesteps)
        self.num_inference_steps = int(diffusion_inference_steps)
        self.recon_loss_weight = float(diffusion_recon_loss_weight)
        self.action_clip = float(diffusion_action_clip)

        self.condition_encoder = FutureConditionEncoder(
            num_observations=num_observations,
            num_motion_observations=num_motion_observations,
            num_priop_observations=num_priop_observations,
            num_motion_steps=num_motion_steps,
            num_future_observations=num_future_observations,
            num_future_steps=num_future_steps,
            motion_latent_dim=motion_latent_dim,
            future_latent_dim=future_latent_dim,
            history_latent_dim=history_latent_dim,
            num_history_steps=num_history_steps,
            activation=activation,
            future_attention_heads=future_attention_heads,
            future_dropout=future_dropout,
            temporal_embedding_dim=temporal_embedding_dim,
        )
        self.condition_dim = self.condition_encoder.output_dim

        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim),
            get_activation(activation),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim),
        )
        self.denoiser = _build_mlp(
            input_dim=self.condition_dim + diffusion_timestep_embed_dim + num_actions,
            hidden_dims=diffusion_hidden_dims,
            output_dim=num_actions,
            activation_name=activation,
            layer_norm=layer_norm,
        )

        if diffusion_beta_schedule == "linear":
            betas = _linear_beta_schedule(self.num_train_timesteps)
        else:
            betas = _cosine_beta_schedule(self.num_train_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1e-8)),
        )

    def reset(self, dones=None):
        return

    def _predict_noise(self, condition, noisy_actions, timesteps):
        timestep_embedding = self.time_embedding(timesteps)
        denoiser_input = torch.cat([condition, noisy_actions, timestep_embedding], dim=-1)
        return self.denoiser(denoiser_input)

    def _predict_start_from_noise(self, noisy_actions, timesteps, pred_noise):
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, noisy_actions.shape)
        sqrt_one_minus = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_actions.shape
        )
        x0 = (noisy_actions - sqrt_one_minus * pred_noise) / torch.clamp(sqrt_alpha, min=1e-8)
        return torch.clamp(x0, -self.action_clip, self.action_clip)

    def _q_sample(self, x_start, timesteps, noise):
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def compute_training_loss(self, observations, target_actions):
        batch_size = observations.shape[0]
        condition = self.condition_encoder(observations)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=observations.device, dtype=torch.long
        )
        noise = torch.randn_like(target_actions)
        noisy_actions = self._q_sample(target_actions, timesteps, noise)
        pred_noise = self._predict_noise(condition, noisy_actions, timesteps)

        denoise_loss = F.mse_loss(pred_noise, noise)
        pred_actions = self._predict_start_from_noise(noisy_actions, timesteps, pred_noise)
        recon_loss = F.mse_loss(pred_actions, target_actions)
        total_loss = denoise_loss + self.recon_loss_weight * recon_loss
        return total_loss, denoise_loss.detach(), recon_loss.detach()

    def _sample_actions(self, observations, stochastic=False):
        condition = self.condition_encoder(observations)
        if stochastic:
            actions = torch.randn(
                observations.shape[0], self.num_actions, device=observations.device, dtype=observations.dtype
            )
        else:
            actions = torch.zeros(
                observations.shape[0], self.num_actions, device=observations.device, dtype=observations.dtype
            )

        step_indices = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            steps=self.num_inference_steps,
            device=observations.device,
        ).long()
        step_indices = torch.unique_consecutive(step_indices)

        for idx, timestep in enumerate(step_indices):
            timestep_batch = torch.full(
                (observations.shape[0],), int(timestep.item()), device=observations.device, dtype=torch.long
            )
            pred_noise = self._predict_noise(condition, actions, timestep_batch)
            pred_actions = self._predict_start_from_noise(actions, timestep_batch, pred_noise)

            if idx == len(step_indices) - 1:
                actions = pred_actions
            else:
                prev_timestep = step_indices[idx + 1]
                alpha_prev = self.alphas_cumprod[prev_timestep]
                alpha_prev = torch.clamp(alpha_prev, min=1e-8)
                actions = (
                    torch.sqrt(alpha_prev) * pred_actions
                    + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=1e-8)) * pred_noise
                )

        return torch.clamp(actions, -self.action_clip, self.action_clip)

    def forward(self, observations, target_actions=None, **kwargs):
        if target_actions is not None:
            return self.compute_training_loss(observations, target_actions)
        return self.act_inference(observations, **kwargs)

    def act(self, observations, stochastic=False, **kwargs):
        return self._sample_actions(observations, stochastic=stochastic)

    def act_inference(self, observations, **kwargs):
        return self._sample_actions(observations, stochastic=False)
