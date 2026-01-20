# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic_future import HistoryEncoder, MotionEncoder, get_activation
from .actor_critic_mimic import ActorCriticMimic


class ActorCriticMimicHyMotion(ActorCriticMimic):
    """Alias class for HYMotion100k teacher policies."""

    pass


class HyFeatureHistoryEncoder(nn.Module):
    """Encode per-frame HY features into a compact latent."""

    def __init__(
        self,
        activation_fn: nn.Module,
        *,
        feature_dim: int,
        num_steps: int,
        proj_hidden: int,
        latent_dim: int,
        conv_layers: int,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_steps = int(num_steps)
        self.latent_dim = int(latent_dim)

        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, int(proj_hidden)),
            activation_fn,
            nn.Linear(int(proj_hidden), self.latent_dim),
            activation_fn,
        )

        conv_layers = int(conv_layers)
        if conv_layers <= 0:
            self.temporal = None
        else:
            layers: list[nn.Module] = []
            for _ in range(conv_layers):
                layers.append(nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1))
                layers.append(activation_fn)
            self.temporal = nn.Sequential(*layers)

    def forward(self, feat_flat: torch.Tensor) -> torch.Tensor:
        b = int(feat_flat.shape[0])
        x = feat_flat.view(b, self.num_steps, self.feature_dim)  # (B, T, D)
        x = self.proj(x)  # (B, T, latent)
        x = x.permute(0, 2, 1)  # (B, latent, T)
        if self.temporal is not None:
            x = self.temporal(x)
        # Simple temporal aggregation.
        x = x.mean(dim=-1)
        return x


class ActorHyFeat(nn.Module):
    def __init__(
        self,
        *,
        num_observations: int,
        num_motion_observations: int,
        num_priop_observations: int,
        num_history_steps: int,
        num_feature_steps: int,
        feature_dim: int,
        motion_latent_dim: int,
        history_latent_dim: int,
        hy_feat_latent_dim: int,
        hy_feat_proj_hidden: int,
        hy_feat_conv_layers: int,
        num_actions: int,
        actor_hidden_dims: list[int],
        activation: nn.Module,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.num_observations = int(num_observations)
        self.num_actions = int(num_actions)

        self.num_motion_observations = int(num_motion_observations)
        self.num_priop_observations = int(num_priop_observations)
        self.num_history_steps = int(num_history_steps)
        self.num_feature_steps = int(num_feature_steps)
        self.feature_dim = int(feature_dim)

        self.num_single_motion_observations = int(self.num_motion_observations)
        self.num_single_priop_observations = int(self.num_priop_observations)
        self.current_size = self.num_motion_observations + self.num_priop_observations
        self.history_size = self.current_size * self.num_history_steps
        self.feature_size = self.num_feature_steps * self.feature_dim

        if self.num_observations != (self.current_size + self.history_size + self.feature_size):
            raise ValueError(
                "Unexpected obs layout for ActorHyFeat: "
                f"num_observations={self.num_observations} != current({self.current_size})"
                f"+history({self.history_size})+feature({self.feature_size})"
            )

        self.motion_encoder = MotionEncoder(activation, self.num_single_motion_observations, 1, int(motion_latent_dim))
        self.history_encoder = HistoryEncoder(activation, self.current_size, self.num_history_steps, int(history_latent_dim))
        self.hy_feat_encoder = HyFeatureHistoryEncoder(
            activation,
            feature_dim=self.feature_dim,
            num_steps=self.num_feature_steps,
            proj_hidden=int(hy_feat_proj_hidden),
            latent_dim=int(hy_feat_latent_dim),
            conv_layers=int(hy_feat_conv_layers),
        )

        input_dim = (
            self.num_single_motion_observations
            + self.num_single_priop_observations
            + int(motion_latent_dim)
            + int(history_latent_dim)
            + int(hy_feat_latent_dim)
        )

        layers: list[nn.Module] = [nn.Linear(input_dim, actor_hidden_dims[0]), activation]
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                layers.append(nn.Linear(actor_hidden_dims[i], self.num_actions))
            else:
                layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                if layer_norm and i == len(actor_hidden_dims) - 2:
                    layers.append(nn.LayerNorm(actor_hidden_dims[i + 1]))
                layers.append(activation)
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, **_kwargs) -> torch.Tensor:
        motion_obs = obs[:, : self.num_motion_observations]
        priop_obs = obs[:, self.num_motion_observations : self.current_size]
        history_obs = obs[:, self.current_size : self.current_size + self.history_size]
        feat_obs = obs[:, self.current_size + self.history_size :]

        motion_latent = self.motion_encoder(motion_obs)
        history_latent = self.history_encoder(history_obs)
        feat_latent = self.hy_feat_encoder(feat_obs)

        backbone = torch.cat([motion_obs, priop_obs, motion_latent, history_latent, feat_latent], dim=1)
        return self.actor(backbone)


class ActorCriticHyFeat(nn.Module):
    """Actor-Critic for student with HY feature history (no future motion obs)."""

    is_recurrent = False

    def __init__(
        self,
        *,
        num_observations: int,
        num_critic_observations: int,
        num_motion_observations: int,
        num_motion_steps: int,
        num_priop_observations: int,
        num_history_steps: int,
        num_actions: int,
        num_feature_dim: int,
        num_feature_steps: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        motion_latent_dim: int = 128,
        history_latent_dim: int = 128,
        hy_feat_latent_dim: int = 160,
        hy_feat_proj_hidden: int = 512,
        hy_feat_conv_layers: int = 2,
        activation: str = "silu",
        init_noise_std: float = 1.0,
        fix_action_std: bool = False,
        action_std=None,
        layer_norm: bool = False,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "ActorCriticHyFeat.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        actor_hidden_dims = actor_hidden_dims or [256, 256, 256]
        critic_hidden_dims = critic_hidden_dims or [256, 256, 256]

        self.fix_action_std = bool(fix_action_std)
        activation_fn = get_activation(activation)

        self.actor_net = ActorHyFeat(
            num_observations=num_observations,
            num_motion_observations=num_motion_observations,
            num_priop_observations=num_priop_observations,
            num_history_steps=num_history_steps,
            num_feature_steps=num_feature_steps,
            feature_dim=num_feature_dim,
            motion_latent_dim=motion_latent_dim,
            history_latent_dim=history_latent_dim,
            hy_feat_latent_dim=hy_feat_latent_dim,
            hy_feat_proj_hidden=hy_feat_proj_hidden,
            hy_feat_conv_layers=hy_feat_conv_layers,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            activation=activation_fn,
            layer_norm=bool(layer_norm),
        )

        # Critic uses privileged observations only.
        self.num_motion_observations = int(num_motion_observations)
        self.num_single_motion_obs = int(num_motion_observations / max(1, int(num_motion_steps)))
        self.critic_motion_encoder = MotionEncoder(activation_fn, self.num_single_motion_obs, int(num_motion_steps), int(motion_latent_dim))

        critic_input_dim = int(num_critic_observations) - int(num_motion_observations) + int(motion_latent_dim) + int(self.num_single_motion_obs)
        critic_layers: list[nn.Module] = [nn.Linear(critic_input_dim, critic_hidden_dims[0]), activation_fn]
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                if layer_norm and i == len(critic_hidden_dims) - 2:
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        if self.fix_action_std:
            if action_std is None:
                raise ValueError("fix_action_std=True requires action_std to be provided")
            self.init_action_std_tensor = torch.tensor(action_std)
            self.std = nn.Parameter(self.init_action_std_tensor, requires_grad=False)
        else:
            self.std = nn.Parameter(float(init_noise_std) * torch.ones(int(num_actions)))

        self.distribution = None
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor_net(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, **kwargs):
        return self.actor_net(observations)

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        motion_obs = critic_observations[:, : self.num_motion_observations]
        motion_single_obs = critic_observations[:, : self.num_single_motion_obs]
        motion_latent = self.critic_motion_encoder(motion_obs)
        backbone_input = torch.cat([critic_observations[:, self.num_motion_observations :], motion_single_obs, motion_latent], dim=1)
        return self.critic(backbone_input)

    def forward(self, observations, critic_observations=None, actions=None, **kwargs):
        self.update_distribution(observations)
        if actions is None:
            actions = self.distribution.sample()
        actions_log_prob = self.get_actions_log_prob(actions)
        entropy = self.entropy
        mu = self.action_mean
        sigma = self.action_std
        value = None
        if critic_observations is not None:
            value = self.evaluate(critic_observations, **kwargs)
        return actions, actions_log_prob, value, mu, sigma, entropy

    def update_std(self, std_coef: float):
        if self.fix_action_std:
            return
        self.std.data[:] = float(std_coef) * self.std.data[:]

    def if_fix_std(self):
        return self.fix_action_std
