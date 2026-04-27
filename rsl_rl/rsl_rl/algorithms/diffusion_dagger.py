import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from rsl_rl.storage import DiffusionRolloutStorage


class DiffusionDagger:
    def __init__(
        self,
        env,
        actor_critic,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        device="cpu",
        precision="float32",
        **kwargs,
    ):
        self.env = env
        self.device = device
        self.actor_critic = actor_critic
        if not hasattr(self.actor_critic, "module"):
            self.actor_critic.to(self.device)

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = DiffusionRolloutStorage.Transition()
        self.storage = None

        self.precision = precision
        self.use_amp = precision in ("float16", "bfloat16")
        self.amp_dtype = None
        if precision == "float16":
            self.amp_dtype = torch.float16
        elif precision == "bfloat16":
            self.amp_dtype = torch.bfloat16
        self.scaler = GradScaler("cuda", enabled=(self.use_amp and precision == "float16"))

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, action_shape):
        self.storage = DiffusionRolloutStorage(
            num_envs, num_transitions_per_env, obs_shape, action_shape, self.device
        )

    def train_mode(self):
        self.actor_critic.train()

    def test_mode(self):
        self.actor_critic.eval()

    def act(self, obs, critic_obs=None, info=None, hist_encoding=False):
        self.transition.observations = obs
        self.transition.target_actions = self.env.get_reference_action_target().detach()
        return self.actor_critic.act(obs).detach()

    def process_env_step(self, rewards, dones, infos):
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        return rewards

    def compute_returns(self, last_critic_obs):
        return

    def update(self):
        mean_total_loss = 0.0
        mean_denoise_loss = 0.0
        mean_recon_loss = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        amp_enabled = self.use_amp and str(self.device).startswith("cuda")
        for obs_batch, target_actions_batch in generator:
            with autocast("cuda", enabled=amp_enabled, dtype=self.amp_dtype):
                total_loss, denoise_loss, recon_loss = self.actor_critic(
                    obs_batch, target_actions=target_actions_batch
                )

            self.optimizer.zero_grad()
            if amp_enabled and self.precision == "float16":
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

            mean_total_loss += total_loss.item()
            mean_denoise_loss += denoise_loss.item()
            mean_recon_loss += recon_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_total_loss /= num_updates
        mean_denoise_loss /= num_updates
        mean_recon_loss /= num_updates
        self.storage.clear()
        return mean_total_loss, mean_denoise_loss, mean_recon_loss
