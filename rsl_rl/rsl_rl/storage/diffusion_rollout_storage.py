import torch


class DiffusionRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.target_actions = None
            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, actions_shape, device="cpu"):
        self.device = device
        self.obs_shape = obs_shape
        self.actions_shape = actions_shape
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=device
        )
        self.target_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)
        self.target_actions[self.step].copy_(transition.target_actions)
        if transition.dones is not None:
            self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        target_actions = self.target_actions.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield observations[batch_idx], target_actions[batch_idx]
