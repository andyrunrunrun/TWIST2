from isaacgym.torch_utils import quat_from_euler_xyz
import torch

from legged_gym.envs.g1.g1_mimic_distill import G1MimicDistill
from legged_gym.envs.g1.g1_mimic_hyfeat_config import G1HyMotion100kStuHyFeatCfg
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos


class G1MimicHyFeat(G1MimicDistill):
    """Mimic env with HY-Motion per-frame feature history and curriculum masking on mimic_obs."""

    def __init__(self, cfg: G1HyMotion100kStuHyFeatCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if self.obs_type != "student_hyfeat":
            return

        # Feature history buffer (stores previous history_len features; current feature is appended on-the-fly).
        self._hy_feat_dim = int(getattr(cfg.env, "hy_feat_dim", 0))
        if self._hy_feat_dim <= 0:
            raise ValueError("cfg.env.hy_feat_dim must be positive for obs_type='student_hyfeat'.")
        self._hy_feat_history_steps = int(getattr(cfg.env, "hy_feat_history_steps", cfg.env.history_len + 1))
        if self._hy_feat_history_steps != int(cfg.env.history_len + 1):
            raise ValueError("Currently only supports hy_feat_history_steps == history_len + 1.")
        self._hy_feat_history_buf = torch.zeros(
            (self.num_envs, int(cfg.env.history_len), int(self._hy_feat_dim)),
            device=self.device,
            dtype=torch.float32,
        )

        # Per-episode mimic_obs masking state (True -> mask mimic_obs).
        self._mask_mimic_obs = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        self._mask_max_iterations = int(getattr(cfg.env, "mask_max_iterations", 0))
        self._mask_reach_ratio = float(getattr(cfg.env, "mask_reach_ratio", 0.8))
        self._mask_shape = str(getattr(cfg.env, "mask_shape", "linear"))
        self._mask_steps_per_iter = int(getattr(cfg.env, "mask_steps_per_iter", 24))

    def _mask_prob(self) -> float:
        max_it = int(self._mask_max_iterations)
        if max_it <= 0:
            return 0.0
        reach = float(self._mask_reach_ratio)
        reach = max(1e-6, min(reach, 1.0))
        steps_per_iter = max(1, int(self._mask_steps_per_iter))
        it = float(self.total_env_steps_counter) / float(steps_per_iter)
        denom = max(1e-6, float(max_it) * reach)
        x = max(0.0, min(it / denom, 1.0))
        shape = self._mask_shape.lower().strip()
        if shape == "linear":
            return x
        if shape in {"square", "quadratic"}:
            return x * x
        if shape == "cosine":
            import math

            return 0.5 * (1.0 - math.cos(math.pi * x))
        return x

    def compute_observations(self):
        if self.obs_type != "student_hyfeat":
            return super().compute_observations()

        # Sample/reset episode-level masking and clear feature history for newly reset envs.
        reset_mask = (self.episode_length_buf <= 1)
        if bool(reset_mask.any()):
            reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self._hy_feat_history_buf[reset_indices] = 0.0

            p = float(self._mask_prob())
            if p <= 0.0:
                self._mask_mimic_obs[reset_indices] = False
            elif p >= 1.0:
                self._mask_mimic_obs[reset_indices] = True
            else:
                self._mask_mimic_obs[reset_indices] = (torch.rand(len(reset_indices), device=self.device) < p)

        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0 * self.yaw, 0 * self.yaw, self.yaw)

        priv_mimic_obs, mimic_obs = self._get_mimic_obs()

        proprio_obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                imu_obs,  # 2
                self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                self.reindex(self.action_history_buf[:, -1]),
            ),
            dim=-1,
        )

        if self.cfg.noise.add_noise and self.headless:
            noise_scale = min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 1.0)
            proprio_obs_buf = proprio_obs_buf + (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * noise_scale
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf = proprio_obs_buf + (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec

        # disable ankle dof velocity
        dof_vel_start_dim = 3 + 2 + self.dof_pos.shape[1]
        ankle_idx = [4, 5, 10, 11]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.0

        # Privileged critic info (unchanged)
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1)

        priv_info = torch.cat(
            (
                self.base_lin_vel,
                self.root_states[:, 0:3],
                self.root_states[:, 3:7],
                key_body_pos,
                self.contact_forces[:, self.feet_indices, 2] > 5.0,
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1,
                self.motor_strength[1] - 1,
            ),
            dim=-1,
        )

        # Curriculum masking: only mask mimic_obs (target-related); proprio and privileged obs stay intact.
        mimic_obs_masked = mimic_obs
        if bool(self._mask_mimic_obs.any()):
            mimic_obs_masked = mimic_obs.clone()
            mimic_obs_masked[self._mask_mimic_obs] = 0.0

        obs_buf = torch.cat((mimic_obs_masked, proprio_obs_buf), dim=-1)

        priv_obs_buf = torch.cat((priv_mimic_obs, proprio_obs_buf, priv_info), dim=-1)
        self.privileged_obs_buf = priv_obs_buf

        # HY feature at current time (interpolated to env dt)
        motion_times = self._get_motion_times()
        if hasattr(self._motion_lib, "prefetch"):
            self._motion_lib.prefetch(self._motion_ids)
        hy_feat = self._motion_lib.calc_hy_feat_frame(self._motion_ids, motion_times)
        hy_feat = hy_feat.to(dtype=torch.float32)

        # Feature sequence: [history (len=H), current] -> (H+1, D)
        hy_seq = torch.cat((self._hy_feat_history_buf, hy_feat.unsqueeze(1)), dim=1)
        hy_flat = hy_seq.reshape(self.num_envs, -1)

        self.obs_buf = torch.cat((obs_buf, self.obs_history_buf.view(self.num_envs, -1), hy_flat), dim=-1)

        # Update history buffers (in-place, same pattern as parent).
        if self.cfg.env.history_len > 0:
            continue_mask = ~reset_mask
            if bool(continue_mask.any()):
                continue_indices = continue_mask.nonzero(as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[continue_indices, :-1] = self.privileged_obs_history_buf[continue_indices, 1:]
                self.privileged_obs_history_buf[continue_indices, -1] = priv_obs_buf[continue_indices]

                self.obs_history_buf[continue_indices, :-1] = self.obs_history_buf[continue_indices, 1:]
                self.obs_history_buf[continue_indices, -1] = obs_buf[continue_indices]

                self._hy_feat_history_buf[continue_indices, :-1] = self._hy_feat_history_buf[continue_indices, 1:]
                self._hy_feat_history_buf[continue_indices, -1] = hy_feat[continue_indices]

            if bool(reset_mask.any()):
                reset_indices = reset_mask.nonzero(as_tuple=False).squeeze(-1)
                self.privileged_obs_history_buf[reset_indices] = priv_obs_buf[reset_indices].unsqueeze(1).expand(
                    -1, self.cfg.env.history_len, -1
                )
                self.obs_history_buf[reset_indices] = obs_buf[reset_indices].unsqueeze(1).expand(-1, self.cfg.env.history_len, -1)
                # feature history stays zeros on reset (current feature is provided in hy_seq's last step).

        return
