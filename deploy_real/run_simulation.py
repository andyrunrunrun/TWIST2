import argparse
import json
import time
import numpy as np
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os
import sys

# Add parent directory to path to allow imports from data_utils and pose
script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(script_dir, '..'))

from data_utils.rot_utils import quatToEuler, euler_from_quaternion_torch, quat_rotate_inverse_torch
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.params import DEFAULT_MIMIC_OBS

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    # avail = ort.get_available_providers() # causing issues on some systems?
    # Simplified provider selection
    if device.startswith('cuda'):
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    # print(f"ONNX policy loaded from {policy_path}")
    return OnnxPolicyWrapper(session, input_name)


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    robot_type: str = "g1",
    mask_indicator: bool = False
):
    """
    Build the mimic_obs at time-step t_step.
    Adapted from server_motion_lib.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.int, device=device)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, -1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, -1, 3)
    root_vel = root_vel.reshape(1, -1, 3)
    root_ang_vel = root_ang_vel.reshape(1, -1, 3)

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])
    
    # Modified for better observability: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos
    if mask_indicator:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)
        # append mask indicator 1
        mask_indicator = torch.ones(1, mimic_obs_buf.shape[1], 1).to(device)
        mimic_obs_buf = torch.cat((mimic_obs_buf, mask_indicator), dim=-1)
    else:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)

    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


class SimulationRunner:
    def __init__(self, 
                 xml_file, 
                 policy_path,
                 motion_file,
                 device='cuda', 
                 vis=False,
                 fall_threshold=0.3
                 ):
        
        self.device = device
        self.vis = vis
        self.fall_threshold = fall_threshold

        # Load Policy
        self.policy = load_onnx_policy(policy_path, device)
        
        # Load Motion Lib
        self.motion_lib = MotionLib(motion_file, device=device)
        self.motion_steps = [1] # future steps for observation, matching default in server_motion_lib
        self.motion_steps_tensor = torch.tensor(self.motion_steps, device=device, dtype=torch.int)

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001 # 1ms simulation step
        self.data = mujoco.MjData(self.model)
        
        if self.vis:
            self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.distance = 2.0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
        else:
            self.viewer = None

        self.num_actions = 29
        self.sim_dt = 0.001
        self.policy_frequency = 100
        self.sim_decimation = int(1 / (self.policy_frequency * self.sim_dt))

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # G1 specific configuration
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (3)
                0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
                0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
            ])

        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]), # root pos
            np.array([1, 0, 0, 0]),  # root rot (quat)
             np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (3)
                0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
                0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
                ])
        ])

        # PD gains
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5, 0.2, 0.2, 0.2,
                5, 5, 5, 5, 0.2, 0.2, 0.2,
            ])
        
        self.torque_limits = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
            ])

        self.action_scale = 0.5

        self.ankle_idx = [4, 5, 10, 11]

        self.n_mimic_obs = 35  
        self.n_proprio = 3 + 2 + 3*29    
        self.n_obs_single = 35 + 3 + 2 + 3*29  
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs 

        # Initialize history
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.mujoco_default_dof_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        n_dof = self.num_actions
        dof_pos = self.data.qpos[7:7+n_dof]
        dof_vel = self.data.qvel[6:6+n_dof]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        return dof_pos, dof_vel, quat, ang_vel

    def run(self):
        # Calculate steps from motion length
        # control_dt = sim_dt * decimation
        policy_dt = self.sim_dt * self.sim_decimation
        motion_id = torch.tensor([0], device=self.device, dtype=torch.long)
        motion_length = self.motion_lib.get_motion_length(motion_id)
        
        # Add some buffer time
        total_duration = motion_length + 2.0 
        total_steps = int(total_duration / self.sim_dt)
        
        self.reset_sim()
        
        # print(f"Simulating for {total_duration:.2f}s ({total_steps} steps)")
        
        # Initial wait
        init_steps = 100
        for _ in range(init_steps):
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()

        # Init state
        pd_target = self.default_dof_pos.copy()
        
        # Loop
        motion_t_step = 0
        
        try:
            for i in tqdm(range(total_steps), desc="Running Simulation", disable=True):
                dof_pos, dof_vel, quat, ang_vel = self.extract_data()
                
                # Check fall
                pelvis_height = self.data.xpos[self.model.body("pelvis").id][2]
                if pelvis_height < self.fall_threshold:
                    print(f"[Run Simulation] Robot fell! Height: {pelvis_height:.3f}")
                    return False

                if i % self.sim_decimation == 0:
                    # 1. Get Mimic Observation (Reference motion)
                    # Note: We use policy_dt for step size here
                    mimic_obs, _, _, _, _, _ = build_mimic_obs(
                        motion_lib=self.motion_lib,
                        t_step=motion_t_step, 
                        control_dt=policy_dt,
                        tar_motion_steps=self.motion_steps_tensor,
                        robot_type="unitree_g1_with_hands"
                    )
                    
                    # 2. Build Proprio Observation
                    rpy = quatToEuler(quat)
                    obs_body_dof_vel = dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        rpy[:2], # roll, pitch
                        (dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])
                    
                    # 3. Construct Full Observation
                    # obs structure: [mimic, proprio, history..., future(mimic)]
                    # Wait, looking at server_low_level:
                    # obs_full = cat([action_mimic, obs_proprio])
                    # obs_buf = cat([obs_full, obs_hist, future_obs])
                    
                    obs_full = np.concatenate([mimic_obs, obs_proprio])
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    
                    future_obs = mimic_obs.copy() # using current mimic as future for now, as in server code
                    
                    obs_buf = np.concatenate([obs_full, obs_hist, future_obs])
                    
                    # Update History
                    self.proprio_history_buf.append(obs_full)
                    
                    # 4. Inference
                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    
                    motion_t_step += 1

                # Low level PD control
                torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)

                if self.vis and i % 20 == 0: # render at ~50fps
                    self.viewer.sync()
                    
                    # Follow robot
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    
        except KeyboardInterrupt:
            pass
        finally:
            if self.viewer:
                self.viewer.close()

        # If we finished the loop without falling, consider it a success
        print("[Run Simulation] Motion completed successfully.")
        return True


def main():
    parser = argparse.ArgumentParser(description='Run TWIST2 Simulation Check')
    parser.add_argument('--xml', type=str, required=True, help='Path to MuJoCo XML file')
    parser.add_argument('--policy', type=str, required=True, help='Path to ONNX policy file')
    parser.add_argument('--motion_file', type=str, required=True, help='Path to .pkl motion file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--vis', action='store_true', help='Visualize simulation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.xml):
        print(f"Error: XML {args.xml} not found")
        sys.exit(1)
    if not os.path.exists(args.policy):
        print(f"Error: Policy {args.policy} not found")
        sys.exit(1)
    if not os.path.exists(args.motion_file):
        print(f"Error: Motion file {args.motion_file} not found")
        sys.exit(1)

    runner = SimulationRunner(
        xml_file=args.xml,
        policy_path=args.policy,
        motion_file=args.motion_file,
        device=args.device,
        vis=args.vis
    )
    
    success = runner.run()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
