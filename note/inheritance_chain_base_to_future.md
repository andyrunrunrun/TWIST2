# TWIST2 环境类继承链详解：从 BaseTask 到 G1MimicFuture

> 本文档详细梳理了 `legged_gym` 中环境类的完整继承关系，从最底层的 Isaac Gym 封装 `BaseTask`，一直到实际训练使用的 `G1MimicFuture`。每一层都配有**代码片段**和**文字解读**，帮助理解各类的分工与演进。

---

## 1. 整体继承树

```text
BaseTask (base_task.py)
    └── LeggedRobot (legged_robot.py)
            ├── Humanoid (humanoid.py)          ← 平行分支：locomotion 步行任务
            └── HumanoidChar (humanoid_char.py)
                    └── HumanoidMimic (humanoid_mimic.py)
                            ├── G1Mimic (g1_mimic.py)
                            └── G1MimicDistill (g1_mimic_distill.py)
                                    └── G1MimicFuture (g1_mimic_future.py)
```

**主链**（通往 `G1MimicFuture`）沿左侧一路向下；`Humanoid` 是一个平行分支，专注于**自主步态 locomotion**，不在主链上，但与 `HumanoidChar` 同级。

---

## 2. 逐层详解：代码 + 文字解读

### Layer 0: `BaseTask`
**文件**：`legged_gym/envs/base/base_task.py`

```python
class BaseTask():
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless
        # env device: GPU only if sim is on GPU and use_gpu_pipeline=True
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        ...
        
        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
```

**主要职责**：
1. **Isaac Gym 底座**：获取 `gym` 句柄，创建物理仿真器 `sim`，准备 `viewer`（若非 headless）。
2. **张量分配**：预先分配 `obs_buf`、`rew_buf`、`reset_buf`、`episode_length_buf` 等基础 PyTorch 张量，这些是后续所有计算的“画布”。
3. **抽象接口**：定义了 `step()`、`reset_idx()`、`reset()` 等抽象方法，强制上层子类实现具体的物理步进与重置逻辑。
4. **交互与渲染**：处理键盘事件（ESC 退出、V 切换同步、F 自由相机、数字键切换跟踪目标等）、相机跟随 `lookat()`、渲染同步 `render()`。

**一句话总结**：`BaseTask` 是“仿真器的启动器和画布的提供者”，它本身不知道机器人长什么样，只负责把 Isaac Gym 跑起来并留好数据位。

---

### Layer 1: `LeggedRobot`
**文件**：`legged_gym/envs/base/legged_robot.py`

```python
class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
```

**主要职责**：
1. **资产加载与解析**：在 `create_sim()` → `_create_envs()` 中加载 URDF/MJCF，解析出 `num_dof`（自由度数量）、`num_bodies`（刚体数量）、`dof_names`、`body_names`、足部索引 `feet_indices`、躯干索引 `torso_idx` 等。
2. **地形系统**：支持 `plane`、`heightfield`、`trimesh` 三种地形，通过 `Terrain` 类生成程序化地形。
3. **PD 控制器**：`_compute_torques()` 实现了位置控制（P）、速度控制（V）、直接力矩控制（T）三种模式，并支持 motor strength 域随机化。
4. **通用奖励与终止**：`compute_reward()` 自动扫描 `cfg.rewards.scales` 中所有非零项，动态组装奖励函数列表；`check_termination()` 检查接触力、高度、超时等基础终止条件。
5. **域随机化（Domain Randomization）**：`_process_rigid_shape_props`（摩擦随机化）、`_process_rigid_body_props`（基础质量/质心随机化）、`_push_robots()`（随机推力）。
6. **环境重置**：`_reset_dofs()`、`_reset_root_states()`、`_resample_commands()` 提供了通用的状态重置逻辑。

**关键代码片段——动态奖励组装**：
```python
def _prepare_reward_function(self):
    for key in list(self.reward_scales.keys()):
        scale = self.reward_scales[key]
        if scale==0:
            self.reward_scales.pop(key)
        else:
            self.reward_scales[key] *= self.dt
    self.reward_functions = []
    self.reward_names = []
    for name, scale in self.reward_scales.items():
        if name=="termination": continue
        self.reward_names.append(name)
        name = '_reward_' + name
        self.reward_functions.append(getattr(self, name))
```
这段代码让子类只需要写 `_reward_xxx()` 方法并在配置里填 scale，就能自动被调用，无需手动注册。

**一句话总结**：`LeggedRobot` 是“腿式机器人的通用操作系统”，知道怎么加载机器人、怎么走路面、怎么算力矩、怎么发奖励，但还不涉及具体任务（locomotion 还是 mimic）。

---

### Layer 2: `HumanoidChar`
**文件**：`legged_gym/envs/base/humanoid_char.py`

```python
class HumanoidChar(LeggedRobot):
    def __init__(self, cfg: HumanoidCharCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        ...
        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
```

**主要职责**：
1. **关键身体部位（Key Bodies）管理**：在 `_init_buffers()` 中根据配置 `cfg.motion.key_bodies` 和 `upper_key_bodies` 构建 `_key_body_ids` 和 `_upper_key_body_ids` 张量。这是后续动作捕捉跟踪的“跟踪点”。
2. **动作延迟课程学习（Action Delay Curriculum）**：`step()` 中根据总步数线性增加动作延迟概率（从 0 到 0.5），让策略学会应对真实世界的控制延迟。
3. **视频录制相机**：`_create_envs()` 中若 `record_video=True`，会为每个环境创建 `camera_sensor`，并在 `render_record()` 中跟随机器人渲染 RGB 图像。
4. **参考动作可视化**：`draw_key_bodies_actual()`（红色球体，画实际机器人关键部位）和 `draw_key_bodies_motion()`（青色/绿色球体，画参考动作关键部位），极大方便了调试 mimic 任务时的跟踪偏差。
5. **观测与后处理**：重写了 `compute_observations()`，加入更丰富的 proprioceptive 观测和 `regularization_scale_curriculum`（正则化奖励的自动缩放）。

**关键代码片段——动作延迟**：
```python
if self.cfg.domain_rand.action_delay:
    start_step = 5000 * 24
    target_step = 20000 * 24
    if self.total_env_steps_counter <= start_step:
        delay_prob = 0.0
    elif self.total_env_steps_counter >= target_step:
        delay_prob = 0.5
    else:
        delay_prob = 0.5 * (self.total_env_steps_counter - start_step) / (target_step - start_step)
    if torch.rand(1, device=self.device) < delay_prob:
        self.delay = torch.tensor(1.0, device=self.device, dtype=torch.float)
    else:
        self.delay = torch.tensor(0.0, device=self.device, dtype=torch.float)
    indices = -self.delay - 1
    action_tensor = self.action_history_buf[:, indices.long()]
```

**一句话总结**：`HumanoidChar` 是“人形角色动画层”，它为 motion mimic 任务引入了关键身体部位跟踪、动作延迟、录制相机和可视化调试能力。

---

### Layer 3: `HumanoidMimic`
**文件**：`legged_gym/envs/base/humanoid_mimic.py`

```python
class HumanoidMimic(HumanoidChar):
    def __init__(self, cfg: HumanoidMimicCfg, sim_params, physics_engine, sim_device, headless):
        self._enable_early_termination = cfg.env.enable_early_termination
        self._pose_termination = cfg.env.pose_termination
        self._tar_motion_steps_priv = cfg.env.tar_motion_steps_priv
        self._tar_motion_steps = cfg.env.tar_motion_steps
        ...
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._load_motions()
        self.motion_difficulty = torch.ones((num_motions), device=self.device) * 100.0
```

**主要职责**：
1. **MotionLib 集成**：在 `_load_motions()` 中实例化 `MotionLib`，加载 `.pkl` 或 `.npz` 格式的参考动作数据集。`MotionLib` 负责管理所有动作片段的采样、插值、缓存。
2. **参考动作状态管理**：通过 `_reset_ref_motion()`、`_update_ref_motion()`、`_get_motion_times()` 维护每个环境当前播放的参考动作 ID (`_motion_ids`) 和时间偏移 (`_motion_time_offsets`)，并实时刷新参考根部位姿、关节角度、关键身体点全局坐标 (`_ref_body_pos`)。
3. **RSI（Random State Initialization）**：`reset_idx()` 中直接将机器人状态重置为参考动作的 `dof_pos`、`root_pos`、`root_rot`、`root_vel`，让训练从动作中的任意一帧开始，极大加速模仿学习收敛。
4. **Motion Curriculum**：`_update_motion_difficulty()` 根据机器人在每个动作上的**完成率**（`episode_length / motion_length`）动态调整 `motion_difficulty`：完成率低则增加难度（更难被采样到），完成率高则降低难度。这实现了自动化的动作难度课程学习。
5. **姿态终止条件（Pose Termination）**：`check_termination()` 中计算实际关键身体点与参考关键身体点的欧氏距离，若偏差过大则提前终止 episode，防止策略在错误轨迹上浪费时间。
6. **观测构建**：`_get_mimic_obs()` 从 `MotionLib` 采样未来/当前帧的参考数据，组装成 mimic 观测（root pos + roll/pitch + root vel + yaw ang vel + dof pos）。

**关键代码片段——Motion Curriculum**：
```python
completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lib.get_motion_length(reset_motion_ids)
motion_completion_rate_sum = torch.zeros(num_motions, device=self.device).scatter_add(0, reset_motion_ids, completion_rate)
motion_completion_rate_count = torch.zeros(num_motions, device=self.device).scatter_add(0, reset_motion_ids, torch.ones_like(completion_rate))
motion_completion_rate = motion_completion_rate_sum / torch.clamp(motion_completion_rate_count, min=1)

add_idx = motion_completion_rate <= 0.5
sub_idx = (motion_completion_rate >= 0.95) & (motion_completion_rate < 0.99)
super_sub_idx = motion_completion_rate >= 0.99

gamma = self.cfg.motion.motion_curriculum_gamma
new_difficulty = self.motion_difficulty.clone()
new_difficulty = torch.where(add_idx, new_difficulty * (1 + gamma), new_difficulty)
new_difficulty = torch.where(sub_idx, new_difficulty * (1 - gamma), new_difficulty)
new_difficulty = torch.where(super_sub_idx, new_difficulty * (1 - gamma * 20), new_difficulty)
self.motion_difficulty = torch.clamp(new_difficulty, min=1., max=10.)
```

**一句话总结**：`HumanoidMimic` 是**动作模仿学习的算法核心层**，它把 `MotionLib` 的参考动作数据流与 RL 环境的 reset/step/observation/reward 循环彻底打通，并引入了 motion curriculum 来自动化难度调度。

---

### Layer 4: `G1Mimic`
**文件**：`legged_gym/envs/g1/g1_mimic.py`

```python
class G1Mimic(HumanoidMimic):
    def __init__(self, cfg: G1MimicCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        ...
```

**主要职责**：
1. **G1 身体索引精确定义**：`_get_body_indices()` 中为 G1 机器人提取 `upper_arm_indices`、`lower_arm_indices`、`torso_indices`、`knee_indices`，用于后续奖励计算和可视化。
2. **Mimic 观测定制**：`_get_mimic_obs()` 组装了 G1 基础 mimic 观测：
   - `root_pos` (3D)
   - `roll`, `pitch`, `yaw` (3D)
   - `root_vel` (3D)
   - `root_ang_vel[..., 2:3]` (yaw only, 1D)
   - `dof_pos` (num_dof)
   总维度：`(9 + num_dof) * num_steps`
3. **观测掩码**：`compute_observations()` 中将脚踝（ankle）的 dof velocity 观测通道置零（索引 4, 5, 10, 11），这是工程经验—— ankle 速度观测噪声大、对策略帮助小，屏蔽后更稳定。
4. **额外奖励函数**：为 G1 定义了腰部/脚踝相关的惩罚项，如 `_reward_waist_dof_acc`、`_reward_ankle_dof_acc`、`_reward_ankle_action`。

**关键代码片段——脚踝速度屏蔽**：
```python
dof_vel_start_dim = mimic_obs.shape[1] + 5 + self.dof_pos.shape[1]
obs_buf[:, [dof_vel_start_dim + 4, dof_vel_start_dim + 5, dof_vel_start_dim + 10, dof_vel_start_dim + 11]] = 0.
```

**一句话总结**：`G1Mimic` 是**G1 机器人的基础 mimic 环境**，它在 `HumanoidMimic` 之上做了 G1 特有的身体索引解析、观测掩码和奖励定制，是早期 G1 模仿学习最直接的入口。

---

### Layer 5: `G1MimicDistill`
**文件**：`legged_gym/envs/g1/g1_mimic_distill.py`

```python
class G1MimicDistill(HumanoidMimic):
    def __init__(self, cfg: G1MimicPrivCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type  # 'priv' or 'student'
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.obs_type == 'student':
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000
```

**主要职责**：
1. **教师-学生（Teacher-Student）蒸馏架构**：通过 `obs_type` 区分两种模式：
   - **`priv`**（教师）：观测包含完整的 privileged mimic 信息（root pos、distance to target、local vel、key body pos 等），用于训练一个拥有“上帝视角”的教师策略。
   - **`student`**（学生）：观测大幅精简（只保留 root xy vel + z pos + roll/pitch + yaw ang vel + dof pos），模拟真实部署时的传感器限制，用于蒸馏学习。
2. **Privileged vs Student 观测分离**：`_get_mimic_obs()` 同时返回 `priv_mimic_obs_buf` 和 `mimic_obs_buf`，前者给 Critic/教师，后者给 Actor/学生。
3. **Limb Weights（肢体权重）**：支持 `use_limb_weights`，可在 reset 时随机生成 `[L_arm, R_arm, L_leg, R_leg]` 的权重，并通过 `_update_dof_err_w()` 动态调整不同肢体关节的跟踪误差权重，实现部分肢体的弱化跟踪训练。
4. **观测历史分离管理**：`_init_buffers()` 中分别维护 `obs_history_buf` 和 `privileged_obs_history_buf`，学生和教师使用不同长度的历史信息。
5. **根部位移缩放**：`_scale_ref_root_pos_delta_local()` 将 MotionLib 中按动作帧率存储的位移缩放为按环境步长 `self.dt` 的位移，保证参考动作与仿真步长对齐。

**关键代码片段——教师 vs 学生 mimic 观测**：
```python
# teacher (privileged)
priv_mimic_obs_buf = torch.cat((
    root_pos,                    # 3 dims
    root_pos_distance_to_target, # 3 dims
    roll, pitch, yaw,            # 3 dims
    root_vel_local,              # 3 dims
    root_ang_vel_local,          # 3 dims
    root_pos_delta_local,        # 3 dims
    root_rot_delta_local,        # 3 dims
    dof_pos,                     # num_dof dims
    whole_key_body_pos if not self.global_obs else whole_key_body_pos_global,
), dim=-1)

# student (observable)
mimic_obs_buf = torch.cat((
    root_vel_local[..., :2],     # 2 dims (xy velocity)
    root_pos[..., 2:3],          # 1 dim (z position)
    roll, pitch,                 # 2 dims
    root_ang_vel_local[..., 2:3],# 1 dim (yaw angular velocity)
    dof_pos,                     # num_dof dims
), dim=-1)[:, self._tar_motion_steps_idx_in_teacher, :]
```

**一句话总结**：`G1MimicDistill` 是**蒸馏训练的核心环境**，它在同一套物理仿真中支持“教师（全知 privileged）”和“学生（受限 observable）”两种观测模式，为后续的 sim-to-real 部署做蒸馏准备。

---

### Layer 6: `G1MimicFuture`
**文件**：`legged_gym/envs/g1/g1_mimic_future.py`

```python
class G1MimicFuture(G1MimicDistill):
    """Student policy environment with future motion support and curriculum masking.
    Extends G1MimicDistill to add future motion capabilities while maintaining 
    all original RL+BC functionality.
    
    Curriculum Masked Privilege Information (CMP)
    """
    def __init__(self, cfg: G1MimicStuFutureCfg, sim_params, physics_engine, sim_device, headless):
        self.future_cfg = cfg.env
        self.evaluation_mode = getattr(cfg.env, 'evaluation_mode', False)
        self.force_full_masking = getattr(cfg.env, 'force_full_masking', False)
        self.enable_force_curriculum = getattr(cfg.env, 'enable_force_curriculum', False)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.obs_type == 'student_future':
            self._tar_motion_steps_future = torch.tensor(
                getattr(cfg.env, 'tar_motion_steps_future', [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
                device=self.device, dtype=torch.long
            )
```

**主要职责**：
1. **未来帧观测（Future Motion Observations）**：
   - 在 `student_future` 模式下，策略不仅看到当前帧参考动作，还能看到未来 `[5, 10, 15, ..., 50]` 步的参考动作。
   - `_get_unified_motion_data()` 是核心优化：它只调用一次 `calc_motion_frame`，同时采样 privileged 帧 + future 帧，然后切片分给 `_get_mimic_obs()` 和 `_build_future_obs_from_data()`，避免重复采样带来的巨大性能开销。
2. **统一数据流与零拷贝切片**：
   - `all_steps = torch.cat([self._tar_motion_steps_priv, self._tar_motion_steps_future])`
   - 对返回的 `root_pos`、`dof_pos` 等统一 reshape，然后 `[:, :num_priv_steps]` 给 privileged，`[:, num_priv_steps:]` 给 future。
3. **FALCON-style 力课程学习（Force Curriculum）**：
   - `enable_force_curriculum=True` 时，在 `_calculate_ee_forces()` 中每隔随机时长（150~250 步）向机器人的末端执行器（如 `left_rubber_hand`、`right_rubber_hand`）施加随机力。
   - `force_scale` 会根据 episode 长度进行课程调整：表现好（episode 长）则增大力的幅度，表现差则减小，实现“越来越难的扰动训练”。
4. **Error Aware Sampling 日志**：`_log_error_aware_sampling_progress()` 和 `_log_max_key_body_error_per_motion()` 定期记录每个动作的 key body 跟踪误差，用于分析哪些动作片段最难跟踪。
5. **观测组装**：`compute_observations()` 在 `student_future` 模式下将 `future_obs` 平铺后拼接到 `obs_buf` 末尾：
   ```python
   if self.obs_type == 'student_future':
       obs_components = [obs_buf, self.obs_history_buf.view(self.num_envs, -1)]
       if future_obs is not None:
           future_obs_flat = future_obs.view(self.num_envs, -1)
           obs_components.append(future_obs_flat)
       self.obs_buf = torch.cat(obs_components, dim=-1)
   ```

**关键代码片段——统一采样与切片**：
```python
def _get_unified_motion_data(self):
    if self.obs_type == 'student_future' and hasattr(self, '_tar_motion_steps_future'):
        all_steps = torch.cat([self._tar_motion_steps_priv, self._tar_motion_steps_future])
        num_priv_steps = self._tar_motion_steps_priv.shape[0]
        num_future_steps = self._tar_motion_steps_future.shape[0]
    else:
        all_steps = self._tar_motion_steps_priv
        ...
    
    motion_times = self._get_motion_times().unsqueeze(-1)
    obs_motion_times = all_steps * self.dt + motion_times
    motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape).flatten()
    obs_motion_times = obs_motion_times.flatten()
    
    root_pos, root_rot, ... = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
    ...
    # 切片分配
    priv_root_pos = root_pos.reshape(num_envs, total_steps, -1)[:, :num_priv_steps]
    future_root_pos = root_pos.reshape(num_envs, total_steps, -1)[:, num_priv_steps:]
```

**关键代码片段——力课程**：
```python
def _calculate_ee_forces(self):
    self.episode_length_counter += 1
    self.force_duration_counter += 1
    need_new_forces = self.force_duration_counter >= self.force_duration_target
    if need_new_forces.any():
        # 重新随机生成力
        force_x = torch.rand(...) * (self.apply_force_x_range[1] - self.apply_force_x_range[0]) + ...
        ...
        self.applied_forces[need_new_forces] = new_forces
    
    # 三角波调制 + 课程缩放
    phase_modulation = torch.where(self.force_phase < 1.0, self.force_phase, 2.0 - self.force_phase)
    final_forces = self.applied_forces.clone()
    for i in range(len(self.force_apply_body_indices)):
        final_forces[:, i] *= self.force_scale.unsqueeze(-1) * phase_modulation.unsqueeze(-1)
    
    # 施加到仿真器
    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(all_forces), None, gymapi.ENV_SPACE)
```

**一句话总结**：`G1MimicFuture` 是**面向未来的学生策略训练环境**，它在蒸馏框架之上引入了“未来动作预测 + 统一采样 + 力课程扰动”，让策略能提前规划动作并具备更强的抗干扰能力，是 TWIST2 项目中实际训练 `g1_stu_future` 任务的最终环境类。

---

## 3. 横向对比：G1Mimic vs G1MimicDistill vs G1MimicFuture

| 特性 | `G1Mimic` | `G1MimicDistill` | `G1MimicFuture` |
|------|-----------|------------------|-----------------|
| **继承父类** | `HumanoidMimic` | `HumanoidMimic` | `G1MimicDistill` |
| **观测模式** | 单一 mimic 观测 | `priv` / `student` 双模式 | `priv` / `student` / `student_future` |
| **Privileged 观测** | 无（只有基础 mimic obs） | 有（完整的 root/body/delta） | 有（与 Distill 一致） |
| **Future 帧** | ❌ | ❌ | ✅（5~50步未来参考动作） |
| **统一采样** | ❌ | ❌ | ✅（单次 `calc_motion_frame`） |
| **力课程（Force Curriculum）** | ❌ | ❌ | ✅（FALCON-style） |
| **Limb Weights** | ❌ | ✅ | ✅ |
| **主要用途** | 基础 G1 mimic 训练 | 教师-学生蒸馏 | 带未来预测的学生训练 + 抗干扰 |

---

## 4. 平行分支：`Humanoid`（自主步行 locomotion）

**文件**：`legged_gym/envs/base/humanoid.py`

```python
class Humanoid(LeggedRobot):
```

`Humanoid` 与 `HumanoidChar` 同级，都继承自 `LeggedRobot`，但它走的是另一条路：
- **不加载 `MotionLib`**，而是基于正弦波生成参考步态（`compute_ref_state()` 中的 `_get_phase()`）。
- 奖励函数围绕**速度跟踪**（`tracking_lin_vel`、`tracking_ang_vel`）、**步态周期**（`feet_air_time`、`feet_contact_number`）、**站立稳定性**（`stand_still`、`stand_base_acc`）设计。
- 观测中包含 `sin_pos`、`cos_pos` 等步态相位信号，让策略学会自主节律行走。

**与主链的关系**：如果把 `LeggedRobot` 比作“所有腿式机器人的基类”，那么 `Humanoid` 是**无参考动作的自主 locomotion 专精**，而 `HumanoidChar → ... → G1MimicFuture` 是**有参考动作的 motion mimic 专精**。两者共享底层物理仿真、PD 控制、域随机化等基础设施，但上层任务逻辑完全分化。

---

## 5. 总结

从 `BaseTask` 到 `G1MimicFuture`，每一层都在前一层的基础上做**最小但精准的增量**：

1. `BaseTask` 搭好 Isaac Gym 的台子。
2. `LeggedRobot` 把腿式机器人的通用物理逻辑（加载、地形、PD、奖励、域随机化）跑通。
3. `HumanoidChar` 为人形 mimic 任务引入 key bodies、动作延迟、相机、可视化。
4. `HumanoidMimic` 接入 `MotionLib`，打通参考动作数据流，实现 RSI、Motion Curriculum、Pose Termination。
5. `G1Mimic` 做 G1 机器人的身体索引和观测掩码定制。
6. `G1MimicDistill` 拆分教师/学生观测，引入 limb weights，为蒸馏做准备。
7. `G1MimicFuture` 最终聚合未来帧预测、统一采样、力课程扰动，形成实际部署前的最强训练环境。

这条继承链体现了典型的**“通用底座 → 任务中间件 → 具体机器人 → 算法变体”**的分层设计思想，既保证了代码复用，又支持了从基础研究到工程落地的渐进式演进。
