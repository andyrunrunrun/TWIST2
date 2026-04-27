# GR00T vs TWIST2：RL 环境奖励函数详细对比

## 1. 对比范围与默认配置

这份笔记只比较两个仓库里**默认训练配置**对应的 RL reward 设计，不比较 auxiliary loss、DAgger/KL 蒸馏损失、终止条件和观测设计。

对比对象：

- `GR00T-WholeBodyControl`
  - 训练配置：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`
  - 奖励组合：`gear_sonic/config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml`
- `TWIST2`
  - teacher：`g1_priv_mimic`
  - student：`g1_stu_future`
  - 奖励实现在 `legged_gym/legged_gym/envs/base/humanoid_mimic.py`

直接相关源码：

- GR00T
  - `gear_sonic/envs/manager_env/mdp/rewards.py`
  - `gear_sonic/config/manager_env/rewards/tracking/base_5point_local_feet_acc.yaml`
  - `gear_sonic/config/manager_env/rewards/terms/*.yaml`
  - `gear_sonic/config/manager_env/base_env.yaml`
  - `gear_sonic/config/manager_env/commands/terms/motion.yaml`
  - `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`
- TWIST2
  - `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
  - `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
  - `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`
  - `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`
  - `legged_gym/legged_gym/envs/base/legged_robot.py`

额外参考：

- Isaac Lab reward manager 文档：`https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/reward_manager.html`
- Isaac Lab 标准 reward 实现：`https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/envs/mdp/rewards.html`

## 2. 总奖励形式与时间步缩放

### 2.1 GR00T

GR00T 使用 Isaac Lab 的 manager-based reward：

\[
R_{\text{GR00T}} = \Delta t \sum_i w_i r_i
\]

其中：

- \(w_i\) 是 yaml 里配置的 reward weight
- \(r_i\) 是每个 reward term 的原始值
- \(\Delta t\) 由 Isaac Lab `RewardManager` 在求和时自动乘上

GR00T 默认环境时间步：

- `sim_dt = 0.005`
- `decimation = 4`
- 所以策略步长：

\[
\Delta t_{\text{GR00T}} = 0.005 \times 4 = 0.02\ \text{s}
\]

### 2.2 TWIST2

TWIST2 在 `legged_robot._prepare_reward_function()` 里显式做：

\[
w_i \leftarrow w_i \cdot \Delta t
\]

因此总奖励也可以写成：

\[
R_{\text{TWIST2}} = \Delta t \sum_i w_i r_i
\]

TWIST2 默认环境时间步：

- `sim.dt = 0.002`
- `control.decimation = 10`
- 所以策略步长：

\[
\Delta t_{\text{TWIST2}} = 0.002 \times 10 = 0.02\ \text{s}
\]

### 2.3 一个重要结论

两边默认策略频率其实都是 **50 Hz**，也就是：

\[
\Delta t_{\text{GR00T}} = \Delta t_{\text{TWIST2}} = 0.02\ \text{s}
\]

所以下面的 reward weight 可以直接做“同量纲”比较。

## 3. 记号说明

为了统一写法，后面使用下面的记号：

- \(p, q, v, \omega\)：分别表示位置、姿态、线速度、角速度
- 上标 `ref`：参考动作
- 上标 `robot`：机器人当前状态
- `anchor`：根部 / pelvis / 主锚点
- \(d_{\text{quat}}(q_1, q_2)\)：四元数角误差
- \([x]_+ = \max(x, 0)\)
- \(B\)：GR00T 中被追踪的 body 数
- \(P\)：GR00T 中 `reward_point_body` 的点数
- \(B_s\)：GR00T 中 anti-shake 约束的 body 数
- \(K\)：TWIST2 中 key body 数
- \(J\)：关节数

## 4. GR00T 默认奖励项

### 4.1 默认激活的 reward term

`sonic_release` 通过 `base_5point_local_feet_acc` 激活如下 reward：

- `tracking_anchor_pos`
- `tracking_anchor_ori`
- `tracking_relative_body_pos`
- `tracking_relative_body_ori`
- `tracking_body_linvel`
- `tracking_body_angvel`
- `action_rate_l2`
- `joint_limit`
- `undesired_contacts`
- `anti_shake_ang_vel`
- `tracking_vr_5point_local`
- `feet_acc`

其中 `sonic_release` 还把 `feet_acc.weight` 从 `-2.5e-7` 覆盖成了 `-2.5e-6`。

### 4.2 公式、含义、默认权重

| 项 | 原始公式 | 配置权重 | 每步等效权重 | 含义 |
|---|---|---:|---:|---|
| `tracking_anchor_pos` | \(\exp(-\|p^{ref}_{anchor}-p^{robot}_{anchor}\|^2 / 0.3^2)\) | `0.5` | `0.01` | 跟踪根部位置 |
| `tracking_anchor_ori` | \(\exp(-d_{\text{quat}}(q^{ref}_{anchor}, q^{robot}_{anchor})^2 / 0.4^2)\) | `0.5` | `0.01` | 跟踪根部姿态 |
| `tracking_relative_body_pos` | \(\exp(-\frac{1}{B}\sum_i \|p^{ref,rel}_i - p^{robot}_i\|^2 / 0.3^2)\) | `1.0` | `0.02` | 跟踪各 body 的 anchor-relative 位置 |
| `tracking_relative_body_ori` | \(\exp(-\frac{1}{B}\sum_i d_{\text{quat}}(q^{ref,rel}_i, q^{robot}_i)^2 / 0.4^2)\) | `1.0` | `0.02` | 跟踪各 body 的 anchor-relative 姿态 |
| `tracking_body_linvel` | \(\exp(-\frac{1}{B}\sum_i \|v^{ref}_i-v^{robot}_i\|^2 / 1.0^2)\) | `1.0` | `0.02` | 跟踪各 body 线速度 |
| `tracking_body_angvel` | \(\exp(-\frac{1}{B}\sum_i \|\omega^{ref}_i-\omega^{robot}_i\|^2 / 3.14^2)\) | `1.0` | `0.02` | 跟踪各 body 角速度 |
| `tracking_vr_5point_local` | \(\exp(-\frac{1}{P}\sum_j \|p^{robot,local}_j-p^{ref,local}_j\|^2 / 0.1^2)\) | `2.0` | `0.04` | 在 root 局部坐标系里跟踪 reward points |
| `action_rate_l2` | \(\sum_j (a_t^j-a_{t-1}^j)^2\) | `-0.1` | `-0.002` | 惩罚动作突变 |
| `joint_limit` | \(\sum_j [q^{min,soft}_j-q_j]_+ + [q_j-q^{max,soft}_j]_+\) | `-10.0` | `-0.2` | 惩罚越过软关节限位 |
| `undesired_contacts` | \(\sum_b \mathbf{1}[\max_t \|F_b(t)\| > 1.0]\) | `-0.1` | `-0.002` | 惩罚不希望发生的碰撞 |
| `anti_shake_ang_vel` | \(\frac{1}{B_s}\sum_b [\|\omega_b\|-1.5]_+^2\) | `-0.005` | `-0.0001` | 惩罚手腕/头部抖动 |
| `feet_acc` | \(\sum_{j\in ankle} \ddot q_j^2\) | `-2.5e-6` | `-5e-8` | 惩罚踝关节加速度，抑制落脚过硬/抖动 |

### 4.3 GR00T 默认 reward 的几个关键细节

#### 4.3.1 GR00T 默认不是 joint-space imitation

GR00T 默认 reward 里**没有**类似 `tracking_joint_dof` / `tracking_joint_vel` 的主 reward。  
它更强调：

- root/anchor 跟踪
- body-level 几何跟踪
- body-level 速度跟踪
- 少量动作平滑与碰撞约束

也就是说，GR00T 更像是在优化“整体几何运动是否像参考动作”，而不是“每个关节角是不是精确贴参考”。

#### 4.3.2 `tracking_vr_5point_local` 在 `sonic_release` 里名字有误导性

虽然 term 名字叫 `tracking_vr_5point_local`，但 `sonic_release` 里把 `reward_point_body` 改成了：

- `torso_link`
- `left_wrist_yaw_link`
- `right_wrist_yaw_link`

所以默认发布配置里这个 term 实际上是：

\[
P = 3
\]

不是传统意义上的 5 点。

#### 4.3.3 `undesired_contacts` 的豁免策略很特别

默认正则表达式把下面这些 body 从惩罚里排除了：

- 左右踝
- 左右手腕
- 左右肘

所以这个 contact penalty 主要在惩罚：

- 躯干
- 大腿
- 小腿
- 其它不该落地的 body

而不是简单地“除了脚，所有接触都罚”。

#### 4.3.4 GR00T 默认开启的是 anti-shake，不是 anti-shuffle

GR00T 默认启用的是：

- `anti_shake_ang_vel`

它的目标是抑制**手腕和头部高频角速度抖动**，不是抑制小碎步。

## 5. TWIST2 默认奖励项

### 5.1 teacher 与 student 的关系

TWIST2 在 reward 设计上，teacher 与 student 基本沿用同一套函数族。  
两者的主要差异不是“reward 家族完全不同”，而是：

- student 额外有 DAgger/KL 蒸馏
- student 观测不同
- student 的少数 reward 权重略有调整

### 5.2 teacher 默认激活项

teacher 默认非零项：

- `tracking_joint_dof`
- `tracking_joint_vel`
- `tracking_root_translation_z`
- `tracking_root_rotation`
- `tracking_root_linear_vel`
- `tracking_root_angular_vel`
- `tracking_keybody_pos`
- `tracking_keybody_pos_global`
- `alive`
- `feet_slip`
- `feet_contact_forces`
- `feet_stumble`
- `dof_pos_limits`
- `dof_torque_limits`
- `dof_vel`
- `dof_acc`
- `action_rate`
- `feet_air_time`
- `ang_vel_xy`
- `ankle_dof_acc`
- `ankle_dof_vel`
- `step_switch_rate`
- `stance_foot_speed`

但是注意：

- `step_switch_rate`
- `stance_foot_speed`

虽然权重非零，**默认并不会生效**，因为 `enable_anti_shuffle_reward = False`。

### 5.3 student 默认激活项

student 默认 reward 家族和 teacher 一样，但有两个重要权重差异：

- `tracking_joint_vel`: teacher `0.3`，student `0.2`
- `action_rate`: teacher `-0.01`，student `-0.05`

也就是：

- student 的 joint velocity imitation 稍微放松
- student 的动作平滑约束更强

### 5.4 公式、含义、teacher/student 默认权重

下面只列默认训练里真正重要的项。

| 项 | 原始公式 | teacher 权重 | student 权重 | 每步等效权重 teacher/student | 含义 |
|---|---|---:|---:|---:|---|
| `alive` | \(1\) | `0.5` | `0.5` | `0.01 / 0.01` | 生存项 |
| `tracking_joint_dof` | \(\exp(-0.15 \sum_j \alpha_j (q^{ref}_j-q_j)^2)\) | `2.0` | `2.0` | `0.04 / 0.04` | 关节位置模仿 |
| `tracking_joint_vel` | \(\exp(-0.01 \sum_j \alpha_j (\dot q^{ref}_j-\dot q_j)^2)\) | `0.3` | `0.2` | `0.006 / 0.004` | 关节速度模仿 |
| `tracking_root_translation_z` | \(\exp(-5 (z^{ref}-z)^2)\) | `1.0` | `1.0` | `0.02 / 0.02` | 根部高度跟踪 |
| `tracking_root_rotation` | \(\exp(-5 d_{\text{quat}}(q^{ref}_{root}, q_{root})^2)\) | `1.0` | `1.0` | `0.02 / 0.02` | 根部姿态跟踪 |
| `tracking_root_linear_vel` | \(\exp(-\|v^{ref,local}_{root}-v^{local}_{root}\|^2)\) | `1.0` | `1.0` | `0.02 / 0.02` | 根部线速度跟踪 |
| `tracking_root_angular_vel` | \(\exp(-\|\omega^{ref,local}_{root}-\omega^{local}_{root}\|^2)\) | `1.0` | `1.0` | `0.02 / 0.02` | 根部角速度跟踪 |
| `tracking_keybody_pos` | \(\exp(-10 \sum_{k=1}^{K} \|p^{yaw-local}_k - p^{ref,yaw-local}_k\|^2)\) | `2.0` | `2.0` | `0.04 / 0.04` | yaw 对齐局部坐标系里的关键 body 位置跟踪 |
| `tracking_keybody_pos_global` | \(\exp(-10 \sum_{k=1}^{K} \|p_k - p^{ref}_k\|^2)\) | `2.0` | `2.0` | `0.04 / 0.04` | 全局关键 body 位置跟踪 |
| `feet_slip` | \(\sum_f \mathbf{1}[F_{z,f}>5] \sqrt{\|v_{f,xy}\|}\) | `-0.1` | `-0.1` | `-0.002 / -0.002` | 惩罚支撑足滑移 |
| `feet_contact_forces` | \(\max(0, \|F_z\|_2 - F_{max})\) | `-5\times10^{-4}` | `-5\times10^{-4}` | `-1e-5 / -1e-5` | 惩罚过大的脚部竖直接触力 |
| `feet_stumble` | \(\mathbf{1}[\exists f:\|F_{xy,f}\| > 4|F_{z,f}|]\) | `-1.25` | `-1.25` | `-0.025 / -0.025` | 惩罚脚绊到障碍/横向冲击过大 |
| `dof_pos_limits` | \(\sum_j [q^{min}_j-q_j]_+ + [q_j-q^{max}_j]_+\) | `-5.0` | `-5.0` | `-0.1 / -0.1` | 惩罚关节越界 |
| `dof_torque_limits` | \(\sum_j [|\tau_j|/\tau^{lim}_j-\beta]_+\) | `-1.0` | `-1.0` | `-0.02 / -0.02` | 惩罚扭矩逼近上限 |
| `dof_vel` | \(\sum_j \dot q_j^2\) | `-1e-4` | `-1e-4` | `-2e-6 / -2e-6` | 惩罚关节速度过大 |
| `dof_acc` | \(\sum_j ((\dot q_{t-1,j}-\dot q_{t,j})/\Delta t)^2\) | `-5e-8` | `-5e-8` | `-1e-9 / -1e-9` | 惩罚关节加速度 |
| `action_rate` | \(\|a_t-a_{t-1}\|_2\) | `-0.01` | `-0.05` | `-2e-4 / -0.001` | 惩罚动作突变 |
| `feet_air_time` | \(\mathbf{1}[\|v^{ref}_{xy}\|>0.05]\sum_f \min(0,\ t^{air}_f-t_{target})\) | `5.0` | `5.0` | `0.1 / 0.1` | 惩罚 swing 太短，鼓励更接近目标腾空时长 |
| `ang_vel_xy` | \(\omega_x^2+\omega_y^2\) | `-0.01` | `-0.01` | `-2e-4 / -2e-4` | 惩罚机体 roll/pitch 角速度 |
| `ankle_dof_acc` | \(\sum_{j\in ankle} ((\dot q_{t-1,j}-\dot q_{t,j})/\Delta t)^2\) | `-1e-7` | `-1e-7` | `-2e-9 / -2e-9` | 额外惩罚踝关节加速度 |
| `ankle_dof_vel` | \(\sum_{j\in ankle} \dot q_j^2\) | `-2e-4` | `-2e-4` | `-4e-6 / -4e-6` | 额外惩罚踝关节速度 |
| `step_switch_rate` | \((\#\text{foot-contact toggles}) \cdot g\) | `-0.20` | `-0.20` | `-0.004 / -0.004` | 小碎步抑制：频繁换脚惩罚 |
| `stance_foot_speed` | \((\sum_f \mathbf{1}[contact_f]\|v_{f,xy}\|)\cdot g\) | `-0.05` | `-0.05` | `-0.001 / -0.001` | 小碎步抑制：支撑足还在滑 |

其中 anti-shuffle 的稳定门控：

\[
g = \mathbf{1}\Big(\|v^{ref}_{root,xy}\| < v_{th}\ \land\ \|g^{proj}_{xy}\| < \theta_{th}\Big)
\]

默认：

- \(v_{th}=0.12\)
- \(\theta_{th}=0.25\)

但是默认：

\[
\texttt{enable\_anti\_shuffle\_reward} = \texttt{False}
\]

所以这两项默认返回 0。

### 5.5 TWIST2 默认 reward 的几个关键细节

#### 5.5.1 TWIST2 默认非常强调 joint-space imitation

TWIST2 默认有两项非常核心：

- `tracking_joint_dof`
- `tracking_joint_vel`

这和 GR00T 完全不同。  
TWIST2 默认是在明确优化“关节轨迹要像参考动作”。

#### 5.5.2 TWIST2 默认没有根部 XY 位置跟踪

默认非零的是：

- `tracking_root_translation_z`

而不是：

- `tracking_root_translation_xy`

这表示 TWIST2 默认并不强行把全局根部 XY 位置锁死到参考轨迹上，而更强调：

- 根部高度
- 根部姿态
- 根部速度
- key body 位置

#### 5.5.3 TWIST2 的 key-body tracking 比 GR00T 更密、更硬

TWIST2 的关键 body 默认有 9 个：

- 左右手
- 左右脚
- 左右膝
- 左右肘
- 头

而且同时有：

- `tracking_keybody_pos`（局部）
- `tracking_keybody_pos_global`（全局）

并且公式里是：

\[
\exp(-10 \sum_k \|e_k\|^2)
\]

不是按 body 数做均值。  
这比 GR00T 那种“对 body error 取 mean 再做 Gaussian”的设计更硬、更容易把姿态钉死。

#### 5.5.4 `feet_air_time` 在 TWIST2 里本质上是 deficit penalty

虽然它的 weight 是正的 `5.0`，但原始项本身其实是：

\[
\min(0,\ t^{air}-t_{target})
\]

也就是：

- 足够长：`0`
- 太短：负值

所以它更像是：

- “惩罚 swing 不够长”

而不是单纯“奖励更长 air time”。

## 6. 一一对应的差异总结

### 6.1 设计哲学差异

| 维度 | GR00T | TWIST2 |
|---|---|---|
| 默认主目标 | body-space / anchor-relative 几何跟踪 | joint-space imitation + root + key-body imitation |
| tracking kernel | 大量 `exp(-mean error / std^2)` | 大量 `exp(-sum error)`，外加很多手工惩罚 |
| 根部约束 | 明确跟踪 `anchor_pos + anchor_ori` | 默认只跟踪 `root z + root rot + root vel`，不跟踪 root XY |
| 上肢/全身表达 | 通过 body-level pose/vel 和 reward points 表达 | 通过 joint+dense key-body 直接约束 |
| locomotion heuristics | 较少 | 很多：slip、stumble、air-time、ankle acc/vel、anti-shuffle |

### 6.2 公式层面的核心区别

#### 区别 1：GR00T 默认没有 `tracking_joint_dof`

这意味着：

- GR00T 奖励的是“整体 body 几何和速度像不像”
- TWIST2 奖励的是“关节角和关节速度也要像”

所以 TWIST2 默认更容易学出**更贴 reference 的低层动作复制器**，GR00T 默认更像**几何/运动学层面的 whole-body tracker**。

#### 区别 2：GR00T 的 body tracking 习惯做 mean，TWIST2 的 key-body tracking 习惯做 sum

GR00T：

\[
\exp\Big(-\frac{1}{B}\sum_i \|e_i\|^2 / \sigma^2\Big)
\]

TWIST2：

\[
\exp\Big(-10\sum_k \|e_k\|^2\Big)
\]

直接后果：

- GR00T 对 body 数目的敏感度更弱
- TWIST2 对多个关键 body 同时失配更敏感

#### 区别 3：GR00T 默认显式跟踪 anchor position，TWIST2 默认不跟踪 root XY

GR00T 默认有：

\[
\exp(-\|p^{ref}_{anchor}-p^{robot}_{anchor}\|^2/\sigma^2)
\]

TWIST2 默认没有对应的 root XY 位置 reward。  
因此：

- GR00T 更容易学出“跟着参考轨迹走”的全局路径一致性
- TWIST2 更偏向“动作形态像、速度像”，但不强绑全局 XY 路径

#### 区别 4：GR00T 默认开启 anti-shake，TWIST2 默认关闭 anti-shuffle

两者不是一回事：

- GR00T `anti_shake_ang_vel`：惩罚腕/头高频角速度抖动
- TWIST2 `step_switch_rate` + `stance_foot_speed`：惩罚慢速稳定段的小碎步

默认状态：

- GR00T：开
- TWIST2：关

#### 区别 5：GR00T 默认 contact penalty 更“全身”，TWIST2 默认 foot penalty 更“步态”

GR00T 默认 contact 逻辑：

- 惩罚不该碰地的 body 接触次数

TWIST2 默认 foot logic：

- 惩罚脚滑
- 惩罚脚绊
- 惩罚脚冲击太大
- 惩罚 swing 不够长
- 可选惩罚小碎步

所以：

- GR00T 更像 whole-body contact hygiene
- TWIST2 更像 locomotion-style regularization

### 6.3 teacher 与 student 的 reward 差异其实很小

TWIST2 teacher/student 在 reward 上的默认差异只有两点：

- `tracking_joint_vel`: `0.3 -> 0.2`
- `action_rate`: `-0.01 -> -0.05`

解释：

- student 的 joint velocity imitation 稍弱
- student 的平滑约束明显更强

这说明 TWIST2 student 的主要变化不在 reward 家族本身，而在：

- 观测压缩
- policy 结构
- teacher KL 蒸馏

## 7. 最后一句话总结

如果只看默认 RL reward：

- **GR00T** 更像“以 root/body 几何一致性为核心的 whole-body tracking reward”，奖励更抽象、更几何化、更统一。
- **TWIST2** 更像“以关节级 imitation 为核心，再叠加大量步态与稳定性先验的 tracking reward”，奖励更低层、更刚性、更工程化。

换句话说：

- GR00T 默认 reward 在问：**你整体动作像不像 reference？**
- TWIST2 默认 reward 在问：**你的每个关节、根部、关键 body、脚步细节，是否都贴近 reference 并且足够稳定？**
