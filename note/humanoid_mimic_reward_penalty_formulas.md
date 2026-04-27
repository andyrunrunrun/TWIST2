# `HumanoidMimic` 奖励与惩罚公式总览

本文对应源码：`legged_gym/legged_gym/envs/base/humanoid_mimic.py`  
总奖励拼装逻辑来自：`legged_gym/legged_gym/envs/base/humanoid_char.py` 的 `compute_reward()`

---

## 1. 总奖励是如何拼装的

记某个环境在时刻 `t` 的第 `k` 个子项输出为 `r_k(t)`（即 `_reward_xxx()` 的返回值），权重为 `w_k`（来自 `cfg.rewards.scales`）。

代码中的总和形式是：

$$
R_{\text{sum}}(t)
=
\sum_{k \in \mathcal{K}_{\text{reg}}}
\alpha_{\text{reg}} \, w_k \, r_k(t)
\;+\;
\sum_{k \notin \mathcal{K}_{\text{reg}}}
w_k \, r_k(t),
$$

其中：

- `\mathcal{K}_{reg}` 对应 `cfg.rewards.regularization_names`
- `\alpha_{reg}` 对应 `cfg.rewards.regularization_scale`

然后会做可选裁剪：

$$
R_1(t)=\max(R_{\text{sum}}(t),0)\quad\text{(若 only\_positive\_rewards=True)}
$$

$$
R_2(t)=\max(R_1(t),-0.5)\quad\text{(若 clip\_rewards=True)}
$$

最后再加终止项（在裁剪之后）：

$$
R(t)=R_2(t)+w_{\text{term}} \cdot r_{\text{term}}(t).
$$

> 关键点：`_reward_xxx()` 函数本身很多返回的是“误差/范数/计数”等非负量。  
> 其是“奖励”还是“惩罚”，主要由 `w_k` 的正负决定（例如 `feet_slip` 常配负权重）。

---

## 2. 符号说明

- `q`：当前关节位置 `dof_pos`，`\dot q`：当前关节速度 `dof_vel`
- `q^*`、`\dot q^*`：参考动作给出的目标关节状态 `_ref_dof_pos/_ref_dof_vel`
- `p_root, R_root, v_root, \omega_root`：当前 root 的位置/旋转/线速度/角速度
- `p_root^*, R_root^*, v_root^*, \omega_root^*`：参考 root 状态
- `\|\cdot\|`：欧氏范数；代码里常见 `sum(x*x)` 即平方范数
- `\exp(-c e)` 型项：误差越小越接近 1，误差越大越接近 0
- `contact`：脚接触布尔（通常 `F_z > 5N`）

---

## 3. 逐项公式与含义

## 3.1 跟踪类（通常正权重）

- `_reward_alive`
  - 公式：`r = 1`
  - 含义：存活常数项。

- `_reward_tracking_joint_dof`
  - 公式：
    $$
    e=\sum_j w_j^{dof}(q_j^*-q_j)^2,\quad r=\exp(-0.15\,e)
    $$
  - 含义：关节角度跟踪。

- `_reward_tracking_joint_vel`
  - 公式：
    $$
    e=\sum_j w_j^{dof}(\dot q_j^*-\dot q_j)^2,\quad r=\exp(-0.01\,e)
    $$
  - 含义：关节速度跟踪。

- `_reward_tracking_root_pose`
  - 公式：
    $$
    e_p=\|p_{root}^*-p_{root}\|_2^2,\;
    e_R=\mathrm{quat\_diff\_angle}(R_{root},R_{root}^*)^2
    $$
    $$
    r=\exp\left(-(e_p+0.1\,e_R)\right)
    $$
  - 含义：root 平移+旋转联合跟踪。

- `_reward_tracking_root_pose_delta_local`
  - 公式：
    $$
    \Delta p_{\text{local}}
    =\mathrm{quat\_rotate\_inverse}(R_{root}^{t-1},\, p_{root}^{t}-p_{root}^{t-1})
    $$
    $$
    e=\|\Delta p_{\text{local}}^*-\Delta p_{\text{local}}\|_2^2,\quad
    r=\exp(-e)
    $$
  - 含义：局部位移增量跟踪（帧间动态一致性）。

- `_reward_tracking_root_rotation_delta_local`
  - 公式（按代码）：
    $$
    \Delta R \approx R_{root}^{t}-R_{root}^{t-1}\;\text{(四元数分量差)}
    $$
    转欧拉后再转到上一步局部系，得到 `\Delta \theta_{local}`，再
    $$
    e=\|\Delta \theta_{local}^*-\Delta \theta_{local}\|_2^2,\quad
    r=\exp(-e)
    $$
  - 含义：局部旋转增量跟踪。

- `_reward_tracking_root_translation`
  - 公式：
    $$
    e=\|p_{root}^*-p_{root}\|_2^2,\quad r=\exp(-5e)
    $$

- `_reward_tracking_root_translation_xy`
  - 公式：
    $$
    e=\|p_{root,xy}^*-p_{root,xy}\|_2^2,\quad r=\exp(-5e)
    $$

- `_reward_tracking_root_translation_z`
  - 公式：
    $$
    e=(p_{root,z}^*-p_{root,z})^2,\quad r=\exp(-5e)
    $$

- `_reward_tracking_root_rotation`
  - 公式：
    $$
    e=\mathrm{quat\_diff\_angle}(R_{root},R_{root}^*)^2,\quad r=\exp(-5e)
    $$

- `_reward_tracking_root_vel`
  - 公式：
    $$
    e_v=\|v^*-v\|_2^2,\quad
    e_\omega=\|\omega^*-\omega\|_2^2,\quad
    r=\exp(-(e_v+0.5e_\omega))
    $$
  - 备注：`global_obs=False` 时会先把参考速度转到局部坐标再比。

- `_reward_tracking_root_linear_vel`
  - 公式：
    $$
    e=\|v^*-v\|_2^2,\quad r=\exp(-e)
    $$

- `_reward_tracking_root_angular_vel`
  - 公式：
    $$
    e=\|\omega^*-\omega\|_2^2,\quad r=\exp(-e)
    $$

- `_reward_tracking_keybody_pos`
  - 公式：
    $$
    e_i=\|p_{key,i}^{local}-p_{key,i}^{*,local}\|_2^2
    $$
    可选 limb 权重后：
    $$
    e=\sum_i \lambda_i e_i,\quad r=\exp(-10e)
    $$
  - 含义：关键身体点在 yaw 对齐局部系下的跟踪。

- `_reward_tracking_keybody_pos_global`
  - 公式：
    $$
    e_i=\|p_{key,i}-p_{key,i}^{*}\|_2^2,\quad
    e=\sum_i \lambda_i e_i,\quad
    r=\exp(-10e)
    $$
  - 含义：关键点全局坐标跟踪。

- `_reward_tracking_feet_height`
  - 代码语义公式：
    - 维护累计脚高 `h_i^{acc}`（通过 `delta_z` 逐步累加）
    - 布尔匹配：
      $$
      b_i=\mathbf{1}\left(|h_i^{acc}-h_i^*|<0.05\right)
      $$
    - 若参考速度很小（原地）则置零：
      $$
      \|v_{root,xy}^*\|<0.1 \Rightarrow b_i=0
      $$
    - 接触脚会清零累计（`h_i^{acc}\leftarrow 0`）
    - 返回：
      $$
      r=\sum_i b_i
      $$
  - 含义：摆动相脚高轨迹匹配。

---

## 3.2 约束/惩罚类（常配负权重）

- `_reward_collision`
  - 公式：
    $$
    r=\sum_b \mathbf{1}\left(\|F_b\|_2>0.1\right)
    $$
  - 含义：惩罚受罚身体部位碰撞次数。

- `_reward_dof_pos_limits`
  - 公式（逐关节越界量）：
    $$
    r=\sum_j \left[\max(q_j^{min}-q_j,0)+\max(q_j-q_j^{max},0)\right]
    $$

- `_reward_dof_torque_limits`
  - 公式：
    $$
    r=\sum_j \max\left(\frac{|\tau_j|}{\tau_j^{lim}}-\tau_{soft},\,0\right)
    $$

- `_reward_feet_stumble`
  - 公式：
    $$
    r=\mathbf{1}\Big(\exists i,\;\|F_{i,xy}\|_2 > 4|F_{i,z}|\Big)
    $$
  - 含义：脚水平冲击过大（绊脚）事件。

- `_reward_feet_contact_forces`
  - 代码语义：
    $$
    s=\|F_{z,\text{feet}}\|_2,\quad
    r=
    \begin{cases}
    0,& s<F_{max}\\
    s-F_{max},& s\ge F_{max}
    \end{cases}
    $$

- `_reward_feet_height`
  - 公式：
    $$
    d_i=|h_i-h_{target}|,\quad d=\min_i d_i,\quad r=\max(d-0.02,0)
    $$
  - 含义：脚高偏离目标带宽的惩罚。

- `_reward_feet_slip`
  - 公式：
    $$
    r=\sum_i \mathbf{1}(contact_i)\sqrt{\|v_{foot,i,xy}\|_2}
    $$
  - 含义：接触相脚滑惩罚。

- `_reward_lin_vel_z`
  - 公式：
    $$
    r=v_{base,z}^2
    $$

- `_reward_ang_vel_xy`
  - 公式：
    $$
    r=\omega_x^2+\omega_y^2
    $$

- `_reward_orientation`
  - 公式：
    $$
    r=g_{proj,x}^2+g_{proj,y}^2
    $$
  - 含义：机身倾斜惩罚（偏离竖直）。

- `_reward_dof_acc`
  - 公式：
    $$
    r=\sum_j \left(\frac{\dot q_j^{t-1}-\dot q_j^t}{\Delta t}\right)^2
    $$

- `_reward_action_rate`
  - 公式：
    $$
    r=\|a_t-a_{t-1}\|_2
    $$

- `_reward_dof_vel`
  - 公式：
    $$
    r=\sum_j \dot q_j^2
    $$

- `_reward_base_acc`
  - 公式：
    $$
    r=\left\|\frac{v_{root}^{t-1}-v_{root}^{t}}{\Delta t}\right\|_2^2
    +
    \left\|\frac{\omega_{root}^{t-1}-\omega_{root}^{t}}{\Delta t}\right\|_2^2
    $$

- `_reward_torque_penalty`
  - 公式：
    $$
    r=\sum_j \tau_j^2
    $$

---

## 3.3 Anti-shuffle 专项

- `_anti_shuffle_stable_gate`
  - 公式：
    $$
    g=
    \mathbf{1}\left(\|v_{root,xy}^*\|<v_{th}\right)
    \cdot
    \mathbf{1}\left(\|g_{proj,xy}\|<\theta_{th}\right)
    $$
  - 含义：仅在“参考慢速且姿态稳定”阶段激活 anti-shuffle 惩罚。

- `_reward_step_switch_rate`
  - 公式：
    $$
    c_t=\mathbf{1}(F_{z,\text{feet}}>F_{th}),
    \quad
    switch=\sum_i \mathrm{XOR}(c_{t,i},c_{t-1,i}),
    \quad
    r=switch\cdot g
    $$
  - 含义：惩罚脚接触状态高频切换（碎步/抖步）。

- `_reward_stance_foot_speed`
  - 公式：
    $$
    c_i=\mathbf{1}(F_{z,i}>F_{th}),\quad
    r=\left(\sum_i c_i\|v_{foot,i,xy}\|_2\right)\cdot g
    $$
  - 含义：惩罚支撑脚在地面上的滑动速度。

---

## 3.4 命令跟踪项

- `_reward_feet_air_time`
  - 代码语义公式：
    - 仅在“首次触地时刻”计分
    - 记目标腾空时间 `T*`，当前累计腾空 `T_i`
      $$
      a_i=\min(T_i-T^*,\,0)
      $$
    - 合计：
      $$
      r=\sum_i a_i
      $$
    - 若参考行走速度过小（`||v^*_{root,xy}|| <= 0.05`）则该项置零
  - 解释：该项原始值通常非正，常用于抑制不合理摆腿节律（是否“奖励”取决于 scale 符号）。

- `_reward_tracking_lin_vel`
  - 公式：
    $$
    e=\|cmd_{xy}-v_{base,xy}\|_2^2,\quad
    r=\exp\left(-\frac{e}{\sigma_{lin}}\right)
    $$

- `_reward_tracking_ang_vel`
  - 公式：
    $$
    e=(cmd_{yaw}-\omega_{base,z})^2,\quad
    r=\exp\left(-\frac{e}{\sigma_{ang}}\right)
    $$

---

## 4. “奖励/惩罚”如何判定（非常重要）

在该实现中，`_reward_xxx()` 多数输出是“非负误差或指标”。  
最终语义取决于 `w_k`：

- `w_k > 0`：鼓励该指标增大（或鼓励误差衰减项接近 1）
- `w_k < 0`：惩罚该指标增大

示例（常见配置）：`tracking_*` 权重为正，`feet_slip`、`dof_acc`、`action_rate` 等为负。

---

## 5. 实践建议（调参时）

- 对 `exp(-c e)` 型项：
  - `c` 越大，容错越小（更“苛刻”）
  - `c` 越小，曲线更平缓（更“宽容”）
- 对惩罚项：
  - 建议先保证量纲可比（例如速度、力矩、加速度）
  - 再逐步放大负权重，避免 early training 直接“惩罚主导”
- 对 anti-shuffle：
  - 先确认 `stable_gate` 条件是否经常满足，否则项几乎不起作用
  - `step_switch_rate` 和 `stance_foot_speed` 建议联调，防止只抑制了切换却带来拖步

---

## 6. 对照清单（函数名）

`HumanoidMimic` 中定义的 reward 项包括：

- `_reward_alive`
- `_reward_tracking_joint_dof`
- `_reward_tracking_joint_vel`
- `_reward_tracking_root_pose`
- `_reward_tracking_root_pose_delta_local`
- `_reward_tracking_root_rotation_delta_local`
- `_reward_tracking_root_translation`
- `_reward_tracking_root_translation_xy`
- `_reward_tracking_root_translation_z`
- `_reward_tracking_root_rotation`
- `_reward_tracking_root_vel`
- `_reward_tracking_root_linear_vel`
- `_reward_tracking_root_angular_vel`
- `_reward_tracking_keybody_pos`
- `_reward_tracking_keybody_pos_global`
- `_reward_tracking_feet_height`
- `_reward_collision`
- `_reward_dof_pos_limits`
- `_reward_dof_torque_limits`
- `_reward_feet_stumble`
- `_reward_feet_contact_forces`
- `_reward_feet_height`
- `_reward_feet_slip`
- `_reward_lin_vel_z`
- `_reward_ang_vel_xy`
- `_reward_orientation`
- `_reward_dof_acc`
- `_reward_action_rate`
- `_reward_dof_vel`
- `_reward_base_acc`
- `_reward_torque_penalty`
- `_reward_step_switch_rate`
- `_reward_stance_foot_speed`
- `_reward_feet_air_time`
- `_reward_tracking_lin_vel`
- `_reward_tracking_ang_vel`

> 若你需要，我可以在下一版附上“当前具体配置文件（如 `g1_mimic_future_config.py`）下的实际权重表”，并自动生成一栏“理论上是奖励还是惩罚”。
