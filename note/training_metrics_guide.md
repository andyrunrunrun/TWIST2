# TWIST2 训练指标说明

## 训练效率指标

| 指标 | 含义 | 趋势 |
|------|------|------|
| `Computation` | 训练速度（每秒步数） | **越大越好** |
| `collection_time` | 数据采集时间 | **越小越好** |
| `learning_time` | 学习更新时间 | **越小越好** |
| `Iteration time` | 单次迭代总时间 | **越小越好** |

---

## 核心损失指标

| 指标 | 含义 | 趋势 |
|------|------|------|
| `Value function loss` | 值函数损失，评估状态价值 | **越小越好** |
| `Surrogate loss` | PPO裁剪目标损失，策略优化目标 | **越小越好**（收敛到稳定值） |

---

## 训练状态指标

| 指标 | 含义 | 趋势 |
|------|------|------|
| `Mean action noise std` | 动作噪声标准差（探索力度） | 随训练逐渐减小（从1.0→0） |
| `Mean reward (total)` | 总奖励（最重要的指标） | **越大越好** |
| `Mean episode length` | 平均episode长度（步数） | **越大越好**（越长越稳定） |

---

## 奖励项 (rew_*)

**所有奖励项都是越大越好**

| 奖励项 | 含义 | 理想值 |
|--------|------|--------|
| `rew_action_rate` | 动作平滑度奖励 | 接近0 |
| `rew_alive` | 存活奖励 | 正值 |
| `rew_ang_vel_xy` | 角速度控制奖励 | 接近0 |
| `rew_ankle_dof_acc` | 踝关节加速度惩罚 | 接近0 |
| `rew_ankle_dof_vel` | 踝关节速度惩罚 | 接近0 |
| `rew_dof_acc` | 关节加速度惩罚 | 接近0 |
| `rew_dof_pos_limits` | 关节位置限制惩罚 | 接近0 |
| `rew_dof_torque_limits` | 关节力矩限制惩罚 | 接近0 |
| `rew_dof_vel` | 关节速度惩罚 | 接近0 |
| `rew_feet_air_time` | 脚部腾空时间奖励 | 正值（走得更好） |
| `rew_feet_contact_forces` | 脚部接触力惩罚 | 接近0 |
| `rew_feet_slip` | 脚部打滑惩罚 | 接近0 |
| `rew_feet_stumble` | 脚部绊倒惩罚 | 接近0 |
| `rew_tracking_joint_dof` | 关节位置跟踪奖励 | **越大越好** |
| `rew_tracking_joint_vel` | 关节速度跟踪奖励 | **越大越好** |
| `rew_tracking_keybody_pos` | 关键身体位置跟踪奖励 | **越大越好** |
| `rew_tracking_keybody_pos_global` | 全局关键身体位置跟踪奖励 | **越大越好** |
| `rew_tracking_root_angular_vel` | 根角速度跟踪奖励 | **越大越好** |
| `rew_tracking_root_linear_vel` | 根线速度跟踪奖励 | **越大越好** |
| `rew_tracking_root_rotation` | 根旋转跟踪奖励 | **越大越好** |
| `rew_tracking_root_translation_z` | 根高度（Z轴）跟踪奖励 | **越大越好** |

---

## 误差项 (error_*)

**所有误差项都是越小越好**

| 误差项 | 含义 | 理想值 |
|--------|------|--------|
| `error_tracking_joint_dof` | 关节位置跟踪误差 | **接近0** |
| `error_tracking_joint_vel` | 关节速度跟踪误差 | **接近0** |
| `error_tracking_keybody_pos` | 关键身体位置跟踪误差 | **接近0** |
| `error_tracking_root_ang_vel` | 根角速度跟踪误差 | **接近0** |
| `error_tracking_root_pose_delta_local` | 根姿态局部变化误差 | **接近0** |
| `error_tracking_root_rotation` | 根旋转跟踪误差 | **接近0** |
| `error_tracking_root_rotation_delta_local` | 根旋转局部变化误差 | **接近0** |
| `error_tracking_root_translation` | 根位置跟踪误差 | **接近0** |
| `error_tracking_root_vel` | 根速度跟踪误差 | **接近0** |

---

## 其他训练指标

| 指标 | 含义 | 趋势 |
|------|------|------|
| `Regularization_scale` | 正则化强度（控制姿态自然度） | 根据需要调整 |
| `Average_episode_length` | 平均episode长度 | **越大越好**（机器人越稳定） |
| `Grad_penalty_coef` | 梯度惩罚系数 | 通常为0 |
| `Mean_motion_difficulty` | 平均动作难度（1-10） | 随训练增加 |

---

## 如何判断训练效果

### ✅ 训练良好的标志
- `Mean reward (total)` 持续上升
- `Mean episode length` 逐渐增加并稳定
- `error_tracking_*` 各项误差逐渐下降
- `rew_tracking_*` 各项跟踪奖励逐渐上升

### ⚠️ 需要注意的信号
- `Mean episode length` 很低（<100）→ 机器人经常摔倒
- `error_tracking_joint_dof` 很高（>1.0）→ 关节跟踪差
- `rew_tracking_*` 为负且绝对值大 → 无法模仿参考动作
- `rew_feet_stumble` 为负且绝对值大 → 经常绊倒

### 📊 训练阶段判断
- **早期（0-5000 iter）**：reward 快速上升，episode length 短
- **中期（5000-20000 iter）**：reward 稳定上升，episode length 逐渐增长
- **后期（20000+ iter）**：reward 趋于稳定，episode length 达到理想值

---

## 快速判断标准

| 指标 | 差 | 中等 | 好 |
|------|-----|-------|-----|
| Mean reward | < 50 | 50-200 | > 200 |
| Mean episode length | < 500 | 500-1500 | > 1500 |
| error_tracking_joint_dof | > 0.5 | 0.2-0.5 | < 0.2 |
| error_tracking_keybody_pos | > 0.2 | 0.05-0.2 | < 0.05 |
