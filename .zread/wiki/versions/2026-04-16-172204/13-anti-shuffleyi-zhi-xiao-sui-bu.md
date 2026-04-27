本文档介绍TWIST2项目中用于抑制机器人在站立或慢速运动时产生高频小碎步（俗称"抖腿"）的Anti-Shuffle机制。该机制通过在奖励函数中引入专门的惩罚项，使策略在保持平衡时倾向于更稳定、有节奏的动作模式，而非持续的小幅抖动。

## 问题背景

在强化学习训练人形机器人进行运动模仿任务时，一个常见的现象是策略在站立或低速运动阶段会出现高频的小碎步动作。这种行为虽然理论上可以维持平衡，但存在以下问题：机械关节频繁启停会增加磨损、能耗效率低、以及视觉效果不自然。传统的追踪奖励（如关节位置追踪、末端执行器位置追踪）无法直接惩罚这种行为，因为小碎步仍然能够较好地追踪参考动作。

## 核心设计思路

Anti-Shuffle机制采用"增量改动"的设计原则——新增的惩罚项默认关闭，不影响现有训练流程。开发者可以通过配置开关和命令行参数按需启用，便于进行A/B测试对比。

该机制包含两个互补的惩罚项：**换脚频率惩罚**（`_reward_step_switch_rate`）直接打击小碎步的核心特征——接触相位的频繁切换；**支撑脚速度惩罚**（`_reward_stance_foot_speed`）则抑制支撑阶段脚底的微滑或微抖。

一个关键的设计是**稳定门控函数**（`_anti_shuffle_stable_gate`），它确保惩罚仅在"慢速+稳态"区间生效。当参考运动速度较高或机器人处于明显失衡状态时，门控输出为零，自动放行大步恢复动作，避免将合理的动态调整也压死。

Sources: [humanoid_mimic.py](legged_gym/legged_gym/envs/base/humanoid_mimic.py#L1340-L1373)

## 实现详解

### 稳定门控函数

稳定门控函数通过两个条件判断当前是否处于"慢速稳态"区间：

```python
def _anti_shuffle_stable_gate(self):
    ref_speed_th = getattr(self.cfg.rewards, "anti_shuffle_ref_vel_th", 0.12)
    tilt_th = getattr(self.cfg.rewards, "anti_shuffle_tilt_th", 0.25)
    
    ref_speed = torch.norm(self._ref_root_vel[:, :2], dim=1)
    tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
    return ((ref_speed < ref_speed_th) & (tilt < tilt_th)).float()
```

`ref_speed_th`（默认0.12 m/s）定义了参考运动的慢速阈值，当参考动作是站立或极慢行走时该条件满足。`tilt_th`（默认0.25）基于投影重力向量的XY范数判断机器人是否处于竖直姿态附近。只有当两个条件同时满足时，门控才输出1.0，激活惩罚项。

Sources: [humanoid_mimic.py](legged_gym/legged_gym/envs/base/humanoid_mimic.py#L1340-L1347)

### 换脚频率惩罚

换脚频率惩罚的核心思路是统计每步触地状态的切换次数。使用XOR操作直接对比当前触地状态与上一时刻状态：

```python
def _reward_step_switch_rate(self):
    if not getattr(self.cfg.rewards, "enable_anti_shuffle_reward", False):
        return torch.zeros(self.num_envs, device=self.device)
    
    contact_th = getattr(self.cfg.rewards, "anti_shuffle_contact_force_th", 5.0)
    contact = self.contact_forces[:, self.feet_indices, 2] > contact_th
    
    switch_cnt = torch.logical_xor(contact, self._anti_shuffle_last_contact).float().sum(dim=1)
    self._anti_shuffle_last_contact[:] = contact
    
    return switch_cnt * self._anti_shuffle_stable_gate()
```

`torch.logical_xor(contact, self._anti_shuffle_last_contact)` 产生一个布尔张量，表示每只脚在当前时刻与上一时刻的触地状态是否不同。求和后得到每个环境的换脚次数，乘以稳定门控后得到最终惩罚值。

Sources: [humanoid_mimic.py](legged_gym/legged_gym/envs/base/humanoid_mimic.py#L1349-L1361)

### 支撑脚速度惩罚

支撑脚速度惩罚针对即使不频繁换脚、也可能存在的"脚底微滑"现象：

```python
def _reward_stance_foot_speed(self):
    if not getattr(self.cfg.rewards, "enable_anti_shuffle_reward", False):
        return torch.zeros(self.num_envs, device=self.device)
    
    contact_th = getattr(self.cfg.rewards, "anti_shuffle_contact_force_th", 5.0)
    contact = (self.contact_forces[:, self.feet_indices, 2] > contact_th).float()
    foot_speed_xy = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
    stance_speed = (foot_speed_xy * contact).sum(dim=1)
    
    return stance_speed * self._anti_shuffle_stable_gate()
```

通过将接触状态作为mask，`foot_speed_xy * contact` 确保只统计支撑脚的速度，避免惩罚摆动脚正常的抬腿动作。

Sources: [humanoid_mimic.py](legged_gym/legged_gym/envs/base/humanoid_mimic.py#L1363-L1373)

### 状态缓冲初始化

`_anti_shuffle_last_contact` 用于存储上一时刻的触地状态，其初始化和reset处理确保状态一致性：

```python
# _init_buffers() 中初始化
self._anti_shuffle_last_contact = torch.zeros(
    (self.num_envs, len(self.feet_indices)),
    device=self.device,
    dtype=torch.bool,
)

# reset_idx() 中同步
anti_shuffle_contact = self.contact_forces[env_ids][:, self.feet_indices, 2] > getattr(
    self.cfg.rewards, "anti_shuffle_contact_force_th", 5.0
)
self._anti_shuffle_last_contact[env_ids] = anti_shuffle_contact
```

reset时将历史状态同步为当前实际接触状态，避免reset导致的接触突变被误判为换脚。

Sources: [humanoid_mimic.py](legged_gym/legged_gym/envs/base/humanoid_mimic.py#L231-L236)
Sources: [humanoid_mimic.py](legged_gym/legged_gym/envs/base/humanoid_mimic.py#L457-L460)

## 配置参数

### 开关与阈值参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_anti_shuffle_reward` | bool | `False` | 总开关，默认关闭以保持向后兼容 |
| `anti_shuffle_ref_vel_th` | float | `0.12` | 参考速度阈值(m/s)，门控判定用 |
| `anti_shuffle_tilt_th` | float | `0.25` | 倾斜阈值(rad)，门控判定用 |
| `anti_shuffle_contact_force_th` | float | `5.0` | 触地力阈值(N)，判断脚是否着地 |

### 奖励权重参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `step_switch_rate` | float | `-0.20` | 换脚频率惩罚权重（负值） |
| `stance_foot_speed` | float | `-0.05` | 支撑脚速度惩罚权重（负值） |

配置文件中这些参数的设置位置如下：

Sources: [g1_mimic_future_config.py](legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py#L98-L159)

## 命令行使用

### 学生策略训练

启用Anti-Shuffle进行学生策略训练：

```bash
bash train.sh 1103_student_single cuda:0 true -0.20 -0.05
```

参数顺序为：`实验ID 设备 启用标志 换脚惩罚权重 支撑脚速度权重`

### 教师策略训练

启用Anti-Shuffle进行教师策略训练：

```bash
bash train_teacher.sh 0201_teacher cuda:0 true -0.20 -0.05
```

### 直接使用命令行参数

训练脚本内部将参数转换为CLI参数：

```bash
python train.py --task g1_stu_future \
    --enable_anti_shuffle_reward \
    --anti_shuffle_ref_vel_th 0.15 \
    --anti_shuffle_tilt_th 0.30 \
    --anti_shuffle_step_switch_scale -0.25 \
    --anti_shuffle_stance_foot_speed_scale -0.08
```

参数注入逻辑在helpers.py中实现：

Sources: [helpers.py](legged_gym/legged_gym/gym_utils/helpers.py#L223-L249)

CLI参数定义如下：

Sources: [helpers.py](legged_gym/legged_gym/gym_utils/helpers.py#L390-L396)

## 参数调优指南

### 起步推荐值

对于大多数场景，建议使用文档中提供的默认起步值：

- `step_switch_rate = -0.20`：惩罚每步换脚约产生-0.2的奖励
- `stance_foot_speed = -0.05`：惩罚支撑脚1 m/s速度约产生-0.05的奖励

这两个值的比例约为4:1，反映了"换脚频率"是小碎步的主要特征。

### 调整策略

当观察到惩罚过于激进（机器人迈步迟疑、容易摔倒）时，可以降低惩罚权重或调整门控阈值：

- 减小`step_switch_rate`绝对值（如改为-0.10）
- 增大`anti_shuffle_ref_vel_th`（如改为0.20）使更多运动阶段被判定为"非慢速"

当观察到惩罚不足（小碎步依然存在）时，可以：

- 增大`step_switch_rate`绝对值（如改为-0.30）
- 减小`anti_shuffle_ref_vel_th`（如改为0.08）使更多阶段进入惩罚区间

### 门控阈值的影响

门控阈值的物理含义：
- `anti_shuffle_ref_vel_th = 0.12` 意味着当参考运动速度低于约0.12 m/s时激活惩罚。这大致对应站立或极其缓慢的原地调整。
- `anti_shuffle_tilt_th = 0.25` 意味着当身体倾斜角度约在15度以内时激活惩罚。更大的倾斜被认为是"失衡恢复"状态。

## 训练脚本实现

训练脚本使用Shell数组构建参数列表，确保参数中可能存在的空格和特殊字符被正确处理：

```bash
extra_args=(
  --anti_shuffle_step_switch_scale "${step_switch_scale}"
  --anti_shuffle_stance_foot_speed_scale "${stance_foot_speed_scale}"
)

if [[ "${enable_anti_shuffle}" == "1" || "${enable_anti_shuffle}" == "true" || "${enable_anti_shuffle}" == "True" ]]; then
  extra_args+=(--enable_anti_shuffle_reward)
fi
```

脚本支持多种布尔值表示形式（`1`/`true`/`True`），降低命令输入错误概率。

Sources: [train.sh](train.sh#L27-L55)
Sources: [train_teacher.sh](train_teacher.sh#L21-L37)

## 技术备注

### 为什么使用增量改动

传统的奖励函数修改方式是在基础奖励上直接叠加新项，但这种方式会改变所有历史实验的奖励分布，导致无法与旧实验进行公平对比。Anti-Shuffle采用"默认关闭+显式开启"的增量改动策略：

1. 配置文件中设置`enable_anti_shuffle_reward = False`
2. 奖励函数开头检查开关状态，未开启时返回零惩罚
3. 通过CLI参数或脚本参数动态开启

这种方式确保了向后兼容性，旧实验继续使用原有奖励配置，新实验按需启用。

### 门控机制的双重作用

稳定门控不仅是简单的条件判断，它实际上定义了一种"何时应该优雅"的策略语义：

- 当参考动作速度高（正常行走/跑步）时，机器人自然会有节奏的迈步，不应被惩罚
- 当参考动作速度低（站立/慢速）时，惩罚引导机器人保持静止或小幅调整，而非频繁换脚
- 当机器人失衡时（倾斜超过阈值），应该允许大步恢复动作

这种语义设计避免了"一刀切"惩罚导致的负迁移问题。