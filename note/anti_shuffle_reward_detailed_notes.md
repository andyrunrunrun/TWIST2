# Anti-Shuffle 改动详解（学生 + 老师训练）

本文是本次“抑制小碎步/抖腿”改动的完整技术备注，覆盖：

1. 改了哪些文件
2. 每段代码为什么要这样写
3. 为什么放在那个位置
4. 为什么它能减少抖腿
5. 如何在 student / teacher 训练中使用新参数

---

## 1. 改动目标与约束

### 目标

- 抑制机器人在站立/慢速段的高频小碎步（抖腿）。
- 允许机器人在明显失衡时迈大步恢复（不把恢复动作也压死）。

### 约束

- 必须是**增量改动**，默认不影响旧实验。
- 不改动现有训练主流程，只增加可控参数。

对应策略：新增奖励 + 开关门控 + 默认关闭。

---

## 2. 核心环境改动（`humanoid_mimic.py`）

文件：`legged_gym/legged_gym/envs/base/humanoid_mimic.py`

---

### 2.1 新增接触历史缓冲：`self._anti_shuffle_last_contact`

代码位置：`_init_buffers()` 中，`super()._init_buffers()` 之后。

```python
self._anti_shuffle_last_contact = torch.zeros(
    (self.num_envs, len(self.feet_indices)),
    device=self.device,
    dtype=torch.bool,
)
```

为什么要这样写：

- `step_switch_rate` 需要比较“当前脚接触状态”与“上一时刻接触状态”。
- 使用 `bool` 节省显存且语义清晰（接触/不接触）。
- 维度是 `[num_envs, n_feet]`，可并行处理所有环境和双脚。

为什么放在这里：

- `_init_buffers()` 是所有按环境维度状态张量的统一初始化位置。
- 放这里能保证 reset 前后都由同一生命周期管理，避免散落在奖励函数里临时创建张量导致性能抖动。

---

### 2.2 在 `reset_idx()` 对齐接触历史，避免 reset 后伪“换脚”

代码位置：`reset_idx()` 的 buffer reset 区域。

```python
anti_shuffle_contact = self.contact_forces[env_ids, self.feet_indices, 2] > getattr(
    self.cfg.rewards, "anti_shuffle_contact_force_th", 5.0
)
self._anti_shuffle_last_contact[env_ids] = anti_shuffle_contact
```

为什么要这样写：

- 如果 reset 后不更新 `last_contact`，下一步会把“reset 导致的接触状态变化”误判为一次换脚，造成假惩罚。
- 这里直接用当前接触状态覆盖历史状态，确保下一步比较是“真实动态变化”。

为什么放在 `reset_idx()`：

- 只有 reset 才会发生状态瞬时重置，这是最可能引入伪信号的位置。
- 在这里修正最直接，不会污染其它流程。

---

### 2.3 新增稳定门控函数：`_anti_shuffle_stable_gate()`

```python
ref_speed = torch.norm(self._ref_root_vel[:, :2], dim=1)
tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
return ((ref_speed < ref_speed_th) & (tilt < tilt_th)).float()
```

为什么要这样写：

- `ref_speed` 小：表示当前参考动作是站立/慢速，最容易出现“原地小碎步刷稳态”。
- `tilt` 小：表示不在明显失衡状态。
- 仅在“慢 + 稳”区间激活 anti-shuffle，失衡时自动放行大步恢复，避免负迁移。

为什么放在独立函数：

- 两个奖励项（换脚惩罚、支撑脚速度惩罚）都要复用同样门控。
- 统一门控可以减少参数漂移，调参时只改一处阈值。

---

### 2.4 新增主项奖励：`_reward_step_switch_rate()`

```python
if not getattr(self.cfg.rewards, "enable_anti_shuffle_reward", False):
    return torch.zeros(self.num_envs, device=self.device)

contact = self.contact_forces[:, self.feet_indices, 2] > contact_th
switch_cnt = torch.logical_xor(contact, self._anti_shuffle_last_contact).float().sum(dim=1)
self._anti_shuffle_last_contact[:] = contact
return switch_cnt * self._anti_shuffle_stable_gate()
```

为什么要这样写：

- `xor` 直接统计接触状态切换次数（最直接对应“小碎步高频换脚”）。
- `sum(dim=1)` 得到每个环境每步切换次数。
- 最前面的开关判断保证默认关闭时返回全 0，行为与旧版一致。

为什么它能抑制小碎步：

- 小碎步的典型特征就是接触相位高频切换。
- 这个项直接对该行为加负激励，等价于“步频成本”。

---

### 2.5 新增辅助项奖励：`_reward_stance_foot_speed()`

```python
if not getattr(self.cfg.rewards, "enable_anti_shuffle_reward", False):
    return torch.zeros(self.num_envs, device=self.device)

contact = (self.contact_forces[:, self.feet_indices, 2] > contact_th).float()
foot_speed_xy = torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2)
stance_speed = (foot_speed_xy * contact).sum(dim=1)
return stance_speed * self._anti_shuffle_stable_gate()
```

为什么要这样写：

- 只看支撑脚（`* contact`），避免惩罚摆动脚正常抬脚动作。
- 惩罚的是支撑脚平面速度，针对“脚底微滑/微挪动”。
- 与 `step_switch_rate` 互补：一个打“换脚频率”，一个打“支撑脚抖动”。

为什么它能抑制抖腿：

- 即使策略不频繁换脚，也可能在接触中通过小幅滑动抖动维稳。
- 该项把这种“接触中乱动”也纳入成本。

---

## 3. 配置改动（学生 + 老师）

---

### 3.1 Student 配置补充

文件：`legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`

新增字段：

- `enable_anti_shuffle_reward = False`
- `anti_shuffle_ref_vel_th = 0.12`
- `anti_shuffle_tilt_th = 0.25`
- `anti_shuffle_contact_force_th = 5.0`

新增 scale：

- `step_switch_rate = -0.20`
- `stance_foot_speed = -0.05`

为什么这样设计：

- 默认 `False`，保证旧实验不受影响。
- 权重默认提供“可用起点”，但只有开关开启才生效。

---

### 3.2 Teacher 配置补充

文件：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`

同样新增了上述开关/阈值/scale。

为什么要在 teacher 也加：

- 用户要求 student 和 teacher 训练代码都支持这套参数。
- teacher 也可能出现“抖步维稳”局部最优，统一机制便于对齐 teacher/student 行为偏好。

---

## 4. 参数解析与训练入口改动

文件：`legged_gym/legged_gym/gym_utils/helpers.py`

---

### 4.1 新增 CLI 参数

新增参数：

- `--enable_anti_shuffle_reward`
- `--disable_anti_shuffle_reward`
- `--anti_shuffle_ref_vel_th`
- `--anti_shuffle_tilt_th`
- `--anti_shuffle_contact_force_th`
- `--anti_shuffle_step_switch_scale`
- `--anti_shuffle_stance_foot_speed_scale`

为什么要加显式参数：

- 之前虽然可以用 dot-notation 覆盖，但不够直观，易拼错。
- 显式参数更适合写训练脚本和文档，也便于实验记录。

---

### 4.2 `update_cfg_from_args()` 映射逻辑

关键逻辑：

1. 先处理 enable/disable 开关
2. 再处理阈值覆盖
3. 最后处理 scale 覆盖
4. 仅在 `env_cfg.rewards` 与对应字段存在时应用，避免对不相关任务误改

为什么放在 `update_cfg_from_args()`：

- 这是仓库统一的 CLI -> cfg 注入入口。
- 放这里可确保 `train.py`、`play.py` 等共享同一覆盖机制。

#### 代码片段与逐段解释（参数注入）

```python
if getattr(args, "enable_anti_shuffle_reward", False):
    if hasattr(env_cfg.rewards, "enable_anti_shuffle_reward"):
        setattr(env_cfg.rewards, "enable_anti_shuffle_reward", True)
if getattr(args, "disable_anti_shuffle_reward", False):
    if hasattr(env_cfg.rewards, "enable_anti_shuffle_reward"):
        setattr(env_cfg.rewards, "enable_anti_shuffle_reward", False)
```

为什么这么写：

- 提供正反两个开关，避免“配置默认开/关”和“命令行期望”冲突时无解。
- `hasattr` 防御式判断，防止在非 mimic 任务上硬写入导致错误。

```python
anti_shuffle_scalar_overrides = [
    ("anti_shuffle_ref_vel_th", "anti_shuffle_ref_vel_th"),
    ("anti_shuffle_tilt_th", "anti_shuffle_tilt_th"),
    ("anti_shuffle_contact_force_th", "anti_shuffle_contact_force_th"),
]
```

为什么这么写：

- 用统一表驱动写法减少重复 `if`，后续新增阈值参数时只需加一行映射。
- 左边是 CLI 参数名，右边是 cfg 字段名，便于保持一一对应关系。

```python
if hasattr(env_cfg.rewards, "scales"):
    step_switch_scale = getattr(args, "anti_shuffle_step_switch_scale", None)
    if step_switch_scale is not None and hasattr(env_cfg.rewards.scales, "step_switch_rate"):
        setattr(env_cfg.rewards.scales, "step_switch_rate", float(step_switch_scale))
```

为什么这么写：

- scale 覆盖必须是“可选”，所以默认 `None`，只有显式传参才覆盖。
- 强制 `float(...)` 避免字符串进入 cfg 后在 reward 计算时报类型错。

#### 代码片段与逐段解释（参数注册）

```python
{"name": "--enable_anti_shuffle_reward", "action": "store_true", "default": False, ...}
{"name": "--anti_shuffle_step_switch_scale", "type": float, "default": None, ...}
```

为什么这么写：

- `enable` 用 `store_true` 更符合 CLI 习惯（出现即开启）。
- scale 用 `default=None`，确保不会隐式覆盖配置文件中的权重。

---

## 5. 训练脚本改动（学生 + 老师）

---

### 5.1 学生脚本 `train.sh`

文件：`train.sh`

新增可选入参：

- 第 3 个：是否启用 anti-shuffle（`true/false`）
- 第 4 个：`step_switch_rate` scale
- 第 5 个：`stance_foot_speed` scale

为什么这样做：

- 保持原调用方式兼容：`bash train.sh <exptid> <device>` 仍可用。
- 需要时再显式打开 anti-shuffle，满足增量改造要求。

#### 代码片段与逐段解释

```bash
enable_anti_shuffle=${3:-false}
step_switch_scale=${4:--0.20}
stance_foot_speed_scale=${5:--0.05}
```

为什么这么写：

- 前两个参数保持原语义不变，新参数从第 3 位开始，不破坏旧脚本调用。
- scale 给出默认建议值，减少首次实验试错成本。

```bash
extra_args=(
  --anti_shuffle_step_switch_scale "${step_switch_scale}"
  --anti_shuffle_stance_foot_speed_scale "${stance_foot_speed_scale}"
)
if [[ "${enable_anti_shuffle}" == "1" || "${enable_anti_shuffle}" == "true" || "${enable_anti_shuffle}" == "True" ]]; then
  extra_args+=(--enable_anti_shuffle_reward)
fi
```

为什么这么写：

- 用 `array` 而不是字符串拼接，避免空格/引号导致参数拆分错误。
- 同时兼容 `1/true/True`，降低命令输入错误概率。

---

### 5.2 老师脚本 `train_teacher.sh`（新增）

文件：`train_teacher.sh`

新增 teacher 封装脚本，参数风格与 `train.sh` 对齐，支持同样 anti-shuffle 控制。

为什么新增而不是只改文档：

- 用户明确要求“更新训练学生和训练老师的代码”。
- teacher 之前没有对等的一键脚本，不利于统一实验流程。

#### 代码片段与逐段解释

```bash
task_name="${robot_name}_priv_mimic"
proj_name="${robot_name}_priv_mimic"
```

为什么这么写：

- teacher 任务固定对应 `g1_priv_mimic`，脚本直接内置可减少命令输入歧义。
- 与 student 脚本参数风格对齐，方便同一批实验管理。

---

## 6. 训练文档改动

文件：

- `train_eval_student.md`
- `train_eval_teacher.md`

改动内容：

1. 增加 anti-shuffle 参数表（含默认值和含义）
2. 增加开启 anti-shuffle 的完整训练命令示例
3. 说明封装脚本如何传入 anti-shuffle 参数

为什么要改文档：

- 参数不是改完代码就能安全使用，必须给出解释与推荐起始值。
- 便于后续复现实验和团队协作。

---

## 7. 为什么这些改动能减少抖腿

合力机制如下：

1. `step_switch_rate`：惩罚高频换脚（直接打击小碎步核心模式）
2. `stance_foot_speed`：惩罚支撑脚乱动（抑制脚底微抖/微滑）
3. 稳定门控：仅在慢速稳态时惩罚；失衡时放行大步恢复
4. 默认关闭：不破坏旧训练；按需启用便于 A/B 测试

---

## 8. 使用建议（起步值）

推荐先用：

- `--enable_anti_shuffle_reward`
- `--anti_shuffle_step_switch_scale -0.20`
- `--anti_shuffle_stance_foot_speed_scale -0.05`

若出现“过于僵硬、恢复慢”：

1. `step_switch_rate` 绝对值减小（如 `-0.12`）
2. `stance_foot_speed` 绝对值减小（如 `-0.02`）
3. `anti_shuffle_tilt_th` 适当调大，扩大“放行恢复动作”范围

---

## 9. 兼容性说明

默认配置下（`enable_anti_shuffle_reward=False`）：

1. 旧命令可直接运行
2. 新奖励函数返回 0，不改变旧 reward 数值结构
3. 这是严格增量改动，可安全回滚或按实验启用
