# `g1_stu_future` 训练流程与继承关系笔记

这份笔记回答两个问题：

1. 运行 `bash train.sh ...` 之后，`g1_stu_future` 任务按什么顺序加载关键 Python 文件？
2. `legged_gym`、`rsl_rl` 里的环境、配置、runner、algorithm、policy 是如何协作完成训练的？

说明：

- 这里讲的是**关键运行链路**，不展开 Python 标准库、Isaac Gym 内部模块和无关依赖。
- 重点聚焦 `g1_stu_future`，但 MoE / Transformer / Diffusion 任务基本复用同一套装配逻辑，只是换了 task 注册项、policy 类或 algorithm 类。

---

## 1. 一句话总览

`g1_stu_future` 的训练链路可以压缩成下面这条主线：

```text
train.sh
-> legged_gym/scripts/train.py
-> import legged_gym.envs (触发 task 注册)
-> task_registry.make_env("g1_stu_future")
-> 实例化 G1MimicFuture
-> task_registry.make_alg_runner(...)
-> 实例化 OnPolicyDaggerRunner
-> runner 内部实例化:
   - teacher: ActorCriticMimic
   - student: ActorCriticFuture
   - algorithm: DaggerPPO
   - storage: RolloutStorage
-> runner.learn()
-> rollout / compute_returns / update / save / log
```

---

## 2. 关键文件加载顺序

下面是运行 `bash train.sh <exptid> cuda:0 ...` 时最重要的文件加载顺序。

### 2.1 Shell 入口：`train.sh`

文件：

- `train.sh`

职责：

- 激活 `twist2` conda 环境
- 设置 `LD_LIBRARY_PATH`
- 把 shell 参数整理成 `python train.py` 的命令行参数

关键点：

- 它固定把 task 设成：
  ```text
  g1_stu_future
  ```
- 也固定把 project 设成：
  ```text
  g1_stu_future
  ```
- 它会传：
  - `--task g1_stu_future`
  - `--proj_name g1_stu_future`
  - `--exptid ...`
  - `--teacher_exptid ...`
  - `--teacher_checkpoint ...`
  - `--max_iterations 150000`

注意：

- `G1MimicStuFutureCfgDAgger.runner.max_iterations` 在配置里默认是 `30001`
- 但 `train.sh` 传了 `--max_iterations 150000`
- 最终以命令行覆盖值为准

---

### 2.2 Python 训练入口：`train.py`

文件：

- `legged_gym/legged_gym/scripts/train.py`

入口逻辑：

```text
get_args()
->_setup_distributed(args)
->train(args)
```

`train(args)` 里最关键的三步：

```text
env, _ = task_registry.make_env(name=args.task, args=args)
runner, train_cfg = task_registry.make_alg_runner(...)
runner.learn(num_learning_iterations=...)
```

也就是说，`train.py` 本身不关心 `g1_stu_future` 的细节，它只做三件事：

1. 解析参数
2. 根据 task 名字向注册表要 env 和 runner
3. 调用 runner 开始训练

---

### 2.3 导入 `legged_gym.envs`：触发任务注册

文件：

- `legged_gym/legged_gym/envs/__init__.py`

这是整条链路里最关键的一步之一，因为 `train.py` 顶部有：

```python
from legged_gym.envs import *
```

这会触发 `envs/__init__.py` 执行，而这个文件做了两件事：

1. 导入所有相关 env / config 类
2. 把 task 名字注册进 `task_registry`

对 `g1_stu_future` 来说，注册语句是：

```text
task_registry.register(
    "g1_stu_future",
    G1MimicFuture,
    G1MimicStuFutureCfg(),
    G1MimicStuFutureCfgDAgger()
)
```

这句的含义非常重要：

- task 名：`"g1_stu_future"`
- 环境类：`G1MimicFuture`
- 环境配置实例：`G1MimicStuFutureCfg()`
- 训练配置实例：`G1MimicStuFutureCfgDAgger()`

也就是说，后面 `task_registry.make_env("g1_stu_future")` 和
`task_registry.make_alg_runner(..., name="g1_stu_future")` 本质上都是从这里取对象。

---

## 3. `task_registry` 是怎么把 task 名字变成真实对象的

文件：

- `legged_gym/legged_gym/gym_utils/task_registry.py`

`TaskRegistry` 内部维护三张表：

```text
self.task_classes[name]
self.env_cfgs[name]
self.train_cfgs[name]
```

对 `g1_stu_future` 来说，注册后表里存的是：

```text
task_classes["g1_stu_future"] = G1MimicFuture
env_cfgs["g1_stu_future"]     = G1MimicStuFutureCfg()         # 已实例化
train_cfgs["g1_stu_future"]   = G1MimicStuFutureCfgDAgger()   # 已实例化
```

注意这里存进去的是**实例**，不是类。

---

### 3.1 `make_env()` 做了什么

`task_registry.make_env(name=args.task, args=args)` 的步骤是：

```text
1. 根据 name 找到 task_class = G1MimicFuture
2. 根据 name 找到 env_cfg = G1MimicStuFutureCfg()
3. 调用 update_cfg_from_args(env_cfg, None, args)
4. set_seed(env_cfg.seed)
5. parse_sim_params(...)
6. 实例化 env = G1MimicFuture(cfg=env_cfg, ...)
```

这里有两个容易忽略的点：

### 点 1：seed 从 train_cfg 复制到 env_cfg

`get_cfgs(name)` 里会做：

```text
env_cfg.seed = train_cfg.seed
```

也就是说：

- 注册时 env_cfg 和 train_cfg 是两份不同实例
- 但 seed 会在取出时从 train_cfg 同步到 env_cfg

### 点 2：命令行覆盖先作用于 env_cfg

`make_env()` 里会先调用：

```text
update_cfg_from_args(env_cfg, None, args)
```

所以：

- `motion.*`
- `env.*`
- terrain / reward / num_envs / seed 等环境参数

都会先在这里改掉，再实例化环境。

---

### 3.2 `make_alg_runner()` 做了什么

`task_registry.make_alg_runner(...)` 的步骤是：

```text
1. 根据 name 找到 train_cfg = G1MimicStuFutureCfgDAgger()
2. 调用 update_cfg_from_args(None, train_cfg, args)
3. class_to_dict(train_cfg) -> train_cfg_dict
4. eval(train_cfg.runner.runner_class_name)
5. 实例化 runner
```

对 `g1_stu_future`，关键配置项是：

```text
runner.runner_class_name    = "OnPolicyDaggerRunner"
runner.algorithm_class_name = "DaggerPPO"
runner.policy_class_name    = "ActorCriticFuture"
teachercfg.runner.policy_class_name = "ActorCriticMimic"
```

所以最后会得到：

```text
runner  = OnPolicyDaggerRunner(...)
student = ActorCriticFuture(...)
teacher = ActorCriticMimic(...)
alg     = DaggerPPO(...)
```

---

## 4. 为什么 `eval("OnPolicyDaggerRunner")` 能找到类

这一点如果不讲清楚，很容易觉得代码“像魔法”。

### 4.1 runner 的 `eval` 来源

在 `task_registry.py` 顶部有：

```python
from rsl_rl.runners import *
```

而：

- `rsl_rl/rsl_rl/runners/__init__.py`

里导出了：

```text
OnPolicyRunner
OnPolicyRunnerMimic
DAggerRunner
OnPolicyDaggerRunner
OnPolicyDiffusionRunner
```

所以 `eval("OnPolicyDaggerRunner")` 能在 `task_registry.py` 的全局命名空间里找到它。

### 4.2 algorithm 和 policy 的 `eval` 来源

在 `on_policy_dagger_runner.py` 顶部有：

```python
from rsl_rl.algorithms import *
from rsl_rl.modules import *
```

因此：

- `eval("DaggerPPO")` 能找到 `rsl_rl.algorithms.DaggerPPO`
- `eval("ActorCriticFuture")` 能找到 `rsl_rl.modules.ActorCriticFuture`
- `eval("ActorCriticMimic")` 能找到 teacher policy 类

这是这套代码动态装配的关键机制。

---

## 5. `g1_stu_future` 的配置继承关系

这一部分要分成两条线看：

1. 环境配置 `env_cfg`
2. 训练配置 `train_cfg`

---

### 5.1 环境配置继承链

对 `g1_stu_future` 的环境配置主链：

```text
BaseConfig
  -> HumanoidCharCfg
    -> HumanoidMimicCfg
      -> G1MimicPrivCfg
        -> G1MimicStuFutureCfg
```

含义：

- `HumanoidCharCfg`：通用 humanoid 角色配置
- `HumanoidMimicCfg`：mimic 任务通用配置
- `G1MimicPrivCfg`：G1 机器人上的 privileged mimic 基础配置
- `G1MimicStuFutureCfg`：在 student 观测里增加 future 分支的特化版本

---

### 5.2 `G1MimicStuFutureCfg` 改了什么

文件：

- `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`

它最重要的改动是：

```text
env.obs_type = "student_future"
```

并且定义了：

- `tar_motion_steps = [0]`
- `tar_motion_steps_future = [0]`
- `n_mimic_obs_single = 6 + 29`
- `n_future_obs_single = 6 + 29`
- `n_proprio = G1MimicPrivCfg.env.n_proprio`
- `num_observations = n_obs_single * (history_len + 1) + n_future_obs`

也就是说，student 观测结构变成：

```text
[当前 student obs] + [历史 student obs] + [future obs]
```

这里有一个很重要的现实细节：

- 机制上它支持 future frames
- 但当前配置里 `TAR_MOTION_STEPS_FUTURE = [0]`
- 也就是说“future 分支”目前默认拿到的是 `0-step` 的参考帧
- 它是“支持未来观测的接口”，但默认配置并不是多步正时间 lookahead

这点很容易误判，笔记里必须记住。

---

### 5.3 训练配置继承链

`g1_stu_future` 的训练配置实例是：

```text
G1MimicStuFutureCfgDAgger()
```

它的继承关系是：

```text
BaseConfig
  -> HumanoidMimicCfgPPO
      -> G1MimicPrivCfgPPO          # 作为 teachercfg / runner 基础

BaseConfig
  -> HumanoidCharCfg
    -> HumanoidMimicCfg
      -> G1MimicPrivCfg
        -> G1MimicStuFutureCfg
          -> G1MimicStuFutureCfgDAgger
```

注意：

- `G1MimicStuFutureCfgDAgger` 本身继承了 `G1MimicStuFutureCfg`
- 所以它既包含 env/motion/rewards，也包含 runner/algorithm/policy/teachercfg
- 但在 `make_alg_runner()` 阶段，真正用到的是它的嵌套类：
  - `runner`
  - `algorithm`
  - `policy`
  - `teachercfg`

---

### 5.4 `G1MimicStuFutureCfgDAgger` 指定了什么

它最关键的几项是：

```text
teachercfg = G1MimicPrivCfgPPO
runner.policy_class_name    = "ActorCriticFuture"
runner.algorithm_class_name = "DaggerPPO"
runner.runner_class_name    = "OnPolicyDaggerRunner"
```

这就是为什么：

- teacher 用的是 privileged mimic teacher
- student 用的是 future-aware student actor-critic
- 训练算法是 `DaggerPPO`
- 外层训练循环是 `OnPolicyDaggerRunner`

---

## 6. 环境类继承关系

`g1_stu_future` 的环境类主继承链是：

```text
BaseTask
  -> LeggedRobot
    -> HumanoidChar
      -> HumanoidMimic
        -> G1MimicDistill
          -> G1MimicFuture
```

每一层的大致职责如下。

### 6.1 `BaseTask`

文件：

- `legged_gym/legged_gym/envs/base/base_task.py`

职责：

- 获取 Isaac Gym handle
- 创建 sim / viewer
- 分配基础 buffer
- 保存 `obs_buf`、`rew_buf`、`reset_buf`、`privileged_obs_buf`

它是最底层的 task 容器。

---

### 6.2 `LeggedRobot`

文件：

- `legged_gym/legged_gym/envs/base/legged_robot.py`

职责：

- 解析基础机器人配置
- 创建 terrain / env / actor
- 定义通用 `step()`
- 在 `post_physics_step()` 里统一做：
  - 刷新 sim tensor
  - 更新 root 状态
  - 检查终止
  - 计算 reward
  - 计算 observation

这是训练循环里“和物理引擎交互”的核心基类。

---

### 6.3 `HumanoidChar`

文件：

- `legged_gym/legged_gym/envs/base/humanoid_char.py`

职责：

- 在 `LeggedRobot` 基础上增加 humanoid character 相关 buffer
- 初始化 key body 索引
- 管理视频录制 camera
- 扩展 humanoid 相关的 reset / step 逻辑

它是 humanoid 角色层。

---

### 6.4 `HumanoidMimic`

文件：

- `legged_gym/legged_gym/envs/base/humanoid_mimic.py`

职责：

- 引入 `MotionLib`
- 管理参考动作 / motion curriculum / motion difficulty
- 提供 mimic 观测、reference motion 更新、mimic reward 所需逻辑
- 维护 `obs_history_buf` / `privileged_obs_history_buf`

这层开始真正进入“模仿学习任务”的核心逻辑。

---

### 6.5 `G1MimicDistill`

文件：

- `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`

职责：

- 把 generic mimic 任务落到 G1 机器人的具体观测与奖励定义上
- 区分 `obs_type == 'priv'` 与 `obs_type == 'student'`
- 构造：
  - `priv_mimic_obs`
  - `mimic_obs`
  - `proprio_obs_buf`
  - `priv_info`
- 生成：
  - `obs_buf`
  - `privileged_obs_buf`

这层是“G1 上的 privileged / student mimic 基础版本”。

---

### 6.6 `G1MimicFuture`

文件：

- `legged_gym/legged_gym/envs/g1/g1_mimic_future.py`

职责：

- 在 `G1MimicDistill` 基础上增加 future observation 分支
- 用 `_get_unified_motion_data()` 一次性采样 privileged steps 和 future steps
- 用 `_build_future_obs_from_data()` 组装 future obs
- 在 `compute_observations()` 中把 future obs 拼进最终 student obs

它就是 `g1_stu_future` 的直接环境类。

---

## 7. `g1_stu_future` 的 observation 是怎么拼出来的

这是这个任务最值得单独讲的一部分。

---

### 7.1 privileged mimic observation

在 `G1MimicFuture._get_mimic_obs()` 里，teacher / critic 用的 `priv_mimic_obs` 来自：

```text
root_pos
+ root_pos_distance_to_target
+ roll/pitch/yaw
+ root_vel_local
+ root_ang_vel_local
+ root_pos_delta_local
+ root_rot_delta_local
+ dof_pos
+ key_body_pos
```

这部分是“更完整、更丰富”的参考信息。

---

### 7.2 student 当前帧 observation

student 当前帧 `mimic_obs` 只取更轻量的一部分：

```text
root_vel_local_xy
+ root_pos_z
+ roll/pitch
+ root_ang_vel_local_yaw
+ dof_pos
```

然后在 `compute_observations()` 里再拼上：

```text
proprio_obs_buf
```

所以当前 student 单帧 observation 是：

```text
obs_buf = mimic_obs + proprio_obs_buf
```

---

### 7.3 future observation

如果 `obs_type == "student_future"`，`G1MimicFuture` 会额外构造：

```text
future_obs
```

这个 `future_obs` 的结构与 student mimic obs 类似，也是从参考 motion 中取出的未来帧信息。

然后最终 student observation 是：

```text
self.obs_buf =
    [当前 obs_buf]
  + [obs_history_buf 展平]
  + [future_obs 展平]
```

也就是说：

```text
student actor 看到的是：
current + history + future
```

---

### 7.4 critic / teacher 看到的是什么

在 runner 里：

```text
critic_obs = privileged_obs if privileged_obs is not None else obs
```

而环境里：

```text
self.privileged_obs_buf = priv_obs_buf
```

注意：

- `privileged_obs_buf` 不包含 `future_obs`
- 它是 `priv_mimic_obs + proprio + priv_info`

因此：

- student actor 输入：`obs_buf`，包含 future 分支
- student critic 输入：`privileged_obs_buf`，不包含 future 分支
- teacher policy 输入：`critic_obs_batch`，本质上也是 privileged obs

这个分工很重要：

```text
student actor      -> 看 current + history + future
student critic     -> 看 privileged obs
teacher actor      -> 看 privileged obs
```

---

## 8. runner / algorithm / policy 是如何装配的

### 8.1 runner：`OnPolicyDaggerRunner`

文件：

- `rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`

它做的事情可以概括为：

```text
读取 train_cfg_dict
-> 初始化 teacher policy
-> 初始化 student policy
-> 初始化 normalizer
-> 初始化 algorithm (DaggerPPO)
-> 初始化 rollout storage
-> 管理 learn_RL() 主循环
```

### 8.2 teacher 是怎么来的

teacher 配置来自：

```text
G1MimicStuFutureCfgDAgger.teachercfg = G1MimicPrivCfgPPO
```

因此 teacher policy class name 是：

```text
ActorCriticMimic
```

Runner 会：

1. 按 `teachercfg` 构造一个 `ActorCriticMimic`
2. 如果 `teacher_experiment_name` 不是 `"None"` / `None` / `"dummy"`
3. 就从日志目录加载 teacher checkpoint

如果 `train.sh` 里 `teacher_exptid` 传的是 `"None"`，那 teacher checkpoint 就不会加载，KL 蒸馏项也会被禁用。

---

### 8.3 student 是怎么来的

student 配置来自：

```text
runner.policy_class_name = "ActorCriticFuture"
```

Runner 会根据 policy 名字判断需要传什么构造参数。

因为 `ActorCriticFuture` 名字里包含 `"Future"`，所以会传入：

- `num_observations`
- `num_critic_observations`
- `num_motion_observations`
- `num_motion_steps`
- `num_priop_observations`
- `num_history_steps`
- `num_actions`
- 以及 `policy_cfg` 中的 future/motion/history 超参数

所以 `ActorCriticFuture` 不是凭空知道 observation 结构的，而是：

- 从 `env.cfg.env` 里拿到维度信息
- 从 `policy_cfg` 里拿到网络超参数
- 然后把 current/history/future 这些分支编码后送入 actor

---

### 8.4 algorithm：`DaggerPPO`

文件：

- `rsl_rl/rsl_rl/algorithms/dagger_ppo.py`

`DaggerPPO` 的职责：

- `act()`：调用 student policy 采样动作，并记录 old log_prob / value / mu / sigma
- `process_env_step()`：把 transition 写入 `RolloutStorage`
- `compute_returns()`：用 GAE 计算 return 和 advantage
- `update()`：做 PPO loss + teacher KL loss + 可选 MoE aux loss

它不是纯监督 DAgger，而是：

```text
PPO 主损失
+ teacher-student KL regularization
```

---

### 8.5 storage：`RolloutStorage`

文件：

- `rsl_rl/rsl_rl/storage/rollout_storage.py`

在 rollout 阶段，它保存：

```text
observations
critic_observations
actions
rewards
dones
values
actions_log_prob
mu
sigma
```

这些量就是 update 阶段重新计算 PPO ratio、value loss、KL 时的依据。

---

## 9. 单次训练 iteration 的完整流程

下面按时间顺序写一遍 `runner.learn_RL()` 每次 iteration 做什么。

---

### 9.1 取初始 observation

进入训练后，runner 会先拿：

```text
obs = env.get_observations()
privileged_obs = env.get_privileged_observations()
critic_obs = privileged_obs if available else obs
```

然后如果配置了 `normalize_obs`，会对 `obs` 和 `critic_obs` 做 normalizer。

---

### 9.2 rollout：跑 `num_steps_per_env` 步

对 `g1_stu_future`，默认：

```text
num_steps_per_env = 24
```

在每一步里：

1. `actions = self.alg.act(obs, critic_obs, ...)`
2. `obs, privileged_obs, rewards, dones, infos = env.step(actions)`
3. 再次构造 `critic_obs`
4. 做 normalizer
5. `self.alg.process_env_step(rewards, dones, infos)`

也就是说：

```text
student 负责真正和环境交互
teacher 不参与 env.step()
teacher 只在 update 阶段参与 KL 蒸馏
```

---

### 9.3 `alg.act()` 做了什么

`DaggerPPO.act()` 里会：

1. `student actor_critic.act(obs)` 采样动作
2. `student actor_critic.evaluate(critic_obs)` 计算 value
3. `get_actions_log_prob(actions)` 记录旧 log_prob
4. 保存旧 `mu` / `sigma`
5. 返回动作给环境

也就是说，rollout 时就已经把 PPO update 需要的“旧策略统计量”都存下来了。

---

### 9.4 `env.step(actions)` 做了什么

环境的 `step()` 主体在 `LeggedRobot` / `HumanoidChar` 一侧：

```text
动作裁剪
-> 计算 PD torque
-> 按 decimation 循环跑物理仿真
-> post_physics_step()
-> 计算 reward / reset / obs
-> 返回 obs_buf, privileged_obs_buf, rew_buf, reset_buf
```

对 `g1_stu_future` 来说，最关键的任务特化逻辑发生在：

- `G1MimicFuture.compute_observations()`
- `G1MimicDistill` / `HumanoidMimic` 的 mimic 相关 reward / reset / motion 更新逻辑

---

### 9.5 rollout 结束后：`compute_returns()`

24 步采样完后，runner 调：

```text
self.alg.compute_returns(critic_obs)
```

它会用最后一个 `critic_obs` 再估一个 `last_values`，然后交给 `RolloutStorage.compute_returns()` 做：

- TD error
- GAE advantage
- returns
- advantage normalization

---

### 9.6 `alg.update()`：真正更新网络

这一步在 `DaggerPPO.update()` 里完成。

对每个 mini-batch：

1. 用当前 student policy 重新评估旧动作
2. 计算 PPO ratio
3. 计算 surrogate loss
4. 计算 value loss
5. 加 entropy regularization
6. 如果 teacher 已加载，则计算 student-teacher KL
7. 如果是 MoE，则再加 load balancing aux loss
8. 反向传播、裁剪梯度、`optimizer.step()`

所以单次更新的本质是：

```text
PPO update
+ optional teacher KL distillation
+ optional MoE aux loss
```

---

### 9.7 保存与日志

update 完成后，runner：

- 写 wandb / console 日志
- 按 `save_interval` 保存 checkpoint
- 训练结束时再保存一次最终模型

teacher checkpoint 只在 runner 初始化时加载一次，不会在每次 iteration 反复加载。

---

## 10. `g1_stu_future` 当前到底实例化了哪些关键对象

如果你跑的是标准：

```bash
bash train.sh <exptid> cuda:0
```

那么关键对象基本可以理解为：

```text
env class:
  G1MimicFuture

env cfg:
  G1MimicStuFutureCfg

runner:
  OnPolicyDaggerRunner

student policy:
  ActorCriticFuture

teacher policy:
  ActorCriticMimic   (如果 teacher_exptid 不是 None)

algorithm:
  DaggerPPO

storage:
  RolloutStorage
```

---

## 11. `g1_stu_future` 与 `g1_stu_mimic` 的关键区别

这两个任务容易混。

### `g1_stu_mimic`

- 环境类：`G1MimicDistill`
- student obs：
  ```text
  current + history
  ```
- policy 往往是更简单的 student actor

### `g1_stu_future`

- 环境类：`G1MimicFuture`
- student obs：
  ```text
  current + history + future
  ```
- student policy：`ActorCriticFuture`
- 训练 runner / algorithm 仍然是 `OnPolicyDaggerRunner + DaggerPPO`

也就是说，`g1_stu_future` 的主要新增点不在 runner，而在：

```text
环境 observation 结构
+ student policy 网络结构
```

---

## 12. 建议的人类阅读顺序

如果你是手动 debug 这条链路，我建议按下面顺序读文件：

### 第 1 轮：先看启动装配

1. `train.sh`
2. `legged_gym/legged_gym/scripts/train.py`
3. `legged_gym/legged_gym/envs/__init__.py`
4. `legged_gym/legged_gym/gym_utils/task_registry.py`
5. `legged_gym/legged_gym/gym_utils/helpers.py`

这一轮解决的问题是：

- task 名字从哪来
- 配置怎么从命令行传进去
- env / runner 怎么被实例化

### 第 2 轮：看配置继承

6. `legged_gym/legged_gym/envs/base/humanoid_mimic_config.py`
7. `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`
8. `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py`

这一轮解决的问题是：

- `g1_stu_future` 到底继承了哪些默认值
- env / runner / algorithm / policy 用的是什么类名

### 第 3 轮：看环境继承与 observation 生成

9. `legged_gym/legged_gym/envs/base/legged_robot.py`
10. `legged_gym/legged_gym/envs/base/humanoid_char.py`
11. `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
12. `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`
13. `legged_gym/legged_gym/envs/g1/g1_mimic_future.py`

这一轮解决的问题是：

- 观测怎么拼
- reference motion 怎么取
- reward / reset / history buffer 怎么更新

### 第 4 轮：看训练循环

14. `rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`
15. `rsl_rl/rsl_rl/algorithms/dagger_ppo.py`
16. `rsl_rl/rsl_rl/storage/rollout_storage.py`
17. `rsl_rl/rsl_rl/modules/actor_critic_future.py`

这一轮解决的问题是：

- rollout 保存了什么
- PPO / DAggerPPO 怎么更新
- student / teacher 怎么交互

---

## 13. 最后给一个脑图式总结

你可以把 `g1_stu_future` 想成下面这个结构：

```text
shell:
  train.sh

python entry:
  train.py

task registration:
  envs/__init__.py
    -> g1_stu_future
       -> env class   = G1MimicFuture
       -> env cfg     = G1MimicStuFutureCfg()
       -> train cfg   = G1MimicStuFutureCfgDAgger()

environment chain:
  BaseTask
    -> LeggedRobot
      -> HumanoidChar
        -> HumanoidMimic
          -> G1MimicDistill
            -> G1MimicFuture

config chain:
  BaseConfig
    -> HumanoidMimicCfg
      -> G1MimicPrivCfg
        -> G1MimicStuFutureCfg

training objects:
  runner    = OnPolicyDaggerRunner
  student   = ActorCriticFuture
  teacher   = ActorCriticMimic
  algorithm = DaggerPPO
  storage   = RolloutStorage

observation split:
  actor obs  = current + history + future
  critic obs = privileged obs
  teacher obs= privileged obs

training loop:
  act -> env.step -> store rollout -> compute_returns -> update -> save/log
```

---

## 14. 你接下来如果要改代码，应该从哪里下手

### 想改 observation 结构

看：

- `g1_mimic_future_config.py`
- `g1_mimic_future.py`

### 想改 teacher / student 装配关系

看：

- `g1_mimic_future_config.py`
- `on_policy_dagger_runner.py`

### 想改 loss / PPO / KL

看：

- `dagger_ppo.py`
- `rollout_storage.py`

### 想改 policy 网络结构

看：

- `actor_critic_future.py`

### 想改 task 名字对应的 env/config

看：

- `envs/__init__.py`
- `task_registry.py`

