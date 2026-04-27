# TWIST2 `ActorCriticFuture.forward()` 数学推导笔记

## 源码位置

- actor-critic 前向与高斯策略接口：
  `rsl_rl/rsl_rl/modules/actor_critic_future.py`
- 这些输出在训练中被使用的位置：
  `rsl_rl/rsl_rl/algorithms/dagger_ppo.py`

---

## 1. 这段代码到底在做什么

这段 `ActorCriticFuture` 的核心思想是：

$$
\boxed{
\pi_\theta(a \mid o)
=
\mathcal{N}\big(\mu_\theta(o), \operatorname{diag}(\sigma^2)\big),
\qquad
V_\phi = V_\phi(o^c)
}
$$

也就是说：

- actor 根据 actor observation `o` 输出动作分布的均值 `\mu_\theta(o)`
- 再结合全局标准差向量 `\sigma` 构造一个对角高斯策略
- critic 根据 critic observation `o^c` 输出状态价值 `V_\phi(o^c)`

这段代码最终返回的是：

$$
\big(
a,\;
\log \pi_\theta(a \mid o),\;
V_\phi(o^c),\;
\mu_\theta(o),\;
\sigma,\;
H(\pi_\theta(\cdot \mid o))
\big).
$$

这 6 个量正好对应 PPO / DAggerPPO 更新需要的核心变量。

---

## 2. 符号定义

对单个样本，记：

$$
\begin{aligned}
o &\in \mathbb{R}^{d_o}, && \text{actor observation}, \\
o^c &\in \mathbb{R}^{d_c}, && \text{critic observation}, \\
a &\in \mathbb{R}^{d_a}, && \text{action}, \\
\mu_\theta(o) &= f_\theta(o), && \text{actor 输出的动作均值}, \\
V_\phi(o^c) &= g_\phi(o^c), && \text{critic 输出的状态价值}, \\
\sigma &= (\sigma_1,\dots,\sigma_{d_a}), && \sigma_i > 0.
\end{aligned}
$$

这里最关键的建模假设是：

$$
\sigma \text{ 不是状态相关的 } \sigma_\theta(o),
\text{ 而是一个全局可学习或固定的标准差向量。}
$$

因此条件动作分布写成：

$$
\pi_\theta(a \mid o)
=
\prod_{i=1}^{d_a}
\mathcal{N}(a_i;\mu_i(o),\sigma_i^2).
$$

这就是“对角高斯策略”的含义：各动作维度条件独立。

---

## 3. `update_distribution(observations)` 的数学含义

源码对应逻辑：

```python
mean = self.actor(observations)
self.distribution = Normal(mean, mean*0. + self.std)
```

### 第一步：actor 输出均值

$$
\mu_\theta(o) = f_\theta(o).
$$

这一步的含义是：

- 给定当前观测 `o`
- actor 网络预测“最中心”的动作

### 第二步：构造高斯分布

$$
\pi_\theta(\cdot \mid o)
=
\mathcal{N}\big(\mu_\theta(o), \operatorname{diag}(\sigma^2)\big).
$$

代码中的

```python
mean*0. + self.std
```

没有新的数学含义，只是在 batch 维度上广播 `self.std`。

如果 batch size 为 `B`，则：

$$
\text{mean shape}=B \times d_a,\qquad
\text{std shape}=d_a,\qquad
\text{broadcast 后}=B \times d_a.
$$

因此对第 `b` 个样本、第 `i` 个动作维度：

$$
a_{b,i} \sim \mathcal{N}(\mu_{b,i}, \sigma_i^2).
$$

---

## 4. `act()` 的数学含义：从策略中采样动作

源码对应逻辑：

```python
self.update_distribution(observations)
return self.distribution.sample()
```

数学上就是：

$$
a \sim \pi_\theta(\cdot \mid o).
$$

也可以写成重参数化形式：

$$
a = \mu_\theta(o) + \sigma \odot \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I).
$$

其中 `\odot` 表示逐元素乘法。

### 这一步的含义

- rollout 时不是直接执行均值动作
- 而是从当前策略分布里采样
- 这样 agent 才有探索能力

因此：

$$
\mu_\theta(o)\ \text{决定动作中心},\qquad
\sigma\ \text{决定探索范围}.
$$

---

## 5. `get_actions_log_prob(actions)` 的数学推导

源码对应逻辑：

```python
return self.distribution.log_prob(actions).sum(dim=-1)
```

### 单个动作维度的 log-prob

对第 `i` 维动作：

$$
\log \mathcal{N}(a_i;\mu_i,\sigma_i^2)
=
-\frac{(a_i-\mu_i)^2}{2\sigma_i^2}
- \log \sigma_i
- \frac{1}{2}\log(2\pi).
$$

### 整个动作向量的 log-prob

由于动作维度独立，联合概率的对数等于逐维求和：

$$
\log \pi_\theta(a \mid o)
=
\sum_{i=1}^{d_a}
\left[
-\frac{(a_i-\mu_i)^2}{2\sigma_i^2}
- \log \sigma_i
- \frac{1}{2}\log(2\pi)
\right].
$$

这正对应代码里的：

$$
\texttt{self.distribution.log\_prob(actions).sum(dim=-1)}.
$$

### 它为什么重要

PPO 的关键比值是：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid o_t)}
=
\exp\Big(
\log \pi_\theta(a_t \mid o_t)
-
\log \pi_{\theta_{\mathrm{old}}}(a_t \mid o_t)
\Big).
$$

因此 `log_prob` 不是附加信息，而是 PPO 更新的核心输入。

---

## 6. `action_mean` 与 `action_std`

### `action_mean`

源码：

```python
return self.distribution.mean
```

数学上：

$$
\mathbb{E}[a \mid o] = \mu_\theta(o).
$$

它表示：

- 当前策略在观测 `o` 下的条件期望动作
- 高斯分布的中心
- 对高斯来说也是 mode

### `action_std`

源码：

```python
return self.distribution.stddev
```

数学上：

$$
\operatorname{Std}[a \mid o] = \sigma.
$$

对每个动作维度 `i`：

$$
\sigma_i \uparrow
\Rightarrow
\text{探索更强，动作更分散};
\qquad
\sigma_i \downarrow
\Rightarrow
\text{探索更弱，动作更集中}.
$$

由于这里的 `\sigma` 不依赖 `o`，所以它更像是一个全局探索强度参数。

---

## 7. `entropy` 的数学推导

源码对应逻辑：

```python
return self.distribution.entropy().sum(dim=-1)
```

### 单维高斯熵

对一维高斯 `\mathcal{N}(\mu_i,\sigma_i^2)`，熵为：

$$
H_i
=
\frac{1}{2}\log(2\pi e \sigma_i^2).
$$

### 对角高斯总熵

各维独立时，总熵为：

$$
H\big(\pi_\theta(\cdot \mid o)\big)
=
\sum_{i=1}^{d_a}
\frac{1}{2}\log(2\pi e \sigma_i^2).
$$

也可以展开成：

$$
H
=
\sum_{i=1}^{d_a}
\left(
\log \sigma_i
+ \frac{1}{2}(1+\log 2\pi)
\right).
$$

### 熵的含义

熵越大，策略越随机；熵越小，策略越确定。

在 PPO 中常见的目标写法是：

$$
L_{\mathrm{total}}
=
L_{\mathrm{policy}}
+ c_v L_{\mathrm{value}}
- c_e H.
$$

因为训练是在最小化 loss，所以减去熵会鼓励策略保留探索。

---

## 8. `act_inference()` 为什么直接返回均值

源码：

```python
actions_mean = self.actor(observations)
return actions_mean
```

数学上：

$$
a_{\mathrm{infer}} = \mu_\theta(o).
$$

### 为什么推理时不用采样

训练时需要探索，因此：

$$
a \sim \pi_\theta(\cdot \mid o).
$$

但在评估、部署、回放时，希望动作：

- 稳定
- 可复现
- 没有额外随机噪声

因此直接使用均值动作：

$$
a = \mu_\theta(o).
$$

对高斯策略来说，均值也是最可能动作，因此这是标准做法。

---

## 9. `evaluate()` 的数学角色

`forward()` 里除了 actor 相关量，还会返回 critic 的 `value`。

源码对应逻辑可抽象为：

$$
V_\phi(o^c) = \mathrm{critic}(x_{\mathrm{critic}}),
$$

其中：

$$
x_{\mathrm{critic}}
=
\operatorname{concat}
\big(
o^c_{\mathrm{priv\_without\_motion}},
\;
o^c_{\mathrm{single\_motion}},
\;
z_{\mathrm{motion}}
\big),
$$

而：

$$
z_{\mathrm{motion}} = \mathrm{Enc}(o^c_{\mathrm{motion}}).
$$

也就是说，critic 不是直接吃原始 `critic_observations`，而是：

1. 提取 motion 部分
2. 用 `critic_motion_encoder` 编码
3. 与其余特权信息、单步 motion 特征拼接
4. 再送入 critic MLP

最终得到：

$$
V_\phi(o^c).
$$

### 它在训练中的作用

critic 用来估计状态值，并参与 advantage 构造：

$$
A_t = \hat{R}_t - V_\phi(o_t^c).
$$

然后：

- actor 用 advantage 更新策略
- critic 自己拟合回报

---

## 10. `forward()` 的完整数学流程

源码大意：

```python
self.update_distribution(observations)

if actions is None:
    actions = self.distribution.sample()

actions_log_prob = self.get_actions_log_prob(actions)
entropy = self.entropy
mu = self.action_mean
sigma = self.action_std

if critic_observations is not None:
    value = self.evaluate(critic_observations)
```

可以分成 6 步。

### 第 1 步：根据 observation 构造策略分布

$$
\mu = \mu_\theta(o),\qquad
\pi_\theta(\cdot \mid o)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2)).
$$

### 第 2 步：如果没有给定动作，就从当前分布采样

$$
a \sim \pi_\theta(\cdot \mid o).
$$

### 第 3 步：计算该动作在当前策略下的 log-prob

$$
\log \pi_\theta(a \mid o).
$$

### 第 4 步：提取当前策略统计量

$$
\mu = \mathbb{E}[a \mid o],\qquad
\sigma = \operatorname{Std}[a \mid o],\qquad
H = H(\pi_\theta(\cdot \mid o)).
$$

### 第 5 步：如果给了 critic observation，则计算状态价值

$$
V_\phi(o^c).
$$

### 第 6 步：返回完整训练所需量

$$
\boxed{
\big(
a,\;
\log \pi_\theta(a \mid o),\;
V_\phi(o^c),\;
\mu,\;
\sigma,\;
H
\big)
}
$$

---

## 11. `forward()` 为什么既能传 `actions`，也能不传

这是这段代码最关键的设计点之一。

### 情况 A：`actions is None`

表示当前处于 rollout 阶段：

$$
a_t \sim \pi_\theta(\cdot \mid o_t).
$$

此时 `forward()` 的作用是：

- 从当前策略采样动作
- 并把当前分布的统计量一起拿出来

### 情况 B：`actions` 已给定

表示当前处于 update 阶段。  
这时不会重新采样，而是用当前策略重新评估旧动作：

$$
\log \pi_\theta(a_t \mid o_t).
$$

这里的 `a_t` 通常是 rollout 时已经存下来的旧动作。

这正是 PPO 所需要的步骤，因为 PPO 要比较：

$$
\pi_\theta(a_t \mid o_t)
\quad \text{和} \quad
\pi_{\theta_{\mathrm{old}}}(a_t \mid o_t).
$$

如果 update 时重新采样动作，就没有办法构造 PPO 的概率比值。

---

## 12. 这段前向和 PPO / DAggerPPO 的连接关系

这段 `forward()` 输出的每一项都有明确用途：

### `actions`

$$
a_t
$$

用于和环境交互。

### `actions_log_prob`

$$
\log \pi_\theta(a_t \mid o_t)
$$

用于 PPO ratio 与策略更新。

### `value`

$$
V_\phi(o_t^c)
$$

用于 value loss、GAE、return 计算。

### `mu`

$$
\mu_\theta(o_t)
$$

用于：

- 日志分析
- KL 计算
- teacher-student distillation

### `sigma`

$$
\sigma
$$

用于：

- 策略熵
- KL 计算
- 探索尺度控制

### `entropy`

$$
H(\pi_\theta(\cdot \mid o_t))
$$

用于熵正则，鼓励探索。

因此这段代码不是一个普通的“前向预测动作”函数，而是一个为 actor-critic / PPO 训练准备完整统计量的接口。

---

## 13. 一句话总结

这段实现最核心的数学本质可以压缩成下面这句话：

$$
\boxed{
\text{actor 输出高斯均值 } \mu_\theta(o),
\text{ 全局 std 给出探索尺度 } \sigma,
\text{ critic 输出 } V_\phi(o^c),
\text{ 然后把 }
(a,\log\pi,V,\mu,\sigma,H)
\text{ 一次性返回。}
}
$$

换句话说，这段代码做的不是“直接输出一个动作”，而是：

$$
\boxed{
\text{定义一个动作分布}
\;\to\;
\text{从中采样或重评估动作}
\;\to\;
\text{同时返回 PPO 更新所需的全部统计量。}
}
$$

