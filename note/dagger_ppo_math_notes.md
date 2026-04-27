# TWIST2 `DaggerPPO` 数学推导笔记

## 源码位置

- 算法主体：
  `rsl_rl/rsl_rl/algorithms/dagger_ppo.py`
- rollout buffer / GAE：
  `rsl_rl/rsl_rl/storage/rollout_storage.py`
- actor-critic 前向接口：
  `rsl_rl/rsl_rl/modules/actor_critic_future.py`

---

## 1. 先说结论：这份 `DaggerPPO` 到底是什么

这份实现**不是经典的 DAgger supervised learning**，而是：

$$
\text{PPO 主损失}
\;+\;
\text{teacher-student KL 正则}
\;+\;
\text{可选的 MoE 辅助损失}.
$$

如果写成“最大化目标”的形式，可以理解为：

$$
\max_\theta\;
J_{\mathrm{clip}}(\theta)
- c_v L_{\mathrm{value}}(\theta)
+ c_e H(\pi_\theta)
- \lambda_{\mathrm{dagger}} D_{\mathrm{KL}}(\pi_{\mathrm{student}} \,\|\, \pi_{\mathrm{teacher}})
- \lambda_{\mathrm{moe}} L_{\mathrm{moe}}.
$$

但代码实际是在**最小化 loss**，所以实现对应的是：

$$
\min_\theta\;
L_{\mathrm{total}}
=
L_{\mathrm{surrogate}}
+ c_v L_{\mathrm{value}}
- c_e H
+ \lambda_{\mathrm{dagger}} L_{\mathrm{KL,teacher}}
+ \lambda_{\mathrm{moe}} L_{\mathrm{moe}}.
$$

其中：

- `L_surrogate` 是 PPO clipped surrogate 的**负号版本**
- `L_value` 是 critic 的回归损失
- `H` 是策略熵
- `L_KL,teacher` 是 student 和 teacher 高斯策略之间的 KL
- `L_moe` 只有 MoE actor 时才会出现

---

## 2. rollout 阶段：先采样、再存旧策略统计量

在 `act()` 里，算法做了下面几件事：

1. 从当前 student 策略采样动作
2. 计算当前 critic value
3. 记录旧策略下该动作的 `log_prob`
4. 记录旧策略的 `mu` 和 `sigma`
5. 把这些量都存进 rollout buffer

记第 `t` 步：

- actor 观测为 `o_t`
- critic 观测为 `o_t^c`
- 动作为 `a_t`

那么 rollout 时保存的是：

$$
\begin{aligned}
a_t &\sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid o_t), \\
V_t &= V_{\phi_{\mathrm{old}}}(o_t^c), \\
\log p_t^{\mathrm{old}} &= \log \pi_{\theta_{\mathrm{old}}}(a_t \mid o_t), \\
\mu_t^{\mathrm{old}} &= \mu_{\theta_{\mathrm{old}}}(o_t), \\
\sigma_t^{\mathrm{old}} &= \sigma_{\theta_{\mathrm{old}}}.
\end{aligned}
$$

这里“old”的含义是：

- 它们是在 rollout 采集数据时的那套参数
- update 时会拿当前参数重新评估这些旧动作

这正是 PPO 的基本结构。

---

## 3. 奖励与 timeout bootstrap

环境执行动作后得到 reward `r_t` 和 done 标记 `d_t`。  
如果发生 `time_outs`，代码会额外做 bootstrap 修正：

$$
r_t \leftarrow r_t + \gamma \, V_t \cdot \mathbf{1}_{\mathrm{timeout}}.
$$

注意：这是**按代码实现**写下来的修正形式。

其目的可以理解为：

- 如果 episode 不是因为真正失败而结束，而是因为 time limit 截断
- 那么单纯把它当终止会低估回报
- 所以需要用 value 做一个 bootstrap 修正

---

## 4. `compute_returns()`：GAE 与 return 的推导

这部分在 `RolloutStorage.compute_returns()` 中实现。

先定义：

- `d_t`：终止标记，终止时为 1，否则为 0
- `V_t`：当前步 value
- `V_{t+1}`：下一步 value
- `r_t`：当前步 reward

代码里的 TD error 是：

$$
\delta_t
=
r_t + (1 - d_t)\gamma V_{t+1} - V_t.
$$

然后用 GAE 递推 advantage：

$$
A_t
=
\delta_t + (1 - d_t)\gamma\lambda A_{t+1}.
$$

最后 return 定义为：

$$
\hat{R}_t = A_t + V_t.
$$

也就是：

$$
\begin{aligned}
\delta_t &= r_t + (1-d_t)\gamma V_{t+1} - V_t, \\
A_t &= \delta_t + (1-d_t)\gamma\lambda A_{t+1}, \\
\hat{R}_t &= A_t + V_t.
\end{aligned}
$$

### advantage 标准化

代码还会把 advantage 做标准化：

$$
\tilde{A}_t
=
\frac{A_t - \mu_A}{\sigma_A + 10^{-8}}.
$$

其中：

$$
\mu_A = \mathrm{mean}(A_t),
\qquad
\sigma_A = \mathrm{std}(A_t).
$$

标准化的作用是：

- 稳定优化
- 减少 advantage 尺度变化带来的梯度震荡

---

## 5. update 阶段：用当前策略重新评估旧动作

进入 `update()` 后，mini-batch 中拿到的是 rollout 时存下来的：

- `obs_batch`
- `critic_obs_batch`
- `actions_batch`
- `returns_batch`
- `advantages_batch`
- `old_actions_log_prob_batch`
- `old_mu_batch`
- `old_sigma_batch`

然后代码调用：

$$
(\_, \log p_t, V_t^{\mathrm{new}}, \mu_t^{\mathrm{new}}, \sigma_t^{\mathrm{new}}, H_t)
=
\text{actor\_critic}(o_t, o_t^c, a_t).
$$

注意这里传入了 `actions_batch`，因此这一步不是重新采样，而是：

$$
\log p_t
=
\log \pi_{\theta}(a_t \mid o_t),
$$

也就是：

- 用**当前参数**去重新计算“旧动作”的概率
- 这是 PPO 概率比值的基础

---

## 6. PPO 概率比值 `ratio`

代码里：

$$
\mathrm{ratio}_t
=
\exp\left(\log p_t - \log p_t^{\mathrm{old}}\right).
$$

也就是：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid o_t)}.
$$

这个量的含义是：

- 如果 `ratio > 1`，说明当前策略比旧策略更偏好这个动作
- 如果 `ratio < 1`，说明当前策略更不偏好这个动作

PPO 就是用这个比值来控制策略更新步长。

---

## 7. PPO clipped surrogate loss

代码里先定义：

$$
\begin{aligned}
\mathrm{surrogate}_t
&=
-\tilde{A}_t \, r_t(\theta), \\
\mathrm{surrogate}_{t}^{\mathrm{clip}}
&=
-\tilde{A}_t \,
\mathrm{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big),
\end{aligned}
$$

其中：

$$
\epsilon = \texttt{clip\_param}.
$$

然后最终 surrogate loss 为：

$$
L_{\mathrm{surrogate}}
=
\mathbb{E}\Big[
\max\big(
\mathrm{surrogate}_t,
\mathrm{surrogate}_{t}^{\mathrm{clip}}
\big)
\Big].
$$

因为代码在做**最小化**，所以它对应 PPO 论文里的最大化形式：

$$
J_{\mathrm{clip}}(\theta)
=
\mathbb{E}\Big[
\min\big(
r_t(\theta)\tilde{A}_t,
\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\tilde{A}_t
\big)
\Big].
$$

两者关系就是：

$$
L_{\mathrm{surrogate}} = -J_{\mathrm{clip}}.
$$

### 直观含义

- 当更新幅度合理时，用真实的 `ratio`
- 当更新过大时，用裁剪后的 `ratio`
- 这样可以避免策略一步走太远

---

## 8. value loss：critic 回归目标

先说明各个符号：

- `target_values_batch`：rollout 时旧 critic 给出的 value，记为 `V_t^{old}`
- `value_batch`：当前 critic 重新预测的 value，记为 `V_t^{new}`
- `returns_batch`：GAE 推出来的回报目标，记为 `\hat{R}_t`

### 不裁剪版本

如果不用 clipped value loss，那么：

$$
L_{\mathrm{value}}
=
\mathbb{E}\Big[
\big(\hat{R}_t - V_t^{\mathrm{new}}\big)^2
\Big].
$$

### 裁剪版本

代码默认启用 clipped value loss。  
先构造：

$$
V_t^{\mathrm{clip}}
=
V_t^{\mathrm{old}}
+
\mathrm{clip}
\big(
V_t^{\mathrm{new}} - V_t^{\mathrm{old}},
-\epsilon, \epsilon
\big).
$$

再计算两个平方误差：

$$
\begin{aligned}
L_t^{\mathrm{value,raw}}
&=
\big(V_t^{\mathrm{new}} - \hat{R}_t\big)^2, \\
L_t^{\mathrm{value,clip}}
&=
\big(V_t^{\mathrm{clip}} - \hat{R}_t\big)^2.
\end{aligned}
$$

最终 value loss 为：

$$
L_{\mathrm{value}}
=
\mathbb{E}\Big[
\max\big(
L_t^{\mathrm{value,raw}},
L_t^{\mathrm{value,clip}}
\big)
\Big].
$$

### 直观含义

这和 PPO 对 actor 做 clip 的思想一样：

- 不希望 critic 一步改太猛
- 用 clipping 限制 value 更新幅度
- 提高训练稳定性

---

## 9. entropy bonus：鼓励探索

前向过程中会得到当前策略熵 `H_t`。  
代码把熵项写成：

$$
- c_e \, \mathbb{E}[H_t],
$$

其中：

$$
c_e = \texttt{entropy\_coef}.
$$

因此 PPO 基础损失是：

$$
L_{\mathrm{ppo\_base}}
=
L_{\mathrm{surrogate}}
+ c_v L_{\mathrm{value}}
- c_e \mathbb{E}[H_t].
$$

其中：

$$
c_v = \texttt{value\_loss\_coef}.
$$

熵越大，策略越随机；最小化这个 loss 时，减去熵会鼓励策略保留探索。

---

## 10. adaptive KL：用旧学生和新学生的 KL 调学习率

这一段**不是 teacher KL**，而是：

- 旧 student 分布 `N(old_mu, old_sigma^2)`
- 新 student 分布 `N(mu, sigma^2)`

之间的 KL。

代码按每个动作维度计算：

$$
D_{\mathrm{KL}}
\big(
\mathcal{N}(\mu^{\mathrm{old}}, (\sigma^{\mathrm{old}})^2)
\;\|\;
\mathcal{N}(\mu^{\mathrm{new}}, (\sigma^{\mathrm{new}})^2)
\big)
$$

单维公式就是：

$$
\log\frac{\sigma^{\mathrm{new}}}{\sigma^{\mathrm{old}}}
+
\frac{
(\sigma^{\mathrm{old}})^2
+
(\mu^{\mathrm{old}}-\mu^{\mathrm{new}})^2
}{
2(\sigma^{\mathrm{new}})^2
}
- \frac{1}{2}.
$$

对动作维度求和，再对 batch 求均值，得到 `kl_mean`。

然后按规则调学习率：

$$
\begin{aligned}
\text{if } \mathrm{kl\_mean} > 2\,\mathrm{desired\_kl},
&\quad \eta \leftarrow \max(10^{-5}, \eta / 1.5), \\
\text{if } \mathrm{kl\_mean} < \mathrm{desired\_kl}/2,
&\quad \eta \leftarrow \min(10^{-2}, 1.5\eta).
\end{aligned}
$$

### 含义

- KL 太大：说明本次更新走太猛了，学习率降低
- KL 太小：说明更新太保守，学习率提高

---

## 11. teacher-student KL：这才是 `DaggerPPO` 里的“Dagger”部分

代码里 teacher loss 不是 action MSE，也不是 ground-truth supervision。  
它是 student policy 和 teacher policy 的**高斯分布 KL**。

记：

- student 当前分布：
  $$
  \pi_s = \mathcal{N}(\mu_s, \sigma_s^2)
  $$
- teacher 分布：
  $$
  \pi_t = \mathcal{N}(\mu_t, \sigma_t^2)
  $$

那么单维 KL 为：

$$
D_{\mathrm{KL}}(\pi_s \,\|\, \pi_t)
=
\log\frac{\sigma_t}{\sigma_s}
+
\frac{\sigma_s^2 + (\mu_s - \mu_t)^2}{2\sigma_t^2}
- \frac{1}{2}.
$$

代码里对所有动作维度和 batch 取平均，再乘权重：

$$
L_{\mathrm{KL,teacher}}
=
\lambda_{\mathrm{dagger}}
\cdot
\mathbb{E}
\Big[
D_{\mathrm{KL}}(\pi_s \,\|\, \pi_t)
\Big].
$$

其中：

$$
\lambda_{\mathrm{dagger}} = \texttt{dagger\_coef}.
$$

### 这一步的含义

它不是要求 student 完全复制 teacher 某个采样动作，而是要求：

- student 的动作均值靠近 teacher
- student 的动作方差结构也靠近 teacher

因此这是一个**分布级蒸馏项**。

---

## 12. 可选的 MoE 辅助损失

如果 actor 是 MoE，并且配置了：

$$
\lambda_{\mathrm{moe}} > 0,
$$

那么总损失还会加上：

$$
\lambda_{\mathrm{moe}} L_{\mathrm{moe}}.
$$

这个项来自 actor 内部的 `get_moe_aux_loss()`，通常用于：

- 负载均衡
- 避免专家塌缩

---

## 13. 最终总损失

把所有部分合起来，代码的总损失可以写成：

$$
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{surrogate}}
+ c_v L_{\mathrm{value}}
- c_e \mathbb{E}[H_t]
+ \lambda_{\mathrm{dagger}} L_{\mathrm{KL,teacher}}
+ \lambda_{\mathrm{moe}} L_{\mathrm{moe}}
}
$$

如果当前不是 MoE，就把最后一项去掉。

如果当前没有加载 teacher，或者 `dagger_coef <= 0`，那么 teacher KL 这一项也是 0。

---

## 14. 梯度更新

得到 `L_total` 之后，代码执行：

1. `optimizer.zero_grad()`
2. `loss.backward()` 或 AMP 的 `scaler.scale(loss).backward()`
3. `clip_grad_norm_(parameters, max_grad_norm)`
4. `optimizer.step()`

梯度裁剪对应：

$$
\|\nabla_\theta L\| \leq \texttt{max\_grad\_norm}.
$$

它的作用是避免梯度爆炸。

---

## 15. `dagger_coef` 的退火：按代码实现写出来

代码里使用：

$$
\texttt{cosine\_decay\_weight}(w, k, T)
=
w \cdot \frac{1}{2}\left(1+\cos\frac{\pi k}{T}\right).
$$

然后每次 update 后做：

$$
\lambda_{\mathrm{dagger}}
\leftarrow
\lambda_{\mathrm{dagger}}
\cdot
\frac{1}{2}\left(1+\cos\frac{\pi k}{T}\right),
\qquad k<T.
$$

当：

$$
k \ge T,
$$

则：

$$
\lambda_{\mathrm{dagger}} \leftarrow \lambda_{\min}.
$$

### 这里要注意

这段实现按代码字面意思看，是一个**递推式衰减**：

$$
\lambda^{(k+1)}
=
\lambda^{(k)}
\cdot
\frac{1}{2}\left(1+\cos\frac{\pi k}{T}\right),
$$

而不是标准教科书里那种“始终相对于初始值 `lambda_0` 的一次性 cosine schedule”。

也就是说，这里的衰减会比很多人直觉里的 cosine decay 更快一些。

---

## 16. 固定标准差时的 `std_schedule`

如果 actor 使用固定标准差，代码还会根据训练进度更新 `std`：

先定义阶段变量：

$$
\mathrm{stage}
=
\mathrm{clip}\left(
\frac{k-k_0}{T_{\mathrm{std}}},
0, 1
\right),
$$

然后：

$$
\mathrm{std\_coef}
=
\mathrm{stage}\cdot(\sigma_{\mathrm{end}}-\sigma_{\mathrm{start}})
+ \sigma_{\mathrm{start}}.
$$

也就是从起始标准差线性插值到目标标准差。

含义：

- 训练初期探索更强
- 训练后期逐渐收敛到更稳定的动作分布

---

## 17. 这份实现和“经典 DAgger”最大的区别

经典 DAgger 通常是：

$$
\min_\theta \mathbb{E}\big[\ell(\pi_\theta(o), \pi_E(o))\big]
$$

也就是：

- 用 expert/teacher 给标签
- 做监督学习
- 常见损失是 action MSE 或交叉熵

但这里不是这样。  
这里的主干仍然是 PPO：

$$
\text{主目标} = \text{PPO},
$$

teacher 只是额外加了一个 KL regularization：

$$
\text{teacher 作用} = \text{policy distillation regularizer}.
$$

所以更准确地说，这份 `DaggerPPO` 是：

$$
\boxed{
\text{PPO} + \text{teacher Gaussian KL distillation}
}
$$

而不是纯粹的 DAgger imitation learning。

---

## 18. 一句话总结

如果把整条训练链路压缩成一句话，可以写成：

$$
\boxed{
\text{用旧策略采样轨迹}
\;\to\;
\text{用 GAE 构造 } \hat{R}_t \text{ 和 } A_t
\;\to\;
\text{用 PPO clipped objective 更新 student}
\;\to\;
\text{再用 teacher KL 把 student 往 teacher 拉回去}
}
$$

所以这份实现的核心不是“直接模仿 teacher 动作”，而是：

$$
\boxed{
\text{reward 驱动的 PPO 更新}
\;+\;
\text{teacher 分布正则}
}
$$

