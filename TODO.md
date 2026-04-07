经过完整审查，以下是我对这个项目的诊断和 2026 年最佳架构建议。

---

## 一、当前架构的关键问题诊断

### 1. **奖励信号极度稀疏且信噪比极低**

这是最致命的问题。当前的奖励设计：

```python
return_tensor[offset:next_offset].fill_(float(episode.reward))
# 即：一整局（数百步决策）共享同一个标量 reward
```

一局麻将约 200-800 步决策，但 **所有步骤获得完全相同的 return**——局末的 rank_bonus + score_term。这意味着：
- 网络无法区分哪步决策是好的、哪步是坏的
- 一局中的关键打牌（如关键舍牌、是否副露）与无关紧要的摸切获得相同的梯度信号
- PPO 的 advantage 在这个设定下本质上只是在学 "哪类局面容易赢"，而非 "该怎么打"

**这就是为什么 clip_fraction 始终为 0.0，policy_loss 接近零** —— 策略实际上几乎没有在更新。

### 2. **网络容量严重不足**

```
输入: 1281 维 flat vector → Linear(1281,256) → ReLU → Linear(256,256) → ReLU → Linear(256,14)
```

一个 2 层 256 宽度的 MLP，参数量约 **40 万**。对于日麻这种信息不完全博弈：
- Suphx (2020) 用的是深层 CNN + attention，数百万参数
- Mortal (2023) 用的是 Transformer encoder，约 600 万参数
- 2026 年的 SOTA 方法普遍使用 **数千万参数级**

40 万参数的 MLP 根本不足以编码日麻的复杂策略空间。

### 3. **特征表示丢失了大量关键信息**

当前特征 (1281 dims) 的严重缺陷：
- **没有对手信息**：不编码其他玩家的舍牌、副露、听牌可能性
- **没有时序信息**：不区分第几巡、不编码舍牌顺序
- **没有危险度信息**：振听、安全牌判断等信息缺失
- **全局历史压缩为稀疏直方图**：仅用 34×3 的直方图表示牌面看到的牌

日麻是一个信息不完全博弈，**对手建模是核心**。当前特征设计基本忽略了这一点。

### 4. **无折扣回报 + 无时序信用分配**

```python
return_tensor[offset:next_offset].fill_(float(episode.reward))
# 没有 gamma 折扣，没有 TD(λ)，没有 GAE
```

advantages 的计算：
```python
advantages = returns - behavior_values
```

这里 returns 对每步都是相同的常数，behavior_values 是行为策略时的 V(s)。这不是标准 PPO 的 GAE advantage，而是一种极其粗糙的蒙特卡洛估计。

### 5. **自对弈没有对手多样性**

4 个座位全部使用同一个策略版本 → 容易陷入策略循环（A beats B, B beats C, C beats A）。没有 league training、没有历史策略池。

---

## 二、2026 年最佳日麻 RL AI 架构

基于 Suphx、Mortal、以及 2024-2026 年游戏 AI 的发展（AlphaStar league training、OpenAI Five、Pluribus），推荐以下架构：

### 阶段一：监督预训练 (Behavioral Cloning)

**这是最关键的起步**。纯 RL 从零训练日麻，在 2026 年仍然不现实。

```
数据: 天凤/雀魂的高段位牌谱 (数百万局)
目标: 模仿学习，学习人类高手的策略分布
预期: 达到大约天凤六段水平
```

这一步解决了冷启动问题，让 RL 在一个已经合理的策略基础上继续优化。

### 阶段二：网络架构

```
┌─────────────────────────────────────────────────┐
│         Observation Encoder (per-player)         │
│                                                  │
│  自家手牌: 34-dim + 赤宝牌标记                      │
│  自家副露: 序列编码                                 │
│  4人舍牌序列: Transformer encoder (巡序 + 位置)     │
│  4人副露历史: 序列编码                              │
│  全局信息: 场风/自风/巡目/宝牌指示牌/本场/供托        │
│  危险度特征: 各家振听/立直状态/推测听牌              │
│                                                  │
│  → Tile-level embedding (34+7 种牌 → d_model)    │
│  → Self-attention layers (6-8 层, d=256/384)      │
│  → Cross-attention: 手牌 attend to 场面            │
│                                                  │
├─────────────────────────────────────────────────┤
│              Policy Head                         │
│  对每个候选动作: Q(s,a) 或 logit                   │
│  动作空间: ~200 (34张舍牌 + pass/pon/chi/kan/     │
│           riichi/tsumo/ron/九种九牌)               │
│                                                  │
├─────────────────────────────────────────────────┤
│              Value Head                          │
│  V(s) → 预测最终顺位期望值 或 4维顺位概率分布        │
│                                                  │
└─────────────────────────────────────────────────┘

参数量: 5M - 20M
```

**为什么用 Transformer 而不是 MLP：**
- 日麻状态是一个**变长序列**（不同巡目，舍牌数量不同）
- 注意力机制天然适合建模 "哪些舍牌暗示了对手的听牌"
- Mortal 已经验证了 Transformer 在日麻上的有效性

### 阶段三：RL 微调 —— 正确的做法

#### (a) 时序信用分配

用 **GAE (Generalized Advantage Estimation)** 替代当前的常数 return：

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

但日麻的中间奖励 $r_t$ 为零（只有终局有分），所以需要：

#### (b) 中间奖励塑形 (Reward Shaping)

```python
# 基于势函数的 reward shaping (保证最优策略不变):
shaped_reward_t = gamma * Phi(s_{t+1}) - Phi(s_t) + terminal_reward

# Phi(s) 可以是:
# - 向听数改善: shanten 降低 → 正奖励
# - 有效牌数变化: ukeire 增加 → 正奖励  
# - 危险度: 放铳概率变化 → 负奖励
# - 期望得点估计: 基于手牌价值的估计
```

#### (c) 对手建模与 League Training

```
策略池 (Policy Pool):
  ├── 当前训练策略 (main agent)
  ├── 历史最佳策略 (best response targets)
  ├── 人类高段位策略 (BC baseline)
  └── 随机历史快照 (diversity)

每局自对弈:
  - 1 个座位: 当前策略
  - 3 个座位: 从策略池中按优先级采样
```

这避免了 self-play 的策略循环问题。

### 阶段四：具体参数建议

| 组件       | 当前值           | 建议值                      | 理由                     |
| ---------- | ---------------- | --------------------------- | ------------------------ |
| 网络       | MLP 256×2 (400K) | Transformer 6L-256d (5-10M) | 容量不足是根本瓶颈       |
| 预训练     | 无               | BC 在百万局牌谱上           | 冷启动 RL 在日麻上不可行 |
| Advantage  | 常数 return      | GAE λ=0.95, γ=0.999         | 时序信用分配             |
| 中间奖励   | 无               | 向听数/有效牌数变化         | 缓解稀疏奖励             |
| Replay     | 32K steps, FIFO  | 100K+ steps                 | 数据效率                 |
| Minibatch  | 2048             | 4096-8192                   | 大 batch 在 PPO 中更稳定 |
| 对手       | 纯自对弈         | League + BC 对手            | 防止策略退化             |
| Value head | 1 维 scalar      | 4 维顺位概率                | 更丰富的价值信号         |
| 熵系数     | 0.01 固定        | 0.01→0.001 衰减             | 早期探索，后期收敛       |

---

## 三、推荐的分步实施路线

```
Phase 0: 修复当前致命 bug (1-2 天)
  ├── 实现 GAE 替代常数 return
  ├── 添加向听数变化的中间 reward shaping
  └── 验证 policy_loss 和 clip_fraction 不再为零

Phase 1: 监督预训练基线 (1-2 周)
  ├── 收集/下载天凤或雀魂高段位牌谱
  ├── 实现 Transformer 策略网络
  ├── BC 预训练至六段水平
  └── 导出为 RL 起点 checkpoint

Phase 2: 正确的 RL 微调 (2-4 周)
  ├── 实现策略池 (policy pool)
  ├── 实现 reward shaping + GAE
  ├── 用 BC 对手 + 自对弈混合训练
  └── 定期与 BC 基线评估

Phase 3: 高级优化 (持续)
  ├── MCTS 辅助搜索 (如 Suphx)
  ├── 对手意图推断网络
  ├── 分布式训练扩展
  └── 竞赛级部署优化
```

---

## 四、总结

**当前项目的核心问题不是计算效率，而是学习信号质量：**

1. 常数回报 → 策略梯度信号淹没在噪声中（这就是 `clip_fraction=0.0` 的根因）
2. 网络太小 → 无法编码日麻的策略复杂度
3. 特征缺失 → 看不到对手信息，无法做防守
4. 无预训练 → 从随机策略通过 RL 学习日麻几乎不可能收敛

**最高优先级修复**：实现 GAE + reward shaping，这个改动最小但收益最大，可以立即验证训练信号是否改善。