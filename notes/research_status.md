# IIGC 研究梳理 — 上一篇论文结果 + 当前实验进展

> 本文件是研究现状的集中整理，帮助理清"上一篇文章已经知道什么"和
> "我们现在在测什么、得到了什么"。
> 最后更新：2026-07-31。

---

## Part 1：上一篇论文（AAAI2027-510k-clear，IIGC）基线

### 1.1 核心主张

部分可观测多智能体中，隐藏关系变量（队友身份/角色/任务分配）会把
"各关系下有用但方向相反"的策略梯度投影到同一局部信息空间并相互抵消。
策略几乎不动可能不是收敛，而是**没有可学习的信号**——短学习轨迹会说谎。

### 1.2 理论贡献

| 贡献 | 内容 |
|---|---|
| IIGC | Information-Induced Gradient Contraction（现象形式化） |
| Pythagorean 分解 | 关系条件梯度能量的共享/差分分解 |
| κ | Relational Gradient Retention Ratio：混合后保留的梯度能量比例（0=全抵消，1=全对齐） |
| Stable-or-Stuck | 路径积分 P（动没动）+ 能量门 E（有无信号）+ κ（信号保留多少） |
| Proposition 4 | κ 是"指定梯度估计器"的函数；反转是边界，不是 value 方法免疫的证据 |

### 1.3 三环境结果

**510K 行为路径积分（22 个 PPO 种子）**

| Mode | n | 路径长度 | 解读 |
|---|---|---|---|
| SINGLE | 5 | 0.456 ± 0.071 | 真探索 |
| STATIC | 5 | 0.329 ± 0.086 | 真探索 |
| OBVIOUS | 8 | 0.328 ± 0.062 | 真探索（dynamic 规则+队友可见） |
| DYNAMIC | 4 | **0.293 ± 0.049** | **假稳定**（最短路径但学不到位） |

- Spearman r=−0.62, p=0.019, Cohen's d=2.67
- 反直觉：合作约束越强 → 训练路径越短
- **OBVIOUS 消融**：OBVIOUS 路径≈STATIC（Δ≈0.001）→ 队友信息可见是稳定化主因

**Toy（极简匹配，10 seeds）**

| 指标 | HIDDEN | REVEALED |
|---|---|---|
| κ (PPO) | 0.039 ± 0.072 | 0.726 ± 0.100 |
| κ (A2C) | 0.194 ± 0.352 | 0.842 ± 0.025 |
| Reward | ~0 | ~+19 |

- 关键：Toy HIDDEN 路径反而**长**（0.48 vs 0.01）→ "路径积分单独用有歧义"

**Overcooked（压力测试）**

| 算法 | STATIC κ | DYNAMIC κ | Reward |
|---|---|---|---|
| PPO | 0.50 | **0.00** | 187 / 0 |
| DQN | 0.473 | 0.645 | 0 / 0 |

- PPO DYNAMIC 完全抵消（κ=0，能量门不过）；DQN 反转但 reward 也为 0

### 1.4 跨算法 κ（510K，反转现象）

| 算法 | 梯度场 | κ_SINGLE | κ_DYNAMIC | 方向 |
|---|---|---|---|---|
| A2C | 策略梯度 | 0.644 ± 0.201 | 0.519 ± 0.060 | **S>D（收缩）** |
| DQN | TD 损失 | 0.797 ± 0.123 | 0.917 ± 0.063 | D>S（反转） |
| SAC | actor | 0.542 ± 0.060 | 0.569 ± 0.017 | D>S（弱） |
| REINFORCE | 策略梯度 | 0.487 ± 0.312 | 0.604 ± 0.238 | 高方差/不收敛 |
| PPO | 策略梯度 | 0.569 (n=1) | 0.444 ± 0.069 | 数据不全 |

### 1.5 论文官方解读（重要）

- **不是**"value 方法免疫隐藏信息"。
- 而是：**κ 依赖于你选的梯度场**。PG 场测"动作偏好冲突"（隐藏下抵消），
  TD 场测"值预测误差"（混合下对齐）。
- 反转被定位为**边界（boundary）**，其**机制是开放的**（Proposition 4）。

### 1.6 三个失败模式（Stable-or-Stuck 的动机）

| 环境 | 隐藏下路径 | 隐藏下 κ | 失败模式 |
|---|---|---|---|
| 510K | 短 | — | 假稳定 |
| Partner | 长 | 0.157 | 徒劳探索 |
| Toy | — | 0.0007 | 梯度死亡 |

### 1.7 局限

- 510K PPO κ 的 single/static 仅 n=1；SAC 仅 2 seeds；REINFORCE 高方差
- Overcooked 缺行为路径；反转机制解释停留在"边界"

---

## Part 2：当前项目（新论文）— 进展

### 2.1 研究问题（承接 Proposition 4 的开放问题）

**反转的机制到底是什么？** 假设：κ 是目标函数"模式/均值几何"
（mode-seeking vs mean-seeking）的可预测后果——
- 硬 PG（REINFORCE）= 对动作分布取模 → 隐藏关系下梯度异号抵消 → κ 低
- 软/均值目标（TD、soft-Q、expected-Q）= 对值/动作平均 → 混合下对齐 → κ 高

### 2.2 核心方法：共同测量基

同一网络 + 同一 rollout 数据 + 同一关系，**只换定义梯度的目标函数**。
这样 κ 的差异只可能来自梯度场本身，而非算法/网络/数据差异。

### 2.3 E1：SAC actor/critic 分离 κ（n=2，已完成）

**设计**：同一 SAC 模型、同一 rollout，actor 场（soft mode-seeking）vs
critic 场（TD / mean-seeking）。预测：actor S>D（收缩），critic D>S（对齐）。

| mode | κ_actor | κ_critic | E_actor | E_critic |
|---|---|---|---|---|
| single | 0.537 ± 0.068 | 0.598 ± 0.025 | 33 | 1.8e5 |
| dynamic | 0.589 ± 0.129 | 0.508 ± 0.039 | 179 | 3.0e6 |

**解读**：简单版假设**未证实**。
- actor 几乎持平（软目标本就是中间态）
- critic 反而 S>D → "TD 按构造聚合一致"过强：批梯度层面 TD 也能收缩

**方法学限制**：actor/critic 在不同参数空间，绝对值不可比，只有场内 S/D 排序可读。

### 2.4 E2：插值谱（n=4，已完成）—— 决定性实验

**设计**：同一 actor 参数 θ，5 种目标：reinforce（硬 mode-seeking）→
awr → softq → expq（mean-seeking），外加 Gibbs 温度 τ 扫描
（π_τ∝softmax(logits/τ)，τ→0 取模，τ→∞ 取均）。

| field | single | dynamic |
|---|---|---|
| reinforce | **0.402** ± 0.085 | **0.477** ± 0.028 |
| awr | 0.500 ± 0.003 | 0.533 ± 0.039 |
| softq | 0.566 ± 0.078 | 0.542 ± 0.103 |
| expq | 0.562 ± 0.059 | 0.520 ± 0.115 |

| Gibbs τ | single | dynamic |
|---|---|---|
| 0.2 | 0.511 | 0.514 |
| 1.0 | 0.562 | 0.520 |
| 5.0 | 0.588 | 0.567 |

**稳健的结果**：**模式/均值轴成立**。reinforce（硬 PG）在两种模式下都是
κ 最低（0.40/0.48），越 mean-seeking κ 越高——"κ 是梯度场的函数"
的最直接证据（同一 θ、同一数据、只换目标）。

**不稳健的结果**：S/D 收缩模式。reinforce/awr 是 D>S，softq/expq 是 S>D，
方差大；n=2 时"dynamic 随 τ 上升"的签名没撑住。

**关键 caveat**：这些是**已适应 dynamic 的收敛 SAC**（reward 6 vs 3，
α 自动调到 ~0 = 全峰值政策）。收敛后 κ 反映"当前解"而非"学习过程"，
所以测不出"隐藏信息→收缩"的 S/D 签名。

### 2.5 Cross-transfer κ 矩阵（n=4，已完成）

**设计**：训练模式 × 测试模式 2×2，解耦"测试环境的关系结构（列）"与
"策略的适应状态（行）"。

**reinforce 场矩阵（mean over seeds 41-44）**：

| 训练\测试 | single | dynamic |
|---|---|---|
| single | 0.402 ± 0.085 | 0.518 ± 0.044 |
| dynamic | 0.487 ± 0.074 | 0.477 ± 0.027 |

awr 全 ≈0.50；softq：S→S 0.57 / S→D 0.52 / D→S 0.60 / D→D 0.54；expq 类似。

**解读**：预测的"未适应策略在隐藏关系下收缩"（κ(S→D)<κ(S→S)）**未出现**——
反而**交叉迁移在两种测试模式下都提升 reinforce κ**（不匹配策略行为同质、
各 deal 回报均匀 → 梯度更一致）。reinforce 场被回报方差主导。

**协议局限**：每个 rollout 仍是"混合分配"（隐藏队友由随机发牌决定），
κ 测的是"两个随机混合的 deal 间一致性"，不是干净的关系条件对比。

### 2.6 Toy 强制分配 κ（已完成）—— IIGC 的干净复现

**设计**：`HiddenMatchingEnv.set_partner()` 每次 rollout 强制单一关系分配
（partner 0 vs 1），用解析真值 Q（匹配 +1 / 不匹配 -1），同一策略、
同一数据、只换关系分配。REVEALED（软策略）vs HIDDEN（随机策略）。

| field | REVEALED | HIDDEN |
|---|---|---|
| reinforce | 0.438 | **0.000** ± 0.000 |
| awr | 0.438 | **0.561** ± 0.020 |
| softq | 0.438 | 0.068 ± 0.044 |
| expq | 0.438 | **0.000** ± 0.000 |

**解读**：
1. **IIGC 在共同测量基上干净复现**：HIDDEN 下两个分配要求相反动作且
   观测不可区分 → reinforce/softq/expq 梯度**完全抵消（κ=0）**；
   REVEALED 下可区分 → 部分对齐（0.44）。同一策略/数据/参数，
   只换分配 → 这是真实信息效应，**不是定义性产物**。
2. **字段轴在隐藏关系下出现**：awr（优势加权）**抵抗收缩**（0.56），
   硬场全灭（0.00）——interpolation 效果在干净条件下可见。
3. Caveat：REVEALED 四场都 =0.438（策略接近正确映射时各场都退化为
   "提高每类 obs 的正确动作概率"，gA-gB 夹角场无关）；awr 的 HIDDEN
   κ 反而高于 REVEALED，awr 无信息对比。

### 2.7 连续揭露实验（510K，21 档 × 6 种子）—— 决定性负面结果

**设计**：`RevealEnv`（OBVIOUS + 按 p 掩码队友位）训练 MaskablePPO，
p ∈ {0.00,...,1.00} 步长 0.05（21 档）× 6 种子（41-46）= 126 个模型，
全信息评估下混合协议测 κ + 方差分解。

**结果（n=6/点）**：

| p | κ | ±std | | p | κ | ±std |
|---|---|---|---|---|---|---|
| 0.00 | 0.548 | 0.055 | | 0.55 | 0.485 | 0.080 |
| 0.05 | 0.498 | 0.040 | | 0.60 | 0.476 | 0.114 |
| 0.10 | 0.501 | 0.121 | | 0.65 | 0.560 | 0.113 |
| 0.15 | 0.477 | 0.064 | | 0.70 | 0.477 | 0.052 |
| 0.20 | 0.462 | 0.074 | | 0.75 | 0.469 | 0.120 |
| 0.25 | 0.520 | 0.098 | | 0.80 | 0.504 | 0.037 |
| 0.30 | 0.521 | 0.045 | | 0.85 | 0.480 | 0.033 |
| 0.35 | 0.469 | 0.080 | | 0.90 | 0.470 | 0.098 |
| 0.40 | 0.518 | 0.075 | | 0.95 | 0.529 | 0.054 |
| 0.45 | 0.450 | 0.111 | | 1.00 | 0.549 | 0.108 |
| 0.50 | 0.533 | 0.046 | | | | |

**统计检验（决定性）**：
- 全局 mean=0.500, std=0.088；**level 间散布 0.031 < 种子内噪声 0.077**
- **ANOVA: F=0.756, p=0.759** —— 揭露比例不能解释 κ 方差
- 逐点 Bonferroni t 检验：21 点全部 p=1.000
- 75% vs 邻居：p=0.608 —— **谷不存在**
- 相邻对：0/20 显著（随机期望 ~1）

**结论**：**κ(p) 统计平坦（~0.50），上一篇的"75% 谷"是 n=2 的采样假象**。
n=6 后完全回归均值。"极值预测 / almost-right penalty" 框架不成立。

**附加发现**：κ 全场 ≈0.50 = "两个 rollout 梯度正交"值 → 混合协议测的 κ
本身被噪声主导（单集方差 ~1.6M 是队间对比 ~2.5e4 的 60 倍），
**510K 混合协议对 κ 不够灵敏**——这是方法学层面的发现。

**配套：a(p) 机制模型已证伪**（`reveal_theory.md` §4.2）——"committed but
wrong"机制预测"互补相消 → κ 平坦"，但 Toy 注入估计错误的验证
（`verify_accuracy_invariance.py`）推翻：E_contrast+σ² 从 486 崩到 2.0、
κ 从 0.37 降到 0.10。原因：±1 奖励的 REINFORCE 下错误动作得 −1 → 梯度反而
推向正确动作，准确率只调制梯度幅值，不产生模型假设的反向梯度。

**510K 应用验证：三次尝试均失败（`stuck_detect/`）**：
1. 强制分配分解（`run_510k_forced_decomp.py`）：拒绝采样控制红A持有者 →
   每关系测 E_shared/E_contrast/σ²。N=30/200 的 REINFORCE 和 advantage 加权
   都分不开 p=0/p=1——per-episode 噪声 σ²~1.5-8M 是信号 200-300 倍。
2. 敏感性探针（`run_510k_sensitivity.py`）：翻转队友位测策略响应。
   p=0 和 p=1 的 L2 敏感性都弱（0.008-0.017）且重叠。
3. **根因**：510K 的策略几乎不依赖队友位——手牌/出牌主导行为，关系是
   弱驱动。**关系条件诊断在 510K 天然区分不了**，不是协议问题。
   应用验证需换环境（Overcooked 角色，见 `notes/app_validation.md`）。

### 2.8 能量-方差分解的实证支撑（A/B/C + 定理）

把"κ = 共享能量占比"从恒等式升级为**带实证保证的工具**（Toy 强制分配）：

**A. 跨样本预测（去循环）**：估计集（200 eps/partner）测的分量，在独立验证集
上预测 κ：REVEALED 误差 0.038、HIDDEN 0.000。分量可外推，不是循环拟合。

**B. 紧凑测量（样本量收敛）**：κ 随 N 收敛，N=50 即 κ=0.415±0.009（真值 0.411），
N=1000 时 ±0.002。κ（比值）比分量的 bootstrap std（~18%）更稳——共同涨落相消。
（注：早期"分量 std~20-30% → κ 可靠"的表述被纠正，正确证据是收敛曲线。）

**C. 尺度不变性**：reward 乘 λ∈{1,10,100} → 梯度乘 λ → κ **精确不变**
（0.4158，机器精度），分量精确按 λ² 缩放（10000 倍）。**κ 对信号强度失明，
分解保留它**——弱信号 run 的 κ 也可以很好看。

**定理（排序保持，`notes/variance_decomp_theory.md`）**：
```
sign(f_i(N) − f_j(N)) = sign( (a_i b_j − a_j b_i)·N + (a_i c_j − a_j c_i) )
```
f_i−f_j 是 N 的**线性函数** → 紧凑测量排序要么恒保持、要么**单次翻转**，
翻转点 N* = (a_j c_i − a_i c_j)/(a_i b_j − a_j b_i)。充分条件：κ_true 与
E_shared/σ² 排序一致 ⟹ 任意 N 保序。合成反例预测 N*=99，实测在 99→100 精确
翻转；真实 reveal 数据 N=10 时 Kendall τ=1.00。
**实用协议**：测 (a,b,c) → 算各对 N* → 取 N > max N*，便宜测量即替代贵测量且有保证。

### 2.9 E3：replay 混合消融（计划，未跑）

DQN 的 TD 场，replay 从"单一关系"到"均匀混合"连续变化 → κ_TD 应随混合度
上升，直接检验"TD 对齐 = 聚合一致性"。

### 2.10 当前图景与开放问题

**已经比较确定**：
1. **κ 是梯度场的函数**：硬 PG（reinforce）场收缩最强（E2 字段轴 n=4 稳健）
2. **IIGC 是真实信息效应**：Toy 强制分配下隐藏关系把 reinforce/softq/expq
   梯度干净地抵消到 0（共同测量基，非定义产物）
3. 上一篇的"反转"部分来自"跨估计器比较"的不可比前提（Proposition 4）
4. **字段轴与信息结构正交**：Toy 显示信息结构决定"是否收缩"，
   E2 显示场的模式/均值几何决定"收缩多少"
5. **能量-方差分解有实证保证**（A/B/C + 排序定理，见 §2.8）：分量可外推、
   κ 可紧凑估计、κ 失明于尺度而分解不、紧凑测量的排序保持有可证边界（N*）

**仍存疑 / 难点**：
1. 在 510K 的收敛 SAC 上测不出"隐藏关系→S>D"签名（E1/E2/cross-transfer
   都失败），因为已适应策略 + 混合分配协议稀释了信号
2. 反转的完整机制（PG 收缩 vs TD 对齐）仍需在干净协议下对 value 场验证

**元观察（研究过程 = IIGC）**：我们的探索过程本身就是 IIGC 的演示——
每个假设（E1/E2/cross-transfer）都被下一个 confound 抵消，看起来"没信号"；
直到强制分配（Toy 的 set_partner）把隐藏 confound 暴露出来，信号立刻出现。
方法论含义：**测不到信号时先怀疑隐藏 confound 在平均你的测量，而不是
断定没有效应——先暴露 confound 的关系变量再下结论。** 这也是新论文的
narrative 钩子。

**下一步候选**（按优先级）：
- [x] **能量-方差分解做实（A/B/C + 排序定理）**：见 §2.8 与
     `toy/verify_variance_decomp.py`、`verify_compactness.py`、
     `verify_scale_invariance.py`、`verify_ranking_flip.py`、
     `notes/variance_decomp_theory.md`
- [ ] 在 Toy 上补充 **HIDDEN vs REVEALED 的对照统计**（多策略初始化、p 值）
- [ ] 把字段轴（awr 抵抗收缩）作为 Toy 的核心图 / 表格
- [ ] E3 replay 混合消融（在 Toy 或 510K 干净协议下）

---

## 数据位置

| 内容 | 路径 |
|---|---|
| 上一篇 κ 汇总 | `AAAI2027-510k-clear/data/kappa_summary.json`（参考目录） |
| E1 结果 | `data/kappa/common_basis_sac_split/results.json` |
| E2 结果 | `data/kappa/common_basis_interp/results.json` |
| Cross-transfer 结果 | `data/kappa/cross_transfer/results.json` |
| Toy 结果 | `data/kappa/toy_fields/results.json` |
| Reveal 全网格结果 | `data/kappa/510k_reveal/results.json` + `figures/` |
| Reveal 测量脚本 | `experiments/common_basis/reveal/run_510k_reveal.py` |
| Reveal 绘图脚本 | `experiments/common_basis/reveal/plot_510k_reveal.py` |
| E1 脚本 | `experiments/common_basis/sac_actor_critic/run_sac_split.py` |
| E2 脚本 | `experiments/common_basis/interpolation/run_interp.py` |
| Cross-transfer 脚本 | `experiments/common_basis/cross_transfer/run_cross_transfer.py` |
| Toy 脚本 | `experiments/common_basis/toy/run_toy_fields.py` |
| 能量分解 A/B/C | `toy/verify_variance_decomp.py` + `verify_compactness.py` + `verify_scale_invariance.py` + `verify_ranking_flip.py` |
| 能量分解结果 | `data/kappa/variance_decomp/` |
| 排序保持定理 | `notes/variance_decomp_theory.md` |
| 共享梯度场 | `src/iigc/metrics/fields.py` |
| 讨论笔记 | `notes/kappa_reversal_discussion.md` |
