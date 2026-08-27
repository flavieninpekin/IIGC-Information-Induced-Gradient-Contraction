# 理论框架（theory）

> 承接 `notes/theory_program.md` C.0-C.5 与 `notes/toy_field_axis_theory.md`。
> 本文是论文 Theory 节的素材：定义、引理、闭式判决、开放项。
> 最后更新：2026-08-19。

---

## 1. 设定（two-action relational bandit，对齐 Toy）

- 单观测 x（Toy 隐藏：常值 obs → 单策略分布 π(a)=softmax(z)_a，z∈ℝ²）。
- 关系 r∈{A,B} 各 1/2，隐藏；Q_A = (r, −r)，Q_B = (−r, r)（镜像）。
- 场 = 关系条件期望梯度 g_r = Σ_a π(a) ∇logπ(a) · w_r(a)，w 由目标函数决定：

| 场 | w_r(a) | 对 Q→−Q 的奇偶性 |
|---|---|---|
| reinforce（无 baseline） | Q_r(a) | 奇 |
| expq（目标 −ΣπQ） | π 加权 Q（含 ∇π 项） | 奇 |
| awr | exp((Q_r(a) − V_r)/τ)，V_r=ΣπQ | 奇+偶混合 |
| awr_nobase | exp(Q_r(a)/τ) | 奇+偶混合 |
| softmax(Q/τ)（framework "gibbs"） | softmax(Q_r(a)/τ) | τ→0 奇主导，τ→∞ 偶主导 |
| softq | 目标 Σ_a π(a)(α logπ(a) − Q(a)) | Q 部分奇、熵部分偶 |
| gibbs（fields.py：E_{π_τ}[Q]） | 权重=π_τ（与 Q 无关） | —（见 §4 O2） |

## 2. 奇偶分解引理（C.2 形式化）

把权重写成 w(Q) = w_odd(Q) + w_even(Q)，
w_even = (w(Q)+w(−Q))/2，w_odd = (w(Q)−w(−Q))/2。
关系 B 的 Q 是 A 的镜像 ⟹ **g_A + g_B = 2·Σ_a π(a)∇logπ(a)·w_even(...)**：
只有偶分量进入共享梯度；奇分量全部进入对比梯度被混叠摧毁。

**两条精确相消推论（Toy 精确验证 ✓，`theory_toy.json`）**：
1. reinforce（无 baseline）：w 奇 → κ=0 精确（实测 E_shared=0.89 ≪ E_contrast=707，
   κ_mean=0.0015 为采样噪声；理论 g_B=−g_A）。
2. expq：∇Σ_a π(a)Q(a) 的两项在镜像下相消 → κ=0 精确（实测 E_shared=0.00）。
3. gibbs（fields.py E_{π_τ}[Q]）：π_τ 与 Q 无关 → g_B=−g_A → **κ≡0（所有 τ）**。

## 3. 闭式判决（✅ 2026-08-26 O1 完成，命题化）

### 3.0 定义钉死（先于一切闭式）

论文的规范定义 = **fields.py 实测协议在期望下的像**：

- 采样型场（reinforce / awr / softmaxq）：g_r = E_{a∼π}[∇_θ(−logπ_θ(a)·w_r(a;θ))]，
  **w 经 V 的梯度照常回传**（fields.py loss_awr 中 v=(probs·q).sum() 可导）；
- 全目标场（expq / softq / gibbs_expq）：g_r = ∇_θ L_r。

镜像带臂机上 π 只依赖 δ=z₀−z₁，所有梯度平行于 u=(1,−1)：记 g_r=c_r·u，则
**κ = (c_A+c_B)² / ((c_A+c_B)² + (c_A−c_B)²)**——一切闭式归约为两个标量。

⚠️ 历史表格（`theory_toy.json` 的 T1 数字）用的是 CE-fit 轨
（∇Σ_a logπ(a)·w(a)，外层均匀权重），是**另一个泛函**；其"非单调""均匀 π 得
0.5"等观察是该轨的性质，不是规范定义的。两轨差异见 `notes/toy_field_axis_theory.md` §6.3。

### 3.1 命题 A（swap 相消·强化版）

**任何逐元素、参数无关权重的场**（reinforce、expq、awr-nobase、softmax(Q/τ)、
E_{π_τ}[Q]）在规范定义下 E[g_B] = −E[g_A] ⟹ κ=0 精确。
（采样协议不破坏它：∇w≡0 时期望采样梯度 = C.1 形式。）

### 3.2 命题 B（softq α 拨盘定理，O1a）✅ 三路互证

设 p=π(A)。L_r = Σ_a π(a)(α logπ(a) − Q_r(a)) 给出

```
c_A = pq(α·ln(p/q) − 2r),   c_B = pq(α·ln(p/q) + 2r)
κ_softq(p, α, r) = A²/(A²+4r²),   A := α·ln(p/(1−p))
```

- α→0 退化 expq（κ→0）；α→∞ 或 π 峰化 → κ→1；
- 对固定 p≠1/2 关于 α 严格单调升；
- **p=1/2 时 κ=0 对一切 α**（熵项在均匀策略处消失）。

验证（`verify_closed_forms_o1.py` → `o1_closed_forms.json`）：
闭式 vs autograd 机器精度（5.0e-16）；MC 锚点一致；
与 `theory_toy.json` 存储 T2 七点**逐位吻合**
（z0: 0.000/0.001/0.023/0.085/0.272/0.700/0.903 ✓；
peaked: 0.038 vs 存表 0.039 为旧表舍入惯例，其余六点全等）。

### 3.3 命题 C（awr-baseline：唯一隐性偶耦合，O1b）✅ 三路互证

w=exp((Q_r−V_r)/τ) 且 V 可导时，∇V 项是唯一的非逐元素耦合。记 x=2r/τ：
A0=e^{xq}, A1=e^{−xp}, B0=e^{−xq}, B1=e^{xp}，
M_A=p·lnp·A0+q·lnq·A1，M_B=p·lnp·B0+q·lnq·B1：

```
c_A = pq(A1 − A0 + x·M_A),   c_B = pq(B1 − B0 − x·M_B)
```

性质：(i) **p=1/2 时 c_B=−c_A 精确 → κ=0 对一切 τ**（MC 验证 0.00000）；
(ii) p≠1/2 时 κ(τ) 有内点极大（两端 τ→0,∞ 都回到反平行）；
(iii) 数值例（r=1）：z0 处峰值 ≈0.42 @ τ≈0.25；peaked 处 ≈0.37 @ τ≈1。
验证：闭式 vs autograd 机器精度（9.2e-16）；600k MC 一致到 ~3SE。

### 3.4 统一读法（"拨盘需要非均匀策略咬合"）

镜像带臂机上，规范协议下所有场的 κ 在均匀策略处精确为 0；幸存只能来自
(i) 显式偶项（softq 熵）或 (ii) 隐式偶耦合（awr 的 ∇V），且二者都随 π 远离
均匀而增强。**字段轴不是纯目标函数的属性，是目标 × 策略锐度的联合属性**——
真实环境策略永不均匀，所以轴在那里可见。这解释了为何 Toy 近均匀 init 的 κ
普遍小而真实环境大。

### 3.5 历史 T1-T3 表（保留存档）

| 任务 | 判决 | 精确数字 |
|---|---|---|
| **T1 awr 锚点** | ❌ 原公式 `1/2+1/(2cosh(2r/τ))` 错 | 已被 §3.3 命题 C 取代；CE-fit 轨数字仅作历史对照 |
| **T2 softq** | ✅ 且升级为定理 | κ=A²/(A²+4r²)，见 §3.2 |
| **T3 gibbs** | ⚠️ 分裂 | softmax(Q/τ) 权重单调 0.081→0.898 ✓（CE-fit 轨）；fields.py E_{π_τ}[Q] 恒 0 ✗（命题 A 覆盖） |

**关键事实（写进论文）**：
1. **κ 是 π 依赖的**：报告 κ(p) 曲线/分布而非单点（§3.2/3.3 给出了解析形式）。
2. **baseline 改变奇偶**（C.4.1 兑现）：awr 带 baseline 是唯一幸存的 PG 型场，
   幸存完全来自 ∇V；无 baseline 版本落在命题 A（κ≡0）。
3. **r 尺度**：闭式统一按 r 参数化（代码 Q=±1 即 r=1）。

## 4. 开放项（写稿前必须解决）

- **O1 正确闭式**：✅ **已完成（2026-08-26）**。规范定义钉死为 fields.py 的
  期望采样梯度；命题 A/B/C 三路互证（闭式 vs autograd 机器精度 vs 大样本 MC），
  softq 与存储 T2 七点逐位吻合。脚本 `experiments/common_basis/toy/verify_closed_forms_o1.py`，
  数据 `data/kappa/toy_fields/o1_closed_forms.json`。详见 §3。
- **O2 gibbs 定义**：✅ **已判决（2026-08-26，读 `run_interp.py:123`）**。
  E2 用的是 `loss_gibbs_expq` = grad −Σ_a π_{θ/τ}(a)·Q_critic(a)，即 fields.py 的
  E_{π_τ}[Q] 场（**不是** framework C.1 的 softmax(Q/τ) 权重）。该场在镜像 bandit 上
  κ≡0（swap 定理），E2 在 510K 上非零且随 τ 单调弱升，因为 **510K 是非镜像多状态结构**
  （deals 不构成精确镜像、critic Q 跨关系不对称）——与 E3 判决"TD/均值场对齐需非镜像
  结构"统一。论文写法：τ 拨盘 = 非镜像结构 × 均值场（条件存在）；softq α 拨盘才是
  普适单调轴。⚠️ 附带 caveat：`run_interp.py:54` 为 deterministic rollout，
  E1/E2/cross-transfer 数字需标注协议伪影风险（`notes/rollout_protocol_artifact.md`）。
- **O3 T4（幸存即失明）**：Overcooked 的 value 场 dynamic κ=0.98 但模型仍学到
  reward~120——"钝端失明"（DQN κ=0.645 & reward=0）在我们的数据里不成立。
  需把"失明"归因到策略（DQN 学不动）而非场，重述 T4。
  **状态：并入 S2 性能后果实验（2026-08-26 计划）**——Toy 关系翻转任务 +
  四目标场从零训练，直接测"盲的行为代价"（O4），命中/偏离都为 T4 重述提供依据。
  设计预注册：`notes/o4_performance_design.md`。

## 5. 跨环境一致性（论文核心图素材）

| 环境 | reinforce 场（奇） | value 场（偶） | 出处 |
|---|---|---|---|
| Toy（精确闭式） | κ→0（相消） | κ→1（对齐） | `theory_toy.json` |
| Overcooked dynamic | 0.015 ± 0.001 | **0.982 ± 0.009** | `oc_field_axis.json` |
| 510K 隐藏（p=0） | 0.017 ± 0.005 | **0.469 ± 0.029** | `510k_field_axis.json` |

三个环境同一现象：**硬策略梯度场在隐藏关系信息下收缩（死），值/均值场对齐（活）**，
包括上一篇声称反转（A2C 收缩 / DQN 反转）的 510K 环境。

## 6. 测量协议（写进 Method / 附录）

1. **forced-partner 伪影**：固定伙伴会关闭 mid-episode 角色切换
   （`overcooked_v3_env.py:102`）→ 对切换型（dynamic）策略 OOD → 0 奖励 → κ=0。
   这是测量伪影而非信号收缩；修正协议只固定起始伙伴、保留切换
   （`SwitchStartEnv`），在此协议下 κ 收缩（dynamic 0.010 vs static 0.31）才可测。
   → 一般化教训：测量协议必须尊重环境自身的动态（切换），否则 κ=0 不可解释。
2. **紧凑测量保证**：排序保持定理 + N* 单次穿越（`notes/variance_decomp_theory.md`），
   1/√N 拟合在 Overcooked static 上 R²=0.99。
3. **κ 是 π 依赖的**：报告 κ(π) 分布而非单点靶子（awr 在 τ=1 时跨 init 0.06-0.51）。
