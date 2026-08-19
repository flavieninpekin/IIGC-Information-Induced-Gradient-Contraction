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

## 3. 闭式判决（2026-08-19 精确验证，`theory_toy.json`）

| 任务 | 判决 | 精确数字 |
|---|---|---|
| **T1 awr 锚点** | ❌ 原公式 `1/2+1/(2cosh(2r/τ))` 错 | 均匀 π + 无 baseline → **κ=0 精确**（非 0.5）；0.5 只在 τ→0 + 带 baseline 出现；带 baseline 近均匀 π 非单调（0.50→0.25@τ=1→0.66），峰化 π 单调（0.50→0.94） |
| **T2 softq** | ✅ | κ(α) 单调 0→0.90（α=0.01→10），精确=采样；α→0=expq(κ=0)，熵主导→1 |
| **T3 gibbs** | ⚠️ 分裂 | softmax(Q/τ) 权重单调 0.081→0.898 ✓；fields.py E_{π_τ}[Q] 恒 0 ✗ |

**关键事实（写进论文）**：
1. **κ 是 π 依赖的**：τ=1、带 baseline 时 awr κ 从 ~0.06（近均匀）到 ~0.51（峰化）。
   论文必须报 κ(π) 分布而非单点（"awr HIDDEN κ=0.561" 是 init 抽样）。
2. **baseline 改变奇偶**（C.4.1 兑现）：无 baseline 的 awr/softmaxq 单调升；
   带 baseline 后近均匀 π 非单调。推导显式区分两版。
3. **r 尺度**：代码 Q=±1（r=1）；notes 的 r=4 与代码不一致，闭式统一按 r 参数化。

## 4. 开放项（写稿前必须解决）

- **O1 正确闭式**：κ(π, τ, baseline) 的符号解。精确计算器已就绪
  （`verify_theory_toy.py::exact_kappa`，对解析目标 autograd 求期望梯度），
  可数值对照任何符号推导。
- **O2 gibbs 定义**：`run_interp.py`（E2）到底用哪个"gibbs"？E2 数字
  （0.511→0.562→0.588）不可能来自 E_{π_τ}[Q]（恒 0），须重查并澄清。
  若 E2 用 softmax(Q/τ) 权重，则 T3 直接成立且跨环境一致。
- **O3 T4（幸存即失明）**：Overcooked 的 value 场 dynamic κ=0.98 但模型仍学到
  reward~120——"钝端失明"（DQN κ=0.645 & reward=0）在我们的数据里不成立。
  需把"失明"归因到策略（DQN 学不动）而非场，重述 T4。

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
