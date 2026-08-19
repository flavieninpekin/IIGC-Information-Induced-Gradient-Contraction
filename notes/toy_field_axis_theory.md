# Toy 场轴闭式验证（T1-T3 判决）

> 目的：对 `theory_program.md` C.1-C.3 的"保留—特异权衡"框架做精确数值验证。
> 用**精确闭式**（对解析目标函数 autograd 求期望梯度）+ **采样**（rollout，
> 每集梯度）双轨，r 参数化，直接对照 T1-T3 的数值靶子与 C.5 证伪判据。
> 最后更新：2026-08-19。脚本：`experiments/../temp verify_theory_toy.py`（结果
> `data/kappa/toy_fields/theory_toy.json`）。

---

## 1. 设定（对齐 C.1）

- Two-action relational bandit（`HiddenMatchingEnv`，hidden：常值 obs → 策略是单个
  2-action softmax π(z)，z=2 logits）。
- 真值 Q：A 下 (a₀→+r, a₁→−r)，B 下镜像（代码 Q=±1，即 r=1；notes 里 r=4 一并核对）。
- 场 = 关系条件期望梯度 g_r = Σ_a π(a)∇logπ(a)·w_r(a)，w 由目标函数决定：
  - reinforce：w=Q（无 baseline）；带 baseline 版见下文
  - awr：w=exp((Q−V)/τ)，V=Σ_a π(a)Q(a)（**带 baseline，匹配 fields.py**）
  - awr_nobase：w=exp(Q/τ)（无 baseline，对照 T1 锚点）
  - softmaxq：w=softmax(Q/τ)（**framework C.1 表真正定义的 "gibbs"** = 归一化 awr）
  - softq：w 隐式，目标 Σ_a π(a)(α logπ(a) − Q(a))
  - expq：目标 −Σ_a π(a)Q(a)
  - gibbs（fields.py 实现）：目标 −Σ_a π_τ(a)Q(a)，π_τ=softmax(z/τ)（E_{π_τ}[Q]）
- κ = E_shared/(E_shared+E_contrast)（无噪声双 rollout），采样版另给 κ_ep 与 σ²。

---

## 2. 结果

### 2.1 基础字段（HIDDEN 5 inits × N=200，r=1，τ=1）

| 场 | κ_ep | κ_mean | E_shared | E_contrast | σ² |
|---|---|---|---|---|---|
| reinforce | 0.0007 | 0.0015 | 0.89 | 706.80 | 547.68 |
| expq | **0.0000** | **0.0000** | 0.00 | 721.50 | 0.00 |
| awr (baseline) | 0.0574 | 0.0626 | 260.49 | 3754.71 | 294.58 |
| softq | 0.0575 | 0.0575 | 43.14 | 721.50 | 0.00 |
| gibbs (E_{π_τ}[Q]) | **0.0000** | **0.0000** | 0.00 | 721.50 | 0.00 |

REVEALED（同一批场，N=200）：reinforce 0.441 / expq 0.440 / awr 0.440 /
softq 0.440 / gibbs 0.440（T5 退化复现，与 `toy_fields/results.json` 的 0.438 一致）。

**C.2 引理复现**：expq 与 gibbs(E_{π_τ}) 的 E_shared **精确为 0**（g_B=−g_A）；
reinforce 的 0.0015 只是采样噪声（E_shared=0.89 ≪ E_contrast=707）。

### 2.2 T1 — awr τ 扫描（r=1，z0=[0.372,−0.239]；τ=[0.1,0.25,0.5,1,2,4,8]）

| 变体 | κ(τ) |
|---|---|
| 精确·带 baseline | 0.499 → 0.458 → 0.359 → **0.253** → 0.239 → 0.383 → 0.663（**非单调**） |
| 精确·无 baseline | 0.081 → 0.081 → 0.086 → 0.131 → 0.291 → 0.594 → 0.850（单调升） |
| 采样 κ_ep（baseline） | 0.458 → 0.392 → 0.241 → 0.099 → 0.033 → 0.013 → 0.008（降） |
| 精确·峰化 π=[2,−2]·baseline | 0.500 → 0.500 → 0.500 → 0.512 → 0.614 → 0.806 → 0.936（单调） |
| 精确·峰化 π=[2,−2]·无 baseline | 0.482 → 0.482 → 0.500 → 0.616 → 0.813 → 0.939 → 0.984（单调） |

**r=4（notes 锚点）：精确·baseline κ(τ=1)=0.458（≠ 0.5）；τ=0.1 时 exp(40) 溢出 → NaN。**

**判决：C.1/T1 的锚点公式「均匀 π、无 baseline → 1/2+1/(2cosh(2r/τ))」是错的。**
- 均匀 π + 无 baseline：g_B = −g_A **精确相消 → κ=0**（对一切 τ），不是 ~0.5。
- 带 baseline 后 0.5 出现在 τ→0 端（z0：0.499@τ=0.1），且**非单调**（τ=1 塌到 0.25）。
- **κ 强烈依赖策略 π**：τ=1、带 baseline 时 κ 从 ~0.06（近均匀）到 ~0.51（峰化）——
  所以 `toy_fields/results.json` 的 "awr HIDDEN κ=0.561±0.020" 是 init 抽样，
  不是稳定靶子。闭式必须写成 κ(π, τ, baseline) 的函数。

### 2.3 T2 — softq α 扫描（r=1，α=[0.01,0.1,0.5,1,2,5,10]）

- 精确 = 采样（σ²=0，确定性梯度）：
  κ(α) = 0.000 → 0.001 → 0.023 → **0.085** → 0.272 → 0.700 → 0.903（**单调升**）
- 峰化 π=[2,−2]：0.000 → 0.039 → 0.500 → 0.800 → 0.941 → 0.990 → 0.998

**判决：T2 成立 ✓**。α→0 退化为 expq（κ=0），熵项主导 → κ→1。
`toy_fields` 的 softq HIDDEN κ=0.068@α=1 与 z0 精确值 0.085 同量级（init 波动）。

### 2.4 T3 — gibbs τ 扫描（r=1，τ=[0.1,0.2,0.5,1,2,5,10]）

| 定义 | κ(τ) |
|---|---|
| **gibbs（fields.py：grad E_{π_τ}[Q]）** | **恒为 0.0（所有 τ）** |
| **softmax(Q/τ) 权重（framework C.1 定义）** | 0.081 → 0.081 → 0.086 → 0.131 → 0.291 → 0.693 → 0.898（**单调升**） |
| 采样 softmaxq κ_ep | 0.0003 → … → 0.0063（升，但被 σ² 压低） |

**判决：T3 分裂 ⚠️。**
- fields.py 的 `gibbs` = E_{π_τ}[Q] 梯度：π_τ 只依赖策略参数、与 Q 无关，关系只经
  Q 符号进入 → **g_B = −g_A 恒成立 → κ≡0**。若论文要的"gibbs 场"是这个，
  **T3"τ 单调升"直接证伪**。
- 框架 C.1 表定义的 gibbs = softmax(Q/τ) 权重（= 归一化 awr）才给出单调升。
- **推论：E2 的 τ 扫描（0.511→0.562→0.588）不可能来自 E_{π_τ}[Q] 场**（恒 0）。
  notes 把两个 "gibbs" 混为一谈；须重查 `run_interp.py` 用的是哪个定义。

---

## 3. 对框架的判决与修正清单

框架**没死**（T2 干净成立、softmax(Q/τ) 单调成立、C.2 相消引理成立），但必须：

1. **T1**：闭式按 κ(π, τ, baseline) 重推——精确计算器已就绪
   （`verify_theory_toy.py::exact_kappa`），可符号化验证；报告 κ(π) 分布而非单点。
2. **T3/gibbs**：明确定义——softmax(Q/τ) 权重（framework 版）单调成立；
   E_{π_τ}[Q]（fields.py 版）恒 0，须从 E2 里澄清并改实现。
3. **baseline 改变奇偶（C.4.1 兑现）**：无 baseline 的 awr/softmaxq 单调，
   带 baseline 后近均匀 π 非单调——推导必须显式区分"带/不带 baseline"两个版本。
4. **单点靶子不可用**："awr HIDDEN κ=0.561" 是 init 抽样；paper 应报 κ(π) 或
   平均 ± 跨 init 标准差，并给精确 κ(π) 曲线。

## 4. 跨环境一致性（Overcooked + 510K，新增证据）

同一框架在真实环境读数一致。**场轴（reinforce 死 / value 活）在三个环境全部复现。**

**Overcooked v3**（GPU 重训 16 模型，切换保持协议；`oc_field_axis.json`）：

| 场 | static κ_ep | dynamic κ_ep |
|---|---|---|
| reinforce（硬 PG，奇） | 0.018 ± 0.003 | 0.015 ± 0.001（死） |
| value（TD/mean-seeking，偶） | 0.549 ± 0.012 | **0.982 ± 0.009（存活）** |

**510K**（上一篇主战场，复用 ppo_reveal 模型；`510k_field_axis.json`，
team 条件方差分解，p=训练时队友位可见度；0.50/1.00 级损坏模型已 GPU 重训补齐，
n=6/级）：

| p | reinforce κ_ep | value κ_ep |
|---|---|---|
| 0.00（隐藏适应） | 0.017 ± 0.005 | **0.469 ± 0.029** |
| 0.50 | 0.017 ± 0.003 | **0.474 ± 0.022** |
| 1.00（可见适应） | 0.014 ± 0.002 | **0.408 ± 0.007** |

- 硬策略梯度场被每集回报方差主导（σ²~1.2-1.9M），κ 塌缩（脆端）；
- 值函数场低方差、且隐藏关系对其不可见/弱依赖 → 对齐（钝端）；
- **p 依赖成立（n=6）**：隐藏适应（p≤0.5）value κ≈0.47 > 可见适应（p=1.0）
  value κ≈0.41 —— 越隐藏越对齐，方向同 Overcooked dynamic vs static。
- 这正是 C.0"脆端/钝端"二分在真实环境的演示；也是上一篇"PPO/A2C κ=0 / DQN κ=0.645"
  反转在共同测量基上的复现（value 场 high-κ ↔ DQN/TD 高 κ）。

## 5. 数据位置

- 本笔记数据：`data/kappa/toy_fields/theory_toy.json`
- 采样脚本（可复用）：`verify_theory_toy.py`
- 交叉验证：`data/kappa/toy_fields/results.json`（run_toy_fields，κ+总能量）
