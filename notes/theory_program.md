# 理论纲领与行动清单 — 2026-08-19

> 目的：一份文件终结"太散、抓不住"。三个部分：
> **A** 生死簿（什么死了、什么活着）→ **B** 今天就按的运行键 → **C** 理论推导任务书（自足，可单独拿去推）。
> 证据数字全部注明出处文件，推导时直接对照。

---

## Part A：生死簿（30 秒版）

### 已经死透的（不要回头挖，补刀记录在案）

| 死亡对象 | 死因 | 埓文 |
|---|---|---|
| "75% 谷" | n=6 后回归均值 | reveal 全网格 ANOVA F=0.756, p=0.759（`data/kappa/510k_reveal/`） |
| "PG 收缩 vs value 反转"作为算法族差异 | 四次独立证伪 | E1（critic 反而 S>D）、E2（S/D 方向不稳健）、cross-transfer（方向反了）、reveal（混合协议 κ≈0.50=正交噪声值） |
| a(p) 机制模型（committed-but-wrong） | 注入实验推翻 | `toy/verify_accuracy_invariance.py`：E_contrast 486→2.0 |
| 510K 上的关系条件诊断 | 三次尝试一致失败 | `stuck_detect/`：σ² 是信号 200-300 倍；策略不依赖队友位（关系弱驱动） |

### 活着的（论文的全部原料）

| 活体 | 证据 | 出处 |
|---|---|---|
| **IIGC 抵消本体**：共同测量基上隐藏关系把硬场梯度打到精确零 | HIDDEN: reinforce κ=0.000±0.000（能量 371.5 全灭）；同一策略/数据/参数只换分配 | `data/kappa/toy_fields/results.json` |
| **字段轴**：reinforce < awr < softq/expq | E2 n=4 稳健（0.40/0.50/0.57）+ Toy 极端分化（0.000 / 0.561 / 0.068 / 0.000）。⚠️ 2026-08-19 精确验证：T2 成立；gibbs 定义需澄清（见 C.5） | `data/kappa/common_basis_interp/results.json`、`notes/toy_field_axis_theory.md` |
| **分解 + 测量学**：P1-P4、N* 定理、N(δ) 闭式 | A/B/C 实验 + 排序翻转 N*=99→100 精确验证 + 1/√N 与实测吻合 | `notes/variance_decomp_theory.md`、`kappa_estimation_theory.md` |
| **Overcooked 诊断区分卡死/收敛** | ⚠️ 2026-08-19 修正：旧"dynamic E_shared=0.000"来自坏 checkpoint + forced 协议伪影（关切换→OOD→0 奖励）。GPU 重训后 dynamic 能学（reward~120）；真实 κ 分离只在切换保持协议下可见（dynamic κ_ep=0.010 vs static 0.31，σ² 主导） | `data/kappa/stuck_detect/overcooked_decomp.json`、`data/kappa/server_tasks/results/` |
| **伪影解剖**：反转 = 跨估计器比较伪影 | 上表第一行右侧整列 | 这是展品不是损失 |

**核心判断**：不是"没找到现象"，是站在一个真现象上面没低头看。
**371.5 的能量配 0.000 的 κ** —— 纯关系信号、全部被混叠摧毁 —— 这个数字在硬盘里躺了两周。

---

## Part B：今天就按的运行键（不需要动脑）

**现状**：`github.com/flavieninpekin/code` 是完整的服务器实验队列（driver 轮询
todos.json → engine 执行 → results 仓库自动 push）。**8 月 6 日 e2e 测试通过后，
队列一次都没真正跑过。13 天。**

### B.1 重启（5 分钟）

```bash
# 服务器上（见 code/README_SERVER.md）：
cd code && git pull
nohup python driver.py &           # IIGC_WORKERS 默认 4，8×B200 可设 8

# 本地盯进度：
cd ../results && git pull && cat status.json
python ../code/aggregate_results.py
```

### B.2 先改 todos.json（避免烧卡）

删掉 `reveal_grid_clean` —— 本地已跑完 126 模型且结论明确（κ(p) 平坦），
服务器重跑纯属浪费 GPU 时。

### B.3 各任务判决什么（8 seeds b1 + 12 seeds b2 ≈ 1-2 天跑完）

| 任务 | 判决的问题 | 为什么关键 |
|---|---|---|
| **oc_mem**（dynamic + 帧记忆 k=4/8/16） | **干预闭环**：记忆 → 推断角色 → E_shared 从 0 回升？ | **全队列最关键**。成功 = "诊断→定位→修复"闭环成立，论文升级；失败 = "诊断有用"线终结，诚实转方法学论文。**无论哪边都终止悬置状态** |
| oc_train b1/b2 + oc_decomp_b1 | 8→20 seeds 下 static/dynamic 分解仍干净区分？ | 补 ICLR 缺口 1（非 Toy 环境可测）的统计强度 |
| oc_baselines_b1 | κ+分解 vs reward 平坦度/梯度方差 | 补 ICLR 缺口 3（无对比基线） |
| oc_n_protocol | Overcooked 上 1/√N + N* 预测/翻转 | P3 定理的跨环境演示 |
| toy_reverify | P1/P2 复核 | 便宜，顺手 |

### B.4 等待期间本地小活（半天）

给 `experiments/common_basis/toy/run_toy_fields.py` 补记 **E_shared / E_contrast**
分量（现在 results.json 只存 κ+总能量）。分量一记，"权衡图"直接出来。

---

## Part C：理论推导任务书（自足，拿去推）

### C.0 候选框架一句话

> **保留—特异权衡（retention–specificity trade-off）**：关系混叠下，
> 混合梯度的幸存成分由目标函数权重对回报的**奇偶分解**决定——
> 奇分量（关系方向信息）全部进入 E_contrast 被混叠摧毁，偶分量幸存为 E_shared；
> 幸存者对"该往哪边推"失明。脆端（奇主导）：κ→0；钝端（偶主导）：
> κ 高但聚合正确、行动无用（DQN κ=0.645 / reward=0）。
> τ 扫描（0.511→0.588 单调）= 拨盘被连续转动。

这是 `notes/kappa_reversal_discussion.md` §4 那句"PG 塌缩（脆），value 钝化（钝）"
的定理化。它同时关闭上一篇 Prop 4 的开放问题：**拨盘位置可由权重函数几何预测**。

### C.1 设定（two-action relational bandit，对齐 Toy 实测参数）

- 单观测 x（或观测分布），关系 r ∈ {A,B}，各 1/2，隐藏。
- 真值 Q：A 下 (a₀→+1, a₁→−1)·r，B 下镜像。**实测用 r=4**（JSON 里 rA=−4, rB=+4），推导保持 r 为符号。
- 策略 π_θ(a) = softmax(logits)，待训。
- 每个场定义为关系条件期望梯度 g_r = Σ_a π(a) ∇logπ(a) · w_r(a)，
  w 由目标函数决定：

| 场 | w_r(a) | 奇偶性（对 Q→−Q） |
|---|---|---|
| reinforce | A_r(a)（含 baseline 则 A=Q−V） | 奇 |
| expq | 场方程不同（见 C.3 T1），效果等价奇 | — |
| awr | exp(A_r(a)/τ) | **奇+偶混合**（sinh+cosh） |
| softq | π 加权 (Q − α log π)，含关系无关的熵项 | Q 部分奇、熵部分偶 |
| gibbs(τ) | softmax(Q/τ) 权重 | τ→0 奇主导，τ→∞ 偶主导 |

### C.2 起步引理（已推到一半，两条干净结果可直接验证）

**奇偶分解引理（待正式化）**：把权重写成 w(Q) = w_odd(Q) + w_even(Q)，
w_even = (w(Q)+w(−Q))/2，w_odd = (w(Q)−w(−Q))/2。关系 B 的 Q 是 A 的镜像
⟹ **g_A + g_B = 2·Σ_a π ∇logπ(a)·w_even(...)**：只有偶分量进入共享梯度。

两条已手的干净推论（建议先写成引理+对数）：

1. **reinforce 精确抵消**：含 baseline 的 A 权重下 g_B = −g_A（奇）
   ⟹ κ = 0 精确。**实测 0.000±0.000 ✓**，且能量 ‖g‖² 大（371.5，
   均匀策略时最大）——"纯关系信号、全灭"的代数原像。
2. **expq 精确抵消**：∇Σ_a π(a)Q(a) 的两项在镜像下相消 ⟹ κ = 0 精确。
   **实测 0.000±0.000 ✓**。

### C.3 任务清单 T1–T5（按优先级）

| # | 任务 | 数值靶子（出处） | 现状（2026-08-19 精确验证） |
|---|---|---|---|
| T1 | **闭式 κ**：五场在 (π, r, τ, α) 下的 κ(π)。锚点：τ≪r 时 κ_AWR → 1/2 + 1/(2cosh(2r/τ))（均匀 π、无 baseline 的标量情形）；解释为何实测略高（0.561，π 非均匀 + baseline 修正） | awr HIDDEN κ=0.561±0.020（`toy_fields/results.json`） | ✅ **2026-08-26 完成（命题 A/B/C）**。规范定义钉死为 fields.py 期望采样梯度；softq κ=A²/(A²+4r²)、A=α·ln(p/q) 与存储 T2 七点逐位吻合；awr-baseline 唯一幸存 PG 型场（∇V 隐性偶耦合），均匀 π 处 κ=0 对一切 τ；旧 CE-fit 轨数字为另一泛函的历史对照。"0.561" 确认为伪影叠加 init 抽样，作废。脚本 `verify_closed_forms_o1.py`、数据 `o1_closed_forms.json` |
| T2 | **softq 微结构**：κ 应随 α 从 0（=expq，κ=0）连续升向熵主导（κ→1）。查 `run_toy_fields.py` 用的 α，解释实测 0.068±0.044 ≠ 0 | softq HIDDEN κ=0.068±0.044 | ✅ **升级为定理（2026-08-26）**：κ_softq = A²/(A²+4r²)，A=α·ln(p/q)；对 p≠1/2 关于 α 严格单调；p=1/2 恒 0；α→∞/峰化 →1。三路互证见 §3.2 引用 |
| T3 | **τ 单调性**：gibbs(τ) 的 κ(τ) 单调升（偶分量占比随 τ 升）。⚠️ E2 数据是已适应 dynamic 的收敛 SAC，只有弱单调——理论应对照 Toy 重跑的 τ 扫描，不是 510K 数字 | τ=0.2/1.0/5.0 → 0.511/0.562/0.588（`common_basis_interp/results.json`） | ✅ **2026-08-26 已判决（O2 取证完成）**。`run_interp.py:123` 的 `loss_gibbs_expq` = grad −Σ_a π_{θ/τ}(a)·Q_critic(a)，即 fields.py 的 E_{π_τ}[Q] 场——**在镜像 bandit 上恒 0（swap 定理），但 E2 的 510K 非零且随 τ 变化，因为 510K 非镜像多状态**（deals 不构成精确镜像、critic Q 跨关系不对称）→ E2 τ 拨盘真实存在但载体是**非镜像结构下的均值场**，与 E3 判决"TD 对齐需非镜像结构"统一。另：`run_interp.py:54` 用 deterministic rollout → E1/E2/cross-transfer 数字带协议 caveat（见 `notes/rollout_protocol_artifact.md`）。论文写法：τ 拨盘=非镜像结构 × 均值场，Toy 镜像下不存在（κ≡0），softq α 拨盘才是普适单调轴（T2） |
| T4 | **幸存即失明**（权衡定理）：证明共享更新方向 = J·w_even 只增加"大 \|Q\| 动作"的概率质量、不区分镜像对 ⟹ 幸存梯度对关系特定进展的投影为 0 或 ε。钝端实证：DQN κ=0.645 & reward=0 | `research_status.md` §1.3 |
| T5 | **REVEALED 退化**：四场全 = 0.438（策略近正确时各场退化为同一夹角）——证明这是 gA−gB 夹角的场无关函数，说明"字段轴只在信息受限时现身" | `toy_fields/results.json` REVEALED 段 |

### C.4 陷阱清单（推之前读一遍）

1. **baseline 减除改变奇偶**：A = Q − V 里 V 依赖 π，不是纯奇权。E2/E1 的
   S/D 不稳健可能部分源于此。推导时明确"带 baseline / 不带"两个版本。
2. **期望 vs 采样**：`fields.py` 用 rollout 采样估计 g_r；闭式对期望。
   对数时用大 N rollout（采样方差 ~1/N，可用 N(δ) 公式定 N）。
3. **Fisher 度量**：κ 的分母是 ‖g‖² 不是 w²；共享/对比能量比会经过
   J_π = Σ_a π ∇logπ∇logπᵀ 加权，均匀 π 时退化为标量情形（C.3 T1 锚点）。
4. **别把两个 0.5 混为一谈**：Toy AWR κ→1/2 是偶奇能量比；510K 混合协议
   κ≈0.50 是两独立梯度正交的期望值。机制不同，论文里要显式区分。

### C.5 证伪判据（对不上就放弃框架，成本两周）

- T1 闭式与 `toy_fields/results.json` 的五个数在种子噪声内对不上 → 死。
- T3 在 Toy 干净重跑的 τ 扫描上不单调 → 死。
- 通过 → 框架升级为定理，ICLR/ICML 之别只是排版。

**2026-08-19 执行结果**（详见 `toy_field_axis_theory.md`）：
- T2 **通过**（κ(α) 单调 0→0.90，精确=采样）；T3 **分裂**（softmax(Q/τ) 单调✓，
  fields.py 的 E_{π_τ}[Q] 恒 0 → 定义澄清后可过）；T1 锚点**公式错误但框架可救**
  （baseline 改变奇偶、κ(π) 需按函数报告）。
- **框架未死**，判定：需按"修正清单"重推 T1 闭式 + 澄清 gibbs 定义后，
  T1-T3 可全部转正。真实环境（Overcooked 场轴）一致：reinforce 死（κ≈0.016）、
  value 存活（κ≈0.98）——框架核心二分的跨环境演示。

详细推导与全部精确/采样表见 `notes/toy_field_axis_theory.md`。

### C.6 可选加餐：监督域最小迁移（半天~一天，不需要 GPU）

同输入、±1 标签按 p 混合，测 BCE（偶主导）vs hinge/perceptron（奇）的梯度保留。
成立 → "现象泛化"从主张变证据（多任务负迁移 / 联邦 client drift / 混合偏好
RLHF 都是同构结构）。失败 → 诚实收窄为 RL 内定理。

---

## Part D：收编表（框架如何统一全部已有结果）

| 已有结果 | 框架读法 |
|---|---|
| Toy HIDDEN reinforce κ=0（能量 371） | 奇场：纯关系信号全灭（脆端极限） |
| Toy HIDDEN awr κ=0.561 | 偶分量幸存（拨盘中段） |
| DQN/TD κ 高 & reward=0 | 钝端：幸存但失明 |
| 上一篇"反转" | 拨盘两端被并列比较 —— Prop 4 开放问题关闭：位置可预测 |
| E2 字段轴 + τ 扫描 | 拨盘连续转动的实证 |
| Reveal 谷死亡 / 510K 诊断失败 | 仪器不灵敏 / 关系弱驱动 —— 负面结果作方法边界 |
| 分解 P1-P4 + N*/N(δ) | 拨盘的测量仪 |
| Overcooked stuck/converged + oc_mem（待跑） | 拨盘在真实环境读数 + 干预闭环。⚠️ 2026-08-19 GPU 重训 16 模型：dynamic **能学**（reward~120，非卡死）；forced 协议对 dynamic 是伪影（关切换→0 奖励→κ=0）；切换保持协议下 κ 分离真实（0.010 vs 0.31）；**oc_mem 阴性**（memory 不恢复 κ，σ² 反随 m 涨）。场轴：reinforce 死 / value 存活 | `data/kappa/server_tasks/results/`、`notes/toy_field_axis_theory.md` §4 |
| 510K 场轴（上一篇主战场，补枪） | ✅ 2026-08-19：ppo_reveal 模型 0.50/1.00 级损坏已 GPU 重训补齐（n=6/级）。**reinforce κ≈0.014-0.017（死）/ value κ≈0.41-0.47（活）——上一篇 A2C vs DQN 反转在共同基上复现**；p 依赖成立：隐藏适应（p≤0.5）value κ≈0.47 > 可见适应（p=1.0）0.41，同 Overcooked 方向 | `data/kappa/server_tasks/results/510k_field_axis.json`、`notes/toy_field_axis_theory.md` §4 |

一句话版（reviewer 复述测试）：
**"隐藏关系混叠下，梯度场按其权重函数的奇偶几何分为会死的和会瞎的；
κ 分解开之后你能看出它死在哪一段；我们给出预测规则、测量仪和修复实验。"**

---

## Part E：日历决策树（2026-08-26 重写——原树的分叉条件已全部落定）

> 原树以 oc_mem 成败 + T1-T3 对错分叉。现在两者都有判决：
> **oc_mem 阴性**（记忆不恢复 κ）、**T2 ✓ / T3 已澄清**（拨盘在 softq α，
> τ 拨盘只在非镜像结构存在）、T1 锚点公式已证错但框架可救。
> 按原树的第四分支走，但材料比预期多：三环境场轴 + 监督域泛化 + witness
> 切片 + 测量伪影两件套。

```
落定状态（2026-08-26）：
  定理侧：swap 相消定理 ✓、softq α 单调定理（T2）✓、awr 闭式待推（O1）
  干预侧：oc_mem 阴性 ✗ → "诊断→修复"闭环缺阳性臂
       │
       ├─ O4 性能后果实验（Toy 自适应 → Overcooked）出阳性
       │     → ICML 2027 完整包：定理 + 场轴 + 行为层权衡 + 诚实边界
       │       （"盲何时有害"从信息层升到行为层，闭环由性能实验补）
       │
       └─ O4 阴性（偶场的盲在自适应任务上无代价）
             → ICML 理论侧/方法学：定理 + 测量学 + 三环境场轴 +
               "权衡无性能后果"作为诚实边界（no-free-lunch 的信息层版本）

ICLR 2027 不进决策树（维持原判断）。
```

> **2026-08-26 判决：走"O4 阴性"分支**。S2-P1 完成：行为差距巨大（7 倍）但
> 中介是信用分配而非关系保留 κ；td κ=0.06 行为最好——κ 与行为解耦。
> 论文定位 = 理论侧/方法学：命题 A/B/C + 三环境场轴 + 测量学（协议伪影两件套、
> N* 定理、分解）+ 监督域泛化 + **κ 适用边界**（新方法学贡献，来自 O4 解耦）。
> S2-P2（Overcooked GPU 臂）降级为可选。

### 当前执行队列（S 计划，2026-08-26 定）

| # | 内容 | 状态 |
|---|---|---|
| S0 | 决策树更新（本节）+ O2 取证写入 + 资源文档同步 + 临时脚本入库 | ✅ 完成 |
| S1 | A1 E2 判决 note；A2 awr/softq 闭式 κ(z,r,τ)/κ(α,z) + exact_kappa 对照 | ✅ 完成（命题 A/B/C 三路互证） |
| S2-P1 | Toy 关系翻转任务 + 四目标场训练 + 预注册预测 | ✅ 完成（阴性判决，见 `notes/o4_performance_verdict.md`） |
| S2-P2 | Overcooked GPU 臂 | 可选（预期收益下降）；若做：DQN/SAC vs PPO 同预算重训 + oracle 上界 |
