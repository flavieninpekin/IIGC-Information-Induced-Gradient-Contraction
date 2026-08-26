# C.6 判决：监督域最小迁移（BCE vs hinge 梯度保留）— 成立

> 实验：`experiments/common_basis/toy/run_supervised_transfer.py`
> 数据：`data/kappa/supervised_transfer/fixed_model.json`（固定模型协议）
> 判决：**奇偶保留轴在监督域成立**——hinge 精确相消（κ=0），BCE/quadratic
> 完全对齐（κ=1，钝化），perceptron 半空间伪影。2026-08-22。

---

## 1. 设计（固定模型镜像协议，对齐 Toy set_partner）

- 同输入 x（高斯，N=4000），关系 = 标签符号翻转（A: y=+1，B: y=−1）。
- **同一固定模型 w0**（随机 init，不训练），同一数据，只换损失。
- g_A = ∇L(f)（y=+1），g_B = ∇L(−f)（y=−1）；κ 测两关系条件梯度保留。
- **不训练是关键**：训练会适应混合（= E2 的"已适应策略"confound，
  把签名洗掉）——固定模型才是"未适应"的干净测量。

## 2. 理论（镜像下 g_B = −g_A ⟺ L′ 为偶函数）

g_A = Σ_i L′(f_i)·x_i，g_B = −Σ_i L′(−f_i)·x_i（链式法则，y=−1 时
df/dw = −x）。故 g_B = −g_A 当且仅当 L′(−u) = L′(u)。

| 损失 | L′(u) | 奇偶性 | 预测 κ |
|---|---|---|---|
| hinge | −I(u<1) | 非偶（|u|<1 时 = −1） | **0（精确相消，脆端）** |
| perceptron | −I(u<0) | 半空间门控 | 半空间均值伪影 |
| BCE | −σ(−u) | 非偶，但 g_A=g_B=−Σσ(−f)x | **1（钝化对齐）** |
| quadratic | 2(u−y) | g_A=g_B=2Σ(wx−1)x | **1（钝化对齐）** |

## 3. 实测（5 seeds，mean∈{0,1}，scale∈{0.01,0.1,1}）

| 损失 | mean=0, s=0.01 | mean=0, s=1 | mean=1, s=0.01 | mean=1, s=1 |
|---|---|---|---|---|
| hinge | **0.0000 ± 0.0000** | 0.9908 ± 0.0033 | **0.0000 ± 0.0000** | 0.2941 ± 0.1815 |
| perceptron | 0.9975 ± 0.0006 | 0.9966 ± 0.0015 | 0.3409 ± 0.1402 | 0.1995 ± 0.0860 |
| bce | **1.0000 ± 0.0000** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| quadratic | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

- **hinge κ=0.0000 精确**（|f|<1 全部活跃时 g_B=−g_A；scale=1 时部分
  margin 外失活 → 相消破坏，κ 回升——与闭式一致）。
- **BCE/quadratic κ=1.0000 精确**：关系不可区分时梯度场**完全对齐**
  （g_A=g_B，模型对标签翻转失明）——这正是"钝化"端的代数形式：
  对齐但不是"好的保留"，是对关系失明。
- perceptron 的 κ≈0.99 是半空间均值伪影（Σ_{f>0}x = c·ŵ），非保留信号。

## 4. 解读（框架泛化的证据）

1. **"现象泛化"从主张变证据**：RL 的脆/钝二分在监督域直接复现——
   hinge（指示权重）= 脆端（κ=0 精确相消）；BCE/quadratic（光滑/均值）
   = 钝端（κ=1 但对关系失明）。同一模型/数据，只换损失，签名立刻出现。
2. **与 RL 表一致**：
   - hinge ↔ reinforce（硬场，κ=0）
   - BCE ↔ TD/softq（均值场，κ 高但 reward=0 的"聚合正确、行动无用"）
3. **泛化结构的共同根源**：权重函数对"关系符号翻转"的奇偶性。
   RL 里关系翻转 = Q→−Q（镜像）；监督里 = y→−y。同一个定理框架。
4. **应用含义**：混合偏好 RLHF（同输入、±1 偏好按 p 混合）与联邦
   client drift 是同构结构——BCE/soft 场钝化、hinge/hard 场崩脆。

## 5. 与 trained-at-p 协议的对比（已适应 confound）

`data/kappa/supervised_transfer/results.json`（先按 p 训练再测）：
p→0.5 时 bce/hinge/perceptron κ→0.004-0.39（模型收敛到 f≈0，梯度死区）
——与 E2 的"已适应策略测不出签名"完全同构。**固定模型协议才是干净测量**。

## 6. 数据位置

- 脚本：`experiments/common_basis/toy/run_supervised_transfer.py`
- 结果：`data/kappa/supervised_transfer/fixed_model.json` + `results.json`
