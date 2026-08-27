# E2 gibbs 判决（O2 取证）— 2026-08-26

> 判决对象：`research_status.md` §2.4 的 τ 扫描（single 0.511→0.562→0.588）到底
> 是哪个场、能否进论文。
> 证据链：代码 `experiments/common_basis/interpolation/run_interp.py` +
> 原始数据 `data/kappa/common_basis_interp/results.json`。

---

## 1. 定义取证（读代码）

`run_interp.py::loss_gibbs_expq`（L123-130）：

```python
pi = F.softmax(logits / tau, dim=-1)          # π_{θ/τ}，只依赖策略参数
q  = min(critic1(obs), critic2(obs))           # no_grad，critic 不回传
loss = -(pi * q).sum(-1).mean()                # 目标 = -E_{π_θ/τ}[Q]
```

即 **E2 的 "gibbs" 就是 fields.py 的 E_{π_τ}[Q] 场**（notes/theory_program.md C.1 表
里的第三种定义），**不是** framework 定义的 softmax(Q/τ) 权重场（=归一化 awr）。

## 2. 数据核对（三条硬证据）

1. **gibbs(τ=1.0) 与 expq 全部 8 个 run 逐位相等**（softmax(z/1)=π，
   两段代码在 τ=1 时数学等价）——定义确认无误，数据内部自洽：

   | mode | seed41 | seed42 | seed43 | seed44 |
   |---|---|---|---|---|
   | single | 0.560340 = 0.560340 ✓ | 0.479977 = 0.479977 ✓ | 0.559876 = 0.559876 ✓ | 0.646300 = 0.646300 ✓ |
   | dynamic | 0.714556 = 0.714556 ✓ | 0.418186 = 0.418186 ✓ | 0.473190 = 0.473190 ✓ | 0.472645 = 0.472645 ✓ |

2. **聚合均值复现报告数字**：single τ∈{0.2,0.5,1,2,5} →
   0.511±0.032 / 0.533±0.024 / 0.562±0.068 / 0.575±0.127 / 0.588±0.133；
   dynamic → 0.514 / 0.525 / 0.520 / 0.526 / 0.567。

3. **逐种子 τ 趋势异质**（关键新发现）：
   - single：seed41 单调升（0.47→0.65）、**seed42 单调降（0.55→0.43）**、
     seed43/44 非单调；
   - dynamic：四个种子全部非单调。
   - **聚合弱单调是种子平均的产物，不是律**——印证 research_status §2.4
     "n=2 时签名没撑住" 的原始怀疑，且 n=4 后仍如此。

## 3. 理论解释（为什么非零、为什么不该有拨盘）

- 镜像 bandit 上（Toy）：π_τ 与 Q 无关、Q 只经符号进入 ⟹ g_B=−g_A 精确，
  **κ≡0 对一切 τ**（swap 相消定理，theory_toy2.json 实测恒 0）。
- 510K 上非零且随 τ 弱变：**非镜像多状态结构**——两个 rollout 批（不同 deal 种子）
  不构成精确镜像，critic Q 跨关系不对称；τ 只调制"策略对 Q 分布的平均宽度"，
  在非对称数据上产生弱的方向效应。
- 这与 E3 判决（replay_mix_verdict.md："TD/均值场的对齐依赖非镜像结构"）
  是同一条结构轴的两面：**镜像处奇性占优（κ→0），非镜像给均值场留对齐空间**。

## 4. 论文写法（判决）

1. **E2 τ 扫描不作为"拨盘"证据**：逐种子方向不一致 + 协议 caveat（下条），
   只能作背景观察。若要展示 τ 效应，必须注明"条件于非镜像结构与确定性 rollout"。
2. **普适单调拨盘只有 softq 的 α**（T2 已定理化，精确=采样）——主叙事用它。
3. **τ 拨盘重新定位为结构轴的读数**：同一均值场在镜像结构 κ≡0、非镜像结构
   κ>0 且可调——这把 E2/E3 统一成"场 × 结构"二维图景的一个格子。
4. **协议 caveat 必须标注**：`run_interp.py:54` 用 deterministic rollout
   （argmax），按 `rollout_protocol_artifact.md`，E1/E2/cross-transfer 的全部
   510K 数字都有协议伪影风险（510K 大动作空间下伪影较弱但排序可能被污染）。
   复现或引用时建议改 stochastic 重测一遍（低成本：模型都在
   `data/models/510k_sac/`）。

## 5. 行动项

- [x] 本判决 note
- [x] `paper/resources/theory.md` O2 状态更新
- [ ] （低优先级，S1 之后）stochastic 重测 E2 四场 + τ 扫描，量化协议伪影幅度

## 6. 数据位置

- 代码：`experiments/common_basis/interpolation/run_interp.py`
- 数据：`data/kappa/common_basis_interp/results.json`
- 模型：`data/models/510k_sac/510k_sac_{single,dynamic}_seed{41-44}.pt`
