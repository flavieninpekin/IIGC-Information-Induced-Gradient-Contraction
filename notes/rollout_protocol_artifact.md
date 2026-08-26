# 协议发现：deterministic rollout 制造假的场轴（2026-08-22）

> 实验：`experiments/common_basis/toy/verify_rollout_protocol.py`
> 数据：`data/kappa/toy_fields/det_vs_stoch.json`
> 状态：**重大方法学发现**——Toy HIDDEN 下的"awr 抗收缩"是测量协议伪影。

---

## 1. 问题

`run_toy_fields.py`（及 `fields.py::rollout_episodes`）用 `deterministic=True`
（argmax）收集 rollout。在 HIDDEN 镜像带臂机上，**两个关系的 argmax 动作相同**
（同一策略、同一 obs）→ 每个关系下所有 step 的动作都是 a*。

对"权重只依赖 Q"的场（awr-nobase / softmaxq / expq / reinforce-nobase），
梯度退化为单点：

```
g_r = ∇[-log π(a*) · w_r(a*)] = -∇logπ(a*) · w_r(a*)   （∇w=0 时）
```

两个关系的 g_A ∥ g_B（**平行同向**）→ κ 等于**权重比**而非梯度保留：

```
κ_det = (w_A(a*)+w_B(a*))² / (2(w_A(a*)²+w_B(a*)²))
```

softmaxq 例：a*=1（π=[0.377,0.623]），w_A(1)=0.119，w_B(1)=0.881
→ κ = (0.119+0.881)²/(2(0.119²+0.881²)) = **0.6329** —— 与实测 0.6329±0.0000 逐位一致。

## 2. 两个协议的实测对比（Toy，HIDDEN 5 inits / REVEALED 训练策略）

| 场 | HIDDEN det | HIDDEN stoch | REVEALED det | REVEALED stoch |
|---|---|---|---|---|
| reinforce | 0.0000 ± 0.0000 | 0.0456 ± 0.0321 | 0.3957 | 0.3962 |
| awr | **0.5514 ± 0.0465** | **0.0281 ± 0.0215** | 0.3957 | 0.3957 |
| softq | 0.0489 ± 0.0926 | 0.0207 ± 0.0153 | 0.3957 | 0.3957 |
| expq | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.3957 | 0.3957 |
| softmaxq | **0.6329 ± 0.0000** | **0.0011 ± 0.0010** | 0.3957 | 0.3958 |

## 3. 结论

1. **Toy HIDDEN 的"awr 抗收缩（κ=0.561）"是 deterministic 协议伪影**：
   它测的是"argmax 动作处权重比"，不是"梯度场保留"。换成正确的
   π-加权随机 rollout 后，awr 收缩到 0.028±0.022 —— 与 swap 相消定理
   一致（只有 baseline 项微弱破坏相消）。
2. **softmaxq κ=0.633 是纯伪影**（闭式 π-加权 κ≡0，随机实测 0.0011）。
3. **REVEALED 协议稳健**（0.396 两种协议一致）——T5 退化（各场同夹角）成立。
4. **IIGC 核心不受影响**：reinforce/expq HIDDEN κ=0 在两种协议下都成立
   （deterministic 下精确 0，stochastic 下 expq 精确 0、reinforce ~0.05 噪声）。

## 4. 对论文的影响

- **字段轴的 Toy 证据要重写**：HIDDEN 下"reinforce<awr<softq/expq"的实验
  表不成立（全 ≈0）。字段轴的真实载体是：
  - 闭式理论（swap 相消定理 + softq α 单调，T2 ✓）
  - 跨环境（Overcooked / 510K：reinforce 死 / value 活，切换保持协议，
    见 `toy_field_axis_theory.md` §4）
- **E1/E2/cross-transfer 的 510K 数字也要重新审视**：它们同样用
  deterministic rollout 测 κ —— 在 510K（大动作空间、逐状态不同 argmax）
  伪影较弱（不是单点退化），但"κ 是场函数"的排序可能部分被协议污染。
  优先级：中等（510K 已定位为弱信号环境，不是论文主证据）。
- **方法学教训（进论文）**：κ 的测量协议必须随机 rollout（π-加权期望），
  deterministic 会退化为"执行动作处的权重比"，制造假场轴。

## 5. 行动

- [x] `verify_rollout_protocol.py`：双协议对比（本文件）
- [x] `run_toy_fields.py` 改为 stochastic 默认 + N=200（重跑场轴表）
- [x] `fields.py::rollout_episodes` 加 `stochastic` 参数
- [ ] **Toy HIDDEN 场轴表从论文中删除**（awr 0.561 / softmaxq 0.633 是伪影）；
      字段轴真实载体 = 闭式定理（swap 相消 + softq α 单调）+ 跨环境场轴
- [ ] E1/E2/cross-transfer 的 510K deterministic κ 需注明协议 caveat
      （510K 是逐状态不同 argmax，伪影较弱但排序可能被部分污染）
- [ ] 论文方法节：协议规范（随机 rollout + N 选择 + 守恒检验）

## 6. 修正后的 Toy 场轴表（stochastic, N=200 eps/关系）

`data/kappa/toy_fields/results.json`（2026-08-22 重跑）：

| 场 | HIDDEN κ | REVEALED κ |
|---|---|---|
| reinforce | 0.0011 ± 0.0013 | 0.4196 |
| awr | 0.0199 ± 0.0193 | 0.4115 |
| softq | 0.0155 ± 0.0134 | 0.4118 |
| expq | 0.0000 ± 0.0000 | 0.4116 |
| softmaxq | 0.0002 ± 0.0002 | 0.4117 |

- HIDDEN：**全场 ≈0**（swap 相消定理的实证确认，awr/softq 的微小残差来自
  baseline/熵项的偶数修正，与闭式一致）。
- REVEALED：全场 ≈0.41（T5 退化稳健复现）。
- **IIGC 核心数字不变**：reinforce/expq HIDDEN κ=0（E_sh=0，E_co>0）。
- **旧表（0.000/0.561/0.068/0.000）作废**，改为上表。
