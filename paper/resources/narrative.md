# 论文骨架（narrative）

> 承接上一篇 AAAI 2027（`AAAI2027-510k-clear`，现象篇：IIGC + κ + 反转作为边界）。
> 本篇（机制篇）回答其开放问题：**反转的机制是什么？**
> 最后更新：2026-08-19。

## 一句话贡献

> 在共同测量基上证明：策略梯度与值方法在隐藏关系信息下的 κ 反转，是**梯度场
> 权重函数的奇偶几何**的可预测后果——硬（mode-seeking）场收缩（κ→0），
> 软/均值（mean-seeking）场对齐（κ 高）。并给出修正的测量协议、
> 三环境（Toy / Overcooked / 510K）的完整证据与一条闭式验证链。

## 候选标题

1. Retention vs Specificity: Why Hidden Relations Contract Policy Gradients but Not Value Gradients
2. The Parity of Gradient Fields: A Unifying Account of the PG/Value Kappa Reversal under Hidden Relational Information
3. Mode-Seeking Dies, Mean-Seeking Survives: Explaining the Policy-Gradient / Value Reversal on a Common Basis

## 摘要草稿（~250 词）

隐藏关系信息（如队友身份/角色/分配）会把"各关系下方向相反"的策略梯度投影到同一
局部信息空间并相互抵消（IIGC，AAAI 2027）。上一篇文章证明 κ（关系梯度保留率）
是梯度估计器的函数，并指出"策略梯度收缩 vs 值方法反转"是一个开放机制。本文
给出机制：设每个目标函数诱导一个关系条件梯度 g_r = Σ_a π(a)∇logπ(a)·w_r(a)，
w 由目标决定。把 w 按 Q 的奇偶分解，隐藏关系（Q 镜像）下**只有偶分量幸存为
共享梯度，奇分量全部进入对比梯度被摧毁**——硬策略梯度（reinforce，奇）精确相消
κ=0；softmax(Q/τ) 与带熵的 softq（偶/混合）幸存。我们用解析 bandit 验证闭式
（softq κ 随 α 单调 0→1；softmax(Q/τ) κ 随 τ 单调），并修正了原锚点公式
（均匀 π 无 baseline 时 awr κ=0 而非 1/2）。跨环境：在 GPU 重训的 Overcooked
与 510K 上，**同一网络同一数据只换目标函数**——reinforce 场 κ≈0.01-0.02（死），
值函数场 κ≈0.4-0.98（活），包括上一篇声称反转的 510K 环境。我们同时报告：
(1) 一个测量陷阱：固定伙伴的测量协议会关闭角色切换，对切换型策略产生 κ=0 的
伪影，需用切换保持协议区分"真收缩"与"测量伪影"；(2) 记忆干预不恢复 κ（阴性）；
(3) 紧凑测量的排序保持定理（N* 单次穿越）为测量提供保证。

## 贡献（Contributions）

1. **机制定理（保留—特异权衡）**：隐藏关系下梯度幸存成分由权重函数的奇偶分解
   决定；硬场精确相消、软场幸存。（支持：Toy 精确闭式 + T2/T3 验证）
2. **测量方法学**：展示固定伙伴协议在切换型（dynamic）环境中会产生 κ=0 伪影
   （固定伙伴 → 关闭角色切换 → 策略 OOD → 0 奖励），给出切换保持协议；
   在该协议下隐藏关系的 κ 收缩（Overcooked dynamic 0.010 vs static 0.31）
   首次可测。
3. **跨环境场轴演示**：Toy / Overcooked / 510K 三环境、同一基、同一结果——
   reinforce 死 / value 活，包括上一篇反转的原始环境。
4. **闭式验证与修正**：softq κ(α) 单调成立；softmax(Q/τ) κ(τ) 单调成立；
   原 awr 锚点公式错误并给出正确 κ(π, τ, baseline) 框架。
5. **阴性结果（诚实边界）**：记忆干预不恢复 κ；κ 是 π 依赖的（报告分布非单点）。

## 章节结构（Outline）

| 节 | 内容 | 证据来源 |
|---|---|---|
| 1 Intro | 反转现象 + 测量伪影的区分（真收缩 vs κ=0 伪影） | 本会话发现 |
| 2 Prelim | κ、能量分解（P1-P4）、紧凑测量 N* 定理 | `notes/variance_decomp_theory.md` |
| 3 Framework | 奇偶分解引理、保留—特异权衡、闭式 | `paper/resources/theory.md` |
| 4 Toy 验证 | T1-T3 精确/采样表、REVEALED 退化 | `theory_toy.json` |
| 5 Overcooked | 切换保持协议下的 κ 分离、场轴、oc_mem 阴性 | `oc_switch_kappa.json`、`oc_field_axis.json` |
| 6 510K | 上一篇环境上的场轴复现 + p 依赖 | `510k_field_axis.json` |
| 7 讨论 | 与 Prop 4 的关系、方法学含义（先暴露 confound 再下结论） | `research_status.md` §2.10 |
| App | 复现脚本与数据清单 | `data_index.md` |

## 关键叙事钩子

- 隐藏/切换环境里"κ=0"有两种截然不同的读法：真收缩（信号被关系混叠平均掉）或
  测量伪影（协议破坏了环境的动态）。本文给出区分方法，并在修正协议下证明
  **真现象（场轴反转）成立且机制可预测**。
- "测不到信号时先怀疑隐藏 confound 在平均你的测量，而不是断定没有效应"——
  这既是方法学教训，也是 IIGC 现象本身的演示（元观察，见 `research_status.md` §2.10）。

## 待定

- [ ] 正确闭式 κ(π, τ, baseline) 的符号推导（见 `theory.md` 开放项 O1）——S1-A2 执行中
- [x] E2 gibbs 定义澄清（O2）✅ 2026-08-26：τ 拨盘 = 非镜像结构 × 均值场（`theory.md` O2）
- [ ] 是否包含旧"75% 谷"死亡 + 510K 混合协议不灵敏（§7 讨论或附录）
- [ ] O4 性能后果实验结果（S2，预注册见 `notes/o4_performance_design.md`）
