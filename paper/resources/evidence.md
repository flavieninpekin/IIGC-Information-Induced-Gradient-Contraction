# 证据表（evidence）

> 论文 Results 用的核心表格。每格给出处；**数字未改动**，复核自 data/*.json。
> 协议与条件见每表下方注记。最后更新：2026-08-19。

---

## 1. Toy：闭式与采样（HIDDEN，5 inits，N=200，r=1，τ=1，α=1）

出处：`data/kappa/toy_fields/theory_toy.json`（HIDDEN/REVEALED 键）。

| 场 | κ_ep | κ_mean | E_shared | E_contrast | σ² |
|---|---|---|---|---|---|
| reinforce（硬 PG） | 0.0007 | 0.0015 | 0.89 | 706.80 | 547.68 |
| expq（均值） | **0.0000** | **0.0000** | 0.00 | 721.50 | 0.00 |
| awr（baseline） | 0.0574 | 0.0626 | 260.49 | 3754.71 | 294.58 |
| softq（α=1） | 0.0575 | 0.0575 | 43.14 | 721.50 | 0.00 |
| gibbs（E_{π_τ}[Q]） | **0.0000** | **0.0000** | 0.00 | 721.50 | 0.00 |

REVEALED（同一批场）：全部 κ≈0.440（0.438-0.441），与旧测量 0.438 一致（T5 退化）。

> 注：expq / gibbs(E_{π_τ}) 的 E_shared **精确为 0**（g_B=−g_A 相消）；reinforce 的
> 0.0015 是采样噪声（E_shared=0.89 ≪ E_contrast=707）。

## 2. Toy：T1-T3 闭式扫描（精确，r=1）

出处：`theory_toy.json`（T1_awr_tau / T2_softq_alpha / T3_gibbs_tau）。

**T1 awr**（z0=[0.372,−0.239]；τ=[0.1,0.25,0.5,1,2,4,8]）：

| 变体 | κ(τ) |
|---|---|
| 精确·带 baseline | 0.499 → 0.458 → 0.359 → **0.253** → 0.239 → 0.383 → 0.663（**非单调**） |
| 精确·无 baseline | 0.081 → 0.081 → 0.086 → 0.131 → 0.291 → 0.594 → 0.850（单调升） |
| 精确·峰化 π=[2,−2]·baseline | 0.500 → 0.500 → 0.500 → 0.512 → 0.614 → 0.806 → 0.936（单调） |

**T2 softq**（α=[0.01,0.1,0.5,1,2,5,10]；精确=采样，σ²=0）：

κ(α) = 0.000 → 0.001 → 0.023 → **0.085** → 0.272 → 0.700 → 0.903（**单调升**）

**T3 gibbs**（τ=[0.1,0.2,0.5,1,2,5,10]）：

| 定义 | κ(τ) |
|---|---|
| gibbs（fields.py：grad E_{π_τ}[Q]） | **恒 0.0（所有 τ）** |
| softmax(Q/τ) 权重（framework C.1） | 0.081 → 0.081 → 0.086 → 0.131 → 0.291 → 0.693 → 0.898（**单调升**） |

> 判决：T2 ✅；T3 分裂（定义澄清后可过）；T1 锚点公式错（均匀 π 无 baseline
> 精确 κ=0）。详见 `paper/resources/theory.md`。

## 3. Overcooked（GPU 重训 16 模型，切换保持协议）

出处：`oc_switch_kappa.json`（N=60/partner/seed）。

**切换保持协议下的分解**（n=8 seeds）：

| 组 | κ_ep | κ_mixed | E_shared | σ² |
|---|---|---|---|---|
| static | **0.310 ± 0.152** | 0.907 ± 0.069 | 41090 | 15298 |
| dynamic | **0.010 ± 0.008** | 0.451 ± 0.188 | 4869 | 445531 |
| mem m4 | 0.012 ± 0.010 | 0.463 ± 0.119 | 6981 | 618677 |
| mem m8 | 0.008 ± 0.003 | 0.559 ± 0.147 | 8558 | 979012 |
| mem m16 | 0.008 ± 0.003 | 0.487 ± 0.041 | 11047 | 1391925 |

> 核心：dynamic κ_ep = 0.010 << static 0.31，**由 σ² 主导**（dynamic σ² ≈ static 的
> 90×）——隐藏角色梯度收缩的真实读数。**记忆干预不恢复 κ**（mem 各 m κ_ep≈0.01，
> σ² 反随 m 涨到 1.39M，阴性）。

**场轴**（n=3 seeds，return 加权梯度；出处 `oc_field_axis.json`）：

| 场 | static κ_ep | dynamic κ_ep |
|---|---|---|
| reinforce（硬 PG） | 0.018 ± 0.003 | 0.015 ± 0.001 |
| awr（优势加权） | 0.015 ± 0.005 | 0.016 ± 0.001 |
| value（TD/mean-seeking） | 0.549 ± 0.012 | **0.982 ± 0.009** |

**标准 grid**（n=8 seeds；`oc_decomp_b1_seed*.json`、`oc_n_protocol.json`）：
static κ_ep 0.06-0.44、κ_mean≈0.50（能量门过）；dynamic 在 forced 协议下全 0
（协议伪影）。oc_n_protocol（seed41）：static κ_true=0.507，1/√N 拟合 R²=0.99
（slope −0.72），κ̂(N) 0.41→0.495。

> 注：forced-partner 协议在 env 中会**关闭 mid-episode 角色切换**
> （`overcooked_v3_env.py:102`）→ 对切换型（dynamic）策略 OOD → 0 奖励 → κ=0
> 是测量伪影；只有切换保持协议下才能区分真收缩（dynamic 0.010 vs static 0.31）。
> dynamic 在自由 rollout 下奖励 ~100-120（与 static 持平，`verify_dynamic_learns.py`），
> 说明收缩是梯度场性质而非"学不动"。

## 4. 510K（上一篇环境，ppo_reveal 模型，n=6/级）

出处：`510k_field_axis.json`（team 条件方差分解，全信息评估）。

| p（训练时队友位可见度） | reinforce κ_ep | value κ_ep |
|---|---|---|
| 0.00（隐藏适应） | 0.017 ± 0.005 | **0.469 ± 0.029** |
| 0.50 | 0.017 ± 0.003 | **0.474 ± 0.022** |
| 1.00（可见适应） | 0.014 ± 0.002 | **0.408 ± 0.007** |

> 场轴分离 ~30×（reinforce 0.015 vs value 0.44，18/18 模型一致）；
> p 依赖成立（隐藏适应 value κ > 可见适应，0.47 vs 0.41）。
> 上一篇跨算法 κ（`AAAI2027-510k-clear/data/kappa_summary.json`）：
> A2C single 0.644 / dynamic 0.519（收缩），DQN single 0.797 / dynamic 0.917（反转）。

## 4.5 Overcooked 受控切片（witness 机制验证）

出处：`data/kappa/overcooked_slice/conflict_witnesses.json`、`witness_field_axis.json`。
设计：固定场景（ready 汤在锅 + 洋葱 + 盘子），脚本化 deliver/cook 两个 option，
角色信用奖励（chef=送餐得分、waiter=放洋葱信用）；状态/horizon/种子跨角色一致。
witness = 两角色 argmax option 不同且 margin>δ。详见 `paper/resources/overcooked_slice.md`。

**Witness 验证**（layout=asymmetric_advantages，H=80，δ=1.0）：

| 指标 | 值 |
|---|---|
| 构造候选 → witness | **24/52（46%）** |
| hidden obs 跨角色恒等 | **True** |
| 自然随机状态 witness 率 | 0/30（0%） |

典型 witness：chef→deliver（Q=20 vs 0）、waiter→cook（Q=6 vs 0）。

**场轴（witness 上，H=40）**：

| 场 | κ_mean | 说明 |
|---|---|---|
| value（TD/mean-seeking） | **0.941-1.000** | 值函数看不见角色 → chef/waiter 值梯度对齐（E_contrast≈0） |
| reinforce / 期望-Q / hard | **N/A** | 冲突在 option 层面；原始动作 Q 平坦或 chef 侧≈0（见 `overcooked_slice.md` §5） |

*原始级场轴在 witness 上不可测（三定义均退化）；原始级收缩由完整环境 rollout
场轴展示（`oc_field_axis.json`：reinforce 0.015 vs value 0.98）。

## 5. 关键阴性/边界结果

| 结果 | 证据 |
|---|---|
| "75% 谷"死亡（n=6 回归均值，ANOVA F=0.756 p=0.759） | `data/kappa/510k_reveal/results.json`、`research_status.md` §2.7 |
| 510K 混合协议不灵敏（σ²~1.6M ≫ 队间 ~2.5e4） | 同上 |
| forced 协议伪影（固定伙伴→关切换→0 奖励→κ=0，仅切换型环境） | `overcooked_v3_env.py:102`、本会话 |
| oc_mem 阴性（记忆不恢复 κ） | `oc_switch_kappa.json` mem_* 键 |
| 510K 关系弱驱动（策略几乎不依赖队友位） | `stuck_detect/sensitivity.json` |

---

## 6. O4 ��ϵ����Ӧʵ�飨S2-P1��2026-08-26��

������`data/kappa/o4_adaptive/`��96 runs��3 modes �� 4 arms �� 8 seeds����
Ԥע�� `notes/o4_performance_design.md`���о� `notes/o4_performance_verdict.md`��

**switch_hidden��oracle=40��**��

| �� | final return | ��ת���� | �ɾ��� �� |
|---|---|---|---|
| td | **36.97 �� 0.06** | 0.75 | 0.06 �� 0.04 |
| softq | 28.62 �� 11.81 | 0.58 | 0.94 �� 0.09 |
| reinforce | 9.48 �� 5.13 | 0.18 | 0.43 �� 0.18 |
| awr | 5.03 �� 2.02 | 0.04 | 0.57 �� 0.08 |

Ҫ�㣺(1) Ŀ�꺯��ѡ����� 7 ����Ϊ��ࣨ��ͬ��ѵ������(2) **�� ����Ϊ����**
��td �� �����Ϊ��á���������������λ���١��� partner ��ǩ��������(3) ����
����Ӧ�޳�ŵ���ۣ�softq/td ��ʱ��Ӧ��reinforce/awr 30k ���ڲ���Ӧ����
(4) O4 �߳�ʵ�߽��֧��"ä����Ϊ����"�ڱ������岻�ɼ����

д�����ģ���Ϊ �� ���ñ߽��֤�� + "������֧������Ӧ����"����������
O3/T4 �������ݣ��� DQN ʧ�ܷǶ۶�ʧ�����£���
