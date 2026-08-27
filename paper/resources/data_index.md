# 数据索引（data_index）

> 论文所有数字的原始出处。复核引用时从这里进。最后更新：2026-08-19。

## 本轮新证据（新论文核心）

| 数据 | 路径 | 内容 |
|---|---|---|
| Toy 闭式+采样 T1-T3 | `data/kappa/toy_fields/theory_toy.json` | exact_kappa（autograd）+ 每集梯度采样、全分量、三扫描 |
| Toy 旧测量 | `data/kappa/toy_fields/results.json` | κ+总能量（5 inits，N=30） |
| Overcooked 分解（切换保持） | `data/kappa/server_tasks/results/oc_switch_kappa.json` | static/dynamic/mem×m 的 E_shared/E_contrast/σ²/κ_ep/κ_mixed，n=8/9 |
| Overcooked 场轴 | `data/kappa/server_tasks/results/oc_field_axis.json` | reinforce/awr/value 场 κ（3 seeds） |
| Overcooked 标准 grid | `data/kappa/server_tasks/results/oc_decomp_b1_seed*.json` | forced 协议静态/动态分解（n=8） |
| Overcooked baselines | `data/kappa/server_tasks/results/oc_baselines_b1*.json` | reward/grad_norm/κ 对比（n=8） |
| Overcooked N-协议 | `data/kappa/server_tasks/results/oc_n_protocol.json` | 1/√N + 排序定理（seed41） |
| 510K 场轴 | `data/kappa/server_tasks/results/510k_field_axis.json` | reinforce/value 场 κ，p∈{0,0.5,1}×n=6 |
| O4 自适应实验 | `data/kappa/o4_adaptive/`（96 runs + aggregate.json） | 四目标场 × 关系翻转任务：行为差距、κ-行为解耦、部署再适应 |
| O1 闭式验证 | `data/kappa/toy_fields/o1_closed_forms.json` | 命题 A/B/C 三路互证（闭式/autograd/MC） |
| Overcooked 训练曲线（快照） | `data/logs/train_*.log`、checkpoints `data/models_overcooked/` | 400K/800K/1M 快照 |

## 本轮新增模型（GPU 重训，替换损坏）

| 数据 | 路径 |
|---|---|
| Overcooked v3（16 个：static/dynamic×41-48） | `data/models_overcooked/overcookedv3_*_final.zip` |
| Overcooked memory（9 个：m4/8/16×41-43） | `data/models_overcooked/overcooked_mem_dynamic_*.zip` |
| 510K reveal 重训（0.50/1.00×缺失 seeds，8 个） | `data/models_reveal/ppo_reveal_{0.50,1.00}_s*.zip` |

## 历史证据（上一篇 + 早期新项目）

| 数据 | 路径 | 内容 |
|---|---|---|
| 上一篇跨算法 κ | `AAAI2027-510k-clear/data/kappa_summary.json` | A2C/DQN/SAC/REINFORCE/PPO × 510K/Overcooked |
| Reveal 全网格 | `data/kappa/510k_reveal/results.json` + `figures/` | 21 档 × 6 seeds，κ(p) 平坦（ANOVA F=0.756 p=0.759） |
| E1 actor/critic | `data/kappa/common_basis_sac_split/results.json` | SAC 场分离（n=2） |
| E2 插值谱 | `data/kappa/common_basis_interp/results.json` | reinforce<awr<softq/expq + gibbs τ |
| Cross-transfer | `data/kappa/cross_transfer/results.json` | 训练×测试 2×2 |
| Toy 场轴（旧） | `data/kappa/toy_fields/results.json` | reinforce 0/awr 0.561/softq 0.068/expq 0 |
| 510K stuck_detect | `data/kappa/stuck_detect/` | forced_decomp/sensitivity/memory_eval |
| 能量分解 A/B/C | `data/kappa/variance_decomp/` | accuracy/compactness/scale/ranking_flip |

## 笔记（理论/方法）

| 笔记 | 路径 |
|---|---|
| 理论纲领 + 行动清单 | `notes/theory_program.md` |
| Toy 场轴闭式验证 | `notes/toy_field_axis_theory.md` |
| 紧凑测量排序保持（N* 定理） | `notes/variance_decomp_theory.md` |
| 研究现状汇总 | `notes/research_status.md` |
| 反转讨论 / 应用验证 / reveal 理论 | `notes/kappa_reversal_discussion.md`、`app_validation.md`、`reveal_theory.md` |

## 复现脚本（✅ 2026-08-26 已入库）

| 脚本 | 作用 |
|---|---|
| `experiments/common_basis/toy/verify_theory_toy.py` | Toy 闭式（exact_kappa，π-加权协议）+ 采样，T1-T3（theory_toy2.json） |
| `experiments/common_basis/toy/verify_theory_toy_v1_cefit.py` | 08-19 原版 CE-fit 轨计算器，仅复现 `theory_toy.json` |
| `experiments/common_basis/server_tasks/run_510k_field_axis.py` | 510K reinforce/value 场 κ（team 条件方差分解） |
| `experiments/common_basis/server_tasks/run_field_axis.py` | Overcooked reinforce/awr/value 场 κ（切换保持） |
| `experiments/common_basis/server_tasks/run_switch_kappa.py` | Overcooked 切换保持分解 + memory 干预 |
| `experiments/common_basis/server_tasks/verify_dynamic_learns.py` | dynamic 训练曲线验证 |
| `experiments/common_basis/server_tasks/repair_checkpoints.py` | SB3 checkpoint 修复（损坏 optimizer） |
| `experiments/common_basis/server_tasks/batch_train_{v3,mem,reveal}.py` | GPU 批量重训 |

> 其余 server_tasks 驱动脚本与说明见 `experiments/common_basis/server_tasks/README.md`。
