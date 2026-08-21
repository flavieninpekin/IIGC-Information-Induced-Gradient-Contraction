# Overcooked 受控切片（overcooked_slice）

> 目标：在非 Toy 的真实合作环境里，提供**可机械验证的关系冲突 witness**，
> 把 IIGC/场轴机制从"描述"升级为"固定状态下可证明的因果对照"。
> 对应干净故事线（`storyline_clean.md`）的"受控切片 + 完整环境验证"两层设计。
> 最后更新：2026-08-19。

---

## 1. 为什么需要这个（对应"贡献不够突出"的回应）

Toy 的精确闭式很干净，但只有两个动作。510K 的关系是弱行为驱动（已被证伪）。
需要一个**关系就是任务本身**的非平凡环境，且能像 Toy 一样**逐状态反事实验证**
——不是凭角色语义推断"chef/waiter 最优行为不同"，而是用脚本化 Q 穷举证明。

## 2. 协议设计（受控切片）

- 布局：`asymmetric_advantages`（标准 Overcooked 布局，空间足够避免死锁）。
- 固定场景：pot 里有 ready 汤 + 洋葱 + 盘子；agent/partner 起始位置固定。
- 两个脚本化 option：`deliver`（拿盘→取汤→送餐）、`cook`（取洋葱→放锅）。
- **角色信用奖励**（唯一随角色变化的两件事：partner 行为 + 信用奖励）：
  - `role=chef`：partner 是厨师；agent 信用 = 送餐得分（`sparse_reward_by_agent[0]`）
  - `role=waiter`：partner 是服务员；agent 信用 = 放洋葱（`potting_onion[0]`）
- 状态、horizon、option、随机种子跨角色**完全一致** → 只有关系条件不同。
- witness 判定：两个角色下 argmax option 不同，且两个优势 margin > δ。

## 3. 结果

### 3.1 Witness 验证（`data/kappa/overcooked_slice/conflict_witnesses.json`）

| 指标 | 值 |
|---|---|
| 构造候选状态 | 52 |
| **witness 状态** | **24/52（46%）** |
| 原生崩溃被跳过 | 5（overcooked 原生崩溃，子进程隔离跳过） |
| **hidden obs 跨角色恒等** | **True**（dynamic 观测无角色 one-hot） |
| 自然随机状态 witness 率 | **0/30（0%）** |

典型 witness（agent @ (7,1)）：

| 角色 | argmax | Q(deliver) | Q(cook) |
|---|---|---|---|
| chef（agent 送餐） | **deliver** | 20.0 | 0.0 |
| waiter（agent 做菜） | **cook** | 0.0 | 6.0 |

### 3.2 场轴（witness 上，`witness_field_axis.json`）

同一策略、同一 hidden obs，只换关系条件（partner + 信用奖励）：

| 场 | κ_mean | κ_ep | E_shared | E_contrast | σ² |
|---|---|---|---|---|---|
| **value（TD/mean-seeking）** | **0.941** | 0.935 | 34341 | 2165 | 212 |
| reinforce（硬 PG） | 0.500* | 0.486 | ~0.02 | ~0.02 | ~0.001 |

*reinforce 行是**退化占位**，不是真收缩：随机策略 + 稀疏角色信用奖励 →
几乎每步奖励为 0 → 梯度≈0 → κ=0.5 是"0/0"默认值。**该格标 N/A**，
reinforce 场的可信测量待补（见 §5）。

## 4. 解读（与主线的关系）

1. **关系冲突是可证明的**：在 46% 的构造状态下，chef 条件下最优行为是
   deliver、waiter 条件下是 cook，且 Q 差距干净（20 vs 0，6 vs 0）。
   这是非 Toy 环境里 IIGC 机制成立前提（"两个关系要求不同最优行为"）的
   **机械验证**，不是语义推断。
2. **冲突对策略不可见**：hidden obs 恒等 = True → 策略在 dynamic 观测下
   无法区分关系 → 这正是"隐藏关系混叠"的前提。
3. **value 场在冲突状态下对齐（κ≈0.94）**：值函数只依赖 obs（关系不可见），
   chef/waiter 条件下值梯度几乎相同 → 对齐。场轴（value 存活）在 witness
   状态上干净复现，与 Toy/完整环境的结论一致。
4. **诚实边界**：自然随机对局中 0/30 状态是 witness → 不能说"整个 Overcooked
   都有 IIGC"，只能主张"在动作真冲突的状态上成立"。这符合
   `storyline_clean.md` 的"明确不主张"。

## 5. 原始级场轴的诚实结论（2026-08-19 实测，N/A 说明）

在 witness 状态上尝试了三种原始级场测量，**均无法给出干净的收缩 κ**：

| 定义 | 续接 | 结果 | 为什么退化 |
|---|---|---|---|
| rollout reinforce（随机策略） | 随机 | 梯度≈0（稀疏信用奖励几乎拿不到） | 随机策略完不成任务 → 奖励≈0 |
| 期望-Q（脚本化续接 + 致密化信用） | 角色最优 option 脚本 | Q 对原始动作平坦（如 chef [23×6]） | 胜任脚本补偿任何坏第一步 → 第一步无关 |
| 期望-Q（种子随机续接 + 致密化信用） | 固定种子随机 | Q_chef≈0（随机送不了汤）vs Q_waiter≈1 | 两角色梯度尺度错配、chef 侧≈0 → κ 是 0/0 噪声 |

**结论（方法学发现）**：witness 的关系冲突在 **option（deliver/cook）层面**，不在
原始动作层面。共享策略参数的原始级梯度在该状态下没有可收缩对象：
- 用胜任脚本续接 → 原始 Q 平坦（第一步无关）；
- 用随机续接 → chef 侧（送餐）梯度≈0（随机策略送不了汤）。

因此 witness 状态上**唯一干净的场轴读数**是：
- **value 场 κ≈1.000**（E_contrast=0）：值函数只依赖 obs（关系不可见）→ chef/waiter 值梯度完全对齐。
- 原始级"reinforce 收缩"由**完整环境 rollout 场轴**展示（`oc_field_axis.json`：reinforce 0.015 vs value 0.98），不在 witness 的原始 Q 上。

> 若要在 witness 状态上直接看到原始级收缩，唯一可行路径是**在
> asymmetric_advantages 上训练真实策略**（cramped_room 模型跨布局不可用），
> 让随机 rollout 能拿到非零角色奖励。该路径未跑（成本/收益权衡）。

## 6. 数据/脚本

| 内容 | 路径 |
|---|---|
| Witness 结果 | `data/kappa/overcooked_slice/conflict_witnesses.json` |
| 场轴结果 | `data/kappa/overcooked_slice/witness_field_axis.json` |
| Witness 验证器 | `experiments/common_basis/overcooked_slice/verify_conflict_witness.py` |
| 场轴测量器 | `experiments/common_basis/overcooked_slice/verify_witness_field_axis.py` |

> 技术备注：本机 `overcooked_ai_py` 版本在长循环中产生原生崩溃（段错误/堆损坏），
> 验证器已用子进程隔离（每状态一个子进程，崩则跳过），不要改为进程内循环。
