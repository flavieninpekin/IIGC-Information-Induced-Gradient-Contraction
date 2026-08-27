# O4 预注册：关系自适应任务的性能后果实验（S2-P1）

> **预注册性质**：以下预测在跑实验之前固定。命中 → 权衡获得行为层证据（O4 关闭）；
> 偏离 → T4 按数据重述，论文走诚实边界分支（`theory_program.md` Part E）。
> 创建：2026-08-26。脚本：`experiments/common_basis/performance/run_toy_adaptive.py`
> （待建）。数据：`data/kappa/o4_adaptive/`。

---

## 1. 动机与定位

- storyline_clean.md 开放项 O4：权衡目前停在信息层（"无法同时保留信号与特异性"），
  性能后果没有实验。
- 约束（设计推演结论）：**平稳隐藏关系 + 无记忆策略下，一切能学的场最终收敛到
  同一个"平均最优固定策略"，终性能无差异**。因此性能差异只能来自：
  (i) 学习速度（样本效率）；(ii) 关系非平稳（episode 内翻转、需跟踪）；
  (iii) 部署时关系可见化后的再适应速度。
- 本实验同时覆盖 (i)+(ii)，用 (iii) 作参照臂。

## 2. 环境：AdaptiveHiddenMatchingEnv

在 `HiddenMatchingEnv` 基础上加：

- **episode 内翻转**：`n_steps=40`，step 20 处 partner 翻转（B↔C）。
  static 臂不翻转（对照）。
- **观测**：常值偏置 + 最近 K=8 步 (action_onehot, reward) 拼接
  （dim = 1 + 8×3 = 25）。关系本身仍不在观测里——但**可从奖励历史推断**
  （reward=+1 ⟹ 当前匹配）。这保留了 IIGC 张力：信息存在但不直接给出，
  各场差异 = 利用可推断信息的效率。
- **revealed 变体**：观测追加当前 partner 指示位（参照上界）。
- 解析量：oracle return = n_steps（全程匹配）；chance = 0；
  最佳无记忆策略 = 0（所以任何正回报都证明跟踪成功）。

## 3. 臂（共同基训练）

同一 MLP 主干 + 同一 replay 缓冲（当前策略随机 rollout）、同步数、同种子。
只换目标函数（对齐 fields.py 定义）：

| 臂 | 目标 | 备注 |
|---|---|---|
| reinforce | ∇logπ(a)·G（MC 回报） | 奇场参照 |
| awr | ∇logπ(a)·exp((G−V)/τ)（V 头可导，对齐 fields.py） | 混合场 |
| softq | ∇Σπ(α logπ − Q)，Q 由 TD(0) 在同一缓冲拟合 | 软场 |
| td | ε-greedy argmax Q，同一 Q 网 TD 拟合 | value 场 |

网格：4 场 × {static-hidden, switch-hidden} × 8 seeds（41–48）
+ {static, switch}-revealed 参照（4 场 × 8 seeds）。
预算：每 run 150k 环境步；Toy 便宜，总量 ~10M 步，CPU 可跑。

## 4. 指标

1. **final return**（eval 500 eps，训练后）与学习曲线（每 10k 步快照）
2. **oracle gap** = 40 − return
3. **翻转恢复率**：E[r_t | t∈(20,28]] − chance（跟踪速度的直接读数）
4. **机制耦合**：训练期每 10k 步在干净基上测 κ（set_partner 强制初始 partner、
   按 episode 初始 partner 分组做 E_shared/E_contrast/σ² 分解）——检验
   "幸存梯度能量 ↔ 行为进展"的场内相关

## 5. 预注册预测

**P1（static-hidden 排序）**：td ≈ softq > awr > reinforce（final return）。
理由：奇场的关系条件梯度在共同基上相消（命题 A），只能靠高方差的历史特征
路径缓慢学会；软/value 场的 Q 结构可直接编码"历史→推断 partner→价值"。

**P2（switch-hidden 排序同 P1，且差距拉大）**：翻转后需抑制旧相位习惯，
软场的 Q 重估（TD 自举跨翻转边界传播）比 MC 回报的混合信用分配更快。
oracle gap 排序反之。

**P3（revealed 参照）**：四场都学好（gap→小），场间排序消失或反转
（信息给足时字段轴退化，T5 在行为层的对应）。

**P4（机制耦合）**：场内 final return 与时间积分幸存梯度能量
∫E_shared dt 正相关；reinforce 臂 κ≈0 全程但行为未必全败（历史特征使
取消不完全）——预期出现"κ 干净基读数低 vs 行为可学"的部分解耦，
若解耦显著，须在论文里明确 κ 的适用边界（测量基不含策略实际使用的信息）。

**证伪判据**（任一成立则 O4 行为层主张收缩）：
- F1：switch-hidden 下 reinforce 进 final return 前二（排序违反 P2）；
- F2：所有场 final return 都 ≈ chance（任务对记忆策略不可学——环境问题，修环境重跑一次；再失败则放弃该环境）；
- F3：场间 final return 差异 < 种子内噪声（无行为后果可谈 → 诚实边界分支）。

## 6. 已知风险

- 历史特征部分解除"隐藏"性：κ 干净基与行为的解耦是特性不是 bug，但要报告（P4）。
- softq 臂的 α 固定 0.2（不做自动温度——避免引入额外自由度）。
- td 臂 ε 从 0.2 线性退到 0.05；评估用随机策略（避免 argmax 协议伪影，
  见 `rollout_protocol_artifact.md`）。

## 7. 判决流程

跑完 → `notes/o4_performance_verdict.md`：逐条 P1-P4 标记 命中/偏离/F1-F3，
更新 `paper/resources/evidence.md` 与 `storyline_clean.md` O4 状态，
决定 Part E 决策树走哪一枝。

---

## 8. ���׶�׷����ƣ�2026-08-26��һ�׶ν����Ԥע�ᣩ

> һ�׶Σ���2-7�������꣬���������޶���
> (a) P4 ǿ������֡���td �ɾ��� ��=0.05����ͣ�ȴ��Ϊ��ã�36.5����
>     ԭ�򣺱��������Ϊ������"��λ����"���ɽ�����ʷ�������� partner ��ǩ��������
>     �� �� �Ĳ�������"��ʼ partner ������"����**���������� �� �������������**��
> (b) P3 δ���С���revealed ������ reinforce(17.5)/awr(8.5) Զδ�� oracle��
>     �ֶβ�����ȫ��Ϣ����Ȼ���ڣ�����Ч�ʻ��ƣ���Ԥ���ڲ�������
>
> һ�׶��о���P1 ? P2 ? P3 ?��Ԥ���ڣ�P4 ǿ�� ?��������෴�Ҹ�����Ϣ������
> �� F1-F3 ������"ä��ʱ�к�"��Ҫ������ä����**����**��������ܼ��飺

### 8.1 ��������Ӧʵ�飨identity-critical ��

- ѵ����static_hidden��һ�׶����ò��䣬nets �����ã�
- �����л��� static_revealed��partner ÿ���ɼ����޷�ת��΢�� 30k ����
  ÿ 6k ������һ�� revealed-static �ر� �� ��Ӧ����
- �����������Ӧ�ٶȣ����� return��32 �Ĳ������� AUC

### 8.2 Ԥע��Ԥ�⣨��ǰ�̶���

- **PA����ֽ���� / ��ŵ���ۣ�**��softq �� static_hidden �����Ҵ��ݶ�����
  ���߳�ŵ������ B���������� �� ����ln(p/q)���������⣩�������������
  reinforce ѧ���ٵ���ŵҲ�� �� **reinforce ����Ӧ����ǰ�ο��� softq/td**
  ��������һ�����ڴ����ڳ���������������"ż���Ҵ�"��һ������Ϊ����������
- **PB���յ��޲�ࣩ**��30k ����������ӽ� oracle��revealed �����ѧ����
  ����ֻ���ٶȲ����յ㡣
- ֤α���� softq/td ����Ӧȫ�̲����� reinforce���޳�ŵ���ۼ��󣩣�
  ��"ä����Ϊ����"�ڱ��������в����ڣ�O4 �߳�ʵ�߽��֧��

### 8.3 ʵ��

`run_toy_adaptive.py` �� `--deploy` �׶Σ�train_config ��Ƕ����
static_hidden ��ѵ����� 30k �� static_revealed ΢�� + ���߼�¼��
�� static_hidden �����������ܣ�32 �� run��Լ 15 ���ӣ���
