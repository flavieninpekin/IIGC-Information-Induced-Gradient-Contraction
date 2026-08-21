# paper/resources — 新论文素材（IIGC 机制篇）

> 目的：把分散在 `notes/`、`data/`、`experiments/` 的结论与数字收编成论文
> 可直接使用的素材。**所有数字都带出处**（data/xxx.json 或 notes/xxx.md），
> 写稿时直接引用，不改数字。
> 最后更新：2026-08-19。

## 目录

| 文件 | 内容 | 写稿时的用途 |
|---|---|---|
| `narrative.md` | 论文骨架：候选标题、摘要草稿、贡献、章节结构、每节要填的证据 | 搭 Outline / 定叙事线 |
| `storyline_clean.md` | 收窄后的干净主线（替代 narrative 的主线） | 定稿叙事 |
| `evidence.md` | 三环境核心证据表（Toy 精确/采样、Overcooked、510K 场轴）+ 修正/阴性结果 | 直接抄进 Results/Tables |
| `overcooked_slice.md` | Overcooked 受控切片：witness 验证 + 场轴（关系冲突的机械证明） | 机制/方法学证据 |
| `theory.md` | 框架定义（保留—特异权衡）、奇偶分解引理、闭式判决（T1-T3）、开放项 | 写 Theory/Propositions |
| `data_index.md` | 全部数据文件位置一览 | 复核引用、找原始数字 |

## 数据源速查（原始）

| 证据 | 文件 |
|---|---|
| Toy 闭式 + 采样（T1-T3） | `data/kappa/toy_fields/theory_toy.json` |
| Toy 旧测量（κ+能量） | `data/kappa/toy_fields/results.json` |
| Overcooked 分解（切换保持） | `data/kappa/server_tasks/results/oc_switch_kappa.json` |
| Overcooked 场轴 | `data/kappa/server_tasks/results/oc_field_axis.json` |
| Overcooked 标准 grid | `data/kappa/server_tasks/results/oc_decomp_b1*.json`、`oc_baselines_b1*.json`、`oc_n_protocol.json` |
| Overcooked 记忆干预 | `data/kappa/server_tasks/results/oc_switch_kappa.json`（mem_* 键） |
| 510K 场轴 | `data/kappa/server_tasks/results/510k_field_axis.json` |
| 510K reveal 旧网格 | `data/kappa/510k_reveal/results.json` |
| 上一篇跨算法 κ | `AAAI2027-510k-clear/data/kappa_summary.json` |
| 新模型 checkpoint | `data/models_overcooked/`（16 v3 + 9 mem）、`data/models_reveal/`（ppo_reveal_*） |

## 写作状态

- [ ] Outline / narrative 定稿
- [ ] Evidence 表核对（每格给出处）
- [ ] Theory 命题编号 + 证明
- [ ] 图（三环境场轴对比 / κ(p) / 训练曲线）
