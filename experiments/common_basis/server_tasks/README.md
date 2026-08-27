# server_tasks — GPU 重训与场轴测量的正式脚本（2026-08-26 从临时目录迁入）

> 这些脚本产出了 `data/kappa/server_tasks/results/` 与 `data/models_overcooked/`、
> `data/models_reveal/` 下的全部数据（2026-08-19 那轮 GPU 重训 + 测量）。
> 原先散落在 `C:\Users\Flavi\AppData\Local\Temp\opencode\`，现已入库。
> 路径假设：从仓库根目录运行（`ROOT = repo root`），依赖 `pip install -e .`。

## 测量类

| 脚本 | 产出 | 说明 |
|---|---|---|
| `run_switch_kappa.py` | `oc_switch_kappa.json` | Overcooked 切换保持协议分解 + memory 干预（mem_* 键） |
| `run_field_axis.py` | `oc_field_axis.json` | Overcooked reinforce/awr/value 场 κ（切换保持） |
| `run_510k_field_axis.py` | `510k_field_axis.json` | 510K reinforce/value 场 κ（team 条件方差分解） |
| `verify_dynamic_learns.py` | （stdout 记录） | dynamic 自由 rollout reward~100-120，排除"学不动"解释 |
| `make_field_axis_figure.py` | `paper/figures/field_axis_comparison.*` | 三环境场轴对比图 |

## 批量训练 / 运维类

| 脚本 | 说明 |
|---|---|
| `batch_train_v3.py` | Overcooked v3 static/dynamic × seeds 41-48（16 模型重训，替换损坏 checkpoint） |
| `batch_train_mem.py` | overcooked_mem_dynamic m4/m8/m16 × s41-43（帧记忆干预，结果阴性） |
| `batch_train_reveal.py` | 510K reveal p∈{0.50,1.00} 缺失 seeds 重训 |
| `repair_checkpoints.py` | SB3 checkpoint optimizer 损坏修复 |
| `run_engine_tasks.py` / `run_ext_tasks.py` / `run_full_grid.py` | 服务器任务引擎的本地驱动/扩展任务 |

## 相关

- 协议要点（切换保持 vs forced-partner 伪影）：`notes/theory_program.md`、`paper/resources/theory.md` §6
- Toy 理论计算器：`../toy/verify_theory_toy.py`（08-22 重建版，π-加权协议）
  与 `../toy/verify_theory_toy_v1_cefit.py`（08-19 原版 CE-fit 轨，
  仅用于复现 `theory_toy.json`；两轨差异见 `notes/toy_field_axis_theory.md` §6.3）
