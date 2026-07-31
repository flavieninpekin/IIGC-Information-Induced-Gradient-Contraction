# Folder Guideline — IIGC（新论文项目）

> 本文件是项目层级导航。忘了某一层是什么就查这里。
> 项目基于上一篇文章（`AAAI2027-510k-clear`，参考目录，不纳入本项目）
> 的中间内容与结果，重测 κ 并解释 PG/value 反转的机制。

## 顶层

```
IIGC/
├── folder_guideline.md      ← 本文件：层级导航
├── README.md                ← 项目简介
├── pyproject.toml           ← `pip install -e .` 安装 iigc 包
├── requirements.txt         ← 依赖
├── .gitignore               ← 排除 AAAI2027-510k-clear*（参考材料）
├── notes/                   ← 研究笔记 / 讨论文档
├── src/                     ← 核心 Python 包（import 为 `iigc`）
├── experiments/             ← 实验（按研究问题分层）
├── data/                    ← 运行时产物（模型/κ结果/日志/曲线）
└── paper/                   ← 新论文写作（当前为空）
```

## src/iigc/ — 核心代码包

```
src/iigc/
├── __init__.py
├── env/                     ← 环境定义（源自 project3/510k-env/env，import 改为相对路径）
│   ├── __init__.py
│   ├── card.py  game.py  patterns.py  scorer.py  obs_utils.py   ← 510K 规则引擎
│   ├── env_510k.py          ← 510K Gym 环境（SINGLE/STATIC/DYNAMIC/OBVIOUS）
│   ├── dqn_wrapper.py       ← 动作掩码 Q 网络 + TD-loss 梯度/kappa
│   ├── discrete_sac.py      ← 离散 SAC（Actor + Critic×2），actor 梯度 kappa
│   ├── features.py          ← 7 维行为特征（path_integral 需要）
│   ├── toy_env.py           ← Toy Matching（HIDDEN/REVEALED）
│   ├── partner_env.py       ← Partner 环境
│   ├── mappo_env.py  mappo_policy.py   ← MAPPO（探索性，未用）
│   └── bots/random_bot.py
├── algos/                   ← 训练脚本（源自 project3/510k-env/train_510k_*.py）
│   ├── __init__.py
│   ├── a2c.py  dqn.py  sac.py  reinforce.py  reinforce_sp.py
└── metrics/                 ← 指标（源自 project3/510k-env/ 对应脚本）
    ├── __init__.py
    ├── kappa_ppo.py         ← PPO κ（需 data/models_selfplay/ 下的模型）
    ├── continuous_reveal.py ← 连续 reveal 干预
    └── path_integral.py     ← 行为路径积分
```

## experiments/ — 实验（按研究问题分层）

```
experiments/
└── common_basis/            ← 新论文核心：共同测量基重测 κ
    ├── design/
    │   └── why_value_reverses.md   ← 反转机制设计（源自 project3/paper）
    ├── sac_actor_critic/    ← 实验1：SAC actor/critic 分离 κ
    ├── interpolation/       ← 实验2：AWR / 温度 / DPG 插值谱
    └── replay_mix/          ← 实验3：DQN replay 关系混合消融
```

## 约定

- **import**：包内一律 `from iigc.env.xxx import ...`（已批量修复；原脚本的
  `sys.path` hack 已移除）。运行前先 `pip install -e .`。
- **数据路径**：脚本默认写 `data/models/<algo>`、`data/kappa/<algo>`、
  `data/logs/<algo>`（已从旧布局 `models_510k_*` 迁移）。
- **未复制的内容**（需要时再从 project3 引入）：
  - IRL 管线（`irl.py`、`transfer.py`、`compare_irl_*`）——废弃方向
  - Overcooked（`overcooked_adapt/`）——如需 stress-test 再加
  - 已训练模型（在 project3/models_* 与参考目录中，可手动拷入 `data/models/` 复用）
- **参考材料**：`AAAI2027-510k-clear/` 与 `AAAI2027-510k-clear.zip` 已被
  gitignore，仅供查阅，不属于本项目。
