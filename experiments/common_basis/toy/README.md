# Toy forced-assignment kappa experiments

HiddenMatchingEnv.set_partner() forces ONE relation per rollout — the cleanest
relation-conditioned measurement basis.

## 1. Field kappa (`run_toy_fields.py`)

Same policy, same data, only the objective changes (reinforce/awr/softq/expq),
with the analytic true Q. REVEALED (soft scaled policy) vs HIDDEN (random init).

| field | REVEALED | HIDDEN |
|-------|----------|--------|
| reinforce | 0.438 | **0.000** +/- 0.000 |
| awr       | 0.438 | **0.561** +/- 0.020 |
| softq     | 0.438 | 0.068  +/- 0.044 |
| expq      | 0.438 | **0.000** +/- 0.000 |

- IIGC reproduced cleanly on a common basis: HIDDEN cancels hard fields to
  kappa = 0, REVEALED gives partial alignment (0.44). Not a definitional
  artifact (same policy/data/params, only the assignment changes).
- Field axis appears under HIDDEN: AWR (advantage-weighted) resists
  cancellation; hard fields cancel.

## 2. Energy-variance decomposition (`verify_energy_decomp.py`)

Verifies kappa is EXACTLY the shared-energy fraction, and that gradient
energy / variance decompose orthogonally into measurable components.

| identity | HIDDEN | REVEALED |
|----------|--------|----------|
| Pythagorean err (E_means = E_shared + E_contrast) | 5.7e-14 | 1.5e-8 |
| Law-of-total-var err (Var_total = Var_between + sigma^2) | 1.8e-5 | 3.6e-7 |
| kappa_mean pred vs meas | 0.000 = 0.000 | 0.360 = 0.360 |

- E_shared = 0 under HIDDEN (perfect cancellation -> kappa 0); > 0 under
  REVEALED (retained energy).
- All components directly measurable (condition means + within-relation
  variances); kappa is not a black box.

Run:
  `python experiments/common_basis/toy/run_toy_fields.py`
  `python experiments/common_basis/toy/verify_energy_decomp.py`
Results: `data/kappa/toy_fields/`, `data/kappa/energy_decomp/`
