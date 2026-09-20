# Canonical Metric Results (mixture-v1)

Status: refreshed measurements under the canonical structural metric

```text
kappa_mix = ||sum_i p_i mu_i||^2 / sum_i p_i ||mu_i||^2.
```

All numbers below are computed from condition means. Episode-noise readouts
(`kappa_ep`) are deliberately excluded here; see the provenance files for the
full decomposition.

## 1. Toy mirror bandit (exact)

- expq / reinforce expected fields: `E_shared = 0`, `kappa_mix = 0` exactly
  (`data/kappa/toy_fields/o1_closed_forms.json`, `kway_geometry.json`).
- softq closed form `kappa_mix = A^2 / (A^2 + 4 r^2)` verified against autograd
  to `5.0e-16` and against MC anchors.
- K-way uniform mixtures cancel for `K = 2, 3, 4` (all below `1.8e-32`).
- Direction sweep at fixed L1 distance `0.45` (K=3): `kappa_mix` in
  `[0.0706, 0.2190]` (`3.10x` spread); K=4: `[0.0513, 0.2095]` (`4.08x`).
  Direction dependence is a simplex-dimension effect, not S3-specific.
- softq interference: minimum `kappa_mix` at `alpha = 1.78` (K=2, p=(0.8,0.2))
  and `alpha = 1.41` (K=3, p=(0.7,0.2,0.1)), confirming that the numerator
  channel cancellation and the ratio minimum are distinct phenomena.
- Controlled visibility contrast (`visibility_control.json`): one policy,
  identical parameters and evaluation seeds, only the observation protocol
  changes. Masked (partner slot zeroed): expq `0`, reinforce `0.0025`, softq
  `0.0045`, softmaxq `0.0005`, differentiable-baseline awr `0.0091`. Revealed:
  all five fields collapse to `0.4397`--`0.4402`. Five random-init policies of
  the same architecture show the same pattern.

## 2. Overcooked field axis (fresh models, switching-preserving)

Source: `data/kappa/server_tasks/results/oc_field_axis.json`, produced by
`run_field_axis.py` with a shared episode batch per checkpoint, the
differentiable-baseline `awr` field (one weight per step, with a global
stop-gradient advantage shift for float32 stability; the shift scales the
field by a positive constant and leaves `kappa_mix` unchanged), and gradients
zero-padded to the full policy parameter vector.

| Mode | Field | kappa_mix (n=3) |
|---|---|---|
| dynamic | reinforce | 0.529 ± 0.034 |
| dynamic | awr | 0.590 ± 0.021 |
| dynamic | value | **0.998 ± 0.000** |
| static | reinforce | 0.515 ± 0.021 |
| static | awr | 0.500 ± 0.000 |
| static | value | 0.568 ± 0.027 |

Interpretation: the value field is strongly aligned with the hidden condition
on dynamic (its condition-mean gradients nearly coincide), while the
differentiable-baseline advantage field, though the least cancelled among
policy-gradient-style fields (`0.590`), stays far below it and only modestly
above the plain policy gradient (`0.529`). The value--policy separation is
about `1.9x`, not the `~65x` implied by comparing noise-dominated `kappa_ep`
readouts. Static shows only a weak separation; the advantage field is at
`0.500` there.

## 3. Overcooked switch kappa (start-partner protocol)

Source: `data/kappa/server_tasks/results/oc_switch_kappa.json`, backfilled.

| Group | kappa_mix (n) |
|---|---|
| static | 0.533 ± 0.068 (8) |
| dynamic | 0.502 ± 0.196 (8) |
| mem m4 | 0.498 ± 0.180 (3) |
| mem m8 | 0.505 ± 0.149 (3) |
| mem m16 | 0.500 ± 0.079 (3) |

Interpretation: under the structural metric the start-partner (chef vs waiter)
condition means are near-orthogonal in every group. The previously reported
"dynamic kappa collapses to 0.010" was an episode-noise statement
(`sigma2` dominance), not structural cancellation.

## 4. 510K field axis (paired protocol, ppo_reveal checkpoints)

Source: `data/kappa/server_tasks/results/510k_field_axis.json`, keys
`s<seed>_paired`, both fields computed from the same episode batch
(`run_510k_field_axis.py`).

| p | reinforce kappa_mix (n=6) | value kappa_mix (n=6) |
|---|---|---|
| 0.00 | 0.514 ± 0.099 | **0.980 ± 0.011** |
| 0.50 | 0.514 ± 0.084 | **0.974 ± 0.018** |
| 1.00 | 0.513 ± 0.092 | **0.948 ± 0.017** |

Interpretation: the value field is aligned, the per-step reward-weighted
policy gradient is near-orthogonal, and the effect is stable across visibility
levels. There is no evidence for a hiddenness trend in `p`; the small
monotone decrease of the value score with `p` should not be over-read.

## 5. Supervised cross-setting check: DICES-350 (three seeds)

Source: `data/kappa/dices350/audit.json`,
`experiments/common_basis/supervised/run_dices_group_audit.py`. The dataset is
cached under `data/external/dices/` (not committed). Setup: 350 adversarial
conversations, all 123 raters rate every item, binary perceived-harm label
(Yes vs No, Unsure dropped), split by `item_id` (245/105), TF-IDF plus shared
logistic model.

| Axis (groups) | `kappa_mix` | `kappa_ep` | `E_contrast` | `sigma2` | shared avg/worst | conditional avg/worst |
|---|---|---|---|---|---|---|
| race (3) | 0.116 ± 0.025 | 0.019 ± 0.004 | 0.0020 | 0.0113 | 0.656 / 0.609 | 0.654 / 0.607 |
| age (3) | 0.033 ± 0.005 | 0.001 ± 0.000 | 0.0003 | 0.0064 | 0.640 / 0.623 | 0.638 / 0.624 |
| gender (2) | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.0005 | 0.0035 | 0.641 / 0.623 | 0.640 / 0.625 |

Interpretation: the race axis carries the strongest condition signal, but its
contrast energy is dominated by within-group item noise
(`E_contrast / sigma2 ~ 0.17`) and conditional models do not improve held-out
average or worst-group accuracy. This is the "no-intervention" side of the
audit, complementing the bandit capacity experiment where contrast dominates
noise and conditional capacity raises the worst condition from 0.003 to 1.97.
Caveats: perceived harm is not objective harm; linear TF-IDF model; no
group-DRO or reweighting baseline; single dataset.

## 6. What Changed Relative to the Old Tables

| Old statement | Canonical reading |
|---|---|
| Toy: reinforce/expt cancellation is exact | Unchanged (exact, mirror model) |
| Toy: hidden/revealed contrast `0` vs `0.41` | Re-measured under one policy and seed schedule (`0.0025` vs `0.4397`--`0.4402`); the old contrast mixed a random hidden policy with a trained revealed policy |
| Overcooked dynamic: reinforce `kappa ~ 0.015` (dead) | `kappa_ep` noise readout; structural `kappa_mix ~ 0.53` |
| Overcooked dynamic: value `kappa ~ 0.98` | Unchanged (structural `~0.998`) |
| Overcooked awr `0.516/0.573` (detached, standardized advantage) | Corrected to the differentiable-baseline field on a shared episode batch: `0.500/0.590` |
| Overcooked awr `0.498/0.854` (first differentiable-baseline rerun) | Shape-broadcast bug (`G` `[T]` minus `V` `[T,1]` produced `[T,T]` weights); fixed to one weight per step with a global stop-gradient shift, giving `0.500/0.590` |
| 510K: reinforce `kappa_ep ~ 0.015` vs value `~0.44` | Paired structural: `0.51` vs `0.95-0.98` |
| "30x/65x field separation" | About `1.9x` under `kappa_mix`; the rest was noise denominator |
| S3-specific direction/interference | K-way simplex geometry |
| softq survives and then devours itself | Retired; entropy term pushes toward uniform under minimization |

## Reproduce

```text
python experiments/common_basis/toy/verify_closed_forms_o1.py
python experiments/common_basis/toy/verify_kway_geometry.py
python experiments/common_basis/toy/verify_s3_survival.py
python experiments/common_basis/toy/verify_visibility_control.py
python experiments/common_basis/server_tasks/run_field_axis.py
python experiments/common_basis/server_tasks/run_510k_field_axis.py
python experiments/common_basis/supervised/run_dices_group_audit.py
```

The DICES-350 CSV is downloaded from the public dataset repository into
`data/external/dices/` (gitignored) and is not redistributed here.
