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
  `[0.0704, 0.2165]` (`3.08x` spread); K=4: `[0.0513, 0.2095]` (`4.08x`).
  Direction dependence is a simplex-dimension effect, not S3-specific.
- softq interference (1001-point scan): minimum `kappa_mix = 2.5e-6` at
  `alpha = 1.72` (K=2, p=(0.8,0.2)) and `4.0e-4` at `alpha = 1.46`
  (K=3, p=(0.7,0.2,0.1)); the shared-energy minimum and the ratio minimum
  coincide on the same grid point (`alpha_scan_minima` in
  `kway_geometry.json`). The earlier 101-point scan's `7.8e-4`/`1.1e-3` were
  grid artifacts; no separate numerator-cancellation location is resolved.
- Controlled visibility contrast (`visibility_control.json`): one policy,
  identical parameters and evaluation seeds, only the observation protocol
  changes. Masked (partner slot zeroed): expq `0`, reinforce `0.0025`, softq
  `0.0045`, softmaxq `0.0005`, differentiable-baseline awr `0.0091`. Revealed:
  all five fields collapse to `0.4397`--`0.4402`. Five random-init policies of
  the same architecture show the same pattern.

## 2. Overcooked field axis (fresh models, switching-preserving)

Source: `data/kappa/server_tasks/results/oc_field_axis.json`, produced by
`run_field_axis.py` with a shared episode batch per checkpoint, the
differentiable-baseline `awr` field (one weight per step, float64, with a
global stop-gradient advantage shift; the shift scales the field by a positive
constant and leaves `kappa_mix` unchanged), and gradients zero-padded to the
full policy parameter vector.

Definition lock: this `awr` is the sampled-return-to-go differentiable-baseline
field (undiscounted `G_t`, `gamma=1`, learned value head kept in the autograd
graph, no clipping, no normalization); the O4 experiment below uses a different
detached/clipped `awr` arm and the two must not be compared as one field. The
JSON also records the post-shift advantage range, the minimum weight, and the
fraction of float32 weights that would underflow to zero
(`awr_zero_weight_fraction_float32`: 20-22% dynamic, 76-78% static). The
float64 re-run reproduces every score to within `1.5e-7`; the float64
computation therefore matters for exact scale invariance, not for the numbers.

| Mode | Field | kappa_mix (n=3) |
|---|---|---|
| dynamic | reinforce | 0.529 ± 0.034 |
| dynamic | awr | 0.590 ± 0.021 |
| dynamic | value | 0.998 ± 0.000 |
| static | reinforce | 0.515 ± 0.021 |
| static | awr | 0.500 ± 0.000 |
| static | value | 0.568 ± 0.027 |

Interpretation: $\kmix$ alone is ambiguous. On dynamic, the value field's
condition-mean gradients nearly coincide (`0.998`) because the condition
contrast is only ~`0.1%` of its shared mass (`E_contrast/mean_noise`
`4.5-16.7` over three seeds): the field is condition-blind, and its score
reflects a dominant condition-independent component. The differentiable-
baseline advantage field is the only policy-gradient-style field with a
resolvable contrast (`E_contrast/mean_noise` `35-176`) at `0.590`; the plain
policy gradient's contrast sits at the estimation-noise floor
(`0.85-1.32`, `0.529`), so its near-orthogonality is an upper bound rather
than a measured cancellation. The value-policy `1.9x` framing is retired;
$\kmix$ is read jointly with the contrast-to-noise ratio and the contrast
share. Static: value contrast/noise is `443-9701` with contrast share
`0.65-0.83` (`0.568`), so the static value field does carry a resolvable
condition-dependent component; the advantage field is at `0.500` with
contrast/noise `6.7-51`.

## 3. Overcooked switch kappa (start-partner protocol)

Source: `data/kappa/server_tasks/results/oc_switch_kappa.json`; re-run
2026-09-23 with `run_switch_kappa.py --force` from the archived checkpoints
(`data/models_overcooked/`), which attaches the `_metadata` block and
reproduces the stored `kappa_mix` values to within `3e-5`. Entries now also
carry `E_mixture`, `E_total`, `kappa_mean_surrogate`, `mean_noise_energy`,
`n_per_condition`, and `weights`.

| Group | kappa_mix (n) |
|---|---|
| static | 0.533 ± 0.068 (8) |
| dynamic | 0.502 ± 0.196 (8) |
| mem m4 | 0.498 ± 0.180 (3) |
| mem m8 | 0.505 ± 0.149 (3) |
| mem m16 | 0.500 ± 0.079 (3) |

Interpretation: under the start-partner protocol the plain-gradient condition
contrast is at the estimation-noise floor in every group without a visible
condition (dynamic `0.17-2.02`, mem m4/m8/m16 `0.62-2.13`), so those
near-orthogonality statements are upper bounds; the static group (visible
partner) has a resolvable contrast in most seeds (`3.1-461`, mean `220`). The
previously reported "dynamic kappa collapses to 0.010" was an episode-noise
statement (`sigma2` dominance), not structural cancellation. Re-run
2026-09-23 from the archived checkpoints reproduces every stored `kappa_mix`
within `3e-5` and attaches the `_metadata` block.

## 4. 510K field axis (paired protocol, ppo_reveal checkpoints)

Source: `data/kappa/server_tasks/results/510k_field_axis.json`, keys
`s<seed>_paired`, both fields computed from the same episode batch with
mask-respecting rollouts (`run_510k_field_axis.py`; action masks are passed
to `get_distribution` so evaluated actions are always legal).

| p | reinforce kappa_mix (n=6) | value kappa_mix (n=6) |
|---|---|---|
| 0.00 | 0.646 ± 0.119 | 0.982 ± 0.005 |
| 0.50 | 0.612 ± 0.223 | 0.985 ± 0.003 |
| 1.00 | 0.667 ± 0.184 | 0.962 ± 0.011 |

Interpretation: neither field has a condition contrast that is both
resolvable and substantial. The value contrast is `1-5%` of its shared mass
at `0.7-3.1x` the mean-estimation noise floor (per-seed ratios; all six seeds
above the floor only at `p=1`, where the teammate is visible), and the
policy-gradient contrast is `0.25-2.0x` the floor (mean-of-ratios
`0.8/0.9/0.9` for reinforce and `1.2/1.0/2.1` for value at `p=0/0.5/1`). The
stable value score tracks a dominant shared component rather than
hidden-condition dependence, and there is no evidence for a hiddenness trend
in `p`. The value-policy `1.5x` framing is retired (the fields have different
supports in the padded parameter vector).

## 5. Supervised cross-setting check: DICES-350 (three seeds)

Source: `data/kappa/dices350/audit.json`,
`experiments/common_basis/supervised/run_dices_group_audit.py`. The dataset is
cached under `data/external/dices/` (not committed). Setup: 350 adversarial
conversations, all 123 raters rate every item, binary perceived-harm label
(Yes vs No, Unsure dropped), split by `item_id` (245/105), TF-IDF plus shared
logistic model.

| Axis (groups) | `kappa_mix` | `kappa_ep` | `E_contrast` | `sigma2` | shared avg/worst | conditional avg/worst |
|---|---|---|---|---|---|---|
| race (3) | 0.156 ± 0.027 | 0.026 ± 0.004 | 0.0019 | 0.0114 | 0.656 / 0.609 | 0.654 / 0.607 |
| age (3) | 0.002 ± 0.000 | 0.000 ± 0.000 | 0.0002 | 0.0058 | 0.640 / 0.623 | 0.638 / 0.624 |
| gender (2) | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.0005 | 0.0035 | 0.641 / 0.623 | 0.640 / 0.625 |

Gradient definition: BCE data gradient at the shared model; the
condition-independent L2 term is excluded (it shifts every condition mean by
the same vector and leaves `E_contrast`/`sigma2` unchanged). Condition weights
are the empirical rater frequencies (race seed41: `[0.358, 0.336, 0.306]`).

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
| Overcooked dynamic: value `kappa ~ 0.98` | Structural `~0.998`, but the contrast share is `~0.1%`: the score is condition-blind, not aligned (see §2) |
| Overcooked awr `0.516/0.573` (detached, standardized advantage) | Corrected to the differentiable-baseline field on a shared episode batch: `0.500/0.590` |
| Overcooked awr `0.498/0.854` (first differentiable-baseline rerun) | Shape-broadcast bug (`G` `[T]` minus `V` `[T,1]` produced `[T,T]` weights); fixed to one weight per step with a global stop-gradient shift, giving `0.500/0.590` |
| Overcooked awr `0.500/0.590` (float32 weights) | Re-run with float64 weights and underflow diagnostics: scores unchanged (max diff `1.5e-7`); 20-22% of dynamic and 76-78% of static float32 weights underflowed to exactly zero |
| 510K: reinforce `kappa_ep ~ 0.015` vs value `~0.44` | Paired structural (mask-respecting): `0.61-0.67` vs `0.96-0.99`; the value score is shared-component-dominated (contrast share `1-5%`) |
| 510K rolling out without action masks | Measurement bug: `get_distribution` was called without `action_masks`, so ~90% of executed actions were silently replaced by random legal plays. Masked-protocol rerun gives reinforce `0.61-0.67` (large seed spread, contrast at noise floor) vs value `0.96-0.99`; the value-policy ratio framing is retired |
| "30x/65x field separation" | Retired: the `1.9x`/`1.5x` ratios compare fields with different supports in the padded parameter vector and do not license an alignment ordering; the `30-65x` came from noise denominators |
| "High `kappa_mix` = condition signal survives" | Retired: `kappa_mix` does not penalize a dominant condition-independent component; read jointly with contrast/noise and the contrast share |
| S3-specific direction/interference | K-way simplex geometry |
| softq survives and then devours itself | Retired; entropy term pushes toward uniform under minimization |
| softq "numerator cancellation point differs from the ratio minimum" | Retired: on the 1001-point scans the two minima coincide on the same grid point (`alpha_scan_minima`); the old `7.8e-4`/`1.1e-3` values were 101-point grid artifacts |
| AWR "interior maximum / non-monotone in tau" | Corrected: monotone decreasing in `tau`, supremum `1/2` as `tau -> 0` (regression test in `tests/test_closed_forms.py`) |
| DICES: uniform condition weights, BCE gradient only | Empirical rater frequencies; the condition-independent L2 term is excluded by definition (it cancels from `E_contrast`/`sigma2`, so the no-intervention verdict is unchanged); race `kappa_mix` `0.116` -> `0.156` |

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
