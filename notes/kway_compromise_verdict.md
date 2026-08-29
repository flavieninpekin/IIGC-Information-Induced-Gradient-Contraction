# K-way geometry and compromise verdict

> Date: 2026-08-28. This note records the first reproducible K-way control
> experiment and the shared-versus-conditional training comparison.

## 1. K-way geometry

Script: `experiments/common_basis/toy/verify_kway_geometry.py`

The experiment uses the exact expected logit gradients of a K-way matching
bandit. The expq closed form agrees with autograd to `2.22e-16`. Uniform
mixtures cancel the expq field for K=2, 3, and 4 (all below `1e-32`). This is a
K-way normalization result, not an S3-specific result.

At fixed L1 distance `0.45` from the uniform mixture, sampled directions give:

| K | uniform-reference range | mixture-reference range |
|---|---:|---:|
| 3 | `0.0748` to `0.1903` (2.54x) | `0.0704` to `0.2165` (3.08x) |
| 4 | `0.0539` to `0.1714` (3.18x) | `0.0513` to `0.2095` (4.08x) |

The first direction-dependent case is therefore K=3 because the mixture
simplex has dimension two. The result does not establish a non-abelian effect.

Softq interference is also not S3-only. With K=2 and condition mixture
`(0.8, 0.2)`, the mixture-reference kappa reaches `0.00078` near
`alpha=1.78`. With K=3 and mixture `(0.7, 0.2, 0.1)`, it reaches `0.00111`
near `alpha=1.41`.

The uniform-reference score is retained for compatibility with earlier files,
but it is not bounded by one for non-uniform mixtures. The mixture-reference
score is the bounded retention fraction and should be the primary metric.

## 2. Shared versus conditional capacity

Script: `experiments/common_basis/toy/run_shared_vs_conditional.py`

This is an exact full-batch training comparison on the same three-condition
matching objective. A shared policy has one logit vector and a conditional
policy has one logit vector per condition. Each seed uses the same base initial
logits; no rollout noise is involved.

With a uniform mixture and the linear objective (`alpha=0`), the shared policy
has zero mixed gradient and remains at average reward `0.667`, while the
conditional policy reaches average reward `1.986` and worst-condition reward
`1.985`. With mild and strong asymmetric mixtures, the shared policy improves
the majority conditions but can leave the worst condition near zero; the
conditional policy keeps high reward across all conditions.

The result supports a compromise interpretation:

> Kappa measures how much condition-conditioned gradient survives aggregation;
> it does not by itself measure task performance. If condition-specific
> behavior matters, the intervention is conditional information or capacity,
> not an arbitrary optimizer change.

## 3. Boundaries and next test

- S3 should be described as the minimal three-condition example, not as proof
  of non-commutative-group-specific behavior.
- The softq alpha minimum is a local gradient-geometry result. Its training
  impact must be measured against the mixed objective and per-condition regret;
  a low kappa alone is not evidence of training failure.
- The next decisive experiment is one real structured-heterogeneity dataset
  with a shared baseline and a condition-aware routing/adapter intervention.
  If the intervention improves held-out worst-condition performance while the
  audit predicts when it is needed, the work can claim an operational method.
  Otherwise the project should remain a theory and measurement paper.
