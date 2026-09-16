# Corrections to the Historical Theory Notes

Status: canonical correction log for claims that appear in
`notes/paper1_propositions.md`, `notes/paper1_group_survival_sketch.md`, and the
archived drafts. The current theory text is `paper/resources/canonical_theory.md`.

## 1. softq dynamics: the sign was wrong

The archived Proposition 5(b) claimed that the surviving softq entropy channel
is mode-seeking under softmax parameterization and drives the policy toward a
peaked distribution. That is incorrect for the implemented objective.

With an equal mirror mixture, the hidden Q terms cancel after averaging:

```text
L_mix = alpha * sum_a pi_a log(pi_a) = -alpha * H(pi).
```

Gradient descent therefore decreases negative entropy and moves the policy
toward the uniform distribution, not toward a peak. In logits,

```text
grad_z L_mix = alpha * pi_b * (ell_b - ell_bar),
d z / d t = -alpha * pi_b * (ell_b - ell_bar).
```

The optimality claim in `paper1_propositions.md` Proposition 5(b) and the
"survive-then-devour" narrative are retired. Any statement about training
dynamics must be re-derived from the actual optimizer step.

## 2. Fisher/KL identity: parameter space must be stated

The archived Proposition 6 stated that the default kappa is a Fisher-metric
ratio. The default kappa in this repository is Euclidean in parameter space.
A Fisher-weighted variant can be defined for categorical logits:

```text
F(pi) = diag(pi) - pi pi^T
kappa_F = (g_bar^T F g_bar) / E_i[g_i^T F g_i]
```

with the infinitesimal KL identity

```text
KL(pi(z) || pi(z + eps v)) = (eps^2 / 2) v^T F v + O(eps^3).
```

The rank-one term `-pi pi^T` must be retained; dropping it is only valid on the
logit tangent space after separately verifying `pi^T g = 0`. The repo's
`verify_kl_kappa.py` already retains the term; the archived proposition text
does not.

## 3. Direction dependence and interference are K-way, not S3

`verify_kway_geometry.py` shows that uniform mixtures cancel for every `K`, that
direction dependence first appears when the condition simplex has dimension two
(so `K >= 3`), and that softq channel interference also occurs for asymmetric
`K=2` mixtures. Statements that these are S3, non-abelian, or
non-commutative-group phenomena are retired. `verify_s3_survival.py` is kept as
the minimal `K=3` example.

## 4. Mixed denominators

Historical files reported both `kappa_uniform_ref` and mixture-reference
numbers, and some scripts called the uniform-reference value `kappa`. The
canonical metric is now

```text
kappa_mix = ||sum_i p_i mu_i||^2 / sum_i p_i ||mu_i||^2,
```

implemented in `src/iigc/metrics/kappa.py` and used by the current experiment
scripts. The historical ratio remains available as `kappa_uniform_ref` for
reading old files, but it is not bounded when `p` is non-uniform.

## 5. Episode noise

`kappa_ep` contains one-episode gradient noise and is a measurement diagnostic.
The structural statement uses condition means (`kappa_mix`). The finite-sample
mean calibration is reported as `kappa_mean_surrogate` and is not a
high-probability ranking guarantee; the earlier `N*` note describes a
deterministic surrogate ratio, not the expectation of the estimator.

## 6. Protocol artifacts kept as method evidence

- Deterministic argmax rollouts measure a single-action weight ratio on the
  mirror bandit, not gradient retention (`rollout_protocol_artifact.md`).
- The Overcooked forced-partner protocol disables mid-episode switching and can
  produce an OOD zero-reward readout. Measurements must use the
  switching-preserving protocol.
- The 510K field-axis script now computes reinforce and value gradients from
  the same episode batch and writes paired results under `s<seed>_paired`.
  Legacy unpaired entries remain in the JSON but must not be pooled with paired
  results.
