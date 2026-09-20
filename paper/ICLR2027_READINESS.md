# ICLR 2027 Readiness and Canonical Claim Registry

Status date: 2026-09-20

## Decision

Do not submit the current repository or `gradient_survival_draft` unchanged.
There is a plausible ICLR paper here, but only after the paper is narrowed to a
single hidden-heterogeneity gradient-audit thesis and all definitions are made
consistent. The current material is not yet a paper: there is no ICLR LaTeX
submission, several source documents contain superseded claims, and the main
metric has multiple denominators and noise conventions.

ICLR 2027 deadlines are the abstract deadline on 2026-09-18 and the full-paper
deadline on 2026-09-25, both 11:59 PM Anywhere on Earth. The main text limit is
9 pages. Sources: `https://iclr.cc/Conferences/2027/CallForPapers` and
`https://iclr.cc/Conferences/2027/AuthorGuidelines`.

The corrected derivations for the Theory section are in
`paper/resources/canonical_theory.md`. The older `theory.md` and long drafts
are retained as provenance only. The current working main text is
`paper/drafts/main_text_v1.md` with a Chinese reading companion at
`paper/drafts/main_text_v1_zh.md`; both use only canonical definitions and the
refreshed numbers. The anonymous ICLR 2027 LaTeX submission is in
`paper/submission/iclr2027/` (`main.tex`), compiled to a 10-page PDF with the
bibliography starting on page 9 (the main text fits the 9-page limit); see
`paper/submission/README.md` for build steps and pre-submission checks.

## Recommended Thesis

> When hidden conditions are averaged, what survives is determined not only by
> symmetry between conditions, but by the geometry of the condition mixture and
> by the objective-induced structure of the gradient field. Retention is a
> structural audit of the specified condition signal, and the audit recommends
> condition-aware capacity only when low retention coincides with contrast
> energy above noise and a task-relevant condition.

This is stronger and safer than claiming that policy-gradient methods are bad
and value methods are good. The contributions are organized as three chapters
of this one question: structure (exact cancellation, soft-Q closed form,
$K$-way geometry and interference), measurement (bounded mixture-reference
retention, protocol and noise separation), and boundaries (real-environment
alignment, DICES-350 negative capacity check, kappa/performance decoupling).

## Novelty Positioning

The paper must not present hidden heterogeneity or gradient conflict as new
problems. The related-work boundary should be explicit:

| Nearby line | What is already established | What this paper can claim |
|---|---|---|
| MGDA, PCGrad, CAGrad, GradDrop, RotoGrad, FAMO and later gradient-surgery work | Conflict is measured from observed task gradients and then reweighted or projected | Hidden-condition symmetry can make cancellation structural before a surgery method is applied; the paper diagnoses the field rather than proposing another surgery |
| Distributional Preference Learning (ICLR 2024), RLHF with hidden context, heterogeneous-preference DPO/RLHF, and recent preference-fairness work | Aggregating unobserved preference/context heterogeneity can distort a reward or policy | The contribution is a gradient-space audit with exact cancellation and objective-specific closed forms, not a new preference model or fairness intervention |
| Group averaging, invariant learning, Reynolds operators, and representation theory | Symmetry averaging projects onto invariant components | The application is the exact condition-gradient field and its measurement consequences; do not claim to invent group averaging |
| The preceding IIGC/AAAI work | Kappa and estimator dependence were introduced and the PG/value reversal was reported as an open mechanism | This paper must add the mechanism, corrected protocol, and new formulas; do not recycle the earlier empirical claim as the main contribution |

The safest novelty sentence is: "We characterize when hidden-condition
gradient aggregation cancels condition-specific signal, derive objective-aware
closed forms in a matching bandit, and show how rollout noise and normalization
can make the resulting diagnostic misleading." Any stronger novelty claim needs
an explicit literature check and a new theorem or experiment.

## Canonical Definitions

For hidden condition `i`, let `g_i(theta) = grad_theta L_i(theta)` denote the
population condition-gradient under a fixed measurement protocol. For a
condition mixture `p`, define

```text
g_bar(p) = sum_i p_i g_i
kappa_mix(p) = ||g_bar(p)||_2^2 / sum_i p_i ||g_i||_2^2
```

For nonnegative normalized `p`, `0 <= kappa_mix <= 1` by Jensen's inequality.
The historical uniform-reference score is a different quantity:

```text
kappa_uniform_ref(p) = ||g_bar(p)||_2^2 / mean_i ||g_i||_2^2
```

It may exceed one when `p` is non-uniform and must not be called a bounded
retention ratio. Use `kappa_mix` for new K-way results.

The following quantities must remain separate:

| Quantity | Meaning | Paper status |
|---|---|---|
| `kappa_mix` | Structural retention under the actual condition mixture | Primary metric |
| `kappa_uniform_ref` | Historical uniform-condition normalization | Compatibility/appendix only |
| `kappa_mean` | Estimate using condition sample means | Structural estimate, with finite-sample caveat |
| `kappa_ep` | Ratio containing episode-level gradient noise | Measurement diagnostic, not structural kappa |
| `kappa_F` | Fisher-weighted variant | Optional appendix only; not the default metric |

The default metric in the code is Euclidean parameter-space kappa. A Fisher or
infinitesimal-KL interpretation requires a separately defined metric and the
full matrix `F = diag(pi) - pi pi^T`; it is not interchangeable with the
default Euclidean score.

## Claims To Keep

### C1. Exact cancellation under symmetric hidden conditions

In the two-action mirror bandit, `Q_A=(r,-r)` and `Q_B=(-r,r)`. For a weight
field that is elementwise in `Q_r` and independent of the policy parameters,
`w_r(a)=f(Q_r(a))`, the expected policy-gradient fields satisfy `g_B=-g_A`,
so the equal mixture has zero shared gradient.
The same cancellation extends to the K-way matching construction under a
uniform condition mixture. This is an exact theorem for the stated model, not
a claim about every neural RL environment.

Evidence: `data/kappa/toy_fields/o1_closed_forms.json`,
`data/kappa/toy_fields/kway_geometry.json`.

### C2. Exact two-action softq closed form

Let `rho=pi(action 0)`, `q=1-rho`, and use
`L_i=sum_a pi_a(alpha log pi_a - Q_i(a))`. Under the equal mirror mixture,

```text
Delta = alpha * log(rho / q)
kappa_mix = Delta^2 / (Delta^2 + 4 r^2)
```

This is a fixed-policy, Euclidean logit-gradient result. It is not a training
dynamics theorem and it does not imply that softq improves task performance.

Evidence: `experiments/common_basis/toy/verify_closed_forms_o1.py` and
`data/kappa/toy_fields/o1_closed_forms.json`.

### C3. K-way geometry and channel interference

For the K-way matching bandit `Q_i(b)=2R delta_ib`, the expected expq field is

```text
g_i[b]     = -2 R pi_b (delta_ib - pi_i)
g_bar[b]  = -2 R pi_b (p_b - <p,pi>)
```

Once the condition simplex has dimension at least two (`K >= 3`), fixed-size
mixtures can differ by direction, not only by distance from uniform. Softq's
entropy and mixture channels can also interfere. Neither phenomenon is
S3-specific, non-abelian-specific, or absent from asymmetric K=2 examples.

Evidence: `data/kappa/toy_fields/kway_geometry.json` and
`notes/kway_compromise_verdict.md`.

### C4. Kappa is a diagnosis, not a sufficient performance statistic

The O4 experiment is useful as a boundary result: in the switching task, TD
has low measured kappa but the best return, while softq has high kappa and lower
return. The correct interpretation is variable mismatch: the measured initial
condition is not the history variable required by the task.

Evidence: `notes/o4_performance_verdict.md` and
`data/kappa/o4_adaptive/aggregate.json`.

### C5. Protocol and noise audit

Deterministic argmax rollouts can turn a gradient-retention measurement into a
single-action weight-ratio measurement. Episode noise can also make a low
`kappa_ep` compatible with a non-low structural `kappa_mean`. Use stochastic
policy-weighted rollouts, report energy and noise, and state the exact gradient
field. The Overcooked forced-partner protocol is a separate environment
artifact: it can disable partner switching and create an OOD zero-reward readout.

Evidence: `notes/rollout_protocol_artifact.md`,
`data/kappa/toy_fields/det_vs_stoch.json`, and
`paper/resources/overcooked_slice.md`. A controlled visibility contrast on the
mirror bandit (same policy, same parameters, same seeds, only the observation
mask changes) confirms the hidden/revealed gap without the earlier policy
confound: `data/kappa/toy_fields/visibility_control.json`.

## Claims To Remove or Downgrade

| Current wording or direction | Required treatment |
|---|---|
| PG universally contracts while value universally survives | Remove; only claim stated fields/protocols and environments |
| Direction dependence or interference is S3/non-abelian-specific | Remove; rewrite as K-way simplex geometry |
| Softq is necessarily mode-seeking or converges to a peaked policy | Remove until the optimization sign and trajectory are re-derived; current `L=alpha sum pi log pi` minimization pushes entropy upward in the mirror case |
| Default Euclidean kappa is a Fisher/KL ratio | Remove from main text; make a separately correct optional variant |
| `kappa_ep` near zero proves structural cancellation | Remove; inspect `sigma2` and `kappa_mean` |
| The N* expression gives a high-probability ranking guarantee | Downgrade to the deterministic surrogate `f_i(N)` unless a ratio-estimator proof is added |
| DPO is directly the same reverse-KL field | Remove or qualify as an analogy with an explicit derivation |
| 510K proves a hiddenness trend | Remove; use it as a weak-drive/protocol boundary |
| Old Toy values `0.561`, `0.068`, or `0.633` are current field-axis evidence | Mark historical/deterministic artifact; do not cite in Results |

## Evidence Priority

| Priority | Evidence | Use |
|---|---|---|
| A | `o1_closed_forms.json` | Exact Z2 formulas and cancellation |
| A | `kway_geometry.json` | K-way formula and direction/interference control |
| A | `det_vs_stoch.json` | Measurement-protocol failure mode |
| A | `o4_adaptive/aggregate.json` | Kappa/performance decoupling boundary |
| A | `notes/canonical_metric_results.md` | Refreshed structural numbers for all real-environment tables |
| A | `data/kappa/dices350/audit.json` | Real three-group supervised check: race axis kappa_mix 0.116, contrast dominated by item noise, no conditional-capacity gain |
| B | `oc_field_axis.json` | Structural `kappa_mix` on a shared episode batch: dynamic value 0.998, differentiable-baseline awr 0.590, reinforce 0.529; static weak |
| B | `510k_field_axis.json` | Paired protocol, keys `s*_paired`: value 0.95-0.98 vs reinforce 0.51 |
| B | `overcooked_slice` | Controlled option-level witness, not primitive-action PG proof |
| C | `theory_toy.json`, `theory_toy2.json` | Historical CE-fit/definition-split records; appendix only |
| C | `common_basis_interp`, `cross_transfer` | Deterministic 510K measurements; do not use as decisive field-axis evidence |

The "field axis" is real but about `1.9x` under `kappa_mix`, not the `30-65x`
separation suggested by comparing noise-dominated `kappa_ep` values. Write the
paper around alignment versus orthogonality, not around "death".

## Minimal Paper Shape

1. Introduction: hidden heterogeneity and why a raw gradient-conflict number is
   insufficient.
2. Definitions: condition gradients, `kappa_mix`, noise decomposition, and
   measurement protocol.
3. Theory: symmetric cancellation, two-action softq closed form, and K-way
   direction/interference formula.
4. Experiments: exact bandit controls, stochastic-vs-deterministic audit, and
   one real-environment supporting study.
5. Boundary: shared versus conditional capacity and the O4 performance
   decoupling result.
6. Limitations and reproducibility.

Do not include all historical experiments in the main narrative. Put old
CE-fit definitions, 510K reveal grids, failed intervention attempts, and
protocol-sensitive results in an appendix or artifact note.

## Pre-submission Blockers

1. Recompute the main tables with one definition registry and metadata for seed,
   rollout protocol, denominator, parameter space, and commit.
   Progress: `src/iigc/metrics/kappa.py` is the single implementation, result
   files carry `measurement_metadata`, and the K=3/4 tables have been
   regenerated under `kappa_mix`. Real-environment tables (Overcooked,
   510K, DICES) are now regenerated with metadata.
2. Resolve the softq optimization-direction claim and remove the unsupported
   mode-seeking/peaking language.
   Progress: `paper/resources/canonical_theory.md` and
   `notes/paper1_propositions_corrections.md` state the corrected direction.
   Remaining: purge the archived drafts from any reused text.
3. Decide whether the main metric is `kappa_mix`; if yes, recompute all K-way
   direction and interference tables under that denominator.
   Progress: decided and implemented for the Toy scripts; the K-way JSON has
   been regenerated. The 510K paired rerun and Overcooked tables have been
   regenerated with the canonical fields.
4. Either remove the Fisher/KL proposition or repair its parameter-space and
   Fisher-matrix derivation.
   Progress: repaired in `canonical_theory.md` (logit-space KL, full Fisher
   matrix, explicit Euclidean-vs-Fisher separation).
5. Re-run or explicitly caveat the 510K common-basis measurements because the
   current field code collects separate rollouts for the fields.
   Progress: `run_510k_field_axis.py` now computes both fields on one paired
   episode batch and stores results under `s<seed>_paired` with metadata; the
   paired rerun over the 18 checkpoints is complete (six seeds per level).
6. Write an anonymous 9-page ICLR paper and an anonymous reproducibility bundle.
7. Check OpenReview profiles, author order, quota, reciprocal-reviewer status,
   double-blind citations, and the required AI-use statement.

## Current Session Status (2026-09-20)

- Canonical metric module: `src/iigc/metrics/kappa.py` with
  `condition_decomposition`, `episode_decomposition`, and
  `measurement_metadata`; unit tests cover boundedness, weighting, noise
  separation, and the metadata contract.
- Closed-form regression tests: `tests/test_closed_forms.py` checks the softq
  closed form against autograd to 1e-10 and exact expq cancellation.
- Corrected theory: `paper/resources/canonical_theory.md`.
- Correction log for superseded claims:
  `notes/paper1_propositions_corrections.md`.
- Active scripts use the canonical decomposition: field axis, switch kappa,
  memory eval, 510K reveal/field axis, K-way geometry, shared vs conditional,
  and the historical interpolation/SAC splits.
- Resolved this session: controlled visibility contrast (one policy, masked vs
  revealed) fixes the toy hidden/revealed confound; the Overcooked field axis
  was re-measured on shared episode batches with the differentiable-baseline
  `awr` and full-parameter zero-padded gradients (dynamic awr 0.590 after
  fixing a `[T]`-vs-`[T,1]` shape-broadcast bug in the first rerun); the 510K
  zero-padding preserves the archived numbers; ICLR PDFs and drafts are synced
  to the new tables.
- Remaining: submission logistics (OpenReview profile, quota, double-blind
  citations), a final read of `main.tex` against `main_text_v1.md`, and the
  discussant questions in `main_discussion.tex`.

## Recommendation

If any submission logistics cannot be resolved before 2026-09-25, do not
submit a placeholder or internally inconsistent paper. The compute-dependent
items are resolved; the remaining risk is only packaging and final proofing. A
narrow, correct theory-and-measurement paper is now in place.
