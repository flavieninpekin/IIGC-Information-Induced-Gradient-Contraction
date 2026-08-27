# Gradient Survival under Hidden Group-Symmetric Heterogeneity: Exact Cancellation, Direction Dependence, and Interference

**Draft v0.5 — for opinion gathering (2026-08-26). All numbers are from verified
experiments; data provenance listed in the Appendix table. Formal proofs of the
general-group statements are in progress; Z2/S3 instances are verified to
machine precision.**

---

## Abstract

A training signal can be silently erased before it ever reaches the weights:
when data is stratified by a *hidden* categorical condition (partner identity,
user group, preference mode), and the conditions' targets are related by a
symmetry group G, averaging updates across conditions cancels every
non-invariant gradient component. We study the algebra of this cancellation in
a minimal three-condition (S3-symmetric) bandit. Three results: (1) exact
cancellation for any objective whose weights are parameter-independent and
elementwise in the target, under symmetric mixtures — the two-condition mirror
result, generalized; (2) *direction dependence*: for fixed mixture asymmetry
(equal L1 distance from uniform), survival rates differ by up to 2.5x across
mixture directions — a phenomenon structurally impossible in the two-condition
case, whose parameter tangent space is one-dimensional; (3) *two-channel
interference*: the entropy-regularized soft-Q objective couples a
policy-intrinsic channel with the mixture-induced channel as vectors in a 2D
space; the two can destructively interfere, making survival non-monotone in
the entropy coefficient, with an exact closed form. A fourth result concerns
the *structure axis*: target asymmetry revives odd (policy-gradient) fields
with closed form (r-s)^2/(2(r^2+s^2)), while value (TD) fields collapse at
their own converged fixed point for *every* asymmetry degree. Implications for
multi-task learning, federated learning, and preference learning are
discussed, including a falsifiable warning: mis-tuned entropy regularization
can destroy what mixture asymmetry provides.

## 1. Introduction

Consider a learning system whose training data is drawn from several
subpopulations — users with conflicting preferences, clients with mirrored
task structure, teammates whose roles are hidden. When the subpopulation
identity is *not observable to the policy* but *changes what the right
behavior is*, the gradients conditioned on each subpopulation may point in
conflicting directions. Averaged updates then cancel: training appears to
progress slowly or stall entirely, without any pathology visible in the loss
curve.

When do such conflicts cancel *exactly*? When do they partially survive? Can
we predict survival *before* training? This paper gives an algebraic account in
a minimal setting where the conflict structure is controlled by a symmetry
group G.

**Running example.** A language model is fine-tuned on preference pairs
collected from heterogeneous users. Three groups disagree systematically:
group A prefers verbose answers, group B concise ones, group C code-style
ones. At inference the user identity is *not part of the prompt* — the policy
faces the same input from all three groups, while "what is correct" differs by
group. Training data mixes the groups in unknown proportions p; per-group
gradients point in different directions; the *mixed* gradient is what the
policy actually takes. This is the phenomenon our bandit abstracts: conditions
= user groups, actions = response styles, Q_i(j) = reward for matching group
i's preference, constant observation = hidden identity, mixture p = group
proportions. Every ingredient of the bandit maps to a feature of the running
example; nothing is arbitrary.

**Why three conditions (not two).** The two-condition (Z2) case is the
well-understood mirror: all gradients are collinear, asymmetry has a single
direction, and survival is a scalar function of asymmetry magnitude (verified
closed forms in §3). Three conditions is the smallest setting with a
two-dimensional irreducible representation, where survival can depend on the
*direction* of the mixture (§3.2) and survival channels can *interfere*
(§3.3). These phenomena are the point of the paper, and a two-condition model
cannot exhibit them; the Z2 results are kept as the verified base case.

**Setup (one-paragraph version).** A constant-observation bandit with |A| = |G|
actions; conditions g ∈ G assign target values Q_g; the policy π_θ is a softmax
that cannot see g. Each objective induces a condition-conditioned expected
gradient g_g = ∇L_g. Training mixes conditions with distribution p; the
effective update is ĝ_p = E_p[g_g]. We study the *survival rate*

    κ(p) = ||ĝ_p||² / E_uniform[||g_g||²],

the fraction of condition-conditioned signal energy surviving the mixture.

**Contributions.**
1. **Exact cancellation generalizes.** Any objective whose per-condition
   gradient is linear in the target with parameter-independent, elementwise
   weights cancels exactly under a symmetric mixture (κ = 0 to 1e-32),
   for G = Z2 *and* G = S3.
2. **Direction dependence (new in S3).** At fixed mixture asymmetry, κ varies
   up to 2.5x with the *direction* of p in the mixture simplex. In Z2 the
   tangent space is one-dimensional, so asymmetry has a single direction —
   this phenomenon is structurally impossible there.
3. **Two-channel interference (new in S3).** The soft-Q objective has two
   survival channels — a policy-intrinsic entropy channel and the
   mixture-induced channel. In the 2D tangent space they add as vectors and
   can cancel; κ(α) is non-monotone with an exact closed form. Warning for
   practice: entropy regularization can erase surviving signal.
4. **Structure axis splits by field type.** Target asymmetry revives
   policy-gradient fields (exact closed form), but TD/value fields collapse at
   their own converged fixed point for all asymmetry degrees (residual
   anti-symmetry e_B = -e_A).

## 2. Setup and definitions

**Three-condition matching bandit.** Conditions i ∈ {0,1,2}; actions j ∈ {0,1,2};
Q_i(j) = 2R·[j=i]. The condition is hidden: the observation is constant, and
π_θ = softmax(z). All expectations are over a∼π_θ (the canonical
π-weighted protocol; deterministic rollouts are known to corrupt such
measurements [protocol-artifact note]).

**Fields.** Condition-conditioned gradients under the *expected per-sample*
definition (matching our measurement code):
- expq / reinforce: weights w = Q (parameter-independent, elementwise);
- softq: L = Σ_a π_a (α log π_a − Q_g(a)) — entropy channel + Q channel;
- awr: w = exp((Q−V)/τ), V differentiable (baseline channel).

**Mixture.** p ∈ Δ² is the training distribution over hidden conditions.
κ(p) := ||Σ_i p_i g_i||² / E_i||g_i||² (uniform reference energy).

## 3. Theory

### 3.1 Exact cancellation (odd fields)

**Proposition 1.** For any field whose per-condition gradient satisfies
g_i = ∇⟨π, Q_i⟩ with Q_i elementwise parameter-independent (reinforce,
expq, awr without baseline, softmax(Q/τ) weights), the uniform mixture
cancels exactly: ĝ = 0, κ = 0.

*Verification (machine precision):* G=Z2 and G=S3, κ ≈ 1e-32 (3 seeds, all
fields). *Proof sketch (Z2)*: Q_B = swap(Q_A) ⟹ g_B = −g_A. *For S3*: Q_i =
2R·e_i and ⟨π,Q_i⟩ = 2R π_i; Σ_i (1/3)∇⟨π,Q_i⟩ = (2R/3)∇Σ_i π_i = 0 because
Σ_i π_i = 1. The S3 mechanism is *different* from the Z2 one (a sum of
gradients that vanishes because Σπ = 1, not because of pairwise mirrors) —
both exact.

### 3.2 Closed form and direction dependence (odd fields)

**Proposition 2.** For expq/reinforce in the S3 bandit:
    ĝ_p[j] = 2R π_j (p_j − ⟨p,π⟩),
    κ(p,π) = Σ_j π_j² (p_j − ⟨p,π⟩)² / E_i Σ_j π_j² (δ_ij − π_i)².
*Verification:* closed form vs autograd, max abs err 6.3e-15 over 200 random
(z,p).

**Direction dependence.** Fix L1 distance from the uniform mixture and rotate
the direction of p (24 directions). At distance 0.45, κ ranges [0.075, 0.192]
— a 2.5x spread determined by direction alone. In Z2 the simplex of mixtures
is a segment: asymmetry has no direction, so survival is a function of
magnitude only. The appearance of direction dependence is a necessary
consequence of the 2D tangent space; it is the empirical signature that the
framework is genuinely group-theoretic and not a repackaging of the Z2 case.

### 3.3 Two-channel interference (softq)

**Proposition 3.** For softq, per-condition gradient and mixture gradient:
    g_i[j] = π_j [ α(ℓ_j − ℓ̄) − 2R(δ_ij − π_i) ],   ℓ = log π, ℓ̄ = ⟨π,ℓ⟩
    ĝ_p[j] = π_j [ α(ℓ_j − ℓ̄) − 2R(p_j − ⟨p,π⟩) ],
    κ_softq = Σ_j π_j² [α(ℓ_j−ℓ̄) − 2R(p_j−⟨p,π⟩)]²
              / mean_i Σ_j π_j² [α(ℓ_j−ℓ̄) − 2R(δ_ij−π_i)]².

The two brackets are the *entropy channel* (direction ℓ−ℓ̄, intrinsic to the
policy) and the *mixture channel* (direction p−⟨p,π⟩, induced by the data).
They live in the same 2D tangent space and add as vectors.

**Non-monotonicity.** For p = (0.7, 0.2, 0.1), κ(α) = 0.396 (α=0.05) →
0.032 (α=2) → 0.415 (α=5): the channels destructively align at intermediate α.
*Verification:* closed form vs autograd, max abs err 3.3e-16.

**Predictive warning.** For any asymmetric p and fixed policy, there exists a
destructive α* where κ collapses. In preference learning (RLHF/DPO), where
entropy/KL coefficients are tuned as hyperparameters, this implies that
regularization strength can silently erase the signal that mixture asymmetry
provides — a falsifiable, actionable prediction.

### 3.4 Structure axis: field-type split

**Proposition 4.** In the two-condition asymmetric bandit Q_A=(r,−r),
Q_B=(−s,+s), uniform mixture:
- odd fields (expq/reinforce): κ = (r−s)² / (2(r²+s²)) — revived by
  asymmetry; verified to 1.4e-17.
- TD fields at their converged fixed point: residual anti-symmetry e_B = −e_A
  for *every* s/r ⟹ κ_TD → 0 for all asymmetry degrees; measured 0.001 ± 0.001
  across s/r ∈ {1.0…0.0}, vs 0.5 for expq at s/r=0.

Interpretation: the *structure axis is field-type dependent*. Structural
asymmetry helps exactly the fields that symmetric design kills, while value
fields are killed by their own convergence — a dynamic mechanism, not a
structural one.

### 3.5 Attractor structure (dynamics; verified, theory in notes)

The single-step geometry does not determine where training goes. Under a
symmetric mixture the three field types have distinct attractors (verified by
gradient-flow simulation and finite-difference checks):
- **Odd fields** are frozen: the gradient is identically zero, so the policy
  never leaves its initialization — "stuck" is exact, not asymptotic.
- **softq** survives only through its entropy channel; under softmax
  parameterization that channel is *mode-seeking* — the flow drives π toward
  the peaked (symmetry-broken) attractor, consuming the very sharpness it
  created. (The π-space intuition that entropy flows push to uniform does not
  survive the softmax reparameterization.)
- **TD** converges to the relation-mean value function, which is G-invariant;
  the induced policy (argmax/softmax of Q̄) degenerates to uniform.

Interpretation: *survival is a single-step property; what survives may eat
itself (softq) or be rendered inert (TD) at the fixed point.* This closes a
loop with the structure-axis result: the value field's collapse at its own
fixed point (§3.4) is the static face of the same attractor.

### 3.6 General-group conjecture (open)

We conjecture the general statement: under a symmetric mixture, the averaged
gradient contains only components invariant under the induced representation
of G; survival decomposes by irreducible representations with closed-form
coefficients; 2D irreps enable direction dependence and interference. Z2 and
S3 are the verified instances. The general proof is in progress.

### 3.7 KL formalization (verified to machine precision)

The Euclidean definition of κ hides a metric fact: under softmax
parameterization, κ is a *Fisher-metric ratio*, and Fisher is the second-order
KL. For tangent-space gradients (Σ_b g_b = 0; also πᵀg = 0 for our fields):

**Prop 6 (κ as infinitesimal-KL ratio).**
    κ_F(p) = (ĝᵀFĝ) / E_r[g_rᵀFg_r] = Σ_b π_b ĝ_b² / E_r[Σ_b π_b g_{r,b}²]
           = lim_{ε→0} KL(π ‖ π+εĝ) / E_r[KL(π ‖ π+εg_r)],
with F = diag(π) − ππᵀ the categorical Fisher metric. *Proof:* πᵀg = 0
implies Fg = diag(π)g on the tangent space; KL(π‖π+εg) = (ε²/2)gᵀFg + O(ε³).
*Verification:* ε-shrinking sequence converges to κ_F (float64 floor).

**Prop 9 (the KL projection is DERIVED from the hidden mixture, not assumed).**
For softq objectives over hidden conditions, the chain is:

    L_r = α·KL(π ‖ π_r*) + const,      π_r*(a) ∝ exp(Q_r(a)/α)
    Σ_r p_r L_r = α·KL(π ‖ π̃) + const,  π̃(a) ∝ exp(Σ_r p_r Q_r(a)/α)
    ⟹  ĝ_p = α·∇KL(π ‖ softmax(Q̄_p/α)),  Q̄_p := Σ_r p_r Q_r.

The first line says each condition's softq objective *is* a reverse KL to that
condition's optimal policy (α·KL(π‖π_r*) = αΣπ logπ − ⟨π,Q_r⟩ + const). The
second is the weighted-mixture Pythagorean identity (Prop 7). Hence the
mixture operator is a geometric-mean projection onto the *mixture-averaged
value*: survival direction = direction of Q̄_p in value space.

*Corollary (mirror).* Mirror conditions under a uniform mixture give
Q̄_p = 0, π̃ = uniform, and ĝ = α·π_b(ℓ_b − ℓ̄) — the entropy channel of §3.3
follows *by derivation* rather than by assumption; Q-linear (odd) fields are
the α→0 degenerate limit with no KL target, recovering exact cancellation.
*Verification (machine precision):* per-condition equality 5.6e-16; mixed
gradient vs projection 5.0e-16; mirror end-to-end exact; asymmetric chain
3.1e-16 (`verify_mixture_to_kl.py`).

**Prop 7 (mixture = geometric-mean projection; KL Pythagorean identity).**
For reverse-KL distillation fields g_r = ∇KL(π‖π_r*) (log-ratio weights):
    ĝ_p = Σ_r p_r g_r = ∇KL(π ‖ π̃),  π̃(a) ∝ exp(Σ_r p_r log π_r*(a)).
*Proof:* Σ_r p_r KL(π‖π_r*) = KL(π‖π̃) + const (weighted-mixture identity;
the log-normalizer is π-independent). *Corollary:* mirror/permuted targets
under a uniform mixture give π̃ = uniform, and the surviving mixed gradient
is exactly the entropy channel π_b(ℓ_b − ℓ̄) of §3.3 — the even component
is, in KL terms, the reverse KL to the uniform target. *Verification:*
3.3e-16.

**Prop 8 (KL direction axis = field axis).** Mode/mean-seeking (the classical
exclusive/inclusive KL dichotomy of variational inference) classifies the
fields: log-ratio (distillation) weights are reverse-KL (mode-seeking; survive
then self-consume), forward-KL fields g_r = π − π_r* are mean-seeking with
mixture gradient π − π̄ (arithmetic mean; align but degenerate), and
Q-linear fields have no KL target (exact cancellation). Verified: forward-KL
mixture identity to 0.00e+00; mirror closed form κ_F = 0.7838 exact.

Interpretation: the original "mode vs mean geometry" intuition for gradient
retention is now a precise statement about which KL direction a field's
weights distill toward — directly checkable in preference learning, where
DPO-style objectives are reverse-KL targets over preference groups.

## 4. Verification summary

| Result | Quantity | Value |
|---|---|---|
| Exact cancellation (S3) | κ at uniform p | 1e-32 |
| expq closed form | max|err| vs autograd | 6.3e-15 |
| softq two-channel closed form | max|err| vs autograd | 3.3e-16 |
| Direction dependence | κ range at L1=0.45 | [0.075, 0.192] (2.5x) |
| Interference | κ(α) at p=(.7,.2,.1) | 0.396 → 0.032 → 0.415 |
| Structure axis | expq vs TD at s/r=0 | 0.500 vs 0.001 |
| Supervised domain | hinge / BCE | κ=0.0000 / κ=1.0000 (exact) |

The supervised-domain line (hinge cancels exactly under label-flip; BCE/mean
losses are fully aligned) confirms the same parity structure outside RL with
the same exactness.

## 5. Implications and related work

**Multi-task learning.** Gradient-conflict literature (PCGrad, CAGrad,
Nash-MTL, …) treats conflicts as empirical quantities to be mitigated. We
characterize when conflicts are *structurally unavoidable* — a prerequisite
for principled mitigation.

**Federated learning.** Drift analyses bound heterogeneity effects via
gradient dissimilarity assumptions; Wang et al. (2022) showed naive measures
overestimate harm. Our decomposition gives the structural rule: symmetric
mirror clients cancel, asymmetric mixtures survive by direction.

**Loss symmetrization (closest neighbor).** The label-noise literature
(unhinged loss; symmetrization of losses, ICLR 2025) works in *risk space*:
symmetric losses keep risk minimizers invariant under stochastic label
flipping. We work in *gradient space* under *structured* heterogeneity (real
subpopulations, not noise): survival rates, direction dependence, and
interference are objects their risk-space theory does not address. The
distinction (stochastic noise vs. structured targets; risk optimality vs.
gradient survival) must be stated explicitly.

**Preference learning (RLHF).** Recent work (2026) documents that preference
averaging under-represents minority groups (procedural fairness) and proposes
empirical mitigations. Our account supplies the algebra: symmetric preference
groups cancel DPO/BCE gradients exactly; asymmetry survives by direction;
entropy coefficients can destructively interfere. A falsifiable experiment
follows directly: measure κ on preference-group-conditioned gradients before
and after tuning β.

**Behavioral decoupling (boundary).** In a companion experiment, clean-basis κ
did not predict task success when the task's requirement (tracking a switching
condition from reward history) was orthogonal to the measured relation
variable — the diagnostic tells you about gradient retention, not about fate.

**Empirical origin (optional citation).** That hidden relational variables can
cancel policy gradients in practice was first observed empirically in
multi-agent games with hidden team assignments and hidden roles. This paper is
self-contained and no result depends on that work; when that observation is
publicly available (arXiv / AAAI 2027), it can be cited here as empirical
motivation.

## 6. Limitations

- Toy constant-observation bandits; real-world demonstrations (preference
  data) are the natural next step and are not yet included.
- Exact statements are for symmetric mixtures and elementwise weight fields;
  general groups are conjectured, not proved.
- The TD collapse statement is asymptotic (at the converged fixed point);
  transient dynamics during training remain to be characterized.

## 7. Conclusion

Under hidden group-symmetric heterogeneity, gradient survival is not an
empirical accident but a computable algebraic quantity: exact cancellation
for odd fields, direction-dependent survival in ≥2D tangent spaces, and
destructive interference between survival channels. The framework converts
"why is my training stalling" into a closed-form question, and yields at
least one actionable warning for practitioners.

---

## Appendix: data provenance

| Data | File |
|---|---|
| S3 survival + closed forms | `data/kappa/toy_fields/s3_survival.json` |
| Structure axis | `data/kappa/toy_fields/structure_axis.json` |
| Z2 closed forms (props A/B/C) | `data/kappa/toy_fields/o1_closed_forms.json` |
| Supervised domain | `data/kappa/supervised_transfer/fixed_model.json` |
| Figures | `paper/figures/s3_survival_landscape.png` |
| Scripts | `experiments/common_basis/toy/verify_s3_survival.py`, `verify_structure_axis.py`, `verify_closed_forms_o1.py` |
| Notes | `notes/paper1_group_survival_sketch.md` |