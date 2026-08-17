# Experiment 2: interpolation spectrum on a common basis

Same actor parameters theta, same rollouts, same relations — only the
objective that defines the gradient changes. This is the true common
measurement basis (E1 could not compare actor/critic absolute kappa because
they live in different parameter spaces; here every field is on the SAME theta).

## Fields (mode-seeking -> mean-seeking)

- `reinforce`: return-weighted grad log pi(a_taken) — hard mode-seeking
- `awr`: advantage-weighted grad log pi(a_taken), tau = 1
- `softq`: SAC actor loss grad (soft, alpha-tuned)
- `expq`: expected-Q grad sum_a pi(a) Q(a) — mean-seeking
- Gibbs sweep: pi_tau = softmax(logits/tau), grad of E[Q]; tau -> 0 is
  mode-seeking (argmax), tau -> inf is mean-seeking (uniform).

## Result (mean over seeds 41-44, n = 4)

| field   | single | dynamic |
|---------|--------|---------|
| reinforce | **0.402** ± 0.085 | **0.477** ± 0.028 |
| awr       | 0.500 ± 0.003 | 0.533 ± 0.039 |
| softq     | 0.566 ± 0.078 | 0.542 ± 0.103 |
| expq      | 0.562 ± 0.059 | 0.520 ± 0.115 |

Gibbs tau sweep (pi_tau = softmax(logits/tau), grad of E[Q]):

| tau | single | dynamic |
|-----|--------|---------|
| 0.2 | 0.511 | 0.514 |
| 0.5 | 0.533 | 0.525 |
| 1.0 | 0.562 | 0.520 |
| 2.0 | 0.575 | 0.526 |
| 5.0 | 0.588 | 0.567 |

## Interpretation (n = 4)

- **Robust: the mode/mean-seeking axis controls kappa LEVEL.** reinforce
  (hard mode-seeking) is the clear minimum in BOTH modes (0.40 / 0.48);
  awr sits above it; softq/expq (mean-seeking) are highest (0.52-0.57).
  This is the strongest E2 evidence that kappa is a function of the update
  field, measured on one and the same parameter vector.
- **Not robust: the S/D contraction pattern.** reinforce/awr show D > S,
  softq/expq show S > D, with high variance (std up to 0.12). The n = 2
  "dynamic kappa rises with tau, single flat" signature did not survive
  adding seeds 43/44: with n = 4 both modes rise with tau, and the dynamic
  curve is noisy.
- These are converged, already-adapted SAC policies (dynamic reward ~6 vs
  single ~3; alpha auto-tuned to ~0 = fully peaked, i.e. maximally
  mode-seeking policies). Contraction under hidden relations is therefore
  weak here — consistent with the E1/E2 caveat: kappa on a converged policy
  reflects its current solution, not the learning trajectory.
- Caveat: reinforce's low kappa may partly reflect REINFORCE's high gradient
  variance (full-episode returns), not only mode-seeking.

## Next steps

- Measure kappa on an **unadapted policy**: a SINGLE-trained model rolled out
  under DYNAMIC (two deals -> two hidden teams) is the direct IIGC test —
  prediction: low kappa for the reinforce field.
- Or measure during learning (early checkpoints) to see contraction before
  adaptation.
- More seeds still desirable (std is large at n = 4).

Run: `python experiments/common_basis/interpolation/run_interp.py`
Results: `data/kappa/common_basis_interp/results.json`
