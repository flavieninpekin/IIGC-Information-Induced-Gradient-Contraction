# Substantiate the variance decomposition (Experiments A/B/C on Toy)

`verify_variance_decomp.py` — de-circularizes and substantiates the
decomposition `kappa = E_shared/(E_shared + E_contrast + sigma^2)`.

## A. Cross-sample prediction (de-circularize)

Components measured on an estimation set (200 eps/partner) predict kappa on a
held-out validation set (200 eps/partner, independent seeds).

| condition | kappa_pred (est) | kappa_meas (val) | error |
|-----------|------------------|------------------|-------|
| REVEALED  | 0.396            | 0.435            | 0.038 |
| HIDDEN    | 0.000            | 0.000            | 0.000 |

The decomposition is a stable, transferable descriptor — not a fit to the
same data.

## B. Compactness: how many episodes to estimate kappa? (corrected)

The correct test for "kappa is estimable from a compact measurement" is a
SAMPLE-SIZE SWEEP (`verify_compactness.py`), not component bootstrap stds.
Bootstrap stds of the components are ~18% relative, but kappa (a ratio) is
much more stable because common fluctuations cancel.

REVEALED (trained policy, true kappa = 0.411):

| N (eps/partner) | kappa_hat | std | err vs true |
|------------------|-----------|------|-------------|
| 10  | 0.428 | 0.019 | 0.016 |
| 50  | 0.415 | 0.009 | 0.004 |
| 100 | 0.414 | 0.007 | 0.003 |
| 200 | 0.412 | 0.005 | 0.001 |
| 1000| 0.412 | 0.002 | 0.000 |

~50 episodes per partner gives kappa within +/- 0.009; relative error falls
from 4.6% (N=10) to 0.5% (N=1000).

HIDDEN (random policy): true kappa = 0.000, estimated exactly at any N
(cancellation is deterministic).

Earlier wording claimed kappa reliability from component bootstrap stds
(~20-30%): that was over-reach. The component stds are ~18% but kappa itself
is more stable; the sample-size sweep is the correct evidence.

## C. kappa is scale-invariant; the decomposition is not (demonstrated)

`verify_scale_invariance.py`: same policy, forced assignments, rewards scaled
by lambda in {1, 10, 100}. Gradients scale by lambda, so components scale by
lambda^2, while kappa (a ratio) is exactly invariant (machine precision).

| lambda | kappa | E_shared | E_contrast | sigma^2 |
|--------|-------|----------|------------|---------|
| 1   | 0.4158 | 0.234 | 0.329 | 1.91 |
| 10  | 0.4158 | 23.43 | 32.92 | 191.1 |
| 100 | 0.4158 | 2343  | 3292  | 19111  |

kappa is identical (0.4158) across a 10000x signal-energy range; the
decomposition reports the difference. kappa is blind to absolute signal
strength (like cosine similarity); the decomposition is not.

Also visible in the natural reveal sweep: E_contrast spans ~2000x (394 -> 0.21)
across reveal levels while kappa_mean only spans 0 -> 0.4.

Practical point: a weak-signal run can have healthy-looking kappa; only the
decomposition reports the absolute gradient energy.

## Notes

- Per-episode kappa formula (kappa_pred_ep) vs empirical pairwise kappa
  (kappa_meas_ep) differ by the expectation-of-ratio effect; the averaged
  (two-rollout) kappa is the clean, formula-matching quantity.

Run: `python experiments/common_basis/toy/verify_variance_decomp.py`
Results: `data/kappa/variance_decomp/results.json`
