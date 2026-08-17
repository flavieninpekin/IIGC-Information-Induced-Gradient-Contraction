# Cross-transfer kappa matrix

Train mode x test mode on the same SAC actor parameters, same two-deal
rollout protocol, only the (train, test) mode pairing changes.

Matrix (reinforce field, mean over seeds 41-44):

| train \ test | single | dynamic |
|--------------|--------|---------|
| single       | 0.402 ± 0.085 | 0.518 ± 0.044 |
| dynamic      | 0.487 ± 0.074 | 0.477 ± 0.027 |

awr: all cells ~0.50 (0.500/0.508, 0.499/0.533).
softq: 0.566/0.524, 0.596/0.542.  expq: 0.562/0.517, 0.581/0.520.

## Interpretation

- The predicted "unadapted policy contracts under hidden relations"
  (kappa(S->D) < kappa(S->S)) is NOT observed: kappa(S->D) = 0.518 >
  kappa(S->S) = 0.402.
- Coherent alternative pattern: **cross-transfer RAISES reinforce kappa in
  both test modes** (D->S > S->S and S->D > D->D). A mismatched policy behaves
  homogeneously regardless of deal (uniformly low-skill play), so its
  return-weighted gradients agree more across deals; a matched policy's
  performance varies with the deal, so its gradients disagree more. The
  reinforce field is dominated by return variance in this protocol.
- softq/expq show the reverse (in-mode higher), and awr is flat at ~0.50.

## Protocol limitation

Each rollout is a *mixture* of assignments (the hidden team depends on the
random deal, and 30 episodes span many different teams). kappa between two
random mixtures measures cross-deal agreement, not a clean
relation-conditioned contrast. The decisive refinement is to FORCE one
assignment per rollout (fix the red-A holder per deal) and compare
assignment A vs assignment B directly.

Run: `python experiments/common_basis/cross_transfer/run_cross_transfer.py`
Results: `data/kappa/cross_transfer/results.json`
