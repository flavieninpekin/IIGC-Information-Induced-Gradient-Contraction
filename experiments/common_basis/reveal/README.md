# Continuous reveal experiments (Toy + 510K)

## Toy prototype

`run_toy_reveal.py`: REINFORCE on RevealToyEnv, forced & mixture protocols.
Validates the measurement machinery (variance decomposition).

| p | kappa_forced | kappa_mix | var_within |
|---|--------------|-----------|------------|
| 0.00 | 0.152 | 0.721 | 0.69 |
| 0.30 | 0.235 | 0.932 | **2.93**  |
| 0.50 | 0.218 | 0.911 | 2.18 |
| 0.70 | 0.199 | 0.911 | 2.05 |
| 0.85 | 0.154 | 0.947 | 1.33 |
| 1.00 | 0.105 | 0.883 | 0.68 |

- Consistency check: law-of-total-variance holds to ~1e-6.
- sigma^2 (var_within) peaks at p=0.3 — "confused at intermediate reveal".
- NO kappa dip under either protocol (forced: monotonic; mix: flat).

## 510K refined (Phase 1–3 + full grid)

`run_510k_reveal.py`: 126 MaskablePPO models (21 levels x 6 seeds),
mixture protocol, per-episode team-label recording + variance decomposition.

**Full grid result (n=6/point)** — see `data/kappa/510k_reveal/`:

| p | kappa | ±std | | p | kappa | ±std |
|---|-------|------|---|-----|-------|------|
| 0.00 | 0.548 | 0.055 | | 0.55 | 0.485 | 0.080 |
| 0.10 | 0.501 | 0.121 | | 0.65 | 0.560 | 0.113 |
| 0.20 | 0.462 | 0.074 | | 0.75 | 0.469 | 0.120 |
| 0.30 | 0.521 | 0.045 | | 0.85 | 0.480 | 0.033 |
| 0.40 | 0.518 | 0.075 | | 0.95 | 0.529 | 0.054 |
| 0.50 | 0.533 | 0.046 | | 1.00 | 0.549 | 0.108 |

**Statistical verdict (definitive)**:
- global mean 0.500, std 0.088; between-level spread (0.031) < within-seed
  noise (0.077)
- **ANOVA F=0.756, p=0.759** — reveal fraction does NOT explain kappa
- per-level Bonferroni t-tests all p=1.000; 75% vs neighbors p=0.608;
  0/20 adjacent pairs significant
- **Conclusion: kappa(p) is statistically flat (~0.50). The original "75%
  dip" was an n=2 sampling artifact.** The "extremum prediction / almost-right
  penalty" framing does not survive adequate sampling.
- Secondary finding: kappa ~ 0.5 everywhere (orthogonal-rollout value) =>
  the mixture protocol is dominated by per-episode gradient noise
  (var ~1.6e6 vs team contrast ~2.5e4), i.e. 510K's mixture-protocol kappa
  is not a sensitive probe.

## Next steps

- The energy/variance decomposition (kappa = E_shared/(E_shared+E_contrast+
  sigma^2), see `notes/reveal_theory.md`) is a candidate meaningful finding:
  verify the identity on Toy forced assignment by measuring E_shared,
  E_contrast, sigma^2 and checking against measured kappa.

## Files

| Script | Description |
|--------|-------------|
| `run_toy_reveal.py` | Toy forced & mixture reveal |
| `run_510k_reveal.py` | 510K baseline + refined measurement |
| `train_reveal.py` | Single-model PPO training |
| `launch_training.py` | Parallel training launcher |
