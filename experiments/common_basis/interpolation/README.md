# Experiment 2: interpolation spectrum (AWR / temperature / DPG)

Goal: place algorithms on a continuous mode-seeking <-> mean-seeking axis and
show kappa interpolates between the two families:

- AWR: TD-consistent value + advantage-weighted policy (expected between).
- Softmax-Gibbs temperature sweep: as tau grows, the actor objective becomes
  more mean-seeking; kappa_actor(tau) should transition PG-like -> TD-like.
- DPG: gradient of Q through the policy parameters (TD-value chain rule).

If kappa moves continuously with the interpolation knob, the reversal is a
property of objective geometry, not a definitional artifact.
