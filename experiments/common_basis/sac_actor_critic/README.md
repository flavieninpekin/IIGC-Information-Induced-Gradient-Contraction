# Experiment 1: SAC actor/critic split kappa

Goal: on the same SAC network and same rollouts, compute kappa on the
**actor** gradient (policy-gradient-like, mode-seeking) and on the **critic**
TD-loss gradient (mean-seeking). Hypothesis: actor shows SINGLE > DYNAMIC
(contracts), critic shows DYNAMIC > SINGLE (aligns) — demonstrating kappa is a
property of the update field, not of the algorithm family.

Assets: `src/iigc/env/discrete_sac.py` (actor + critic1/critic2), `src/iigc/algos/sac.py`
(training + kappa on actor gradient). Need to add a critic-gradient kappa path.
