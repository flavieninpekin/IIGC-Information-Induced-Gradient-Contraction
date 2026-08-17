# Experiment 1: SAC actor/critic split kappa

On the same SAC network and same rollouts, compute kappa on the **actor**
gradient (soft mode-seeking) and the **critic** TD-loss gradient (mean-seeking),
to test whether kappa is a function of the update field.

## Method

- Models: pretrained `data/models/510k_sac/{single,dynamic} × {seed41,42}`
- Two rollouts per model (30 episodes each, distinct deal seeds)
- `kappa_actor`  : SAC actor loss gradient (`DiscreteSAC.actor_gradient`)
- `kappa_critic` : TD-loss gradient on critic1 with SAC soft target
  (`DiscreteSAC.critic_gradient`, added for this experiment)
- Energy gate reported (kappa is undefined when gradient energy ~ 0)

## Result (preliminary, n = 2)

| mode    | kappa_actor        | kappa_critic       | E_actor | E_critic |
|---------|--------------------|--------------------|---------|----------|
| single  | 0.537 ± 0.068      | **0.598** ± 0.025  | 33      | 1.8e5    |
| dynamic | 0.589 ± 0.129      | **0.508** ± 0.039  | 179     | 3.0e6    |

## Interpretation

- The **simple hypothesis is NOT confirmed**: the critic does not align in
  DYNAMIC (S > D), and the soft actor is essentially flat (S < D by a small
  margin, within noise).
- The SAC actor's flat kappa is expected from the "intermediate" view: the
  discrete SAC actor is a soft-maximum over Q (temperature-controlled), not
  hard REINFORCE, so it sits between the two families.
- The critic contracting in DYNAMIC suggests the TD target is NOT simply
  aggregate-consistent: under hidden relations the value landscape is
  multi-modal across assignments, so batch TD gradients from different
  assignment-mixtures can still point apart.
- `kappa_actor` and `kappa_critic` live in different parameter spaces, so only
  the *within-field* S/D ordering is interpretable, not the absolute values.

## Limitations / next steps

- n = 2 and high variance; add more seeds / checkpoints.
- Each rollout is a *mixture* of assignments (the team depends on the random
  deal), which dilutes the relation-conditioned signal — prefer forced
  assignment rollouts.
- The decisive common-basis test is the interpolation spectrum (Exp 2):
  REINFORCE / AWR / soft-Q / DPG on the **same** actor parameters.

Run: `python experiments/common_basis/sac_actor_critic/run_sac_split.py`
Results: `data/kappa/common_basis_sac_split/results.json`
