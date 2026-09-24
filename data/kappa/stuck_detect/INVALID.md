# INVALID / stale — do not cite

Marked 2026-09-23 during the ICLR 2027 environment audit.

- `overcooked_decomp.json`, `overcooked_memory_eval.json`: measured with
  `_force_partner`, which disables mid-episode role switching. Dynamic-trained
  policies are out of distribution under that protocol (they collect zero
  reward, so the decomposition is trivially zero). Every "dynamic stuck /
  all-zero" readout here is a protocol artefact, not a stuck state.
- `forced_decomp.json`: 510K forced decomposition measured before the
  action-mask fix. `run_510k_forced_decomp.py` now passes `action_masks`
  (commit `1f10f12`), but this file has not been re-run.
- `sensitivity.json`: valid (mask-aware sampling in
  `run_510k_sensitivity.py`).
- `figures/`: generated from the files above; treat as historical.

Superseded by the switching-preserving, start-partner measurements in
`data/kappa/server_tasks/results/oc_switch_kappa.json`.
