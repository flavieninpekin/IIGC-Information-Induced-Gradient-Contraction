# INVALID — do not cite

`results.json` (21-level visibility grid) was measured before the 510K
action-mask fix: `run_510k_reveal.py` sampled from unmasked distributions, so
the environment silently replaced executed actions with random legal plays
(the same defect measured at ~90% replacement in the paired field-axis
protocol). The script now passes `action_masks` (commit `1f10f12`), but the
grid has not been re-run.

Marked 2026-09-23. Use
`data/kappa/server_tasks/results/510k_field_axis.json` (mask-respecting, paired
protocol) for 510K evidence.
