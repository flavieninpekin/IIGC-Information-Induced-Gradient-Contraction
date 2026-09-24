# INVALID — do not cite

`reveal_kappa.json` in this directory is invalid (marked 2026-09-23 during the
ICLR 2027 environment audit). Reasons:

- `run_overcooked_reveal.py` applied the visibility mask only when
  `len(obs) == 99`, but the static observation is 98-dimensional (96 state
  features + 2 partner one-hot bits; `PARTNER_TYPES` has two entries). The mask
  therefore never executed (runtime probe: `mask_changed=False`), so the file
  is a no-mask curve.
- The field computed there (`reinforce_grad`) is the value field
  `-sum_t V(s_t)`, not a REINFORCE/advantage field, and the protocol never
  switches partners (`RevealMaskEnv` defaults to `mode='static'`).
- Consequence: the flat `kappa(p)` curve must not be cited as evidence about
  teammate visibility.

Superseded by the switching-preserving measurements in
`data/kappa/server_tasks/results/oc_field_axis.json` and
`oc_switch_kappa.json`.
