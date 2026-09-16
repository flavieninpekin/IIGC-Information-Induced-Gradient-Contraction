# IIGC - Hidden-heterogeneity gradient audit

This repository contains the experiments and analysis for hidden-condition
gradient aggregation, kappa measurement, and objective-dependent cancellation.
The current research scope is narrower than the original PG/value reversal
story: kappa diagnoses retention of a specified condition signal and is not a
general performance predictor.

The single source of truth for the ICLR 2027 decision, canonical definitions,
claim status, evidence priority, and blockers is
`paper/ICLR2027_READINESS.md`. The canonical metric implementation is
`src/iigc/metrics/kappa.py` (`kappa_mix`); refreshed numbers across Toy,
Overcooked, and 510K are collected in `notes/canonical_metric_results.md`.

Use `folder_guideline.md` for the directory layout. The `notes/` and
`paper/drafts/` directories retain historical reasoning and superseded claims;
they are not automatically valid paper text. The previous paper repo
(`AAAI2027-510k-clear`, kept here as a reference) should be treated as related
work and checked for overlap before any submission.
