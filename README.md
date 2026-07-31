# IIGC — Common-basis kappa re-measurement

New paper project: explain the policy-gradient vs value-based kappa reversal
from the geometry of the update objective (mode-seeking vs mean-seeking),
measured under a common measurement basis (same network, same data, only the
objective changes).

The previous paper repo (`AAAI2027-510k-clear`, kept in this folder as a
read-only reference) established that kappa is estimator-specific; this project
builds on its environments, algorithms, and data to explain *why* the reversal
happens and to predict it.

See `folder_guideline.md` for the directory layout, `notes/` for the research
discussion, and `experiments/common_basis/design/` for the experiment design.
