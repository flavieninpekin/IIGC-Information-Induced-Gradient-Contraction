"""K-way hidden-mixture geometry checks.

This is the dimension-aware replacement for the S3-only experiment.  It keeps
the matching bandit but varies the number of hidden conditions K.  The script
checks three facts:

1. Uniform matching mixtures cancel the linear/expq field for K=2, 3, and 4.
2. Direction dependence appears once the condition simplex has dimension at
   least two, without requiring a non-abelian group.
3. Softq channels can interfere for K=2 as well as K=3 when the condition
   mixture is asymmetric.

Two normalizations are reported. ``kappa_uniform_ref`` uses the historical
uniform reference energy and may exceed one for an asymmetric mixture.
``kappa_mixture_ref`` uses the actual mixture energy and is bounded by one.

Output: data/kappa/toy_fields/kway_geometry.json
"""
import json
import os

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "data", "kappa", "toy_fields", "kway_geometry.json")
R = 1.0


def softmax_np(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z))
    return e / e.sum()


def q_matrix(k):
    return 2.0 * np.eye(k, dtype=float)


def expected_grads(z, k, field, alpha=1.0):
    """Return one exact logit gradient per hidden condition."""
    qs = q_matrix(k)
    grads = []
    for q in qs:
        zt = torch.tensor(np.asarray(z)[:k], dtype=torch.double,
                          requires_grad=True)
        pi = torch.softmax(zt, dim=0)
        qt = torch.tensor(q, dtype=torch.double)
        if field == "expq":
            loss = -(pi * qt).sum()
        elif field == "softq":
            loss = (pi * (alpha * torch.log(pi) - qt)).sum()
        else:
            raise ValueError(field)
        grads.append(torch.autograd.grad(loss, zt)[0].detach().numpy())
    return np.asarray(grads)


def kappa_values(grads, p):
    """Return shared/contrast energies under two denominator conventions."""
    grads = np.asarray(grads, dtype=float)
    p = np.asarray(p, dtype=float)
    mixed = p @ grads
    shared = float(mixed @ mixed)
    uniform_energy = float(np.mean(np.sum(grads * grads, axis=1)))
    mixture_energy = float(np.sum(p[:, None] * grads * grads))
    contrast = float(np.sum(p[:, None] * (grads - mixed) ** 2))
    return {
        "E_shared": shared,
        "E_contrast_p": contrast,
        "E_uniform": uniform_energy,
        "E_mixture": mixture_energy,
        "kappa_uniform_ref": shared / max(uniform_energy, 1e-300),
        "kappa_mixture_ref": shared / max(mixture_energy, 1e-300),
    }


def expq_closed(z, p, k):
    """Closed form for the expq gradient of the K-way matching bandit."""
    pi = softmax_np(np.asarray(z)[:k])
    p = np.asarray(p, dtype=float)
    center = float(p @ pi)
    mixed = -2.0 * R * pi * (p - center)
    gs = np.asarray([
        -2.0 * R * pi * (np.eye(k)[i] - pi[i])
        for i in range(k)
    ])
    return mixed, gs


def valid_fixed_l1_points(k, distance, rng, n=200):
    """Sample simplex points exactly distance from uniform in L1 norm."""
    center = np.ones(k) / k
    points = []
    while len(points) < n:
        direction = rng.normal(size=k)
        direction -= direction.mean()
        l1 = np.abs(direction).sum()
        if l1 < 1e-12:
            continue
        step = distance / l1
        candidate = center + step * direction
        if np.min(candidate) >= 0.0:
            points.append(candidate)
    return np.asarray(points)


def summarize_direction_test(z, k, distance=0.45, n=200, seed=41):
    rng = np.random.default_rng(seed)
    points = valid_fixed_l1_points(k, distance, rng, n=n)
    grads = expected_grads(z, k, "expq")
    values = [kappa_values(grads, p) for p in points]
    uniform = np.asarray([x["kappa_uniform_ref"] for x in values])
    mixture = np.asarray([x["kappa_mixture_ref"] for x in values])
    return {
        "k": k,
        "l1_distance": distance,
        "n_directions": len(points),
        "uniform_ref": {
            "min": float(uniform.min()),
            "max": float(uniform.max()),
            "ratio": float(uniform.max() / max(uniform.min(), 1e-300)),
        },
        "mixture_ref": {
            "min": float(mixture.min()),
            "max": float(mixture.max()),
            "ratio": float(mixture.max() / max(mixture.min(), 1e-300)),
        },
    }


def alpha_scan(z, p, k, alphas):
    rows = []
    for alpha in alphas:
        vals = kappa_values(expected_grads(z, k, "softq", alpha), p)
        rows.append({"alpha": float(alpha), **vals})
    return rows


def main():
    z = np.array([0.5, -0.2, -0.3, -0.1])
    out = {
        "setup": "K-way matching bandit with exact expected logit gradients",
        "validation": {},
        "uniform_cancellation": {},
        "direction_tests": {},
        "alpha_scans": {},
    }

    worst = 0.0
    for k in (2, 3, 4):
        rng = np.random.default_rng(100 + k)
        for _ in range(100):
            zk = rng.normal(size=k)
            p = rng.dirichlet(np.ones(k))
            mixed_auto = p @ expected_grads(zk, k, "expq")
            mixed_closed, _ = expq_closed(zk, p, k)
            worst = max(worst, float(np.max(np.abs(mixed_auto - mixed_closed))))
    out["validation"]["expq_closed_max_abs_err"] = worst

    for k in (2, 3, 4):
        p = np.ones(k) / k
        expq = kappa_values(expected_grads(z, k, "expq"), p)
        softq = kappa_values(expected_grads(z, k, "softq", alpha=1.0), p)
        out["uniform_cancellation"][str(k)] = {
            "expq": expq,
            "softq_alpha1": softq,
        }

    for k in (3, 4):
        out["direction_tests"][str(k)] = summarize_direction_test(z, k)

    alphas = np.logspace(-3, 2, 101)
    out["alpha_scans"]["K2_p_0.8_0.2"] = alpha_scan(
        z, np.array([0.8, 0.2]), 2, alphas)
    out["alpha_scans"]["K3_p_0.7_0.2_0.1"] = alpha_scan(
        z, np.array([0.7, 0.2, 0.1]), 3, alphas)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"expq closed max abs err = {worst:.3e}")
    for k in (2, 3, 4):
        row = out["uniform_cancellation"][str(k)]
        print(f"K={k}: expq uniform kappa="
              f"{row['expq']['kappa_mixture_ref']:.3e}; "
              f"softq alpha=1="
              f"{row['softq_alpha1']['kappa_mixture_ref']:.4f}")
    for name, rows in out["alpha_scans"].items():
        best = min(rows, key=lambda x: x["kappa_mixture_ref"])
        print(f"{name}: min mixture-ref kappa="
              f"{best['kappa_mixture_ref']:.6f} at alpha={best['alpha']:.4g}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
