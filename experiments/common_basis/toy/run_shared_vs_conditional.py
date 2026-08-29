"""Exact training comparison for hidden K-way matching objectives.

The experiment separates two questions that kappa alone cannot answer:

* A shared policy must compromise across hidden conditions.
* A condition-aware policy can keep a separate policy for each condition.

Training uses exact full-batch objectives, so the result is not affected by
rollout noise.  The same base initial logits are used for all models in a
seed; the conditional model starts with one copy per condition.
The linear objective is the alpha=0 baseline; softq uses the exact objective
from the K-way geometry script.

Output: data/kappa/compromise_training/results.json
"""
import argparse
import json
import os

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "kappa", "compromise_training")
R = 1.0
K = 3
Q = 2.0 * torch.eye(K, dtype=torch.float64)
SCENARIOS = {
    "uniform": np.array([1.0 / 3.0] * 3),
    "mild": np.array([0.6, 0.3, 0.1]),
    "strong": np.array([0.8, 0.15, 0.05]),
}
ALPHAS = [0.0, 0.2, 1.0, 2.0]


def exact_loss(logits, p, alpha):
    """Full-batch shared or conditional softq loss."""
    pi = torch.softmax(logits, dim=-1)
    log_pi = torch.log(pi)
    p_t = torch.as_tensor(p, dtype=torch.float64)
    if logits.ndim == 1:
        qbar = p_t @ Q
        return (pi * (alpha * log_pi - qbar)).sum()
    per_condition = (pi * (alpha * log_pi - Q)).sum(dim=1)
    return (p_t * per_condition).sum()


def policy_metrics(logits, p):
    p_t = torch.as_tensor(p, dtype=torch.float64)
    pi = torch.softmax(logits, dim=-1).detach().numpy()
    rewards = np.sum(pi * Q.numpy(), axis=1) if logits.ndim == 2 else None
    if rewards is None:
        rewards = pi @ Q.numpy()
        per_condition = rewards
    else:
        per_condition = rewards
    return {
        "policy": pi.tolist(),
        "per_condition_reward": np.asarray(per_condition).tolist(),
        "average_reward": float(p_t.numpy() @ np.asarray(per_condition)),
        "worst_condition_reward": float(np.min(per_condition)),
        "average_regret": float(2.0 - p_t.numpy() @ np.asarray(per_condition)),
        "worst_regret": float(2.0 - np.min(per_condition)),
    }


def initial_kappa(logits, p, alpha):
    """Measure condition-gradient retention in the current parameter space.

    For conditional logits, each condition has a separate parameter block, so
    this is a parameter-space diagnostic rather than a directly comparable
    shared-policy kappa.
    """
    grads = []
    for i, q in enumerate(Q):
        z = logits.detach().clone().requires_grad_(True)
        if z.ndim == 1:
            pi = torch.softmax(z, dim=-1)
            loss = (pi * (alpha * torch.log(pi) - q)).sum()
        else:
            pi = torch.softmax(z[i], dim=-1)
            loss = (pi * (alpha * torch.log(pi) - q)).sum()
        grads.append(torch.autograd.grad(loss, z)[0].detach().numpy())
    gs = np.asarray(grads).reshape(K, -1)
    mixed = np.asarray(p) @ gs
    uniform_energy = np.mean(np.sum(gs * gs, axis=1))
    mixture_energy = np.sum(np.asarray(p)[:, None] * gs * gs)
    return {
        "E_shared": float(mixed @ mixed),
        "E_uniform": float(uniform_energy),
        "E_mixture": float(mixture_energy),
        "kappa_uniform_ref": float((mixed @ mixed) /
                                    max(uniform_energy, 1e-300)),
        "kappa_mixture_ref": float((mixed @ mixed) /
                                    max(mixture_energy, 1e-300)),
        "mixed_gradient_norm": float(np.linalg.norm(mixed)),
    }


def train_one(p, alpha, conditional, seed, steps, lr):
    torch.manual_seed(seed)
    base = torch.randn(K) * 0.4
    initial = base.repeat(K, 1) if conditional else base
    logits = initial.clone().requires_grad_(True)
    initial_metrics = policy_metrics(logits, p)
    if not conditional:
        kappa_info = initial_kappa(logits, p, alpha)
    else:
        # The conditional model has separate parameter blocks; measure its
        # retention in that full parameter space for reference only.
        kappa_info = initial_kappa(logits, p, alpha)

    for _ in range(steps):
        loss = exact_loss(logits, p, alpha)
        grad = torch.autograd.grad(loss, logits)[0]
        with torch.no_grad():
            logits -= lr * grad
        logits = logits.detach().requires_grad_(True)

    final_metrics = policy_metrics(logits, p)
    return {
        "seed": seed,
        "conditional": conditional,
        "alpha": alpha,
        "initial": initial_metrics,
        "initial_kappa": kappa_info,
        "final": final_metrics,
        "final_loss": float(exact_loss(logits, p, alpha).detach()),
    }


def summarize(rows):
    def stats(values):
        values = np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std())}

    return {
        "n_seeds": len(rows),
        "average_reward": stats([r["final"]["average_reward"] for r in rows]),
        "worst_condition_reward": stats(
            [r["final"]["worst_condition_reward"] for r in rows]),
        "average_regret": stats([r["final"]["average_regret"] for r in rows]),
        "mixed_gradient_norm_initial": stats(
            [r["initial_kappa"]["mixed_gradient_norm"] for r in rows]),
        "kappa_mixture_initial": stats(
            [r["initial_kappa"]["kappa_mixture_ref"] for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args()

    raw = []
    summary = {}
    for scenario, p in SCENARIOS.items():
        summary[scenario] = {}
        for conditional in (False, True):
            for alpha in ALPHAS:
                key = f"{'conditional' if conditional else 'shared'}_alpha{alpha}"
                rows = [train_one(p, alpha, conditional, seed,
                                  args.steps, args.lr)
                        for seed in range(41, 41 + args.seeds)]
                raw.extend([{"scenario": scenario, **row} for row in rows])
                summary[scenario][key] = summarize(rows)
                s = summary[scenario][key]
                print(f"{scenario:7s} {key:24s} "
                      f"reward={s['average_reward']['mean']:.4f} "
                      f"worst={s['worst_condition_reward']['mean']:.4f} "
                      f"k0={s['kappa_mixture_initial']['mean']:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "config": {"K": K, "scenarios": {k: v.tolist()
                                           for k, v in SCENARIOS.items()},
                   "alphas": ALPHAS, "steps": args.steps, "lr": args.lr,
                   "n_seeds": args.seeds,
                   "kappa_note": "conditional kappa uses separate parameter blocks and is not directly comparable to shared kappa"},
        "summary": summary,
        "raw": raw,
    }
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved:", path)


if __name__ == "__main__":
    main()
