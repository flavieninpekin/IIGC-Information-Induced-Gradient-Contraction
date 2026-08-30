"""Task 1: revive the reveal experiment as a prediction-validation figure.

Two curves, one figure:
  (a) per-p trained: train a shared-MLP policy at each reveal level p
      (mirror bandit, expq/reinforce objective, analytic Q), measure kappa(p)
      at the trained policy  ->  "training-measurement closed loop"
  (b) fixed-policy measurement basis: train at p=1 only, then measure kappa
      at ALL p levels with that fixed policy  ->  Prop 10 predicts monotone
      increasing (shared net: verified numerically in verify_reveal_kappa V2)

Environment: 2 conditions (mirror Q=(+1,-1)/(-1,+1)), 3 observation states
(hidden / rev-A / rev-B), shared MLP 3->16->2. Policy gradient via analytic
expected gradients (no sampling noise in measurement).

Output:
  data/kappa/toy_fields/reveal_training.json
  paper/figures/reveal_kappa_training.png
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_JSON = os.path.join(ROOT, 'data', 'kappa', 'toy_fields',
                        'reveal_training.json')
OUT_PNG = os.path.join(ROOT, 'paper', 'figures', 'reveal_kappa_training.png')

PS = np.round(np.linspace(0.0, 1.0, 21), 2)
SEEDS = [41, 42, 43]
Q_A = np.array([1.0, -1.0])
Q_B = np.array([-1.0, 1.0])
OBS_HID = np.array([1.0, 0.0, 0.0])
OBS_A = np.array([0.0, 1.0, 0.0])
OBS_B = np.array([0.0, 0.0, 1.0])
N_TRAIN_EP = 4000
BATCH = 64
LR = 0.05


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 2))

    def logits(self, obs):
        return self.net(torch.tensor(obs, dtype=torch.double))


def train(net, p, seed):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(net.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    for it in range(N_TRAIN_EP // BATCH):
        obs_b, act_b, rew_b = [], [], []
        for _ in range(BATCH):
            r = int(rng.integers(0, 2))
            if rng.random() < p:
                obs = OBS_A if r == 0 else OBS_B
            else:
                obs = OBS_HID
            logits = net.logits(obs)
            probs = torch.softmax(logits, 0).detach().numpy()
            a = int(rng.choice(2, p=probs))
            q = Q_A if r == 0 else Q_B
            rew = q[a]
            obs_b.append(obs)
            act_b.append(a)
            rew_b.append(rew)
        obs_t = torch.tensor(np.array(obs_b), dtype=torch.double)
        acts_t = torch.tensor(act_b, dtype=torch.long)
        rew_t = torch.tensor(rew_b, dtype=torch.float32)
        lp = torch.log_softmax(net.logits(obs_t), -1)
        loss = -(lp[range(len(acts_t)), acts_t] * rew_t).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def expected_grad(net, obs, q):
    """-grad_theta <pi(obs), q>  (expq/reinforce expected gradient)."""
    for prm in net.parameters():
        prm.requires_grad = True
    net.zero_grad()
    logits = net.logits(obs)
    pi = torch.softmax(logits, 0)
    qt = torch.tensor(q, dtype=torch.float32)
    (-(pi * qt).sum()).backward()
    g = torch.cat([p.grad.detach().flatten() for p in net.parameters()
                   if p.grad is not None])
    return g.numpy()


def kappa_of(net, p):
    """kappa(p) = ||(g_A+g_B)/2||^2 / avg(||g_A||^2,||g_B||^2)."""
    ga_h = expected_grad(net, OBS_HID, Q_A)
    gb_h = expected_grad(net, OBS_HID, Q_B)
    ga_r = expected_grad(net, OBS_A, Q_A)
    gb_r = expected_grad(net, OBS_B, Q_B)
    ga = (1 - p) * ga_h + p * ga_r
    gb = (1 - p) * gb_h + p * gb_r
    m = (ga + gb) / 2
    denom = (ga @ ga + gb @ gb) / 2
    return (m @ m) / max(denom, 1e-300)


def main():
    # (a) per-p trained closed loop
    per_p = {str(p): [] for p in PS}
    for p in PS:
        for seed in SEEDS:
            net = Net()
            net.double()
            # convert to double for consistency
            net = train(net.double(), float(p), seed)
            per_p[str(p)].append(kappa_of(net, float(p)))
    per_p_mean = [float(np.mean(per_p[str(p)])) for p in PS]
    per_p_std = [float(np.std(per_p[str(p)])) for p in PS]

    # (b) fixed-policy measurement basis: train at p=1, measure at all p
    fixed = {str(p): [] for p in PS}
    for seed in SEEDS:
        net = Net().double()
        net = train(net, 1.0, seed)
        for p in PS:
            fixed[str(p)].append(kappa_of(net, float(p)))
    fixed_mean = [float(np.mean(fixed[str(p)])) for p in PS]
    fixed_std = [float(np.std(fixed[str(p)])) for p in PS]

    # monotonicity checks
    def is_inc(v):
        return all(v[i] <= v[i + 1] + 1e-9 for i in range(len(v) - 1))

    out = {
        'p': PS.tolist(),
        'per_p_trained': {'mean': per_p_mean, 'std': per_p_std,
                          'monotone': is_inc(per_p_mean)},
        'fixed_policy_measured_all_p': {'mean': fixed_mean, 'std': fixed_std,
                                        'monotone': is_inc(fixed_mean)},
        'note': ('per_p_trained = train at each p then measure kappa(p); '
                 'fixed_policy = train at p=1 only, measure kappa at all p '
                 '(Prop 10 predicts monotone for the fixed-policy curve)'),
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=1)
    print('per-p trained monotone:', is_inc(per_p_mean),
          [round(x, 3) for x in per_p_mean])
    print('fixed-policy monotone:', is_inc(fixed_mean),
          [round(x, 3) for x in fixed_mean])

    # figure
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(PS, per_p_mean, yerr=per_p_std, fmt='o-', ms=5, lw=1.5,
                label='per-$p$ trained (closed loop)')
    ax.errorbar(PS, fixed_mean, yerr=fixed_std, fmt='s--', ms=5, lw=1.5,
                label='fixed policy ($p$=1 trained), measured at all $p$')
    ax.axhline(0.5, color='gray', ls=':', lw=1,
               label='old 510K mixed-protocol level (~0.5, noise-dominated)')
    ax.set_xlabel('reveal probability $p$')
    ax.set_ylabel(r'$\kappa(p)$')
    ax.set_title('Reveal experiment revived: kappa(p) prediction vs training')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print('saved:', OUT_JSON)
    print('saved:', OUT_PNG)


if __name__ == '__main__':
    main()