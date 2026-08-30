"""Reveal curves: prove they are the closed-form RATIONAL function, not
exponentials, and that their coefficients are computed from the net.

Exact formula for a trained net (mirror bandit, shared or block params):
  ga_h = -gb_h  (exact, Q_A = -Q_B)
  ga = (1-p) ga_h + p ga_r ;  gb = (1-p) gb_h + p gb_r
  kappa(p) = p^2 A / [ (1-p)^2 E_hid + p^2 B + p(1-p) C_x ]
    A    = ||(ga_r + gb_r)/2||^2
    E_hid= ||ga_h||^2
    B    = (||ga_r||^2 + ||gb_r||^2)/2
    C_x  = ga_h . (ga_r - gb_r)          [cross term; =0 iff block-orthogonal]

Checks:
  C1. closed-form (with net-computed A/B/C_x/E_hid) == direct kappa_of
      measurement, machine precision, for every p.
  C2. exponential fit loses badly vs rational fit (SSE comparison).
  C3. C_x != 0 -> the shared-net curve differs from the block-orthogonal
      prediction; quantify.
Output: paper/figures/reveal_kappa_rational.png (measured vs closed form)
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
DATA = json.load(open(os.path.join(ROOT, 'data', 'kappa', 'toy_fields',
                                   'reveal_training.json')))
OUT_PNG = os.path.join(ROOT, 'paper', 'figures', 'reveal_kappa_rational.png')

PS = np.array(DATA['p'])
Q_A = np.array([1.0, -1.0])
Q_B = np.array([-1.0, 1.0])
OBS_HID = np.array([1.0, 0.0, 0.0])
OBS_A = np.array([0.0, 1.0, 0.0])
OBS_B = np.array([0.0, 0.0, 1.0])
SEED = 41


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 2))

    def logits(self, obs):
        return self.net(torch.tensor(obs, dtype=torch.double))


def train(net, p, seed, n_ep=4000, batch=64, lr=0.05):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(n_ep // batch):
        obs_b, act_b, rew_b = [], [], []
        for _ in range(batch):
            r = int(rng.integers(0, 2))
            obs = (OBS_A if r == 0 else OBS_B) if rng.random() < p else OBS_HID
            probs = torch.softmax(net.logits(obs), 0).detach().numpy()
            a = int(rng.choice(2, p=probs))
            rew_b.append((Q_A if r == 0 else Q_B)[a])
            obs_b.append(obs)
            act_b.append(a)
        obs_t = torch.tensor(np.array(obs_b), dtype=torch.double)
        acts_t = torch.tensor(act_b, dtype=torch.long)
        rew_t = torch.tensor(rew_b, dtype=torch.double)
        lp = torch.log_softmax(net.logits(obs_t), -1)
        loss = -(lp[range(len(acts_t)), acts_t] * rew_t).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def grads(net):
    out = {}
    for name, obs, q in [('hid', OBS_HID, Q_A), ('hidB', OBS_HID, Q_B),
                         ('a', OBS_A, Q_A), ('b', OBS_B, Q_B)]:
        net.zero_grad()
        pi = torch.softmax(net.logits(obs), 0)
        qt = torch.tensor(q, dtype=torch.double)
        (-(pi * qt).sum()).backward()
        out[name] = np.concatenate(
            [p.grad.detach().numpy().flatten() for p in net.parameters()])
    return out


def kappa_direct(gs, p):
    ga = (1 - p) * gs['hid'] + p * gs['a']
    gb = (1 - p) * gs['hidB'] + p * gs['b']
    m = (ga + gb) / 2
    return (m @ m) / max((ga @ ga + gb @ gb) / 2, 1e-300)


def kappa_closed(gs, p):
    ga_h, gb_h, ga_r, gb_r = gs['hid'], gs['hidB'], gs['a'], gs['b']
    A = float(((ga_r + gb_r) / 2) @ ((ga_r + gb_r) / 2))
    E_hid = float(ga_h @ ga_h)
    B = float((ga_r @ ga_r + gb_r @ gb_r) / 2)
    C_x = float(ga_h @ (ga_r - gb_r))
    den = (1 - p) ** 2 * E_hid + p ** 2 * B + p * (1 - p) * C_x
    return p ** 2 * A / max(den, 1e-300), A, E_hid, B, C_x


def fit_exp(p, k):
    # k = a * (1 - exp(-b p)), grid search a in (0,1], b>0
    best = None
    for a in np.linspace(0.01, 1.5, 300):
        for b in np.linspace(0.1, 20, 200):
            pred = a * (1 - np.exp(-b * p))
            sse = float(np.sum((pred - k) ** 2))
            if best is None or sse < best[0]:
                best = (sse, a, b)
    return best


def fit_rational(p, k):
    # k = p^2 A / [(1-p)^2 E + p^2 B + p(1-p) C], free A,B,C,E>0
    best = None
    for A in np.linspace(0.01, 2, 200):
        for E in np.linspace(0.01, 5, 250):
            for B in np.linspace(0.01, 2, 200):
                den = (1 - p) ** 2 * E + p ** 2 * B
                pred = np.where(den > 0, p ** 2 * A / den, 0.0)
                sse = float(np.sum((pred - k) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, A, E, B)
    return best


def main():
    # fixed-policy net (trained at p=1)
    net = Net().double()
    net = train(net, 1.0, SEED)
    gs = grads(net)

    print('C3 cross term:')
    _, A, E_hid, B, C_x = kappa_closed(gs, 0.5)
    print(f'  A={A:.4f}  E_hid={E_hid:.4f}  B={B:.4f}  C_x={C_x:.4f}')
    print(f'  |C_x|/B = {abs(C_x) / B:.3f}  (0 => block-orthogonal shape)')

    print('\nC1 closed-form == direct measurement (machine precision):')
    worst = 0.0
    for p in PS:
        k_d = kappa_direct(gs, float(p))
        k_c, *_ = kappa_closed(gs, float(p))
        worst = max(worst, abs(k_d - k_c))
    print(f'  max |direct - closed| over 21 p = {worst:.2e}')

    # measured fixed-policy curve (same seed path as the JSON: seed 41)
    ks_meas = [kappa_direct(gs, float(p)) for p in PS]

    print('\nC2 rational vs exponential fit on measured fixed-policy curve:')
    re = fit_exp(PS, np.array(ks_meas))
    rr = fit_rational(PS, np.array(ks_meas))
    print(f'  exponential SSE = {re[0]:.5f}  (a={re[1]:.2f}, b={re[2]:.2f})')
    print(f'  rational   SSE = {rr[0]:.5f}  (A={rr[1]:.2f}, E={rr[2]:.2f}, '
          f'B={rr[3]:.2f})')
    print(f'  rational fit beats exponential by '
          f'{re[0] / max(rr[0], 1e-12):.0f}x')

    # figure: measured points vs net-computed closed form (no fitting)
    kc = [kappa_closed(gs, float(p))[0] for p in PS]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(PS, ks_meas, 'o-', ms=6, lw=1.6, label='measured (fixed policy, '
            'p=1 trained, seed 41)')
    ax.plot(PS, kc, 'x--', ms=6, lw=1.4, color='crimson',
            label='closed form (A/B/C$_x$/E_hid computed from net)')
    ax.set_xlabel('reveal probability $p$')
    ax.set_ylabel(r'$\kappa(p)$')
    ax.set_title('Not exponentials: rational closed form, coefficients '
                 'computed from the net\n(max |direct - closed| = '
                 f'{worst:.1e})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print('\nsaved:', OUT_PNG)


if __name__ == '__main__':
    main()