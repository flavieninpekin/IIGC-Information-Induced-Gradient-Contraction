"""Structure-axis interpolation (Paper 1, sketch note item #5).

Z2 bandit with ASYMMETRIC targets: Q_A = (r, -r), Q_B = (-s, +s), s/r in
[0,1]. s/r = 1 is the pure mirror (everything dies for odd fields); s -> 0
removes condition B's conflict.

Part A (exact, canonical expected gradients):
  expq/reinforce closed form  kappa(r,s) = (r-s)^2 / (2 (r^2 + s^2))
  (survival revives with asymmetry; verify vs autograd, machine precision)

Part B (value field, trained Q-net, E3 protocol):
  Does kappa_TD revive with asymmetry like PG fields, or collapse at its own
  fixed point? Hand derivation: at the converged relation-mean Q-bar, per-
  condition TD residuals satisfy e_B = -e_A for EVERY s/r under a uniform
  mixture => g_B = -g_A => kappa_TD -> 0 for all asymmetry degrees.
  Prediction to test: kappa_TD ~ 0 post-convergence at all s/r (contrast
  with expq revival); transiently nonzero during training.

Output: data/kappa/toy_fields/structure_axis.json
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'data', 'kappa', 'toy_fields', 'structure_axis.json')

R = 1.0
S_RATIOS = [1.0, 0.75, 0.5, 0.25, 0.0]
SEEDS = [41, 42, 43, 44, 45]


# ----------------------------------------------------------------------------
# Part A: exact expected gradients (canonical definition)
# ----------------------------------------------------------------------------

def q_vec_a(r):
    return np.array([r, -r])


def q_vec_b(s):
    return np.array([-s, s])


def expected_grad(z, q, field, alpha=1.0):
    zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
    pi = torch.softmax(zt, dim=0)
    qt = torch.tensor(q, dtype=torch.double)
    if field == 'expq':
        loss = -(pi * qt).sum()
    elif field == 'softq':
        lp = torch.log(torch.clamp(pi, min=1e-300))
        loss = (pi * (alpha * lp - qt)).sum()
    elif field == 'awr':
        tau = 1.0
        v = (pi * qt).sum()
        lp = torch.log(torch.clamp(pi, min=1e-300))
        ws = torch.exp((qt - v) / tau)
        total = torch.zeros_like(zt)
        pi_d = pi.detach()
        for a in range(2):
            if zt.grad is not None:
                zt.grad.zero_()
            (-lp[a] * ws[a]).backward(retain_graph=True)
            total = total + float(pi_d[a]) * zt.grad.detach().clone()
        return total.numpy()
    else:
        raise ValueError(field)
    loss.backward()
    return zt.grad.detach().numpy()


def kappa_pair(ga, gb):
    m = (ga + gb) / 2.0
    d = (ga - gb) / 2.0
    es, ec = float(m @ m), float(d @ d)
    return es / max(es + ec, 1e-300)


def closed_kappa_expq(s):
    return (R - s) ** 2 / (2 * (R ** 2 + s ** 2))


def part_a(z):
    rows = {}
    worst = 0.0
    for s in S_RATIOS:
        k_closed = closed_kappa_expq(s)
        k_auto = kappa_pair(expected_grad(z, q_vec_a(R), 'expq'),
                            expected_grad(z, q_vec_b(s), 'expq'))
        rows[str(s)] = {'closed': k_closed, 'autograd': k_auto,
                        'abs_err': abs(k_closed - k_auto)}
        worst = max(worst, abs(k_closed - k_auto))
        k_softq = kappa_pair(expected_grad(z, q_vec_a(R), 'softq'),
                             expected_grad(z, q_vec_b(s), 'softq'))
        rows[str(s)]['softq_alpha1'] = k_softq
    return {'z': z.tolist(), 'kappa_expq_curve': {k: v['closed'] for k, v in rows.items()},
            'rows': rows, 'max_abs_err_closed_vs_autograd': worst}


# ----------------------------------------------------------------------------
# Part B: trained Q-net TD field (E3 protocol, asymmetric targets)
# ----------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self, in_dim=1, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def make_data(s, n=2000, seed=0, gamma=0.9):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rel = int(rng.integers(0, 2))
        q = q_vec_a(R) if rel == 0 else q_vec_b(s)
        a = int(rng.integers(0, 2))
        rw = q[a]
        nobs = np.array([1.0], dtype=np.float32)
        rows.append((np.array([1.0], dtype=np.float32), a, rw, nobs))
    return rows


def td_grad(net, data, gamma=0.9):
    obs = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    acts = torch.tensor([d[1] for d in data], dtype=torch.long)
    rews = torch.tensor([d[2] for d in data], dtype=torch.float32)
    nobs = torch.tensor(np.array([d[3] for d in data]), dtype=torch.float32)
    with torch.no_grad():
        target = rews + gamma * net(nobs).max(dim=-1).values
    loss = F.mse_loss(net(obs).gather(1, acts[:, None]).squeeze(1), target)
    net.zero_grad()
    loss.backward()
    gv = torch.cat([p.grad.detach().flatten()
                    for p in net.parameters() if p.grad is not None])
    return gv.numpy()


def train_net(s, n_train, seed, gamma=0.9):
    torch.manual_seed(seed)
    net = QNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    data = make_data(s, n_train, seed, gamma)
    obs = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    acts = torch.tensor([d[1] for d in data], dtype=torch.long)
    rews = torch.tensor([d[2] for d in data], dtype=torch.float32)
    nobs = torch.tensor(np.array([d[3] for d in data]), dtype=torch.float32)
    for _ in range(3000):
        with torch.no_grad():
            target = rews + gamma * net(nobs).max(dim=-1).values
        loss = F.mse_loss(net(obs).gather(1, acts[:, None]).squeeze(1), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def part_b(s_ratios=S_RATIOS, seeds=SEEDS, n_train=2000, n_meas=1000):
    out = {}
    for s in s_ratios:
        ks_final, ks_mid = [], []
        for seed in seeds:
            net = train_net(s, n_train, seed)
            data_a = make_data(s, n_meas, seed + 1000, 0.9)
            data_b = make_data(s, n_meas, seed + 2000, 0.9)
            rel_a = [d for d in data_a if d[2] == d[2]]  # keep all (unused)
            # condition-specific batches: build per-condition meas data
            da = [(np.array([1.0], dtype=np.float32), a, q_vec_a(R)[a],
                   np.array([1.0], dtype=np.float32))
                  for a in range(2) for _ in range(n_meas // 2)]
            db = [(np.array([1.0], dtype=np.float32), a, q_vec_b(s)[a],
                   np.array([1.0], dtype=np.float32))
                  for a in range(2) for _ in range(n_meas // 2)]
            ga = td_grad(net, da)
            gb = td_grad(net, db)
            ks_final.append(kappa_pair(ga, gb))
        out[str(s)] = {'kappa_td_mean': float(np.mean(ks_final)),
                       'kappa_td_std': float(np.std(ks_final)),
                       'kappa_td_seeds': ks_final}
        print(f's/r={s}: kappa_TD = {np.mean(ks_final):.4f} +- '
              f'{np.std(ks_final):.4f}')
    return out


def transient_kappa_td(s, steps=(50, 200, 800, 3000), seeds=SEEDS[:3],
                       gamma=0.9):
    """kappa_TD measured DURING training: transient survival then collapse."""
    torch.set_num_threads(1)
    out = {}
    for step in steps:
        ks = []
        for seed in seeds:
            net = QNet()
            opt = torch.optim.Adam(net.parameters(), lr=1e-2)
            data = make_data(s, 2000, seed, gamma)
            obs = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
            acts = torch.tensor([d[1] for d in data], dtype=torch.long)
            rews = torch.tensor([d[2] for d in data], dtype=torch.float32)
            nobs = torch.tensor(np.array([d[3] for d in data]), dtype=torch.float32)
            for it in range(step):
                with torch.no_grad():
                    target = rews + gamma * net(nobs).max(dim=-1).values
                loss = F.mse_loss(net(obs).gather(1, acts[:, None]).squeeze(1),
                                  target)
                opt.zero_grad()
                loss.backward()
                opt.step()
            da = [(np.array([1.0], dtype=np.float32), a, q_vec_a(R)[a],
                   np.array([1.0], dtype=np.float32))
                  for a in range(2) for _ in range(500)]
            db = [(np.array([1.0], dtype=np.float32), a, q_vec_b(s)[a],
                   np.array([1.0], dtype=np.float32))
                  for a in range(2) for _ in range(500)]
            ga = td_grad(net, da)
            gb = td_grad(net, db)
            ks.append(kappa_pair(ga, gb))
        out[str(step)] = {'mean': float(np.mean(ks)), 'std': float(np.std(ks)),
                          'seeds': ks}
        print(f'  s/r={s} step={step}: kappa_TD = {np.mean(ks):.4f} +- '
              f'{np.std(ks):.4f}')
    return out


def main():
    z = np.array([0.4, -0.4])
    pa = part_a(z)
    print('Part A (expq closed form vs autograd):')
    for k, v in pa['rows'].items():
        print(f'  s/r={k}: closed={v["closed"]:.4f} auto={v["autograd"]:.4f} '
              f'(err {v["abs_err"]:.1e}) softq={v["softq_alpha1"]:.4f}')
    print(f'  max abs err = {pa["max_abs_err_closed_vs_autograd"]:.3e}')

    pb = part_b()
    print('TD transient (s/r=0.5):')
    pt = transient_kappa_td(0.5)
    out = {'part_a': pa, 'part_b': pb, 'transient_td_s05': pt,
           'note': ('expq: kappa revives with asymmetry (r-s)^2/2(r^2+s^2); '
                    'TD: collapses at its own converged fixed point for all '
                    's/r (e_B = -e_A under uniform mixture); transient '
                    'survival during training explains real-env value-field '
                    'readings')}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
