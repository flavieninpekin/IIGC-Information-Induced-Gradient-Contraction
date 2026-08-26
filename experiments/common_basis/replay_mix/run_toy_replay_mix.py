"""E3: DQN replay-buffer relation-mix ablation on the Toy mirror bandit.

Hypothesis (from kappa_reversal_discussion.md §6): TD-field kappa should RISE
with the fraction of relation-B data in the replay buffer ("TD reconciliation"
= aggregation consistency).

Closed-form prediction on the Toy mirror bandit (constant obs, 2 actions):
the Q network cannot distinguish relations, so at mixture rho it converges to
the relation-mean Q: Q0 = 1-2rho, Q1 = 2rho-1 (any gamma; the bootstrap term
cancels in the residuals). The relation-conditional TD gradients then satisfy
g_B = -g_A * (1-rho)/rho, giving

    kappa_TD(rho) = (1-2rho)^2 / (2 (rho^2 + (1-rho)^2))

which DECREASES from 0.5 (rho=0) to exactly 0 (rho=0.5): on the mirror bandit
the TD field collapses at the aggregated Q. This rejects the "TD 调和" claim
for the clean mirror setting and is the bandit-level counterpart of the
Overcooked/510K result "TD alignment requires non-mirror structure".

Outputs:
  data/kappa/replay_mix/toy_mix_results.json
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'replay_mix')
os.makedirs(OUT_DIR, exist_ok=True)

RHOS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
N_TRANS = 2000          # training transitions per rho
N_MEAS = 1000           # relation-conditional transitions for measurement
TRAIN_STEPS = 800
LR = 5e-2
SEEDS = 3


class QNet(nn.Module):
    def __init__(self, obs_dim=1, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        return self.net(obs)


def sample_batch(n, rho, seed):
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, 1), dtype=np.float32)
    act = rng.integers(0, 2, size=n)
    rel = (rng.random(n) < rho).astype(int)   # 0=A, 1=B
    partner = rel
    rew = np.where(act == partner, 1.0, -1.0)
    return torch.FloatTensor(obs), torch.LongTensor(act), torch.FloatTensor(rew)


def td_loss(qnet, obs, act, rew, gamma=0.0):
    q = qnet(obs)
    qa = q[range(len(act)), act]
    with torch.no_grad():
        target = rew + gamma * q.max(dim=-1).values
    return ((qa - target) ** 2).mean()


def kappa_from_grads(gA, gB):
    gA = np.asarray(gA, dtype=float)
    gB = np.asarray(gB, dtype=float)
    m = (gA + gB) / 2.0
    e_sh = float(m @ m)
    e_co = float(((gA - gB) / 2.0) @ ((gA - gB) / 2.0))
    return e_sh / max(e_sh + e_co, 1e-30), e_sh, e_co


def closed_form(rho):
    return (1 - 2 * rho) ** 2 / (2 * (rho ** 2 + (1 - rho) ** 2))


def run():
    results = {}
    for gamma in [0.0, 0.9]:
        rows = []
        for rho in RHOS:
            kappas, q0s, q1s = [], [], []
            for s in range(SEEDS):
                obs, act, rew = sample_batch(N_TRANS, rho, seed=1000 + s)
                qnet = QNet()
                opt = torch.optim.Adam(qnet.parameters(), lr=LR)
                idx = torch.randperm(N_TRANS)[:N_TRANS]
                for step in range(TRAIN_STEPS):
                    opt.zero_grad()
                    loss = td_loss(qnet, obs[idx[:256]], act[idx[:256]],
                                   rew[idx[:256]], gamma)
                    loss.backward()
                    opt.step()
                # relation-conditional measurement batches
                obsA, actA, rewA = sample_batch(N_MEAS, 0.0, seed=2000 + s)
                obsB, actB, rewB = sample_batch(N_MEAS, 1.0, seed=3000 + s)
                gA = torch.autograd.grad(td_loss(qnet, obsA, actA, rewA, gamma),
                                         qnet.parameters())
                gB = torch.autograd.grad(td_loss(qnet, obsB, actB, rewB, gamma),
                                         qnet.parameters())
                gAv = torch.cat([g.detach().flatten() for g in gA])
                gBv = torch.cat([g.detach().flatten() for g in gB])
                k, _, _ = kappa_from_grads(gAv.numpy(), gBv.numpy())
                with torch.no_grad():
                    q = qnet(torch.zeros(1, 1))
                    q0s.append(q[0, 0].item())
                    q1s.append(q[0, 1].item())
                kappas.append(k)
            rows.append({
                'rho': rho,
                'kappa_mean': float(np.mean(kappas)),
                'kappa_std': float(np.std(kappas)),
                'closed_form': closed_form(rho),
                'Q0': float(np.mean(q0s)),
                'Q1': float(np.mean(q1s)),
                'Q0_pred': 1 - 2 * rho,
                'Q1_pred': 2 * rho - 1,
            })
            print(f'gamma={gamma} rho={rho:.2f}: '
                  f'kappa={np.mean(kappas):.4f} +/- {np.std(kappas):.4f} '
                  f'(closed form {closed_form(rho):.4f})')
        results[f'gamma_{gamma}'] = rows

    with open(os.path.join(OUT_DIR, 'toy_mix_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nsaved: {os.path.join(OUT_DIR, "toy_mix_results.json")}')


if __name__ == '__main__':
    run()
