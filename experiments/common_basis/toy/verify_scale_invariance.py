"""Experiment C made concrete: kappa is scale-invariant, the decomposition is not.

Same policy, forced assignments, rewards scaled by lambda in {1, 10, 100}.
Gradients scale by lambda, so (E_shared, E_contrast, sigma^2) scale by
lambda^2, while kappa (a ratio) is exactly invariant.

=> kappa cannot distinguish the scaled conditions; the decomposition can
   report the absolute signal strength. This is the "decomposition is richer
   than kappa" claim, demonstrated by construction.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iigc.envs._toy.toy_env import HiddenMatchingEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'variance_decomp')
os.makedirs(OUT_DIR, exist_ok=True)

N_STEPS = 10
N_EPS = 500
LAMBDAS = [1, 10, 100]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        return self.net(obs)


def train_revealed(steps=250, ent_coef=0.15, seed=1):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = PolicyNet()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(steps):
        p = np.random.randint(0, 2)
        obs = torch.FloatTensor([1.0 - p, p]).unsqueeze(0)
        logits = policy(obs)
        probs = F.softmax(logits, dim=-1)
        lp = F.log_softmax(logits, dim=-1)
        ent = -(probs * lp).sum()
        loss = F.cross_entropy(logits, torch.tensor([p])) - ent_coef * ent
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


def ep_grad(policy, env, lam):
    obs, info = env.reset()
    done = False
    lps, rews = [], []
    while not done:
        probs = F.softmax(policy(torch.FloatTensor(obs).unsqueeze(0)), dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        lps.append(dist.log_prob(a))
        obs, r, done, trunc, info = env.step(a.item())
        rews.append(r)
    loss = -sum(lp * (lam * r) for lp, r in zip(lps, rews))
    policy.zero_grad(); loss.backward()
    gv = [p.grad.detach().clone().flatten() for p in policy.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


def collect(policy, partner, lam, n_eps=N_EPS, base_seed=100):
    env = HiddenMatchingEnv(revealed=True, n_steps=N_STEPS)
    env.set_partner(partner)
    gs = []
    for i in range(n_eps):
        torch.manual_seed(base_seed + i); np.random.seed(base_seed + i)
        gs.append(ep_grad(policy, env, lam))
    env.close()
    return torch.stack(gs)


def components(gA, gB):
    muA = gA.mean(0); muB = gB.mean(0)
    mu = (muA + muB) / 2.0
    E_shared = mu.norm().pow(2).item()
    E_contrast = ((muA - muB) / 2.0).norm().pow(2).item()
    varA = (gA - muA).norm(dim=1).pow(2).mean().item()
    varB = (gB - muB).norm(dim=1).pow(2).mean().item()
    sigma2 = (varA + varB) / 2.0
    return E_shared, E_contrast, sigma2


def kappa_of_means(g1, g2):
    avg = (g1 + g2) / 2.0
    e = (g1.norm()**2 + g2.norm()**2) / 2.0
    return (avg.norm()**2 / max(e, 1e-12)).item()


def main():
    policy = train_revealed(seed=1)
    print(f'{"lambda":>7} {"kappa":>7} {"E_shared":>9} {"E_contrast":>11} '
          f'{"sigma2":>9} {"E_scaled/l^2":>12}')
    results = {}
    base = None
    for lam in LAMBDAS:
        gA = collect(policy, 0, lam)
        gB = collect(policy, 1, lam)
        E_sh, E_co, s2 = components(gA, gB)
        k = kappa_of_means(gA.mean(0), gB.mean(0))
        results[lam] = {'kappa': k, 'E_shared': E_sh, 'E_contrast': E_co,
                        'sigma2': s2}
        rel = '' if base is None else f'{E_sh/base["E_shared"]:>6.0f}'
        print(f'{lam:>7} {k:>7.4f} {E_sh:>9.4f} {E_co:>11.4f} '
              f'{s2:>9.4f}  {rel:>9}')
        if base is None:
            base = {'E_shared': E_sh, 'E_contrast': E_co, 'sigma2': s2}

    # kappa invariance check
    ks = [results[lam]['kappa'] for lam in LAMBDAS]
    print(f'\nkappa across lambdas: {["%.4f" % k for k in ks]}  '
          f'(max-min = {max(ks)-min(ks):.2e})')
    print(f'E_shared at lambda=1 vs 100: {results[1]["E_shared"]:.4f} vs '
          f'{results[100]["E_shared"]:.4f}  (ratio = '
          f'{results[100]["E_shared"]/results[1]["E_shared"]:.0f}, '
          f'expected lambda^2 = 10000)')

    with open(os.path.join(OUT_DIR, 'scale_invariance.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "scale_invariance.json")}')


if __name__ == '__main__':
    main()
