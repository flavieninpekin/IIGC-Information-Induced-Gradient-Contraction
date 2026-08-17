"""How many episodes are needed to estimate kappa reliably?

Corrects the over-reach in Experiment B: reporting component bootstrap stds
does NOT establish that kappa is reliably estimable from a compact
measurement. The right test is a SAMPLE-SIZE SWEEP:

  - collect a large pool of per-episode gradients per partner (2000 eps)
  - for N in [10, 20, 50, 100, 200, 500]: bootstrap-sample N eps per partner,
    compute the measured two-rollout kappa (partner-0 mean vs partner-1 mean),
    report kappa_hat(N) +/- std, and error vs the large-sample true kappa.

kappa(N) should converge to E_shared/(E_shared+E_contrast) as N grows.
This answers: what N gives kappa within a given tolerance?
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
POOL = 2000          # episodes per partner in the pool
BOOT = 300
SIZES = [10, 20, 50, 100, 200, 500, 1000]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        return self.net(obs)


class RevealToyEnv(HiddenMatchingEnv):
    def __init__(self, reveal_fraction=1.0, n_steps=N_STEPS):
        self.reveal_fraction = reveal_fraction
        super().__init__(revealed=True, n_steps=n_steps)

    def _obs(self):
        obs = super()._obs()
        if self.reveal_fraction < 1.0:
            keep = (np.random.random(2) < self.reveal_fraction).astype(np.float32)
            obs = obs * keep
        return obs


def train_revealed(steps=250, ent_coef=0.15, seed=1):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = PolicyNet(2)
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


def ep_grad(policy, env):
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
    loss = -sum(lp * r for lp, r in zip(lps, rews))
    policy.zero_grad(); loss.backward()
    gv = [p.grad.detach().clone().flatten() for p in policy.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


def collect_pool(policy, reveal_frac, n=POOL):
    env = RevealToyEnv(reveal_fraction=reveal_frac)
    env.set_partner(0)
    poolA = torch.stack([ep_grad(policy, env) for _ in range(n)])
    env.set_partner(1)
    poolB = torch.stack([ep_grad(policy, env) for _ in range(n)])
    env.close()
    return poolA, poolB


def kappa_of_means(gA, gB):
    g1 = gA.mean(0); g2 = gB.mean(0)
    avg = (g1 + g2) / 2.0
    e = (g1.norm()**2 + g2.norm()**2) / 2.0
    return (avg.norm()**2 / max(e, 1e-12)).item()


def sweep(poolA, poolB, rng, sizes=SIZES, B=BOOT):
    """kappa_hat(N) +/- std via bootstrap, and true (large-N) kappa."""
    nA, nB = poolA.shape[0], poolB.shape[0]
    true = kappa_of_means(poolA, poolB)
    rows = []
    for N in sizes:
        ks = []
        for _ in range(B):
            ia = rng.integers(0, nA, N)
            ib = rng.integers(0, nB, N)
            ks.append(kappa_of_means(poolA[ia], poolB[ib]))
        ks = np.array(ks)
        rows.append({'N': N, 'kappa_hat': ks.mean(), 'std': ks.std(),
                     'err_vs_true': abs(ks.mean() - true)})
    return true, rows


def report(tag, true, rows):
    print(f'\n--- {tag} ---')
    print(f'  true kappa (large-N) = {true:.4f}')
    print(f'  {"N":>6} {"kappa_hat":>9} {"std":>7} {"err":>7}  rel-std')
    for r in rows:
        rel = r['std'] / max(true, 1e-9)
        print(f'  {r["N"]:>6} {r["kappa_hat"]:>9.4f} {r["std"]:>7.4f} '
              f'{r["err_vs_true"]:>7.4f}  {rel:.1%}')
    return rows


def main():
    rng = np.random.default_rng(0)
    results = {}

    # REVEALED (trained policy)
    pol_r = train_revealed(seed=1)
    pa, pb = collect_pool(pol_r, 1.0)
    true_r, rows_r = sweep(pa, pb, rng)
    results['revealed'] = {'true': true_r, 'sweep': rows_r}
    report('REVEALED (trained policy)', true_r, rows_r)

    # HIDDEN (random policy)
    pol_h = PolicyNet(2)
    pa2, pb2 = collect_pool(pol_h, 0.0)
    true_h, rows_h = sweep(pa2, pb2, rng)
    results['hidden'] = {'true': true_h, 'sweep': rows_h}
    report('HIDDEN (random policy)', true_h, rows_h)

    with open(os.path.join(OUT_DIR, 'compactness.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "compactness.json")}')


if __name__ == '__main__':
    main()
