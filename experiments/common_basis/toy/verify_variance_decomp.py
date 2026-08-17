"""Experiments A/B/C: substantiate the variance decomposition on Toy.

A. Cross-sample prediction: components measured on an estimation set predict
   kappa on a held-out validation set (de-circularizes the identity check).
B. Component stability: bootstrap estimation error of E_shared, E_contrast,
   sigma^2 (makes kappa estimable from a compact measurement).
C. Decomposition richer than kappa: kappa_mean is scale-invariant (constant in
   Toy's symmetric reveal), while the components (signal strength) vary widely —
   kappa cannot distinguish conditions that the decomposition can.

All measurements use forced assignments (set_partner) and the REINFORCE field.
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
EST_EPS = 200      # estimation-set episodes per partner
VAL_EPS = 200      # validation-set episodes per partner
BOOT = 500         # bootstrap resamples
REVEAL_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED_EST = 1000
SEED_VAL = 5000


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


def train_revealed(steps=250, ent_coef=0.15, seed=0):
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


def collect(policy, reveal_frac, partner, n_eps, base_seed):
    env = RevealToyEnv(reveal_fraction=reveal_frac)
    env.set_partner(partner)
    gs = []
    for i in range(n_eps):
        torch.manual_seed(base_seed + i); np.random.seed(base_seed + i)
        gs.append(ep_grad(policy, env))
    env.close()
    return torch.stack(gs)


def components(gA, gB):
    """E_shared, E_contrast, sigma^2 from per-partner gradients."""
    muA = gA.mean(0); muB = gB.mean(0)
    mu = (muA + muB) / 2.0
    E_shared = mu.norm().pow(2).item()
    E_contrast = ((muA - muB) / 2.0).norm().pow(2).item()
    varA = (gA - muA).norm(dim=1).pow(2).mean().item()
    varB = (gB - muB).norm(dim=1).pow(2).mean().item()
    sigma2 = (varA + varB) / 2.0
    return E_shared, E_contrast, sigma2


def kappa_two_rollout(gA, gB, n_rollout):
    """kappa between the partner-0 mean gradient and partner-1 mean gradient."""
    g1 = gA[:n_rollout].mean(0)
    g2 = gB[:n_rollout].mean(0)
    avg = (g1 + g2) / 2.0
    e = (g1.norm()**2 + g2.norm()**2) / 2.0
    return (avg.norm()**2 / max(e, 1e-12)).item()


def measure_condition(policy, reveal_frac, field='reinforce'):
    """Collect est/val per-partner gradients, return components + kappa."""
    out = {}
    gA_est = collect(policy, reveal_frac, 0, EST_EPS, SEED_EST)
    gB_est = collect(policy, reveal_frac, 1, EST_EPS, SEED_EST + 100000)
    E_sh, E_co, s2 = components(gA_est, gB_est)
    out['E_shared'] = E_sh
    out['E_contrast'] = E_co
    out['sigma2'] = s2
    N = VAL_EPS // 2
    out['kappa_pred'] = E_sh / (E_sh + E_co + s2 / N)
    out['kappa_pred_ep'] = E_sh / (E_sh + E_co + s2)

    gA_val = collect(policy, reveal_frac, 0, VAL_EPS, SEED_VAL)
    gB_val = collect(policy, reveal_frac, 1, VAL_EPS, SEED_VAL + 100000)
    out['kappa_meas'] = kappa_two_rollout(gA_val, gB_val, VAL_EPS // 2)
    pairs = []
    for _ in range(2000):
        i = np.random.randint(VAL_EPS); j = np.random.randint(VAL_EPS)
        gi, gj = gA_val[i], gB_val[j]
        avg = (gi + gj) / 2.0
        e = (gi.norm()**2 + gj.norm()**2) / 2.0
        pairs.append(avg.norm().pow(2).item() / max(e.item(), 1e-12))
    out['kappa_meas_ep'] = float(np.mean(pairs))
    return out


def bootstrap_stability(gA, gB, B=BOOT):
    """Bootstrap estimation error of the components."""
    n = gA.shape[0]
    rng = np.random.default_rng(0)
    E_sh, E_co, s2 = [], [], []
    for _ in range(B):
        idxA = rng.integers(0, n, n)
        idxB = rng.integers(0, n, n)
        e1, e2, s = components(gA[idxA], gB[idxB])
        E_sh.append(e1); E_co.append(e2); s2.append(s)
    return (np.std(E_sh), np.std(E_co), np.std(s2))


def main():
    results = {}
    print('=' * 64)
    print('EXPERIMENT A + B: REVEALED (trained policy, reinforce field)')
    print('=' * 64)
    pol_r = train_revealed(seed=1)
    A = measure_condition(pol_r, 1.0)
    results['A_revealed'] = A
    print(f'  E_shared={A["E_shared"]:.3f}  E_contrast={A["E_contrast"]:.3f}  '
          f'sigma2={A["sigma2"]:.3f}')
    print(f'  kappa_pred (N=100) = {A["kappa_pred"]:.4f}')
    print(f'  kappa_meas (val)   = {A["kappa_meas"]:.4f}')
    print(f'  |pred - meas|      = {abs(A["kappa_pred"] - A["kappa_meas"]):.4f}')
    print(f'  kappa_pred_ep      = {A["kappa_pred_ep"]:.4f}   '
          f'kappa_meas_ep = {A["kappa_meas_ep"]:.4f}')
    # bootstrap stability
    gA = collect(pol_r, 1.0, 0, EST_EPS, SEED_EST)
    gB = collect(pol_r, 1.0, 1, EST_EPS, SEED_EST + 100000)
    st = bootstrap_stability(gA, gB)
    results['B_revealed'] = {'E_shared_std': st[0], 'E_contrast_std': st[1],
                             'sigma2_std': st[2]}
    print(f'  bootstrap std: E_shared={st[0]:.3f}  E_contrast={st[1]:.3f}  '
          f'sigma2={st[2]:.3f}')

    print('\n' + '=' * 64)
    print('EXPERIMENT A + B: HIDDEN (random policy, reinforce field)')
    print('=' * 64)
    pol_h = PolicyNet(2)  # obs is 2-dim [0,0] when masked
    A2 = measure_condition(pol_h, 0.0)
    results['A_hidden'] = A2
    print(f'  E_shared={A2["E_shared"]:.3f}  E_contrast={A2["E_contrast"]:.3f}  '
          f'sigma2={A2["sigma2"]:.3f}')
    print(f'  kappa_pred (N=100) = {A2["kappa_pred"]:.4f}')
    print(f'  kappa_meas (val)   = {A2["kappa_meas"]:.4f}')
    print(f'  |pred - meas|      = {abs(A2["kappa_pred"] - A2["kappa_meas"]):.4f}')
    gA2 = collect(pol_h, 0.0, 0, EST_EPS, SEED_EST)
    gB2 = collect(pol_h, 0.0, 1, EST_EPS, SEED_EST + 100000)
    st2 = bootstrap_stability(gA2, gB2)
    results['B_hidden'] = {'E_shared_std': st2[0], 'E_contrast_std': st2[1],
                           'sigma2_std': st2[2]}
    print(f'  bootstrap std: E_shared={st2[0]:.3f}  E_contrast={st2[1]:.3f}  '
          f'sigma2={st2[2]:.3f}')

    print('\n' + '=' * 64)
    print('EXPERIMENT C: kappa is scale-invariant; decomposition is not')
    print('=' * 64)
    print(f'{"p":>5} {"kappa_mean":>10} {"kappa_ep":>9} {"E_shared":>9} '
          f'{"E_contrast":>11} {"sigma2":>9}')
    C = {}
    for p in REVEAL_LEVELS:
        pol = train_revealed(seed=2)
        gA = collect(pol, p, 0, EST_EPS, SEED_EST)
        gB = collect(pol, p, 1, EST_EPS, SEED_EST + 100000)
        E_sh, E_co, s2 = components(gA, gB)
        k_mean = E_sh / (E_sh + E_co)
        k_ep = E_sh / (E_sh + E_co + s2)
        C[p] = {'E_shared': E_sh, 'E_contrast': E_co, 'sigma2': s2,
                'kappa_mean': k_mean, 'kappa_ep': k_ep}
        print(f'{p:>5.2f} {k_mean:>10.4f} {k_ep:>9.4f} {E_sh:>9.3f} '
              f'{E_co:>11.3f} {s2:>9.3f}')
    results['C'] = C

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    main()
