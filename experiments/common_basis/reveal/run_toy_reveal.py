"""Toy continuous-reveal prototype: kappa variance decomposition.

Trains a REINFORCE policy on RevealToyEnv (partner bits kept with prob p,
masked to [0,0] otherwise — masking is unobservable to the policy), then at
FULL-info eval measures, per reveal level p:

  - mu_A, mu_B            : condition-mean gradients (partner forced via
                            set_partner) — the clean relation contrast
  - Var_between           : across-partner variance   (deal contrast)
  - Var_within            : within-partner variance   (= sigma^2 + action noise)
  - Var_total             : total gradient variance
  - kappa(p)              : ||mu||^2 / (||mu||^2 + Var_total)

and checks the law-of-total-variance identity:
    Var_total == Var_between + Var_within

This is the operational protocol for the reveal-theory model
(see notes/reveal_theory.md): kappa's dip is Var/||mu||^2 peaking.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iigc.envs._toy.toy_env import HiddenMatchingEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'toy_reveal')
os.makedirs(OUT_DIR, exist_ok=True)

REVEAL_LEVELS = [round(i / 20, 2) for i in range(21)]  # 0.00 ... 1.00 step 0.05
N_STEPS = 10
TRAIN_STEPS = 600
N_EPS = 60
LR = 1e-2
N_INITS = 3  # policy inits per level (mean +/- std for robust sigma^2 peak)


class RevealToyEnv(HiddenMatchingEnv):
    """HiddenMatchingEnv (revealed=True) with partner bits kept w.p. p.

    Masked obs becomes [0,0], indistinguishable from 'no info'.
    """

    def __init__(self, reveal_fraction=1.0, n_steps=N_STEPS):
        self.reveal_fraction = reveal_fraction
        super().__init__(revealed=True, n_steps=n_steps)

    def _obs(self):
        obs = super()._obs()  # [1-partner, partner] one-hot
        if self.reveal_fraction < 1.0:
            keep = (np.random.random(2) < self.reveal_fraction).astype(np.float32)
            obs = obs * keep
        return obs


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        return self.net(obs)


def train_reinforce(p, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = PolicyNet()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    env = RevealToyEnv(reveal_fraction=p)
    for _ in range(TRAIN_STEPS):
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
        G = sum(rews)
        loss = -sum(lp * G for lp in lps) / max(len(lps), 1)
        opt.zero_grad(); loss.backward(); opt.step()
    env.close()
    return policy


def episode_gradient(policy, env):
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


def measure(policy, n_eps=N_EPS):
    env = RevealToyEnv(reveal_fraction=1.0)  # full-info eval
    grads = {0: [], 1: []}
    for partner in (0, 1):
        env.set_partner(partner)
        for _ in range(n_eps):
            grads[partner].append(episode_gradient(policy, env))
    env.close()

    gA = torch.stack(grads[0]); gB = torch.stack(grads[1])
    muA = gA.mean(0); muB = gB.mean(0)
    mu = (muA + muB) / 2.0
    varA = (gA - muA).norm(dim=1).pow(2).mean().item()
    varB = (gB - muB).norm(dim=1).pow(2).mean().item()
    var_within = (varA + varB) / 2.0
    var_between = 0.5 * (muA - mu).norm().pow(2).item() + 0.5 * (muB - mu).norm().pow(2).item()
    allg = torch.cat([gA, gB])
    var_total = (allg - mu).norm(dim=1).pow(2).mean().item()
    mu2 = mu.norm().pow(2).item()
    kappa = mu2 / (mu2 + var_total) if (mu2 + var_total) > 0 else float('nan')
    return {
        'mu2': mu2, 'var_total': var_total, 'var_between': var_between,
        'var_within': var_within, 'var_sum': var_between + var_within,
        'kappa': kappa, 'consistency': abs(var_total - (var_between + var_within)),
    }


def measure_mixture(policy, n_eps=30, n_rollouts=2):
    """510K-style mixture protocol: each rollout averages over episodes with
    RANDOM partners (mirrors the deal mixtures of the 510K reveal experiment)."""
    env = RevealToyEnv(reveal_fraction=1.0)
    grads = []
    for _ in range(n_rollouts):
        g = None; n = 0
        for _ in range(n_eps):
            env.set_partner(np.random.randint(0, 2))
            ge = episode_gradient(policy, env)
            g = ge if g is None else g + ge
            n += 1
        grads.append(g / max(n, 1))
    env.close()
    gA, gB = grads
    avg = (gA + gB) / 2.0
    e = (gA.norm() ** 2 + gB.norm() ** 2) / 2.0
    kappa = (avg.norm() ** 2 / max(e, 1e-10)).item()
    return kappa


def main():
    results = {}
    print(f'{"p":>5} {"kappa_f":>8} {"kappa_mix":>9} {"mu2":>9} '
          f'{"var_total":>10} {"var_between":>12} {"var_within_mean":>14} '
          f'{"var_within_std":>14} {"n":>3}')
    for p in REVEAL_LEVELS:
        ms = [measure(train_reinforce(p, seed=i)) for i in range(N_INITS)]
        vm = np.mean([m['var_within'] for m in ms])
        vs = np.std([m['var_within'] for m in ms])
        # use the first init for the scalar columns
        m0 = ms[0]
        k_mix = np.mean([measure_mixture(train_reinforce(p, seed=i)) for i in range(N_INITS)])
        results[p] = {
            'var_within_mean': vm, 'var_within_std': vs, 'n': N_INITS,
            'inits': ms, 'kappa_mix_mean': k_mix,
        }
        print(f'{p:>5.2f} {m0["kappa"]:>8.4f} {k_mix:>9.4f} {m0["mu2"]:>9.2e} '
              f'{m0["var_total"]:>10.2e} {m0["var_between"]:>12.2e} '
              f'{vm:>14.4f} {vs:>14.4f} {N_INITS:>3}')

    # locate the sigma^2 peak
    peak_p = max(REVEAL_LEVELS, key=lambda p: results[p]['var_within_mean'])
    print(f'\n--- sigma^2 peak at p={peak_p:.2f} '
          f'(var_within_mean={results[peak_p]["var_within_mean"]:.4f}) ---')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    main()
