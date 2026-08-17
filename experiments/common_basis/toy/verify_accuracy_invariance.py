"""Verify the a(p) mechanism prediction on Toy (fixed policy, injected error).

Theory (reveal_theory.md §4.2 corrected): for a fixed policy whose relation
estimate has accuracy a, injecting estimation errors should give
  E_contrast = (2a-1)^2 c^2,  sigma^2 = 4a(1-a) c^2
so that E_contrast + sigma^2 = c^2 is INVARIANT in a (kappa flat).

Injection: under forced partner A, with prob (1-a) the policy samples its
action from pi(.|obs_B) (acts as if it estimated the wrong relation); the
REINFORCE gradient uses log-pi(a_t | obs_A) and reward +1 if a_t matches the
true partner.

We measure E_shared, E_contrast, sigma^2, kappa across a in [0.5, 1.0].
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
N_EPS = 400
ACCS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def sample_action(policy, obs):
    probs = F.softmax(policy(torch.FloatTensor(obs).unsqueeze(0)), dim=-1)
    return torch.distributions.Categorical(probs).sample().item(), probs


def ep_grad_injected(policy, partner, a, base_seed):
    """One episode with estimate accuracy a (injected). obs_A = true partner obs,
    obs_B = the OTHER partner's obs (used when the estimate is wrong)."""
    env = HiddenMatchingEnv(revealed=True, n_steps=N_STEPS)
    env.set_partner(partner)
    obs_A = env._obs()
    other = 1 - partner
    obs_B = np.array([1 - other, other], dtype=np.float32)
    lps, rews = [], []
    for _ in range(N_STEPS):
        if np.random.rand() < a:
            act, _ = sample_action(policy, obs_A)          # correct estimate
        else:
            act, _ = sample_action(policy, obs_B)          # wrong estimate
        # gradient uses log-pi(act | obs_A); reward matches TRUE partner
        probs = F.softmax(policy(torch.FloatTensor(obs_A).unsqueeze(0)), dim=-1)
        dist = torch.distributions.Categorical(probs)
        lps.append(dist.log_prob(torch.tensor(act)))
        r = 1.0 if act == partner else -1.0
        rews.append(r)
    env.close()
    loss = -sum(lp * r for lp, r in zip(lps, rews))
    policy.zero_grad(); loss.backward()
    gv = [p.grad.detach().clone().flatten() for p in policy.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


def collect(policy, partner, a, n_eps=N_EPS, base_seed=100):
    gs = []
    for i in range(n_eps):
        torch.manual_seed(base_seed + i); np.random.seed(base_seed + i)
        gs.append(ep_grad_injected(policy, partner, a, base_seed))
    return torch.stack(gs)


def components(gA, gB):
    muA = gA.mean(0); muB = gB.mean(0)
    mu = (muA + muB) / 2.0
    E_shared = mu.norm().pow(2).item()
    E_contrast = ((muA - muB) / 2.0).norm().pow(2).item()
    varA = (gA - muA).norm(dim=1).pow(2).mean().item()
    varB = (gB - muB).norm(dim=1).pow(2).mean().item()
    sigma2 = (varA + varB) / 2.0
    k = E_shared / (E_shared + E_contrast + sigma2) if (E_shared + E_contrast + sigma2) > 0 else 0
    return E_shared, E_contrast, sigma2, k


def main():
    policy = train_revealed(seed=1)
    print(f'{"a":>5} {"E_shared":>9} {"E_contrast":>11} {"sigma2":>9} '
          f'{"sum(Ec+s2)":>11} {"kappa":>7}')
    results = {}
    for a in ACCS:
        gA = collect(policy, 0, a)
        gB = collect(policy, 1, a)
        E_sh, E_co, s2, k = components(gA, gB)
        results[a] = {'E_shared': E_sh, 'E_contrast': E_co, 'sigma2': s2,
                      'sum': E_co + s2, 'kappa': k}
        print(f'{a:>5.2f} {E_sh:>9.4f} {E_co:>11.4f} {s2:>9.4f} '
              f'{E_co + s2:>11.4f} {k:>7.4f}')

    with open(os.path.join(OUT_DIR, 'accuracy_invariance.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "accuracy_invariance.json")}')


if __name__ == '__main__':
    main()
