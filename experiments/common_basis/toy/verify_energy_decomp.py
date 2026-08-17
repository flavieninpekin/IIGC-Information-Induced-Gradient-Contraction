"""Verify the energy-variance decomposition on Toy (forced assignments).

Under set_partner, per-partner condition means and within-partner variances are
directly measurable. We check the exact identities:

  Pythagorean (noise-free):
      (||g_A||^2 + ||g_B||^2)/2  ==  E_shared + E_contrast
      E_shared  = ||(g_A+g_B)/2||^2
      E_contrast= ||(g_A-g_B)/2||^2

  Law of total variance:
      Var_total == Var_between + Var_within
      Var_between = (||g_A-mu||^2 + ||g_B-mu||^2)/2
      Var_within  = (E[||g-mu_A||^2|A] + E[||g-mu_B||^2|B])/2  = sigma^2

  kappa predictions:
      kappa_mean = E_shared/(E_shared + E_contrast)          (two-rollout, noise suppressed)
      kappa_ep   = E_shared/(E_shared + E_contrast + sigma^2) (per-episode, with noise)

If all hold to machine precision, kappa is exactly the retained (shared) fraction
of gradient energy, orthogonally decomposable into shared/contrast/noise.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iigc.envs._toy.toy_env import HiddenMatchingEnv
from iigc.metrics.fields import rollout_episodes, kappa_and_energy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'energy_decomp')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 200
N_STEPS = 10
SEED_A, SEED_B = 100, 200


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        return self.net(obs)


class ToyAgent:
    def __init__(self, policy):
        self.actor = policy
        self.device = 'cpu'
        self.log_alpha = torch.zeros(1)


def train_revealed(steps=250, ent_coef=0.15):
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


def analyze(policy, revealed, tag):
    env = HiddenMatchingEnv(revealed=revealed, n_steps=N_STEPS)
    gs = {0: [], 1: []}
    for partner in (0, 1):
        env.set_partner(partner)
        for _ in range(N_EPS):
            gs[partner].append(ep_grad(policy, env))
    env.close()

    gA = torch.stack(gs[0]); gB = torch.stack(gs[1])
    muA = gA.mean(0); muB = gB.mean(0)
    mu = (muA + muB) / 2.0

    # energies (noise-free, from condition means)
    E_shared = mu.norm().pow(2).item()
    E_contrast = ((muA - muB) / 2.0).norm().pow(2).item()
    E_means = (muA.norm().pow(2).item() + muB.norm().pow(2).item()) / 2.0

    # variances
    varA = (gA - muA).norm(dim=1).pow(2).mean().item()
    varB = (gB - muB).norm(dim=1).pow(2).mean().item()
    sigma2 = (varA + varB) / 2.0
    var_between = ((muA - mu).norm().pow(2).item() + (muB - mu).norm().pow(2).item()) / 2.0
    allg = torch.cat([gA, gB])
    var_total = (allg - mu).norm(dim=1).pow(2).mean().item()

    # kappa predictions
    k_mean_pred = E_shared / (E_shared + E_contrast)
    k_ep_pred = E_shared / (E_shared + E_contrast + sigma2)
    k_ep_def = E_shared / (E_shared + var_total)  # ||mu||^2/(||mu||^2+Var_total)
    # measured
    k_mean_meas = kappa_and_energy(muA, muB)[0]   # two-rollout (means)
    # empirical per-episode kappa: mean over random (A,B) episode pairs
    pairs = []
    for _ in range(2000):
        i = np.random.randint(N_EPS); j = np.random.randint(N_EPS)
        gi, gj = gA[i], gB[j]
        avg = (gi + gj) / 2.0
        e = (gi.norm()**2 + gj.norm()**2) / 2.0
        pairs.append(avg.norm().pow(2).item() / max(e.item(), 1e-12))
    k_ep_emp = float(np.mean(pairs))

    row = {
        'tag': tag, 'E_shared': E_shared, 'E_contrast': E_contrast,
        'E_means': E_means, 'pythag_E': E_shared + E_contrast,
        'sigma2': sigma2, 'var_between': var_between, 'var_total': var_total,
        'var_sum': var_between + sigma2,
        'kappa_mean_pred': k_mean_pred, 'kappa_mean_meas': k_mean_meas,
        'kappa_ep_pred': k_ep_pred, 'kappa_ep_def': k_ep_def, 'kappa_ep_emp': k_ep_emp,
        'err_pythag': abs(E_means - (E_shared + E_contrast)),
        'err_varlaw': abs(var_total - (var_between + sigma2)),
    }
    return row


def report(row):
    print(f"\n--- {row['tag']} ---")
    print(f"  E_shared={row['E_shared']:.3f}  E_contrast={row['E_contrast']:.3f}  "
          f"E_means={row['E_means']:.3f}  E_shared+E_contrast={row['pythag_E']:.3f}")
    print(f"  pythagorean err = {row['err_pythag']:.2e}")
    print(f"  sigma2={row['sigma2']:.3f}  var_between={row['var_between']:.3f}  "
          f"var_total={row['var_total']:.3f}  var_between+sigma2={row['var_sum']:.3f}")
    print(f"  law-of-total-var err = {row['err_varlaw']:.2e}")
    print(f"  kappa_mean: pred={row['kappa_mean_pred']:.4f}  meas={row['kappa_mean_meas']:.4f}")
    print(f"  kappa_ep  : formula={row['kappa_ep_pred']:.4f}  def={row['kappa_ep_def']:.4f}  "
          f"empirical={row['kappa_ep_emp']:.4f}")


def main():
    torch.manual_seed(0); np.random.seed(0)
    results = {}

    # Condition 1: random-init policy on HIDDEN obs (relation indistinguishable)
    policy_h = PolicyNet(1)
    r1 = analyze(policy_h, revealed=False, tag='HIDDEN (random policy)')
    results['hidden'] = r1
    report(r1)

    # Condition 2: trained policy on REVEALED obs (relation distinguishable)
    policy_r = train_revealed()
    r2 = analyze(policy_r, revealed=True, tag='REVEALED (trained policy)')
    results['revealed'] = r2
    report(r2)

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    main()
