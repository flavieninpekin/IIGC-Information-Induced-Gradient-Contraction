"""Overcooked feasibility probe: does the decomposition separate stuck from
converged runs?

Ground truth (from the previous paper): PPO STATIC (role visible) kappa=0.50,
reward 187; DYNAMIC (role hidden) kappa=0.00, reward 0. The role is the task
(chef -> deliver, waiter -> cook), so the relation is a strong behavioral
driver — unlike 510K.

We force the partner role per rollout (env._force_partner) and measure the
gradient decomposition (E_shared / E_contrast / sigma^2 / kappa).

Hypothesis:
  DYNAMIC (stuck) -> E_shared ~ 0, kappa ~ 0, healthy E_total (IIGC signature)
  STATIC (converged) -> E_shared > 0
"""
import os, json
import numpy as np
import torch

from stable_baselines3 import PPO
from iigc.envs._overcooked import OvercookedV3Env

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_overcooked')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'stuck_detect')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 50
MODELS = [('static', s) for s in [41, 42, 43, 44]] + \
         [('dynamic', s) for s in [41, 42, 43, 44]]


def episode_gradient(model, env):
    """One episode's per-step reward-weighted REINFORCE gradient."""
    obs, info = env.reset()
    g = None
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        d = model.policy.get_distribution(ot)
        a = d.get_actions().item()
        next_obs, r, done, trunc, info = env.step(a)
        d2 = model.policy.get_distribution(torch.FloatTensor(obs).unsqueeze(0))
        lp = d2.log_prob(torch.tensor([a]))
        model.policy.zero_grad()
        (-lp * r).backward()
        gv = torch.cat([p.grad.detach().clone().flatten()
                        for p in model.policy.parameters() if p.grad is not None])
        g = gv if g is None else g + gv
        obs = next_obs
    return g if g is not None else torch.zeros(1)


def collect(model, mode, partner, n_eps=N_EPS, base_seed=100):
    env = OvercookedV3Env(mode=mode)
    env._force_partner = partner
    gs = []
    for i in range(n_eps):
        torch.manual_seed(base_seed + i); np.random.seed(base_seed + i)
        gs.append(episode_gradient(model, env))
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
    E_total = E_shared + E_contrast + sigma2
    k_ep = E_shared / E_total if E_total > 0 else 0.0
    N = gA.shape[0]
    denom = E_shared + E_contrast + sigma2 / N
    k_mean = E_shared / denom if denom > 0 else 0.0
    return dict(E_shared=E_shared, E_contrast=E_contrast, sigma2=sigma2,
                E_total=E_total, kappa_ep=k_ep, kappa_mean=k_mean)


def main():
    print(f'{"model":>14} {"E_shared":>9} {"E_contrast":>11} {"sigma2":>9} '
          f'{"E_total":>10} {"k_ep":>7} {"k_mean":>8}')
    results = {}
    for mode, seed in MODELS:
        fp = os.path.join(MODEL_DIR, f'overcookedv3_{mode}_seed{seed}_final.zip')
        if not os.path.exists(fp):
            print(f'MISSING {fp}')
            continue
        model = PPO.load(fp, device='cpu')
        model.policy.eval()
        g_chef = collect(model, mode, 'chef')
        g_waiter = collect(model, mode, 'waiter')
        c = components(g_chef, g_waiter)
        c['mode'] = mode; c['seed'] = seed
        results[f'{mode}_s{seed}'] = c
        print(f'{mode:>8} s{seed}  {c["E_shared"]:>9.3f} {c["E_contrast"]:>11.3f} '
              f'{c["sigma2"]:>9.3f} {c["E_total"]:>10.3f} '
              f'{c["kappa_ep"]:>7.4f} {c["kappa_mean"]:>8.4f}')

    # summary by mode
    print('\n--- summary (mean over seeds) ---')
    for mode in ['static', 'dynamic']:
        rows = [v for k, v in results.items() if v['mode'] == mode]
        if rows:
            print(f'{mode:>8}: E_shared={np.mean([r["E_shared"] for r in rows]):.2f} '
                  f'k_mean={np.mean([r["kappa_mean"] for r in rows]):.4f} '
                  f'sigma2={np.mean([r["sigma2"] for r in rows]):.1f}')

    with open(os.path.join(OUT_DIR, 'overcooked_decomp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "overcooked_decomp.json")}')


if __name__ == '__main__':
    main()
