"""Evaluate the memory intervention: did dynamic+memory escape the stuck state?

Compares, under forced chef/waiter rollouts:
  - dynamic no-memory (baseline, expected stuck: reward ~0, E_shared=0)
  - dynamic + memory (intervention, expected rescued if memory lets the
    policy infer the hidden role)

Metrics: mean reward per role, and the gradient decomposition
(E_shared / E_contrast / sigma2 / kappa) + energy gate.
"""
import os, json
import numpy as np
import torch

from stable_baselines3 import PPO
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env
from iigc.envs._overcooked.overcooked_memory_env import OvercookedMemoryEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_overcooked')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'stuck_detect')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 50
MEMORY_MODELS = [('mem_dynamic_m4', s) for s in [41, 42, 43]]


def episode_gradient(model, env):
    obs, info = env.reset()
    g = None
    done = False
    rews = []
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        d = model.policy.get_distribution(ot)
        a = d.get_actions().item()
        next_obs, r, done, trunc, info = env.step(a)
        rews.append(r)
        d2 = model.policy.get_distribution(torch.FloatTensor(obs).unsqueeze(0))
        lp = d2.log_prob(torch.tensor([a]))
        model.policy.zero_grad()
        (-lp * r).backward()
        gv = torch.cat([p.grad.detach().clone().flatten()
                        for p in model.policy.parameters() if p.grad is not None])
        g = gv if g is None else g + gv
        obs = next_obs
    return g if g is not None else torch.zeros(1), sum(rews)


def collect(model, env, partner, n_eps=N_EPS, base_seed=100):
    env._force_partner = partner
    gs, rews = [], []
    for i in range(n_eps):
        torch.manual_seed(base_seed + i); np.random.seed(base_seed + i)
        g, r = episode_gradient(model, env)
        gs.append(g); rews.append(r)
    env._force_partner = None
    return torch.stack(gs), np.mean(rews)


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
    results = {}
    print(f'{"model":>18} {"r_chef":>7} {"r_waiter":>8} {"E_shared":>9} '
          f'{"k_mean":>7} {"E_total":>9}')

    # baseline: dynamic no-memory (stuck)
    for seed in [41, 42, 43]:
        fp = os.path.join(MODEL_DIR, f'overcookedv3_dynamic_seed{seed}_final.zip')
        if not os.path.exists(fp):
            continue
        model = PPO.load(fp, device='cpu'); model.policy.eval()
        env = OvercookedV3Env(mode='dynamic')
        g_c, r_c = collect(model, env, 'chef')
        g_w, r_w = collect(model, env, 'waiter')
        c = components(g_c, g_w)
        c.update(mode='dynamic_no_mem', seed=seed, reward_chef=r_c,
                 reward_waiter=r_w)
        results[f'dyn_no_mem_s{seed}'] = c
        print(f'{"dyn_no_mem":>18} {r_c:>7.1f} {r_w:>8.1f} '
              f'{c["E_shared"]:>9.2f} {c["kappa_mean"]:>7.4f} {c["E_total"]:>9.2f}')
        env.close()

    # intervention: dynamic + memory
    for tag, seed in MEMORY_MODELS:
        fp = os.path.join(MODEL_DIR, f'overcooked_mem_dynamic_m4_s{seed}.zip')
        if not os.path.exists(fp):
            print(f'MISSING {fp}')
            continue
        model = PPO.load(fp, device='cpu'); model.policy.eval()
        env = OvercookedMemoryEnv(mode='dynamic', memory=4)
        g_c, r_c = collect(model, env, 'chef')
        g_w, r_w = collect(model, env, 'waiter')
        c = components(g_c, g_w)
        c.update(mode='dynamic_mem', seed=seed, reward_chef=r_c,
                 reward_waiter=r_w)
        results[f'dyn_mem_s{seed}'] = c
        print(f'{"dyn_mem_m4":>18} {r_c:>7.1f} {r_w:>8.1f} '
              f'{c["E_shared"]:>9.2f} {c["kappa_mean"]:>7.4f} {c["E_total"]:>9.2f}')
        env.close()

    with open(os.path.join(OUT_DIR, 'overcooked_memory_eval.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "overcooked_memory_eval.json")}')


if __name__ == '__main__':
    main()
