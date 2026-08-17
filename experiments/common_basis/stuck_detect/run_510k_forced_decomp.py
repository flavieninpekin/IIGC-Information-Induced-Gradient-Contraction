"""510K forced-assignment decomposition: does it separate stuck-prone runs?

The mixture protocol was shown to be kappa-insensitive in 510K (noise-
dominated). Here we measure the gradient decomposition under FORCED
assignments (rejection-sample deals so player 0 is on the red-A team vs not),
which gives clean per-relation rollouts.

Models: reveal-trained MaskablePPO at p=0.0 (never saw team info = stuck-prone
candidate) vs p=1.0 (fully visible = converged candidate).

Prediction: p=0.0 -> E_shared ~ 0 (gradients cancel, STUCK signature),
p=1.0 -> E_shared > 0.
"""
import os, json
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from iigc.envs._510k.env import FiveTenKEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_reveal')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'stuck_detect')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 200
MODELS = [(0.00, 41), (0.00, 42), (1.00, 41), (1.00, 42)]


class FixedAssignEnv:
    """Rejection-sample deals so player 0 is on (or not on) the red-A team."""

    def __init__(self, mode='obvious', agent_on_red=True, max_tries=200):
        self.env = FiveTenKEnv(mode=mode)
        self.agent_on_red = agent_on_red
        self.max_tries = max_tries

    def reset(self, seed=None, options=None):
        for _ in range(self.max_tries):
            obs, info = self.env.reset(seed=seed)
            game = self.env.unwrapped.game
            if game.red_a_team is None:
                continue
            on_red = 0 in game.red_a_team
            if on_red == self.agent_on_red:
                return obs, info
        raise RuntimeError('could not force assignment')

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def episode_gradient(model, env, advantage=True, gamma=0.99):
    """One episode's gradient. advantage=True: per-step advantage-weighted
    (A_t = return_t - V(s_t)) for much lower variance than raw REINFORCE."""
    obs, info = env.reset()
    olist, alist, rews = [], [], []
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        d = model.policy.get_distribution(ot)
        a = d.get_actions().item()
        next_obs, r, done, trunc, info = env.step(a)
        olist.append(obs)
        alist.append(a)
        rews.append(r)
        obs = next_obs

    g = None
    if advantage:
        # returns and advantages
        T = len(rews)
        rets = [0.0] * T
        g_ = 0.0
        for t in range(T - 1, -1, -1):
            g_ = rews[t] + gamma * g_
            rets[t] = g_
        for t in range(T):
            ot = torch.FloatTensor(olist[t]).unsqueeze(0)
            with torch.no_grad():
                v = model.policy.predict_values(ot).item()
            adv = rets[t] - v
            d2 = model.policy.get_distribution(ot)
            lp = d2.log_prob(torch.tensor([alist[t]]))
            model.policy.zero_grad()
            (-lp * adv).backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                            for p in model.policy.parameters() if p.grad is not None])
            g = gv if g is None else g + gv
    else:
        for t in range(len(olist)):
            ot = torch.FloatTensor(olist[t]).unsqueeze(0)
            d2 = model.policy.get_distribution(ot)
            lp = d2.log_prob(torch.tensor([alist[t]]))
            model.policy.zero_grad()
            (-lp * rews[t]).backward()
            gv = torch.cat([p.grad.detach().clone().flatten()
                            for p in model.policy.parameters() if p.grad is not None])
            g = gv if g is None else g + gv
    return g if g is not None else torch.zeros(1)


def collect(model, agent_on_red, n_eps=N_EPS, base_seed=100):
    env = FixedAssignEnv(agent_on_red=agent_on_red)
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
    # two-rollout (averaged) kappa: sigma2 reduced by N eps per relation
    N = gA.shape[0]
    k_mean = E_shared / (E_shared + E_contrast + sigma2 / N) \
        if (E_shared + E_contrast + sigma2 / N) > 0 else 0.0
    return dict(E_shared=E_shared, E_contrast=E_contrast, sigma2=sigma2,
                E_total=E_total, kappa_ep=k_ep, kappa_mean=k_mean)


def main():
    print(f'{"model":>14} {"E_shared":>9} {"E_contrast":>11} {"sigma2":>9} '
          f'{"kappa_ep":>8} {"kappa_mean":>10}')
    results = {}
    for p, seed in MODELS:
        fp = os.path.join(MODEL_DIR, f'ppo_reveal_{p:.2f}_s{seed}.zip')
        if not os.path.exists(fp):
            print(f'MISSING {fp}')
            continue
        model = MaskablePPO.load(fp, device='cpu')
        model.policy.eval()
        g_red = collect(model, True)
        g_non = collect(model, False)
        c = components(g_red, g_non)
        c['model'] = f'p={p:.2f} s{seed}'
        results[f'p{p:.2f}_s{seed}'] = c
        print(f'p={p:.2f} s{seed}  {c["E_shared"]:>9.3f} {c["E_contrast"]:>11.3f} '
              f'{c["sigma2"]:>9.3f} {c["kappa_ep"]:>8.4f} {c["kappa_mean"]:>10.4f}')

    with open(os.path.join(OUT_DIR, 'forced_decomp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "forced_decomp.json")}')


if __name__ == '__main__':
    main()
