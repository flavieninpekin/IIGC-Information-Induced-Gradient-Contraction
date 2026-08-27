"""Field-axis kappa on Overcooked fresh models (common basis).

Same network, same switching-preserving rollouts, only the gradient objective
changes:
  - reinforce : grad sum_t log pi(a_t) * G_t           (hard mode-seeking)
  - awr       : grad sum_t log pi(a_t) * exp(adv_t/tau) (advantage-weighted)
  - value     : grad sum_t V(s_t)                       (mean-seeking / TD-like)

Within each field we compare static vs dynamic kappa. Prediction: value field
shows HIGH dynamic kappa (reversal recreated on a common basis).
"""
import json
import os
import sys

import numpy as np
import torch

import torch._dynamo  # noqa: F401  pre-import before gym/overcooked

sys.path.insert(0, r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code")
sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

from stable_baselines3 import PPO  # noqa: E402
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env, PARTNER_TYPES  # noqa: E402

CHKPT = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"
OUT = r"C:\Users\Flavi\opencode\IIGC\data\kappa\server_tasks\results\oc_field_axis.json"
N_EPS = 40
AWR_TAU = 1.0


class SwitchStartEnv(OvercookedV3Env):
    """Start with a fixed partner, keep mid-episode switching on."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._force_start = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
        self.base_env.reset()
        for a in self.pool.values():
            a.reset(); a.set_agent_index(1); a.set_mdp(self.mdp)
        if self._force_start is not None:
            self._partner_idx = PARTNER_TYPES.index(self._force_start)
        else:
            self._partner_idx = np.random.randint(len(PARTNER_TYPES))
        self._switch_timer = 0
        self._steps = 0
        return self._get_obs(), {}


def rollout(model, env):
    """One episode: return (obs_list, action_list, reward_list, value_list)."""
    obs, info = env.reset()
    obs_l, act_l, rew_l, val_l = [], [], [], []
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        dist = model.policy.get_distribution(ot)
        a = dist.sample().item()
        with torch.no_grad():
            v = model.policy.predict_values(ot).item()
        obs_l.append(obs)
        act_l.append(a)
        rew_l.append(0.0)
        val_l.append(v)
        obs, r, done, trunc, info = env.step(a)
        rew_l[-1] = r
    return obs_l, act_l, rew_l, val_l


def ep_grad_field(model, obs_l, act_l, rew_l, val_l, field, tau=AWR_TAU):
    G = np.cumsum(rew_l[::-1])[::-1].copy()
    obs = torch.FloatTensor(np.array(obs_l))
    act = torch.tensor(act_l)
    dist = model.policy.get_distribution(obs)
    lp = dist.log_prob(act)

    model.policy.zero_grad()
    if field == "reinforce":
        loss = -(lp * torch.FloatTensor(G)).sum()
    elif field == "awr":
        adv = torch.FloatTensor(G) - torch.FloatTensor(val_l)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        w = torch.exp(torch.clamp(adv / tau, -10.0, 10.0))
        loss = -(lp * w).sum()
    elif field == "value":
        vv = model.policy.predict_values(obs)
        loss = -vv.sum()
    else:
        raise ValueError(field)
    loss.backward()
    gv = [p.grad.detach().clone().flatten()
          for p in model.policy.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


def collect(model, env, partner, field, n=N_EPS):
    env._force_start = partner
    gs = []
    for i in range(n):
        torch.manual_seed(100 + i); np.random.seed(100 + i)
        gs.append(ep_grad_field(model, *rollout(model, env), field))
    env._force_start = None
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
    denom = E_shared + E_contrast + sigma2 / gA.shape[0]
    k_mean = E_shared / denom if denom > 0 else 0.0
    return dict(E_shared=E_shared, E_contrast=E_contrast, sigma2=sigma2,
                E_total=E_total, kappa_ep=k_ep, kappa_mean=k_mean)


def main():
    fields = ["reinforce", "awr", "value"]
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    for mode in ("static", "dynamic"):
        for s in (41, 44, 48):
            fp = os.path.join(CHKPT, f"overcookedv3_{mode}_seed{s}_final.zip")
            model = PPO.load(fp, device="cpu")
            model.policy.eval()
            env = SwitchStartEnv(mode=mode)
            for field in fields:
                key = f"{mode}_s{s}_{field}"
                if key in out:
                    continue
                gA = collect(model, env, "chef", field)
                gB = collect(model, env, "waiter", field)
                comp = components(gA, gB)
                out[key] = comp
                print(f"{mode} s{s} {field:9s}: kappa_ep={comp['kappa_ep']:.3f} "
                      f"E_shared={comp['E_shared']:10.1f} sigma2={comp['sigma2']:12.1f}",
                      flush=True)
            env.close()
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2, default=float)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
