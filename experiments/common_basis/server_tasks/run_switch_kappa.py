"""Supplementary: dynamic kappa under a switching-preserving protocol.

The engine/paper forced-partner protocol disables mid-episode role switching,
which is out-of-distribution for dynamic-trained policies (they get 0 reward ->
kappa trivially 0). Here we keep switching ON but fix only the episode START
partner, then compute the standard components (E_shared/E_contrast/sigma2/kappa)
between chef-start and waiter-start episode gradients.

Also computes free mixed-protocol kappa (random partner each episode, two random
mixes) as in the 510K reveal protocol.
"""
import json
import os
import sys

import numpy as np
import torch

import torch._dynamo  # noqa: F401  pre-import before gym/overcooked

sys.path.insert(0, r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code")
sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

import engine  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env, PARTNER_TYPES  # noqa: E402
from iigc.envs._overcooked.overcooked_memory_env import OvercookedMemoryEnv  # noqa: E402

CHKPT = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"
OUT = r"C:\Users\Flavi\opencode\IIGC\data\kappa\server_tasks\results\oc_switch_kappa.json"

N_EPS = 60


class SwitchStartEnv(OvercookedV3Env):
    """Start with a fixed partner, but keep mid-episode switching on."""

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


class SwitchStartMemoryEnv(OvercookedMemoryEnv):
    """Memory env + fixed START partner, switching kept on."""

    def reset(self, seed=None, options=None):
        self._history.clear()
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


def collect(model, env, partner, n):
    env._force_start = partner
    gs = []
    for i in range(n):
        torch.manual_seed(100 + i); np.random.seed(100 + i)
        gs.append(engine._ep_grad(model, env))
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


def mixed_kappa(model, env, n=2 * N_EPS, seed=7):
    rng = np.random.default_rng(seed)
    gs = []
    for i in range(n):
        torch.manual_seed(100 + i); np.random.seed(100 + i)
        gs.append(engine._ep_grad(model, env))
    g = torch.stack(gs)
    halves = []
    for _ in range(50):
        idx = rng.permutation(n)[: n // 2]
        halves.append(g[idx].mean(0))
    hs = torch.stack(halves)
    k = []
    for i in range(25):
        k.append(hs[2 * i].dot(hs[2 * i + 1]).item() /
                 max((hs[2 * i].norm() ** 2 + hs[2 * i + 1].norm() ** 2) / 2, 1e-12))
    return float(np.mean(k))


def main():
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))

    def measure(key, model, env):
        if key in out:
            print(f"skip {key} (done)", flush=True)
            return
        gA = collect(model, env, "chef", N_EPS)
        gB = collect(model, env, "waiter", N_EPS)
        free = mixed_kappa(model, env)
        env.close()
        comp = components(gA, gB)
        comp["kappa_mixed_free"] = free
        out[key] = comp
        print(f"{key}: kappa_ep={comp['kappa_ep']:.3f} "
              f"E_shared={comp['E_shared']:.1f} E_contrast={comp['E_contrast']:.1f} "
              f"sigma2={comp['sigma2']:.1f} kappa_mixed={free:.3f}", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, default=float)

    for mode in ("static", "dynamic"):
        for s in range(41, 49):
            fp = os.path.join(CHKPT, f"overcookedv3_{mode}_seed{s}_final.zip")
            model = PPO.load(fp, device="cpu"); model.policy.eval()
            env = SwitchStartEnv(mode=mode)
            measure(f"{mode}_s{s}", model, env)

    for m in (4, 8, 16):
        for s in (41, 42, 43):
            fp = os.path.join(CHKPT, f"overcooked_mem_dynamic_m{m}_s{s}.zip")
            model = PPO.load(fp, device="cpu"); model.policy.eval()
            env = SwitchStartMemoryEnv(memory=m)
            measure(f"mem_m{m}_s{s}", model, env)

    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
