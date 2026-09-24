"""Supplementary: dynamic kappa under a switching-preserving protocol.

The engine/paper forced-partner protocol disables mid-episode role switching,
which is out-of-distribution for dynamic-trained policies (they get 0 reward ->
kappa trivially 0). Here we keep switching ON but fix only the episode START
partner, then compute the standard components (E_shared/E_contrast/sigma2/kappa)
between chef-start and waiter-start episode gradients.

Also computes free mixed-protocol kappa (random partner each episode, two random
mixes) as in the 510K reveal protocol.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

import torch._dynamo  # noqa: F401  pre-import before gym/overcooked

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO  # noqa: E402
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env, PARTNER_TYPES  # noqa: E402
from iigc.envs._overcooked.overcooked_memory_env import OvercookedMemoryEnv  # noqa: E402
from iigc.metrics.kappa import episode_decomposition, measurement_metadata  # noqa: E402

from episode_grad import ep_grad  # noqa: E402

CHKPT = os.environ.get(
    "IIGC_OC_CHKPT", os.path.join(ROOT, "data", "models_overcooked"))
OUT = os.path.join(ROOT, "data", "kappa", "server_tasks", "results",
                   "oc_switch_kappa.json")

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

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._force_start = None

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
        gs.append(ep_grad(model, env))
    env._force_start = None
    return torch.stack(gs)


def components(gA, gB):
    result = episode_decomposition([gA, gB])
    result['E_total'] = result['E_mixture'] + result['sigma2']
    return result


def mixed_kappa(model, env, n=2 * N_EPS, seed=7):
    rng = np.random.default_rng(seed)
    gs = []
    for i in range(n):
        torch.manual_seed(100 + i); np.random.seed(100 + i)
        gs.append(ep_grad(model, env))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="ignore any existing JSON and recompute everything")
    ap.add_argument("--only", default=None,
                    help="comma-separated key substrings to recompute")
    args = ap.parse_args()
    only = [s for s in (args.only.split(",") if args.only else []) if s]

    out = {
        "_metadata": measurement_metadata(
            "episode reinforce gradient by start partner",
            "equal_two_conditions", "stochastic_policy_switch_preserving",
            "euclidean", "episode_noise_separate")
    }
    if os.path.exists(args.out) and not args.force:
        previous = json.load(open(args.out))
        out.update(previous)

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
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)

    for mode in ("static", "dynamic"):
        for s in range(41, 49):
            key = f"{mode}_s{s}"
            if only and not any(o in key for o in only):
                continue
            fp = os.path.join(CHKPT, f"overcookedv3_{mode}_seed{s}_final.zip")
            model = PPO.load(fp, device="cpu"); model.policy.eval()
            env = SwitchStartEnv(mode=mode)
            measure(key, model, env)

    for m in (4, 8, 16):
        for s in (41, 42, 43):
            key = f"mem_m{m}_s{s}"
            if only and not any(o in key for o in only):
                continue
            fp = os.path.join(CHKPT, f"overcooked_mem_dynamic_m{m}_s{s}.zip")
            model = PPO.load(fp, device="cpu"); model.policy.eval()
            env = SwitchStartMemoryEnv(memory=m)
            measure(key, model, env)

    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
