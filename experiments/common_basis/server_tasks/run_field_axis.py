"""Overcooked field measurements on switching-preserving rollouts.

The ``value`` entry is the diagnostic field ``-sum_t V(s_t)``. It is not a TD
residual and should not be labelled as a universal mean-seeking field.

Protocol: one episode batch per (checkpoint, partner, seed) is collected and
shared by every field; only the scalar objective changes. ``awr`` keeps the
policy-dependent baseline ``V(s)`` in the autograd graph (the
differentiable-baseline field of the main text). Every per-episode gradient is
zero-padded to the full policy parameter vector, so all fields live in one
coordinate system. The metadata block records this protocol.

Pass ``--force`` to recompute cached entries.
"""
import json
import os
import sys

import numpy as np
import torch

import torch._dynamo  # noqa: F401  pre-import before gym/overcooked

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KITCHEN_CODE = os.environ.get(
    "IIGC_KITCHEN_CODE",
    r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code")

sys.path.insert(0, KITCHEN_CODE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from stable_baselines3 import PPO  # noqa: E402
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env, PARTNER_TYPES  # noqa: E402
from iigc.metrics.kappa import episode_decomposition, measurement_metadata  # noqa: E402

CHKPT = os.environ.get(
    "IIGC_OC_CHKPT", os.path.join(ROOT, "data", "models_overcooked"))
OUT = os.path.join(ROOT, "data", "kappa", "server_tasks", "results",
                   "oc_field_axis.json")
N_EPS = 40
AWR_TAU = 1.0
FIELDS = ("reinforce", "awr", "value")


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
    """One episode: return (obs_list, action_list, reward_list)."""
    obs, info = env.reset()
    obs_l, act_l, rew_l = [], [], []
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        dist = model.policy.get_distribution(ot)
        a = dist.sample().item()
        obs_l.append(obs)
        act_l.append(a)
        rew_l.append(0.0)
        obs, r, done, trunc, info = env.step(a)
        rew_l[-1] = r
    return obs_l, act_l, rew_l


def flat_full_grad(model):
    """Concatenate the policy gradient in the full parameter space."""
    parts = []
    for p in model.policy.parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        parts.append(g.detach().clone().flatten())
    return torch.cat(parts)


def ep_grad_field(model, obs_l, act_l, rew_l, field, tau=AWR_TAU):
    G = np.cumsum(rew_l[::-1])[::-1].copy()
    obs = torch.FloatTensor(np.array(obs_l))
    act = torch.tensor(act_l)
    dist = model.policy.get_distribution(obs)
    lp = dist.log_prob(act)

    model.policy.zero_grad()
    if field == "reinforce":
        loss = -(lp * torch.FloatTensor(G)).sum()
    elif field == "awr":
        v = model.policy.predict_values(obs)  # differentiable baseline
        adv = torch.FloatTensor(G) - v
        w = torch.exp(torch.clamp(adv / tau, -20.0, 20.0))
        loss = -(lp * w).sum()
    elif field == "value":
        vv = model.policy.predict_values(obs)
        loss = -vv.sum()
    else:
        raise ValueError(field)
    loss.backward()
    return flat_full_grad(model)


def collect_batch(model, env, partner, n=N_EPS):
    """One episode batch per (partner, seed), shared by all fields."""
    env._force_start = partner
    batch = []
    for i in range(n):
        torch.manual_seed(100 + i); np.random.seed(100 + i)
        batch.append(rollout(model, env))
    env._force_start = None
    return batch


def components(gA, gB):
    result = episode_decomposition([gA, gB])
    result['E_total'] = result['E_mixture'] + result['sigma2']
    return result


def main():
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    if "--force" in sys.argv:
        out = {k: v for k, v in out.items() if k == "_metadata"}
    out["_metadata"] = measurement_metadata(
        "reinforce/awr/value per-episode field gradient",
        "equal_two_conditions", "shared_episode_batch_switch_preserving",
        "full_policy_parameter_vector_euclidean", "episode_noise_separate")
    for mode in ("static", "dynamic"):
        for s in (41, 44, 48):
            keys = [f"{mode}_s{s}_{f}" for f in FIELDS]
            if all(k in out for k in keys):
                continue
            fp = os.path.join(CHKPT, f"overcookedv3_{mode}_seed{s}_final.zip")
            model = PPO.load(fp, device="cpu")
            model.policy.eval()
            env = SwitchStartEnv(mode=mode)
            batches = {pt: collect_batch(model, env, pt)
                       for pt in ("chef", "waiter")}
            for field in FIELDS:
                key = f"{mode}_s{s}_{field}"
                if key in out:
                    continue
                gA = torch.stack([ep_grad_field(model, *r, field)
                                  for r in batches["chef"]])
                gB = torch.stack([ep_grad_field(model, *r, field)
                                  for r in batches["waiter"]])
                comp = components(gA, gB)
                out[key] = comp
                print(f"{mode} s{s} {field:9s}: kappa_mix={comp['kappa_mix']:.3f} "
                      f"E_shared={comp['E_shared']:10.1f} sigma2={comp['sigma2']:12.1f}",
                      flush=True)
            env.close()
            del model
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2, default=float)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
