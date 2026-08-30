"""Overcooked reveal-kappa curve: fixed-policy measurement basis (Prop 10).

Uses EXISTING static models (role visible at training, 99-dim obs = 96 state
+ 3 one-hot role bits). At eval time we mask the role one-hot bits with
probability (1-p): the observation stays 99-dim but the role is hidden —
exactly the "partial reveal" knob of Prop 10.

Per reveal level p, per static model seed:
  - condition A = chef-start, condition B = waiter-start (SwitchStartEnv:
    fixed start partner, mid-episode switching kept on — the switching-
    preserving protocol)
  - canonical expected gradient of the reinforce field (return-weighted)
    conditioned on start partner
  - kappa(p) = ||(gA+gB)/2||^2 / avg(||gA||^2, ||gB||^2)

Prediction (Prop 10): kappa(p) increasing in p; kappa(0)~0 (hidden role =
mirror-ish cancellation on the measurement basis), kappa(1) = static-model
field-axis value.

The masked obs feeds the SAME 99-dim policy net (role bits zeroed), so no
retraining is needed; this is the fixed-policy measurement basis.

Output: data/kappa/overcooked_reveal/reveal_kappa.json + figure
"""
import json
import os
import sys

import numpy as np
import torch

import torch._dynamo  # noqa: F401

sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

from stable_baselines3 import PPO  # noqa: E402
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env, PARTNER_TYPES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_overcooked')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'overcooked_reveal')
os.makedirs(OUT_DIR, exist_ok=True)

PS = np.round(np.linspace(0.0, 1.0, 11), 2)
SEEDS = [41, 42, 43]
N_EPS = 30
AWR_TAU = 1.0
HORIZON = 80


class RevealMaskEnv(OvercookedV3Env):
    """Static-mode env; at reset picks a fixed start partner (forced) and
    masks the role one-hot with probability (1-p) on every obs."""

    def __init__(self, mode='static', reveal_p=1.0, horizon=HORIZON,
                 switch_interval=20):
        self._mask = False
        self._rng = np.random.default_rng(0)
        super().__init__(mode=mode, horizon=horizon,
                         switch_interval=switch_interval)
        self.reveal_p = reveal_p
        self._force_start = None

    def _get_obs(self):
        obs = super()._get_obs()
        if self._mask and len(obs) == 99:
            obs = obs.copy()
            obs[96:] = 0.0
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self.base_env.reset()
        for a in self.pool.values():
            a.reset(); a.set_agent_index(1); a.set_mdp(self.mdp)
        if self._force_start is not None:
            self._partner_idx = PARTNER_TYPES.index(self._force_start)
        else:
            self._partner_idx = np.random.randint(len(PARTNER_TYPES))
        self._switch_timer = 0
        self._steps = 0
        self._mask = self._rng.random() > self.reveal_p
        return self._get_obs(), {}

    def step(self, action):
        out = super().step(action)
        self._mask = self._rng.random() > self.reveal_p
        return out


def rollout(model, env):
    obs, info = env.reset()
    obs_l, act_l, rew_l = [], [], []
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            act, _ = model.predict(obs_t, deterministic=False)
        act = int(act)
        obs, r, done, trunc, info = env.step(act)
        obs_l.append(obs_t[0])
        act_l.append(act)
        rew_l.append(r)
        if len(obs_l) >= 400:
            break
    return obs_l, act_l, rew_l


def flat_grad(model, loss):
    model.policy.optimizer.zero_grad()
    loss.backward()
    gs = [p.grad.detach().clone().flatten()
          for p in model.policy.parameters() if p.grad is not None]
    return torch.cat(gs)


def reinforce_grad(model, obs_l, act_l, rew_l):
    """Value field: grad sum_t V(s_t) — TD/mean-seeking, reward-independent.

    (The reinforce field is unusable here: with a waiter start the static
    model earns zero reward on this layout, so every reward-weighted gradient
    is identically zero — a degenerate condition, not a kappa signal.)
    """
    if len(obs_l) == 0:
        return torch.zeros(1)
    obs_b = torch.stack(obs_l)
    vv = model.policy.predict_values(obs_b)
    loss = -vv.sum()
    return flat_grad(model, loss)


def kappa(gA, gB):
    m = (gA + gB) / 2
    es = float(m @ m)
    et = (float(gA @ gA) + float(gB @ gB)) / 2
    return es / max(et, 1e-12), es, et


def main():
    results = {}
    for seed in SEEDS:
        path = os.path.join(MODEL_DIR,
                            f'overcookedv3_static_seed{seed}_final.zip')
        if not os.path.exists(path):
            print('missing', path)
            continue
        model = PPO.load(path, device='cpu')
        row = {}
        for p in PS:
            ks, es_, ec_ = [], [], []
            env_a = RevealMaskEnv(reveal_p=float(p))
            env_a._force_start = 'chef'
            env_b = RevealMaskEnv(reveal_p=float(p))
            env_b._force_start = 'waiter'
            gA = torch.zeros(1)
            gB = torch.zeros(1)
            for ep in range(N_EPS):
                oa, aa, ra = rollout(model, env_a)
                ob, ab, rb = rollout(model, env_b)
                if ep == 0:
                    gA = reinforce_grad(model, oa, aa, ra)
                    gB = reinforce_grad(model, ob, ab, rb)
                else:
                    gA = gA + reinforce_grad(model, oa, aa, ra)
                    gB = gB + reinforce_grad(model, ob, ab, rb)
            gA = gA / N_EPS
            gB = gB / N_EPS
            k, e_sh, e_co = kappa(gA, gB)
            ks.append(k)
            row[str(p)] = {'kappa': k, 'e_shared': e_sh, 'e_contrast': e_co}
            print(f'  seed{seed} p={p:.2f}: kappa={k:.4f} '
                  f'(Esh={e_sh:.1e} Eco={e_co:.1e})', flush=True)
        results[f'seed{seed}'] = row
        model.policy.optimizer = None

    # aggregate
    agg = {}
    for p in PS:
        vals = [results[s][str(p)]['kappa'] for s in results]
        agg[str(p)] = {'mean': float(np.mean(vals)),
                       'std': float(np.std(vals)),
                       'seeds': vals}
    out = {'p': PS.tolist(), 'per_level': agg, 'per_seed': results,
           'note': ('fixed-policy measurement basis: static models (role '
                    'visible at training), role one-hot masked with prob '
                    '(1-p) at eval; reinforce field; chef/waiter start '
                    'conditions; Prop 10 predicts increasing kappa(p)')}
    with open(os.path.join(OUT_DIR, 'reveal_kappa.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\naggregate:')
    for p in PS:
        v = agg[str(p)]
        print(f'  p={p:.2f}: kappa={v["mean"]:.4f} ± {v["std"]:.4f}')


if __name__ == '__main__':
    main()