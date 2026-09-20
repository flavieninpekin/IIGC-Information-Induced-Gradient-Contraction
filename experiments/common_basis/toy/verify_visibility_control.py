"""Controlled visibility comparison on the mirror bandit.

``run_toy_fields.py`` measures the hidden condition with random-init
``PolicyNet(1)`` policies and the revealed condition with a trained
``ScaledPolicy``, so its archived ``0 -> 0.41`` contrast is not controlled by
policy parameters. This script fixes the protocol: ONE policy with obs_dim=2
is evaluated under two observation protocols with identical parameters and an
identical evaluation seed schedule.

  revealed protocol : obs = one-hot partner (condition visible)
  masked protocol   : obs = zeros (condition not visible); the reward and the
                      analytic Q still depend on the forced relation

Fields and analytic Q definitions follow ``run_toy_fields.py``. Output:
``data/kappa/toy_fields/visibility_control.json``.
"""
import json
import os

import numpy as np
import torch

from run_toy_fields import (
    PolicyNet, ScaledPolicy, ToyAgent, train_revealed, make_q_fn,
    kappa_components, N_EPS, N_STEPS, SEED_A, SEED_B,
)
from iigc.envs._toy.toy_env import HiddenMatchingEnv
from iigc.metrics.fields import FIELDS, rollout_episodes, compute_grad, field_loss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'toy_fields')
os.makedirs(OUT_DIR, exist_ok=True)

N_RANDOM = 5
USE_Q = {'awr', 'softq', 'expq', 'softmaxq'}


class MaskedMatchingEnv(HiddenMatchingEnv):
    """2-dim observation layout in both protocols; masking zeroes the slot."""

    def __init__(self, masked, n_steps=N_STEPS):
        self.masked = masked
        super().__init__(revealed=True, n_steps=n_steps)

    def _obs(self):
        obs = super()._obs()
        return obs * 0.0 if self.masked else obs


def measure(policy, masked):
    """Same policy, same seeds; only the observation protocol changes."""
    env = MaskedMatchingEnv(masked=masked)
    agent = ToyAgent(policy)
    env.set_partner(0)
    eps_a = rollout_episodes(agent, env, n_eps=N_EPS, base_seed=SEED_A,
                             stochastic=True)
    env.set_partner(1)
    eps_b = rollout_episodes(agent, env, n_eps=N_EPS, base_seed=SEED_B,
                             stochastic=True)
    env.close()

    qA = make_q_fn(not masked, 'A')
    qB = make_q_fn(not masked, 'B')
    out = {}
    for name in FIELDS:
        use_q = name in USE_Q
        gA = compute_grad(agent, field_loss(agent, eps_a, name,
                                            q_fn=qA if use_q else None))
        gB = compute_grad(agent, field_loss(agent, eps_b, name,
                                            q_fn=qB if use_q else None))
        out[name] = kappa_components(gA, gB)
    ra = float(np.mean([sum(t[2] for t in traj) for traj in eps_a]))
    rb = float(np.mean([sum(t[2] for t in traj) for traj in eps_b]))
    return out, ra, rb


def summarize(rows):
    return {name: {
        'kappa_mean': float(np.mean([r[name]['kappa'] for r in rows])),
        'kappa_std': float(np.std([r[name]['kappa'] for r in rows])),
        'E_shared': float(np.mean([r[name]['E_shared'] for r in rows])),
        'E_contrast': float(np.mean([r[name]['E_contrast'] for r in rows])),
        'E_total': float(np.mean([r[name]['E_total'] for r in rows])),
    } for name in FIELDS}


def main():
    torch.manual_seed(0); np.random.seed(0)
    base = train_revealed()

    policies = {'trained_scaled': ScaledPolicy(base, tau=5.0)}
    for i in range(N_RANDOM):
        torch.manual_seed(1000 + i)
        policies[f'random_{i}'] = PolicyNet(2)

    results = {}
    for tag, policy in policies.items():
        row = {}
        for masked in (False, True):
            key = 'masked' if masked else 'revealed'
            out, ra, rb = measure(policy, masked)
            row[key] = {'fields': out, 'rA': ra, 'rB': rb}
        results[tag] = row
        print(f'--- {tag} (same parameters, same seeds) ---')
        print(f'  {"protocol":9} ' + ' '.join(f'{n:>10}' for n in FIELDS))
        for key in ('revealed', 'masked'):
            vals = row[key]['fields']
            print(f'  {key:9} ' + ' '.join(
                f'{vals[n]["kappa"]:>10.4f}' for n in FIELDS))
        print(f'  reward revealed: rA={row["revealed"]["rA"]:.2f} '
              f'rB={row["revealed"]["rB"]:.2f}; '
              f'masked: rA={row["masked"]["rA"]:.2f} rB={row["masked"]["rB"]:.2f}')

    summary = {
        'protocol': 'controlled_visibility_same_policy_same_seeds',
        'n_eps': N_EPS, 'n_steps': N_STEPS,
        'seed_a': SEED_A, 'seed_b': SEED_B,
        'trained': {
            'revealed': results['trained_scaled']['revealed'],
            'masked': results['trained_scaled']['masked'],
        },
        'random_mean': {
            'revealed': {
                'fields': summarize([results[f'random_{i}']['revealed']['fields']
                                     for i in range(N_RANDOM)]),
                'rA': float(np.mean([results[f'random_{i}']['revealed']['rA']
                                     for i in range(N_RANDOM)])),
                'rB': float(np.mean([results[f'random_{i}']['revealed']['rB']
                                     for i in range(N_RANDOM)])),
            },
            'masked': {
                'fields': summarize([results[f'random_{i}']['masked']['fields']
                                     for i in range(N_RANDOM)]),
                'rA': float(np.mean([results[f'random_{i}']['masked']['rA']
                                     for i in range(N_RANDOM)])),
                'rB': float(np.mean([results[f'random_{i}']['masked']['rB']
                                     for i in range(N_RANDOM)])),
            },
            'n_policies': N_RANDOM,
        },
        'per_policy': results,
    }
    fp = os.path.join(OUT_DIR, 'visibility_control.json')
    with open(fp, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'\nSaved: {fp}')


if __name__ == '__main__':
    main()
