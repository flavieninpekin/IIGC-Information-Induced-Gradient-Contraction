"""Deterministic vs stochastic rollout comparison for the Toy field axis.

Finding (2026-08-22): run_toy_fields.py uses rollout_episodes() with
deterministic=True (argmax). Under the HIDDEN mirror bandit, the argmax action
is the same for both relations -> the empirical gradients gA, gB become
PARALLEL (same direction -grad log pi(a*), weights w_A(a*), w_B(a*)) ->
kappa_softmaxq = (w_A+w_B)^2 / (2(w_A^2+w_B^2)) = 0.633 exactly, even though
the pi-weighted expected kappa is 0. Deterministic rollouts measure the weight
ratio at the executed action, NOT the retention of the full gradient field.

This script measures all fields under both protocols (HIDDEN and REVEALED) to
quantify the artifact and to determine which protocol the paper should use.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F

from iigc.envs._toy.toy_env import HiddenMatchingEnv
from iigc.metrics.fields import FIELDS, compute_grad, field_loss, kappa_and_energy

from run_toy_fields import (PolicyNet, ToyAgent, make_q_fn, N_EPS, N_STEPS,
                            SEED_A, SEED_B, HIDDEN_INITS, train_revealed,
                            ScaledPolicy)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'data', 'kappa', 'toy_fields', 'det_vs_stoch.json')


def rollout(agent, env, n_eps, base_seed, stochastic):
    episodes = []
    for ep in range(n_eps):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        traj = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            probs = F.softmax(agent.actor(obs_t), dim=-1)
            if stochastic:
                act = torch.distributions.Categorical(probs).sample().item()
            else:
                act = probs.argmax(dim=-1).item()
            next_obs, reward, done, trunc, info = env.step(act)
            traj.append((obs, act, reward, next_obs, done))
            obs = next_obs
        episodes.append(traj)
    return episodes


def measure(policy, revealed, stochastic):
    env = HiddenMatchingEnv(revealed=revealed, n_steps=N_STEPS)
    agent = ToyAgent(policy)
    env.set_partner(0)
    eps_a = rollout(agent, env, N_EPS, SEED_A, stochastic)
    env.set_partner(1)
    eps_b = rollout(agent, env, N_EPS, SEED_B, stochastic)
    env.close()
    qA = make_q_fn(revealed, 'A')
    qB = make_q_fn(revealed, 'B')
    out = {}
    for name in FIELDS:
        use_q = name in ('awr', 'softq', 'expq', 'softmaxq')
        gA = compute_grad(agent, field_loss(agent, eps_a, name, q_fn=qA if use_q else None))
        gB = compute_grad(agent, field_loss(agent, eps_b, name, q_fn=qB if use_q else None))
        k, e = kappa_and_energy(gA, gB)
        out[name] = {'kappa': k, 'energy': e}
    return out


def main():
    results = {'HIDDEN': {}, 'REVEALED': {}}
    print('Training REVEALED policy...')
    policy_r = ScaledPolicy(train_revealed(), tau=5.0)
    for stoch in [False, True]:
        tag = 'stochastic' if stoch else 'deterministic'
        results['REVEALED'][tag] = measure(policy_r, True, stoch)
        print(f'\n--- REVEALED {tag} ---')
        for name in FIELDS:
            f = results['REVEALED'][tag][name]
            print(f'  {name:10}: kappa={f["kappa"]:.4f}')

    for stoch in [False, True]:
        tag = 'stochastic' if stoch else 'deterministic'
        hid = []
        for i in range(HIDDEN_INITS):
            policy_h = PolicyNet(1)
            hid.append(measure(policy_h, False, stoch))
        results['HIDDEN'][tag] = {name: {
            'kappa_mean': float(np.mean([f[name]['kappa'] for f in hid])),
            'kappa_std': float(np.std([f[name]['kappa'] for f in hid]))}
            for name in FIELDS}
        print(f'\n--- HIDDEN {tag} (mean over {HIDDEN_INITS} inits) ---')
        for name in FIELDS:
            s = results['HIDDEN'][tag][name]
            print(f'  {name:10}: kappa={s["kappa_mean"]:.4f} +/- {s["kappa_std"]:.4f}')

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nsaved: {OUT}')


if __name__ == '__main__':
    main()
