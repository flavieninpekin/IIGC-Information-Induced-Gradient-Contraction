"""Toy forced-assignment kappa: the clean relation-conditioned measurement.

HiddenMatchingEnv.set_partner() forces ONE relation per rollout, so each
rollout is a single clean assignment (partner 0 vs partner 1) — unlike the
510K deal-mixture protocol.

Conditions:
  - REVEALED : obs reveals the partner -> the two assignments are
               distinguishable, gradients should AGREE -> high kappa
  - HIDDEN   : obs is constant -> assignments are indistinguishable,
               assignment A demands action 0 and B demands action 1 ->
               mode-seeking gradients CONFLICT -> low kappa

Fields (same policy params, same data, only the objective changes):
  reinforce / awr / softq / expq, using the ANALYTIC true Q of the matching
  game (Q = +1 at the matching action, -1 otherwise). This avoids any
  critic-learning confound.

Prediction: REVEALED kappa high for all fields; HIDDEN kappa low. Whether the
mode/mean axis modulates the HIDDEN contraction is an open question the data
answers.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iigc.envs._toy.toy_env import HiddenMatchingEnv
from iigc.metrics.fields import (
    FIELDS, rollout_episodes, kappa_and_energy, compute_grad, field_loss,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'toy_fields')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 30
N_STEPS = 20
SEED_A, SEED_B = 100, 200
HIDDEN_INITS = 5


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs, deterministic=True):
        probs = F.softmax(self.forward(obs), dim=-1)
        if deterministic:
            return probs.argmax(dim=-1), probs
        return torch.distributions.Categorical(probs).sample(), probs


class ScaledPolicy(nn.Module):
    """Softens a trained policy by scaling logits (pi ~ softmax(logits/tau))."""

    def __init__(self, base, tau=5.0):
        super().__init__()
        self.base = base
        self.tau = tau

    def forward(self, obs):
        return self.base(obs) / self.tau

    def get_action(self, obs, deterministic=True):
        probs = F.softmax(self.forward(obs), dim=-1)
        if deterministic:
            return probs.argmax(dim=-1), probs
        return torch.distributions.Categorical(probs).sample(), probs


class ToyAgent:
    """Minimal wrapper so iigc.metrics.fields works with a bare policy."""

    def __init__(self, policy):
        self.actor = policy
        self.device = 'cpu'
        self.log_alpha = torch.zeros(1)  # alpha = 1.0 for softq


def train_revealed(obs_dim=2, steps=250, lr=1e-2, ent_coef=0.15):
    policy = PolicyNet(obs_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        p = np.random.randint(0, 2)
        obs = torch.FloatTensor([1.0 - p, p]).unsqueeze(0)
        logits = policy(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        ent = -(probs * log_probs).sum()
        loss = F.cross_entropy(logits, torch.tensor([p])) - ent_coef * ent
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


def make_q_fn(revealed, assignment):
    """Analytic true Q: +1 at the matching action, -1 otherwise."""
    if revealed:
        def q_fn(obs):
            partner = obs.argmax(dim=1)  # obs = [1-p, p]
            q = torch.full_like(obs, -1.0)
            q[range(obs.shape[0]), partner] = 1.0
            return q
        return q_fn
    p = 0 if assignment == 'A' else 1

    def q_fn(obs):
        q = torch.full((obs.shape[0], 2), -1.0)
        q[:, p] = 1.0
        return q
    return q_fn


def measure(policy, revealed):
    env = HiddenMatchingEnv(revealed=revealed, n_steps=N_STEPS)
    agent = ToyAgent(policy)

    env.set_partner(0)
    eps_a = rollout_episodes(agent, env, n_eps=N_EPS, base_seed=SEED_A)
    env.set_partner(1)
    eps_b = rollout_episodes(agent, env, n_eps=N_EPS, base_seed=SEED_B)
    env.close()

    qA = make_q_fn(revealed, 'A')
    qB = make_q_fn(revealed, 'B')
    out = {}
    for name in FIELDS:
        use_q = name in ('awr', 'softq', 'expq')
        gA = compute_grad(agent, field_loss(agent, eps_a, name, q_fn=qA if use_q else None))
        gB = compute_grad(agent, field_loss(agent, eps_b, name, q_fn=qB if use_q else None))
        k, e = kappa_and_energy(gA, gB)
        out[name] = {'kappa': k, 'energy': e}
    ra = np.mean([sum(t[2] for t in traj) for traj in eps_a])
    rb = np.mean([sum(t[2] for t in traj) for traj in eps_b])
    return out, ra, rb


def summarize(fields_list):
    return {name: {'kappa_mean': np.mean([f[name]['kappa'] for f in fields_list]),
                   'kappa_std': np.std([f[name]['kappa'] for f in fields_list]),
                   'energy': np.mean([f[name]['energy'] for f in fields_list])}
            for name in FIELDS}


def main():
    print('Training REVEALED policy...')
    policy_r = ScaledPolicy(train_revealed(), tau=5.0)
    with torch.no_grad():
        for p in [0, 1]:
            obs = torch.FloatTensor([1 - p, p]).unsqueeze(0)
            pr = F.softmax(policy_r(obs), dim=-1).tolist()
            print(f'  obs=[{1-p},{p}] -> probs={pr}')

    results = {}
    out_r, ra, rb = measure(policy_r, True)
    results['REVEALED'] = {'fields': out_r, 'rA': ra, 'rB': rb}
    print('\n--- REVEALED (trained policy) ---')
    for name in FIELDS:
        print(f'  {name:10}: kappa={out_r[name]["kappa"]:.4f}  E={out_r[name]["energy"]:.2e}')
    print(f'  reward: rA={ra:.2f} rB={rb:.2f}')

    hid = []
    hid_r = []
    for i in range(HIDDEN_INITS):
        policy_h = PolicyNet(1)
        out_h, ra_h, rb_h = measure(policy_h, False)
        hid.append(out_h)
        hid_r.append((ra_h, rb_h))
        print(f'  hidden init {i}: ' + '  '.join(
            f'{n}={out_h[n]["kappa"]:.3f}' for n in FIELDS))
    results['HIDDEN'] = {'fields': summarize(hid),
                         'rA': np.mean([r[0] for r in hid_r]),
                         'rB': np.mean([r[1] for r in hid_r]),
                         'n_inits': HIDDEN_INITS}
    print('\n--- HIDDEN (random-init policy, mean over %d inits) ---' % HIDDEN_INITS)
    for name in FIELDS:
        s = results['HIDDEN']['fields'][name]
        print(f'  {name:10}: kappa={s["kappa_mean"]:.4f} +/- {s["kappa_std"]:.4f}  '
              f'E={s["energy"]:.2e}')
    print(f'  reward: rA={results["HIDDEN"]["rA"]:.2f} rB={results["HIDDEN"]["rB"]:.2f}')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    main()
