"""Historical SAC actor/critic field measurements.

The actor and critic fields live in different parameter spaces and are not
directly comparable in magnitude. The critic measurement is a TD-loss field,
not a universal ``mean-seeking`` field. The deterministic rollout protocol
also requires the caveat recorded in ``notes/rollout_protocol_artifact.md``.
Energy is reported as a validity diagnostic for each field.
"""
import os, json
import numpy as np
import torch

from iigc.envs._510k.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from iigc.envs._510k.discrete_sac import DiscreteSAC
from iigc.metrics.kappa import condition_decomposition

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models', '510k_sac')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'common_basis_sac_split')
os.makedirs(OUT_DIR, exist_ok=True)

MODES = ['single', 'dynamic']
SEEDS = [41, 42]
N_EPS = 30
SEED_A, SEED_B = 1000, 2000


def kappa_and_energy(gA, gB):
    """Compatibility wrapper around the canonical mixture-reference kappa."""
    result = condition_decomposition([gA, gB])
    return result['kappa_mix'], result['E_mixture']


def rollout(model, env, n_eps=N_EPS, base_seed=0):
    transitions = []
    for ep in range(n_eps):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                act, _ = model.actor.get_action(obs_t, deterministic=True)
            action = act.item()
            next_obs, reward, done, trunc, info = env.step(action)
            transitions.append((obs, action, reward, next_obs, done))
            obs = next_obs
    return transitions


def run():
    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'510k_sac_{mode}_seed{seed}.pt')
            if not os.path.exists(fp):
                print(f'MISSING {fp}')
                continue

            sac = DiscreteSAC(112 + MASK_DIM, MASK_DIM, MAX_ACTIONS, device='cpu')
            sac.load(fp)
            sac.actor.eval()
            sac.critic1.eval()
            sac.critic2.eval()

            env_a = FiveTenKMaskedEnv(mode=mode)
            ta = rollout(sac, env_a, base_seed=SEED_A); env_a.close()
            env_b = FiveTenKMaskedEnv(mode=mode)
            tb = rollout(sac, env_b, base_seed=SEED_B); env_b.close()

            obs_a = [t[0] for t in ta]; act_a = [t[1] for t in ta]
            obs_b = [t[0] for t in tb]; act_b = [t[1] for t in tb]

            # Actor diagnostic field; do not infer a universal geometry label.
            k_actor, e_actor = kappa_and_energy(
                sac.actor_gradient(obs_a, act_a), sac.actor_gradient(obs_b, act_b))

            # Critic TD-loss field in a different parameter space.
            k_critic, e_critic = kappa_and_energy(
                sac.critic_gradient(ta), sac.critic_gradient(tb))

            ra = np.mean([t[2] for t in ta])
            rb = np.mean([t[2] for t in tb])
            results[mode][f'seed{seed}'] = {
                'kappa_actor': k_actor, 'energy_actor': e_actor,
                'kappa_critic': k_critic, 'energy_critic': e_critic,
                'rA': ra, 'rB': rb,
            }
            print(f'SAC {mode} s{seed}: κ_actor={k_actor:.4f} (E={e_actor:.1e})  '
                  f'κ_critic={k_critic:.4f} (E={e_critic:.1e})  r={ra:.2f}/{rb:.2f}')

    print(f'\n{"="*70}\nSUMMARY')
    for mode in MODES:
        for field in ['kappa_actor', 'kappa_critic', 'energy_actor', 'energy_critic']:
            vals = [v[field] for v in results[mode].values()]
            if vals:
                print(f'{mode:8} {field:14}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    run()
