"""kappa as a training forecaster: does kappa(t) predict future improvement?

Uses historical SAC checkpoints (100K granularity) from project3.
For each (mode, seed) run and each checkpoint t, measure:
  - kappa(t)   : reinforce-field retention ratio (energy-gated)
  - R(t)       : mean reward under full-info rollouts
Then test whether low kappa(t) predicts low future improvement.

Prediction (if kappa is a health probe): corr(kappa(t), R_end - R(t)) > 0.
"""
import os, glob
import numpy as np
import torch

from iigc.envs._510k.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from iigc.envs._510k.discrete_sac import DiscreteSAC
from iigc.metrics.fields import (
    rollout_episodes, kappa_and_energy, compute_grad, field_loss,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC = r'C:\Users\Flavi\llmprojects\project3\models_510k_sac'
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'forecast')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 30
SEED_A, SEED_B = 1000, 2000
RUNS = [('single', 41), ('single', 42), ('dynamic', 41), ('dynamic', 42)]


def checkpoint_path(mode, seed, steps):
    if steps == 'final':
        return os.path.join(SRC, f'510k_sac_{mode}_seed{seed}.pt')
    return os.path.join(SRC, f'510k_sac_{mode}_seed{seed}_{steps}_steps.pt')


def measure_ckpt(fp, mode):
    sac = DiscreteSAC(112 + MASK_DIM, MASK_DIM, MAX_ACTIONS, device='cpu')
    sac.load(fp)
    sac.actor.eval()
    sac.critic1.eval()
    sac.critic2.eval()

    env = FiveTenKMaskedEnv(mode=mode)
    eps_a = rollout_episodes(sac, env, n_eps=N_EPS, base_seed=SEED_A); env.close()
    env = FiveTenKMaskedEnv(mode=mode)
    eps_b = rollout_episodes(sac, env, n_eps=N_EPS, base_seed=SEED_B); env.close()

    gA = compute_grad(sac, field_loss(sac, eps_a, 'reinforce'))
    gB = compute_grad(sac, field_loss(sac, eps_b, 'reinforce'))
    k, e = kappa_and_energy(gA, gB)
    ra = np.mean([sum(t[2] for t in traj) for traj in eps_a])
    rb = np.mean([sum(t[2] for t in traj) for traj in eps_b])
    return k, e, (ra + rb) / 2.0


def main():
    rows = []  # (mode, seed, steps, kappa, energy, reward)
    for mode, seed in RUNS:
        steps_list = [100000, 200000, 300000, 400000, 500000]
        for steps in steps_list:
            fp = checkpoint_path(mode, seed, steps)
            if not os.path.exists(fp):
                continue
            k, e, r = measure_ckpt(fp, mode)
            rows.append({'mode': mode, 'seed': seed, 'steps': steps,
                         'kappa': k, 'energy': e, 'reward': r})
            print(f'{mode} s{seed} {steps//1000:>4}k: kappa={k:.4f} '
                  f'E={e:.1e} R={r:.2f}')
        fp = checkpoint_path(mode, seed, 'final')
        if os.path.exists(fp):
            k, e, r = measure_ckpt(fp, mode)
            rows.append({'mode': mode, 'seed': seed, 'steps': 'final',
                         'kappa': k, 'energy': e, 'reward': r})
            print(f'{mode} s{seed} final:   kappa={k:.4f} E={e:.1e} R={r:.2f}')

    # Forecast: for each (run, t), future improvement = R(last) - R(t)
    # last checkpoint per run = the final marker (or the max-step entry)
    final_R = {}
    for mode, seed in RUNS:
        run_rows = [r for r in rows if r['mode'] == mode and r['seed'] == seed]
        if not run_rows:
            continue
        last = max(run_rows, key=lambda r: (r['steps'] != 'final', r['steps']))
        final_R[(mode, seed)] = last['reward']

    pred, target = [], []
    for r in rows:
        fr = final_R.get((r['mode'], r['seed']))
        if fr is None:
            continue
        is_last = (r['steps'] == 'final') or (
            r['steps'] == max(x['steps'] for x in rows
                              if x['mode'] == r['mode'] and x['seed'] == r['seed']
                              and x['steps'] != 'final'))
        if is_last:
            continue
        pred.append(r['kappa'])
        target.append(fr - r['reward'])

    pred = np.array(pred); target = np.array(target)
    print(f'\n=== FORECAST ===')
    print(f'n = {len(pred)} points')
    if len(pred) >= 3:
        corr = np.corrcoef(pred, target)[0, 1]
        print(f'corr(kappa(t), future improvement) = {corr:.3f}')
        med = np.median(pred)
        low = target[pred <= med]
        high = target[pred > med]
        print(f'  kappa <= med ({med:.3f}): future improvement = {low.mean():+.3f} +/- {low.std():.3f} (n={len(low)})')
        print(f'  kappa >  med       : future improvement = {high.mean():+.3f} +/- {high.std():.3f} (n={len(high)})')
    else:
        print('too few points')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        import json
        json.dump(rows, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    main()
