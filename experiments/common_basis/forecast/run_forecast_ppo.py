"""kappa as a training forecaster on self-play PPO runs.

The SAC 510K runs converge within 100K (flat reward -> nothing to forecast).
The self-play PPO runs (project3/models_selfplay) have genuine learning
trajectories (reward ~29 -> 50+ over 100K-900K). Use those.

For each (mode, seed) run and checkpoint t:
  - kappa(t): REINFORCE-field retention (energy-gated), full-info eval
  - R(t)    : mean reward under the same rollouts
Forecast test: corr(kappa(t), R_end - R(t)) should be > 0 if kappa is a
health probe (low kappa -> stuck -> little future improvement).
"""
import os, json
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from iigc.envs._510k.env import FiveTenKEnv
from iigc.metrics.kappa_ppo import rollout_ppo, kappa_ppo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC = r'C:\Users\Flavi\llmprojects\project3\models_selfplay'
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'forecast')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 30
RUNS = [('single', 41), ('static', 41), ('dynamic', 41), ('static', 42)]


def ckpt_path(mode, seed, steps):
    if steps == 'final':
        return os.path.join(SRC, f'510k_{mode}_seed{seed}_final.zip')
    return os.path.join(SRC, f'510k_{mode}_seed{seed}_{steps}_steps.zip')


def measure(fp, mode):
    model = MaskablePPO.load(fp, device='cpu')
    model.policy.eval()
    env_a = FiveTenKEnv(mode=mode)
    ta = rollout_ppo(model, env_a, n_eps=N_EPS); env_a.close()
    env_b = FiveTenKEnv(mode=mode)
    tb = rollout_ppo(model, env_b, n_eps=N_EPS); env_b.close()
    k = kappa_ppo(model, ta, tb)
    ra = np.mean([sum(rl) for _, _, rl in ta])
    rb = np.mean([sum(rl) for _, _, rl in tb])
    return k, (ra + rb) / 2.0


def main():
    rows = []
    for mode, seed in RUNS:
        steps = 100000
        while True:
            fp = ckpt_path(mode, seed, steps)
            if not os.path.exists(fp):
                break
            k, r = measure(fp, mode)
            rows.append({'mode': mode, 'seed': seed, 'steps': steps,
                         'kappa': k, 'reward': r})
            print(f'{mode} s{seed} {steps//1000:>4}k: kappa={k:.4f} R={r:.2f}')
            steps += 100000
        fp = ckpt_path(mode, seed, 'final')
        if os.path.exists(fp):
            k, r = measure(fp, mode)
            rows.append({'mode': mode, 'seed': seed, 'steps': 'final',
                         'kappa': k, 'reward': r})
            print(f'{mode} s{seed} final:   kappa={k:.4f} R={r:.2f}')

    # forecast: per run, future improvement = R(last) - R(t)
    final_R = {}
    for mode, seed in RUNS:
        rr = [x for x in rows if x['mode'] == mode and x['seed'] == seed]
        if rr:
            last = max(rr, key=lambda x: (x['steps'] != 'final', x['steps']))
            final_R[(mode, seed)] = last['reward']

    pred, target = [], []
    for r in rows:
        fr = final_R.get((r['mode'], r['seed']))
        if fr is None:
            continue
        is_last = r['steps'] == 'final' or (
            r['steps'] == max(x['steps'] for x in rows
                              if x['mode'] == r['mode'] and x['seed'] == r['seed']
                              and x['steps'] != 'final'))
        if is_last:
            continue
        pred.append(r['kappa'])
        target.append(fr - r['reward'])

    pred = np.array(pred); target = np.array(target)
    print(f'\n=== FORECAST (self-play PPO) ===')
    print(f'n = {len(pred)} points')
    if len(pred) >= 3:
        corr = np.corrcoef(pred, target)[0, 1]
        print(f'corr(kappa(t), future improvement) = {corr:.3f}')
        med = np.median(pred)
        low = target[pred <= med]
        high = target[pred > med]
        print(f'  kappa <= med ({med:.3f}): improvement = {low.mean():+.3f} +/- {low.std():.3f} (n={len(low)})')
        print(f'  kappa >  med       : improvement = {high.mean():+.3f} +/- {high.std():.3f} (n={len(high)})')

    with open(os.path.join(OUT_DIR, 'forecast_ppo.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "forecast_ppo.json")}')


if __name__ == '__main__':
    main()
