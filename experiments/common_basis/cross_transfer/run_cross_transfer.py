"""Cross-transfer kappa matrix: train mode x test mode.

For each (train_mode, test_mode) cell, load a SAC model TRAINED in train_mode,
roll it out in TEST mode under two distinct deal seeds (in DYNAMIC these are
two hidden-team assignments), and compute kappa for each gradient field.

Matrix layout:

              test SINGLE   test DYNAMIC
train SINGLE   S -> S        S -> D   <- unadapted policy + hidden relations
train DYNAMIC  D -> S        D -> D   <- adapted policy

This separates two factors:
  - column (test mode): the relation structure at evaluation
  - row  (train mode):  the policy's adaptation state

Prediction (reinforce field): kappa(S->D) is the LOWEST cell -- an unadapted
policy's gradients contract under hidden relations; comparing against
kappa(S->S) and kappa(D->D) disentangles environment vs adaptation.
"""
import os, json
import numpy as np

from iigc.envs._510k.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from iigc.envs._510k.discrete_sac import DiscreteSAC
from iigc.metrics.fields import (
    FIELDS, rollout_episodes, kappa_and_energy, compute_grad, field_loss,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models', '510k_sac')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'cross_transfer')
os.makedirs(OUT_DIR, exist_ok=True)

MODES = ['single', 'dynamic']
SEEDS = [41, 42, 43, 44]
N_EPS = 30
SEED_A, SEED_B = 1000, 2000


def run():
    # results[train][test][seed][field] = {'kappa':.., 'energy':..}
    results = {t: {e: {} for e in MODES} for t in MODES}
    for train in MODES:
        for test in MODES:
            for seed in SEEDS:
                fp = os.path.join(MODEL_DIR, f'510k_sac_{train}_seed{seed}.pt')
                if not os.path.exists(fp):
                    print(f'MISSING {fp}')
                    continue
                sac = DiscreteSAC(112 + MASK_DIM, MASK_DIM, MAX_ACTIONS, device='cpu')
                sac.load(fp)
                sac.actor.eval()

                env_a = FiveTenKMaskedEnv(mode=test)
                eps_a = rollout_episodes(sac, env_a, n_eps=N_EPS, base_seed=SEED_A); env_a.close()
                env_b = FiveTenKMaskedEnv(mode=test)
                eps_b = rollout_episodes(sac, env_b, n_eps=N_EPS, base_seed=SEED_B); env_b.close()

                ra = np.mean([sum(t[2] for t in traj) for traj in eps_a])
                rb = np.mean([sum(t[2] for t in traj) for traj in eps_b])

                row = {'rA': ra, 'rB': rb, 'fields': {}}
                for name in FIELDS:
                    gA = compute_grad(sac, field_loss(sac, eps_a, name))
                    gB = compute_grad(sac, field_loss(sac, eps_b, name))
                    k, e = kappa_and_energy(gA, gB)
                    row['fields'][name] = {'kappa': k, 'energy': e}
                results[train][test][f'seed{seed}'] = row
                kline = '  '.join(
                    f'{n}={row["fields"][n]["kappa"]:.3f}' for n in FIELDS)
                print(f'{train:7}->{test:7} s{seed}: r={ra:.2f}/{rb:.2f}  {kline}')

    print(f'\n{"="*76}\nKAPPA MATRIX (mean over seeds, n={len(SEEDS)})')
    for name in FIELDS:
        print(f'\n--- field: {name} ---')
        header = '          ' + ''.join(f'{t:>12}' for t in MODES)
        print(header)
        for train in MODES:
            cells = []
            for test in MODES:
                vals = [results[train][test][s]['fields'][name]['kappa']
                        for s in results[train][test] if 'fields' in results[train][test][s]]
                cells.append(f'{np.mean(vals):.3f}±{np.std(vals):.3f}' if vals else '  n/a   ')
            print(f'  {train:8}' + ''.join(f'{c:>12}' for c in cells))

    # Energy gate summary
    print(f'\n--- min gradient energy per cell (diagnostic) ---')
    for train in MODES:
        for test in MODES:
            for name in FIELDS:
                es = [results[train][test][s]['fields'][name]['energy']
                      for s in results[train][test] if 'fields' in results[train][test][s]]
                if es:
                    print(f'  {train}->{test} {name:10}: minE={min(es):.1e}')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    run()
