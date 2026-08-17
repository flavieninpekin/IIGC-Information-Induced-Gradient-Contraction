"""Train additional SAC seeds for the common-basis experiments.

Usage:
    python train_more_seeds.py single 43 500000
    python train_more_seeds.py dynamic 44 300000

Saves to data/models/510k_sac/510k_sac_<mode>_seed<seed>.pt (same convention
as src/iigc/algos/sac.py) so the E1/E2 measurement scripts pick them up.
"""
import os, sys, time
import numpy as np

from iigc.envs._510k.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from iigc.envs._510k.discrete_sac import DiscreteSAC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models', '510k_sac')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_one(mode, seed, total_steps=500_000):
    fp = os.path.join(MODEL_DIR, f'510k_sac_{mode}_seed{seed}.pt')
    if os.path.exists(fp):
        print(f'SKIP {mode} seed{seed} (model exists)')
        return

    print(f'TRAIN SAC {mode} seed{seed} {total_steps} steps...', flush=True)
    env = FiveTenKMaskedEnv(mode=mode)
    obs_dim = env.observation_space.shape[0]
    sac = DiscreteSAC(obs_dim, MASK_DIM, MAX_ACTIONS, lr=3e-4, device='cuda')

    obs, _ = env.reset()
    step = 0
    t0 = time.time()
    r_buffer = []
    try:
        while step < total_steps:
            action = sac.select_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            r_buffer.append(reward)
            sac.buffer.add(obs, action, reward, next_obs, done,
                           info.get('action_mask', np.ones(MASK_DIM, dtype=np.float32)))
            sac.update(batch_size=64)
            obs = next_obs
            step += 1
            if done or trunc:
                obs, _ = env.reset()
            if step % 100_000 == 0:
                fps = step / (time.time() - t0)
                avg_r = np.mean(r_buffer[-10000:]) if r_buffer else 0
                print(f'  {mode} s{seed}: {step//1000}k steps  {fps:.0f}fps  '
                      f'r={avg_r:.3f}  a={sac.alpha_val:.3f}', flush=True)
        sac.save(fp)
        print(f'  DONE {mode} seed{seed} in {(time.time()-t0)/60:.1f} min', flush=True)
    except Exception as e:
        print(f'  CRASH {mode} seed{seed}: {e}', flush=True)
        import traceback; traceback.print_exc()
    finally:
        env.close()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'single'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 43
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 500_000
    train_one(mode, seed, steps)
