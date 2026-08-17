"""Train a single PPO model on RevealEnv for a (reveal_fraction, seed) pair.

Usage: python train_reveal.py <reveal_fraction> <seed> [--steps 1000000]

Saves to data/models_reveal/ppo_reveal_<frac>.s<seed>.zip — same naming as
the original reveal_fine.py so the Phase 1 measurement script picks them up.
"""
import os, sys, time, multiprocessing
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from iigc.envs._510k.env import FiveTenKEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_reveal')
os.makedirs(MODEL_DIR, exist_ok=True)


class RevealEnv(FiveTenKEnv):
    """510K OBVIOUS mode with team bits randomly masked at training time."""
    def __init__(self, reveal_fraction=1.0, **kw):
        super().__init__(mode='obvious', **kw)
        self.reveal_fraction = reveal_fraction

    def _get_obs(self):
        obs = super()._get_obs()  # 112 + 4 team bits
        if self.reveal_fraction < 1.0 and self.game:
            team = obs[-4:].copy()
            keep = (np.random.random(4) < self.reveal_fraction).astype(np.float32)
            obs[-4:] = team * keep
        return obs


def train_one(frac, seed, steps=1_000_000):
    name = f'{frac:.2f}'
    fp = os.path.join(MODEL_DIR, f'ppo_reveal_{name}_s{seed}.zip')
    if os.path.exists(fp):
        print(f'SKIP {name} s{seed} (exists)', flush=True)
        return

    print(f'TRAIN reveal={name} s{seed}  {steps} steps...', flush=True)

    def make():
        e = RevealEnv(reveal_fraction=frac)
        return ActionMasker(e, lambda env: env.unwrapped._get_action_mask())

    env = SubprocVecEnv([make for _ in range(8)], start_method='spawn')
    env = VecMonitor(env)
    model = MaskablePPO(
        MaskableActorCriticPolicy, env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0, seed=seed, device='cuda',
    )
    t0 = time.time()
    model.learn(total_timesteps=steps)
    model.save(fp)
    env.close()
    elapsed = time.time() - t0
    print(f'DONE reveal={name} s{seed}  {elapsed/60:.1f} min', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    frac = float(sys.argv[1])
    seed = int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
    train_one(frac, seed, steps)
