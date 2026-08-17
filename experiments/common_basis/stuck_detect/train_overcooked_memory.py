"""Train Overcooked dynamic + memory (intervention).

Intervention test: the diagnostic says the dynamic run is stuck because the
role (a strong behavioral driver) is hidden and not inferable from the current
memoryless observation. Giving the policy MEMORY (a history stack) lets it
infer the role from behavior — a fix that is NOT "reveal the role" (= static).

Usage: python train_overcooked_memory.py dynamic <memory> <seed>
Saves to data/models_overcooked/overcooked_mem_dynamic_m<mem>_s<seed>.zip
"""
import os, sys, time, multiprocessing
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from iigc.envs._overcooked.overcooked_memory_env import OvercookedMemoryEnv, memory_obs_dim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_overcooked')
os.makedirs(MODEL_DIR, exist_ok=True)

N_ENVS = 8
TOTAL_STEPS = 1_000_000
MEMORY = 4


def train(mode='dynamic', memory=MEMORY, seed=42):
    name = f'overcooked_mem_{mode}_m{memory}_s{seed}'
    fp = os.path.join(MODEL_DIR, f'{name}.zip')
    if os.path.exists(fp):
        print(f'SKIP {name} (exists)')
        return fp

    def make():
        return OvercookedMemoryEnv(mode=mode, memory=memory)

    env = SubprocVecEnv([make for _ in range(N_ENVS)], start_method='spawn')
    env = VecMonitor(env)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256,
                batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                verbose=0, seed=seed, device='cuda')
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS)
    model.save(fp)
    env.close()
    print(f'DONE {name} in {(time.time()-t0)/60:.1f} min')
    return fp


if __name__ == '__main__':
    multiprocessing.freeze_support()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dynamic'
    memory = int(sys.argv[2]) if len(sys.argv) > 2 else MEMORY
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    train(mode, memory, seed)
