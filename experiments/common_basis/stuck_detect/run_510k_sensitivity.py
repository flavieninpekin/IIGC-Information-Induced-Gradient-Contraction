"""Policy sensitivity to the relation bits: a low-variance stuck detector.

For a fixed game state, flip the team bits in the observation and measure how
the policy's action distribution changes: ||pi(a|obs) - pi(a|obs_flipped)||.

- A policy that IGNORES the relation (p=0.0 reveal model) -> sensitivity ~ 0
- A policy that USES it (p=1.0) -> sensitivity > 0

Pure forward passes, no per-episode gradient noise — a direct probe of
"does the policy condition on the hidden relation" (the stuck signature).
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F

from sb3_contrib import MaskablePPO
from iigc.envs._510k.env import FiveTenKEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_reveal')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'stuck_detect')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 20
MODELS = [(0.00, 41), (0.00, 42), (0.50, 41), (1.00, 41), (1.00, 42)]


def policy_logits(model, obs):
    """Raw policy-network logits (bypasses get_distribution which ignores the
    relation-bit difference in MaskablePPO)."""
    t = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        feat = model.policy.extract_features(t)
        latent = model.policy.mlp_extractor.forward_actor(feat)
        return model.policy.action_net(latent)


def sensitivity(model, env, n_eps=N_EPS, base_seed=100):
    """Mean over states of ||pi(a|obs) - pi(a|obs_flipped)|| (L2)."""
    diffs = []
    for ep in range(n_eps):
        torch.manual_seed(base_seed + ep); np.random.seed(base_seed + ep)
        obs, info = env.reset()
        done = False
        while not done:
            obs_f = obs.copy()
            obs_f[-4:] = 1.0 - obs_f[-4:]   # flip team bits
            p1 = F.softmax(policy_logits(model, obs), dim=-1)
            p2 = F.softmax(policy_logits(model, obs_f), dim=-1)
            diffs.append((p1 - p2).norm(dim=1).item())
            mask = env.unwrapped._get_action_mask()
            a, _ = model.predict(obs, action_masks=mask, deterministic=False)
            obs, r, done, trunc, info = env.step(int(a))
    env.close()
    return {'L2': float(np.mean(diffs)), 'max': float(max(diffs)),
            'n_states': len(diffs)}


def _logits(d):
    """Extract action logits from an SB3 maskable distribution object."""
    cat = getattr(d, 'distribution', None)
    if cat is not None:
        logits = getattr(cat, 'logits', None)
        if logits is not None:
            return logits
        probs = getattr(cat, 'probs', None)
        if probs is not None:
            return torch.log(probs + 1e-10)
    raise TypeError(type(d))


def main():
    print(f'{"model":>14} {"L2 sens":>9} {"KL sens":>9} {"n_states":>9}')
    results = {}
    for p, seed in MODELS:
        fp = os.path.join(MODEL_DIR, f'ppo_reveal_{p:.2f}_s{seed}.zip')
        if not os.path.exists(fp):
            print(f'MISSING {fp}')
            continue
        model = MaskablePPO.load(fp, device='cpu')
        model.policy.eval()
        env = FiveTenKEnv(mode='obvious')
        s = sensitivity(model, env)
        env.close()
        results[f'p{p:.2f}_s{seed}'] = s
        print(f'p={p:.2f} s{seed}  L2={s["L2"]:>9.4f}  max={s["max"]:>9.4f}  n={s["n_states"]}')

    with open(os.path.join(OUT_DIR, 'sensitivity.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "sensitivity.json")}')


if __name__ == '__main__':
    main()
