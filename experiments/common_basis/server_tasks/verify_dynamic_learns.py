"""Verify dynamic actually learns: free-rollout reward across training snapshots."""
import sys
import numpy as np
import torch

import torch._dynamo  # noqa: F401

sys.path.insert(0, r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code")
sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

import engine  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env  # noqa: E402

CHKPT = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"


def eval_free(mode, s, steps, n=10):
    if steps == 1000000:
        fp = rf"{CHKPT}\overcookedv3_{mode}_seed{s}_final.zip"
    else:
        fp = rf"{CHKPT}\overcookedv3_{mode}_seed{s}_{steps}_steps.zip"
    model = PPO.load(fp, device="cpu")
    model.policy.eval()
    env = OvercookedV3Env(mode=mode)
    rews = []
    for i in range(n):
        torch.manual_seed(100 + i)
        np.random.seed(100 + i)
        g, r = engine._ep_grad_rew(model, env)
        rews.append(r)
    env.close()
    return float(np.mean(rews))


def main():
    print("=== dynamic learning curve (free-rollout reward) ===")
    for s in (41, 44, 48):
        a = eval_free("dynamic", s, 400000)
        b = eval_free("dynamic", s, 800000)
        c = eval_free("dynamic", s, 1000000)
        print(f"dyn s{s}: 400K={a:6.1f}  800K={b:6.1f}  1M={c:6.1f}")
    print("=== static learning curve ===")
    for s in (41, 44, 48):
        a = eval_free("static", s, 400000)
        b = eval_free("static", s, 800000)
        c = eval_free("static", s, 1000000)
        print(f"sta s{s}: 400K={a:6.1f}  800K={b:6.1f}  1M={c:6.1f}")


if __name__ == "__main__":
    main()
