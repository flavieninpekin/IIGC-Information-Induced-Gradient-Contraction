"""Extend oc_decomp / oc_baselines to checkpoints outside the 2-seed intersection,
using the exact engine measurement helpers (_collect, _components, _collect_rew).

Writes oc_decomp_ext_seed{s}.json and oc_baselines_b1_ext.json alongside the
engine-produced files in the server_tasks results dir.
"""
import json
import os
import sys

CLONE = r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code"
sys.path.insert(0, CLONE)
sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

import engine  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env  # noqa: E402

CHKPT = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"
RESULTS = r"C:\Users\Flavi\opencode\IIGC\data\kappa\server_tasks\results"

EXTRA = [("static", 43), ("dynamic", 47), ("dynamic", 48)]


def main():
    for mode, s in EXTRA:
        fp = os.path.join(CHKPT, f"overcookedv3_{mode}_seed{s}_final.zip")
        model = PPO.load(fp, device="cpu")
        model.policy.eval()
        env = OvercookedV3Env(mode=mode)
        g_c = engine._collect(model, env, "chef", 50)
        g_w = engine._collect(model, env, "waiter", 50)
        env.close()
        out = engine._components(g_c, g_w)
        path = os.path.join(RESULTS, f"oc_decomp_ext_seed{s}.json")
        with open(path, "w") as f:
            json.dump({mode: out}, f, indent=2, default=float)
        print(f"[oc_decomp_ext] s{s} {mode}: {out}", flush=True)

    out = {}
    for mode, s in EXTRA:
        fp = os.path.join(CHKPT, f"overcookedv3_{mode}_seed{s}_final.zip")
        model = PPO.load(fp, device="cpu")
        model.policy.eval()
        env = OvercookedV3Env(mode=mode)
        g_c, r_c = engine._collect_rew(model, env, "chef", 60)
        g_w, r_w = engine._collect_rew(model, env, "waiter", 60)
        env.close()
        comp = engine._components(g_c, g_w)
        allg = [g_c, g_w]
        import torch
        allg = torch.cat(allg)
        allr = torch.cat([r_c, r_w])
        out[f"{mode}_s{s}"] = {
            **comp,
            "grad_norm_mean": allg.norm(dim=1).mean().item(),
            "ep_reward_mean": allr.float().mean().item(),
            "ep_reward_std": allr.float().std().item(),
        }
        print(f"[oc_baselines_ext] {mode}_s{s}: kappa={comp['kappa_ep']:.3f} "
              f"rew={out[f'{mode}_s{s}']['ep_reward_mean']:.1f}", flush=True)
    with open(os.path.join(RESULTS, "oc_baselines_b1_ext.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("[oc_baselines_ext] saved", flush=True)


if __name__ == "__main__":
    main()
