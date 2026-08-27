"""Post-training measurement suite on the full 16-model grid (fresh checkpoints).

Runs the exact engine task functions with todos.json configs.
"""
import os
import sys

import torch._dynamo  # pre-import before gym/overcooked to avoid torch 2.11 flake

CLONE = r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code"
sys.path.insert(0, CLONE)
sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

import engine  # noqa: E402

BASE = r"C:\Users\Flavi\opencode\IIGC\data\kappa\server_tasks"
engine.RESULTS = os.path.join(BASE, "results")
engine.CHKPT = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"
engine.META = os.path.join(BASE, ".meta")
engine.LOGS = os.path.join(BASE, "logs")
os.makedirs(engine.RESULTS, exist_ok=True)
os.makedirs(engine.LOGS, exist_ok=True)

SEEDS = list(range(41, 49))

TASKS = {
    "oc_decomp": {"id": "oc_decomp_b1", "type": "oc_decomp",
                  "seeds": SEEDS, "n_eps": 50},
    "oc_baselines": {"id": "oc_baselines_b1", "type": "oc_baselines",
                     "seeds": SEEDS, "n_eps": 60},
    "oc_n_protocol": {"id": "oc_n_protocol", "type": "oc_n_protocol",
                      "seeds": [41], "pool": 500, "boot": 100,
                      "sizes": [10, 20, 50, 100, 200, 500]},
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(TASKS)
    for t in wanted:
        cfg = TASKS[t]
        print(f"\n=== {t}: {cfg}", flush=True)
        wrapped = engine.REGISTRY[cfg["type"]].__wrapped__
        wrapped(cfg)
        print(f"=== {t} done", flush=True)
