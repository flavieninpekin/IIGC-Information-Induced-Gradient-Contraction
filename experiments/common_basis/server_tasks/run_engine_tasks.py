"""Run IIGC server engine tasks locally with the available checkpoints.

Uses the exact engine task functions from the cloned flavieninpekin/code repo,
with RESULTS/CHKPT redirected to local dirs. Tasks with checkpoints missing
for some seeds are run on the intersection seeds, then a supplement script
extends them using the same engine measurement helpers.

Usage: python run_engine_tasks.py <task> [more tasks...]
"""
import os
import sys

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

TASKS = {
    "oc_decomp": {"id": "oc_decomp_b1", "type": "oc_decomp",
                  "seeds": [41, 42], "n_eps": 50},
    "oc_baselines": {"id": "oc_baselines_b1", "type": "oc_baselines",
                     "seeds": [41, 42], "n_eps": 60},
    "oc_n_protocol": {"id": "oc_n_protocol", "type": "oc_n_protocol",
                      "seeds": [41], "pool": 500, "boot": 100,
                      "sizes": [10, 20, 50, 100, 200, 500]},
    "oc_mem": {"id": "oc_mem", "type": "oc_mem", "seeds": [41, 42, 43],
               "memory": [4]},
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(TASKS)
    for t in wanted:
        cfg = TASKS[t]
        print(f"\n=== {t}: {cfg}", flush=True)
        wrapped = engine.REGISTRY[cfg["type"]].__wrapped__
        wrapped(cfg)
        print(f"=== {t} done", flush=True)
