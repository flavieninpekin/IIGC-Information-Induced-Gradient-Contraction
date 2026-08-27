"""Batch-train the 8 missing 510K reveal models for the field-axis grid.

Levels 0.50/1.00 x seeds. Uses local train_reveal.py (positional steps arg).
"""
import os
import subprocess
import sys
import time

MODEL_DIR = r"C:\Users\Flavi\opencode\IIGC\data\models_reveal"
SCRIPT = r"C:\Users\Flavi\opencode\IIGC\experiments\common_basis\reveal\train_reveal.py"
PYTHONPATH = r"C:\Users\Flavi\opencode\IIGC\src"

TARGETS = [(0.50, 41), (0.50, 42), (0.50, 45), (0.50, 46),
           (1.00, 43), (1.00, 44), (1.00, 45), (1.00, 46)]

env = {**os.environ, "PYTHONPATH": PYTHONPATH}

for frac, seed in TARGETS:
    fp = os.path.join(MODEL_DIR, f"ppo_reveal_{frac:.2f}_s{seed}.zip")
    if os.path.exists(fp):
        os.remove(fp)
        print("removed corrupt", os.path.basename(fp), flush=True)

for frac, seed in TARGETS:
    t0 = time.time()
    r = subprocess.run([sys.executable, SCRIPT, f"{frac:.2f}", str(seed), "1000000"],
                       env=env, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    fp = os.path.join(MODEL_DIR, f"ppo_reveal_{frac:.2f}_s{seed}.zip")
    ok = r.returncode == 0 and os.path.exists(fp)
    print(f"[{dt:5.1f} min] p={frac:.2f} s{seed}: exit={r.returncode} "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print(r.stdout[-1200:], flush=True)
        print(r.stderr[-1200:], flush=True)
print("ALL DONE", flush=True)
