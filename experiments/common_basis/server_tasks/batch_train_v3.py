"""Batch-train Overcooked V3 static/dynamic x seeds 41-48 with the server-exact
script, overwriting old (degenerate) checkpoints in the working dir."""
import os
import subprocess
import sys
import time

CHKPT = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"
SCRIPT = (r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code"
          r"\experiments\common_basis\stuck_detect\train_overcooked_ppo.py")
PY = r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code"
PYTHONPATH = r"C:\Users\Flavi\opencode\IIGC\src" + os.pathsep + PY

for name in os.listdir(CHKPT):
    if name.startswith("overcookedv3_") and name.endswith(".zip"):
        os.remove(os.path.join(CHKPT, name))
        print("removed old", name, flush=True)

env = {**os.environ, "PYTHONPATH": PYTHONPATH}
order = []
for mode in ("static", "dynamic"):
    for seed in range(41, 49):
        order.append((mode, seed))

for mode, seed in order:
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, SCRIPT, mode, str(seed),
         "--steps", "1000000", "--model-dir", CHKPT],
        env=env, capture_output=True, text=True)
    dt = (time.time() - t0) / 60
    ok = r.returncode == 0
    print(f"[{dt:5.1f} min] {mode} s{seed}: exit={r.returncode} "
          f"{'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        print(r.stdout[-1500:], flush=True)
        print(r.stderr[-1500:], flush=True)
print("ALL DONE", flush=True)
