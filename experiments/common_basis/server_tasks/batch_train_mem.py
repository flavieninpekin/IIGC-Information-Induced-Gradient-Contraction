"""Batch-train Overcooked dynamic+memory models (m=4/8/16 x seeds 41-43)."""
import os
import subprocess
import sys
import time

CHKPT = r"C:\Users\Flavi\AppData\Local\Temp\opencode\chkpt_clean"
SCRIPT = (r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code"
          r"\experiments\common_basis\stuck_detect\train_overcooked_memory.py")
PYTHONPATH = r"C:\Users\Flavi\opencode\IIGC\src" + os.pathsep + \
    r"C:\Users\Flavi\AppData\Local\Temp\opencode\flavien-code"

env = {**os.environ, "PYTHONPATH": PYTHONPATH}

for mem in (4, 8, 16):
    for seed in (41, 42, 43):
        name = f"overcooked_mem_dynamic_m{mem}_s{seed}.zip"
        fp = os.path.join(CHKPT, name)
        if os.path.exists(fp):
            print(f"skip {name} (exists)", flush=True)
            continue
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, SCRIPT, "dynamic", str(mem), str(seed),
             "--model-dir", CHKPT],
            env=env, capture_output=True, text=True)
        dt = (time.time() - t0) / 60
        ok = r.returncode == 0 and os.path.exists(fp)
        print(f"[{dt:5.1f} min] m{mem} s{seed}: exit={r.returncode} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(r.stdout[-1500:], flush=True)
            print(r.stderr[-1500:], flush=True)
print("ALL DONE", flush=True)
