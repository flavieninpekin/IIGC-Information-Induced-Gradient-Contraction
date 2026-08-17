"""Launch reveal-model training in parallel (max 4 concurrent).

Trains the full grid: reveal p in {0.00, 0.05, ..., 1.00} (21 levels)
x seeds {41..46} (6 seeds). Skips models that already exist.
"""
import os, sys, subprocess, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_reveal')
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), 'train_reveal.py')
MAX_WORKERS = 4
PYTHON_EXE = sys.executable

LEVELS = [round(i / 20, 2) for i in range(21)]      # 0.00, 0.05, ..., 1.00
SEEDS = [41, 42, 43, 44, 45, 46]

JOBS = [(frac, seed) for frac in LEVELS for seed in SEEDS]

remaining = []
for frac, seed in JOBS:
    name = f'{frac:.2f}'
    fp = os.path.join(MODEL_DIR, f'ppo_reveal_{name}_s{seed}.zip')
    if os.path.exists(fp):
        print(f'SKIP  {name} s{seed} (exists)')
    else:
        remaining.append((frac, seed))

print(f'\n{len(JOBS)} total, {len(JOBS) - len(remaining)} done, '
      f'{len(remaining)} to train\n')

if not remaining:
    print('All models already exist.')
    sys.exit(0)

procs = []
for frac, seed in remaining:
    while sum(1 for p in procs if p.poll() is None) >= MAX_WORKERS:
        time.sleep(5)

    name = f'{frac:.2f}'
    log = os.path.join(ROOT, 'data', 'logs', f'train_reveal_{name}_s{seed}.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    lf = open(log, 'w')
    print(f'LAUNCH reveal={name} s{seed}', flush=True)
    p = subprocess.Popen(
        [PYTHON_EXE, TRAIN_SCRIPT, str(frac), str(seed)],
        stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': os.path.join(ROOT, 'src')},
    )
    procs.append(p)

for p in procs:
    p.wait()

print('\nAll training done.')
