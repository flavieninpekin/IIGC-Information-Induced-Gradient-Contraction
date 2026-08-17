"""Plot the refined 510K reveal curve.

Reads data/kappa/510k_reveal/results.json and produces:
  1. aggregate  : mean kappa(p) +/- std band
  2. per-seed   : each seed's kappa(p) curve (overlaid, and small-multiples)
  3. variance   : mean var_within / var_between vs p (the sigma^2 decomposition)

Saves PNGs to data/kappa/510k_reveal/figures/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RES = os.path.join(ROOT, 'data', 'kappa', '510k_reveal', 'results.json')
FIG_DIR = os.path.join(ROOT, 'data', 'kappa', '510k_reveal', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

with open(RES) as f:
    data = json.load(f)

levels = sorted([float(k) for k in data.keys()])
seeds = sorted({int(s[1:]) for lv in data for s in data[lv]})


def col(level, seed, key, default=None):
    d = data.get(str(level), {}).get(f's{seed}', {})
    v = d.get(key)
    return v if v is not None else default


# ---- Aggregate: mean +/- std of kappa_mix ----
means, stds, ns = [], [], []
for lv in levels:
    vals = [col(lv, s, 'kappa_mix') for s in seeds]
    vals = [v for v in vals if v is not None]
    if vals:
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        ns.append(len(vals))
    else:
        means.append(np.nan); stds.append(np.nan); ns.append(0)
means = np.array(means); stds = np.array(stds); ns = np.array(ns)
lv = np.array(levels)

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(lv, means, yerr=stds, fmt='o-', capsize=3, label='mean +/- std')
for i, n in enumerate(ns):
    ax.annotate(f'n={n}', (lv[i], means[i]), textcoords='offset points',
                xytext=(0, 6), fontsize=7, ha='center')
ax.set_xlabel('reveal fraction p')
ax.set_ylabel(r'$\kappa$ (mixture protocol)')
ax.set_title('510K reveal: aggregate kappa curve')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'kappa_aggregate.png'), dpi=200)
print('saved kappa_aggregate.png')

# ---- Per-seed curves (overlaid) ----
fig, ax = plt.subplots(figsize=(9, 5))
for s in seeds:
    y = [col(lv, s, 'kappa_mix') for lv in levels]
    ax.plot(lv, y, 'o-', ms=4, lw=1.2, alpha=0.8, label=f's{s}')
ax.axhline(0.5, color='k', lw=0.5, ls='--', alpha=0.5)
ax.set_xlabel('reveal fraction p')
ax.set_ylabel(r'$\kappa$')
ax.set_title('510K reveal: per-seed kappa curves')
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'kappa_per_seed.png'), dpi=200)
print('saved kappa_per_seed.png')

# ---- Per-seed small multiples ----
ncol = 3
nrow = int(np.ceil(len(seeds) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 2.8 * nrow), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
for i, s in enumerate(seeds):
    ax = axes[i]
    y = [col(lv, s, 'kappa_mix') for lv in levels]
    ax.plot(lv, y, 'o-', ms=3, lw=1)
    ax.set_title(f'seed {s}', fontsize=9)
    ax.grid(alpha=0.3)
for j in range(len(seeds), len(axes)):
    axes[j].axis('off')
fig.suptitle('510K reveal: kappa per seed', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'kappa_small_multiples.png'), dpi=200)
print('saved kappa_small_multiples.png')

# ---- Variance decomposition (mean +/- std) ----
for key, ylab, fname in [
    ('var_within', r'$Var_{within}$ (sigma^2)', 'var_within.png'),
    ('var_between', r'$Var_{between}$', 'var_between.png'),
]:
    vm, vs, vn = [], [], []
    for p in levels:
        vals = [col(p, s, key) for s in seeds]
        vals = [v for v in vals if v is not None]
        if vals:
            vm.append(np.mean(vals)); vs.append(np.std(vals)); vn.append(len(vals))
        else:
            vm.append(np.nan); vs.append(np.nan); vn.append(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(lv, vm, yerr=vs, fmt='o-', capsize=3)
    ax.set_xlabel('reveal fraction p')
    ax.set_ylabel(ylab)
    ax.set_title(f'510K reveal: {key} vs p')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200)
    print(f'saved {fname}')

# ---- Per-seed var_within ----
fig, ax = plt.subplots(figsize=(9, 5))
for s in seeds:
    y = [col(lv, s, 'var_within') for lv in levels]
    ax.plot(lv, y, 'o-', ms=4, lw=1.2, alpha=0.8, label=f's{s}')
ax.set_xlabel('reveal fraction p')
ax.set_ylabel(r'$Var_{within}$')
ax.set_title('510K reveal: per-seed var_within')
ax.grid(alpha=0.3)
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'var_within_per_seed.png'), dpi=200)
print('saved var_within_per_seed.png')

print('\nAll figures saved to', FIG_DIR)
