"""Survival-landscape figure for the S3 bandit (Paper 1, Fig. draft).

Panel A: kappa_expq(p) over the condition-mixture simplex (closed form,
exact). Direction dependence of survival is visible as non-circular contours.
Panel B: two-channel interference — softq kappa(alpha) at a fixed asymmetric
mixture, closed form vs autograd (they coincide to machine precision).

Reads data/kappa/toy_fields/s3_survival.json; recomputes grids via the
verified closed forms for smoothness.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys_path = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

R = 1.0
N = 3


def pi_of(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def kappa_expq_closed(pi, p):
    dot = float(p @ pi)
    num = float(np.sum(pi ** 2 * (p - dot) ** 2))
    den = np.mean([float(np.sum(pi ** 2 * (np.eye(N)[i] - pi[i]) ** 2))
                   for i in range(N)])
    return num / max(den, 1e-300)


def kappa_softq_closed(pi, p, alpha):
    ell = np.log(pi)
    lbar = float(pi @ ell)
    dotp = float(p @ pi)
    nv = alpha * (ell - lbar) - 2 * R * (p - dotp)
    num = float(np.sum(pi ** 2 * nv ** 2))
    parts = [float(np.sum(pi ** 2 * (alpha * (ell - lbar)
                                     - 2 * R * (np.eye(N)[i] - pi[i])) ** 2))
             for i in range(N)]
    return num / max(float(np.mean(parts)), 1e-300)


def bary(pa, pb):
    """barycentric (pa,pb,pc) -> 2D plot coords."""
    pc = 1.0 - pa - pb
    return pb + pc / 2.0, pc * np.sqrt(3) / 2.0


def main():
    z = np.array([0.5, -0.2, -0.3])
    pi = pi_of(z)

    # ---- grid over simplex ----
    n = 220
    xs, ys, vals_e, vals_s = [], [], [], []
    alphas_grid = np.linspace(0.02, 0.98, n)
    A, Bg = np.meshgrid(alphas_grid, alphas_grid)
    for ia in range(n):
        for ib in range(n):
            pa, pb = A[ia, ib], Bg[ia, ib]
            pc = 1.0 - pa - pb
            if pc < 0.005:
                continue
            p = np.array([pa, pb, pc])
            x, y = bary(pa, pb)
            xs.append(x)
            ys.append(y)
            vals_e.append(kappa_expq_closed(pi, p))
            vals_s.append(kappa_softq_closed(pi, p, 1.0))
    xs, ys = np.array(xs), np.array(ys)
    ve, vs_ = np.array(vals_e), np.array(vals_s)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # Panel A: expq landscape
    ax = axes[0]
    cf = ax.tricontourf(xs, ys, ve, levels=18, cmap='viridis')
    plt.colorbar(cf, ax=ax, label=r'$\kappa_{\mathrm{expq}}(p)$')
    for j, lab in enumerate(['A', 'B', 'C']):
        x, y = bary(*(np.eye(N)[j][:2]))
        ax.annotate(f'p_{lab}=1', (x, y), ha='center', va='bottom',
                    fontsize=9, xytext=(x, y + 0.03))
    ax.set_title('(a) odd-field survival over mixture simplex\n'
                 r'(fixed $\pi$; direction-dependent)')
    ax.axis('off')

    # Panel B: softq landscape (interference visible)
    ax = axes[1]
    cf = ax.tricontourf(xs, ys, vs_, levels=18, cmap='magma')
    plt.colorbar(cf, ax=ax, label=r'$\kappa_{\mathrm{softq}}(p;\alpha{=}1)$')
    ax.set_title('(b) two-channel interference\n'
                 '(entropy channel can cancel mixture channel)')
    ax.axis('off')

    # Panel C: interference dip along alpha
    ax = axes[2]
    p_asym = np.array([0.7, 0.2, 0.1])
    alphas = np.logspace(-2, 1, 80)
    ks = [kappa_softq_closed(pi, p_asym, a) for a in alphas]
    data = json.load(open(os.path.join(ROOT, 'data', 'kappa', 'toy_fields',
                                       's3_survival.json')))
    scan_a = [float(a) for a in data['softq_scan']['alphas']]
    scan_k = [data['softq_scan']['kappa'][str(a)] for a in scan_a]
    ax.semilogx(alphas, ks, '-', lw=2, label='two-channel closed form')
    ax.plot(scan_a, scan_k, 'o', ms=7, label='autograd (canonical def.)')
    ax.axhline(kappa_expq_closed(pi, p_asym), color='gray', ls='--', lw=1,
               label='expq (no entropy channel)')
    ax.set_xlabel(r'entropy coefficient $\alpha$')
    ax.set_ylabel(r'$\kappa$')
    ax.set_title(r'(c) destructive interference: $\kappa(\alpha)$ non-monotone'
                 '\n' r'$p=(0.7,0.2,0.1)$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('Gradient survival under hidden S3-symmetric conditions '
                 '(three-condition matching bandit)', y=1.02)
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, 's3_survival_landscape.png')
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    print('saved:', out_png)


if __name__ == '__main__':
    main()
