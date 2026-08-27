"""S3 survival test: does a 3-condition (non-abelian-representation) bandit
reveal structure invisible in the Z2 mirror bandit?

Setup: 3 hidden conditions i in {0,1,2}, 3 actions, softmax policy pi(z),
analytic Q_i(a) = r*(2*[a==i] - 1)  ("match your partner", S3-permutation
symmetric). Canonical expected gradients (matches fields.py semantics):

  expq      : g_i = grad(-<pi, Q_i>)                       (full objective)
  reinforce : g_i = E_a~pi[ grad(-log pi(a) * Q_i(a)) ]    (theta-free weight;
              equals -grad<pi,Q> in expectation => same as expq up to sign)
  softq     : g_i = grad sum_a pi(a)(alpha log pi(a) - Q_i(a))

Mixture over conditions p (the training distribution of the hidden variable);
kappa(p) = ||E_i g_i||^2 / E_i ||g_i||^2   (uniform reference over conditions).

Hand-derived closed form (expq):
  g_i[b]  = 2 r pi_b (delta_ib - pi_i)
  hat g_p[b] = 2 r pi_b (p_b - <p, pi>)
  kappa_closed(p, pi) = sum_b pi_b^2 (p_b - <p,pi>)^2
                        / E_i sum_b pi_b^2 (delta_ib - pi_i)^2

THE QUESTION (Paper-1 death clause): in Z2 there is only ONE direction of
asymmetry away from the uniform mixture, so kappa depends on |amplitude| only.
In S3 the simplex of mixtures is 2-dimensional: two asymmetric distributions
EQUIDISTANT from uniform may excite the 2D standard representation differently.
If kappa differs across such directions (beyond noise) => genuine S3-only
structure => Paper 1 stands. If kappa is direction-invariant => the framework
collapses to "orthogonal components average out" => stop loss.

Output: data/kappa/toy_fields/s3_survival.json
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'data', 'kappa', 'toy_fields', 's3_survival.json')

R = 1.0
N_COND = 3


def q_vec(i):
    return np.array([r_ * R for r_ in (2 * np.eye(N_COND)[i])])


def expected_grad(z, i, field, alpha=0.2):
    """Canonical expected gradient under condition i."""
    q = q_vec(i)
    zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
    pi = torch.softmax(zt, dim=0)
    qt = torch.tensor(q, dtype=torch.double)
    if field == 'expq':
        loss = -(pi * qt).sum()
    elif field == 'softq':
        lp = torch.log(torch.clamp(pi, min=1e-300))
        loss = (pi * (alpha * lp - qt)).sum()
    elif field == 'reinforce':
        lp = torch.log(torch.clamp(pi, min=1e-300))
        # E_a~pi[ grad(-log pi(a) w(a)) ], theta-free w: sum_a pi_a grad(-lp_a w_a)
        ws = qt.detach()
        total = torch.zeros_like(zt)
        pi_d = pi.detach()
        for a in range(N_COND):
            if zt.grad is not None:
                zt.grad.zero_()
            (-lp[a] * ws[a]).backward(retain_graph=True)
            total = total + float(pi_d[a]) * zt.grad.detach().clone()
        return total.numpy()
    else:
        raise ValueError(field)
    loss.backward()
    return zt.grad.detach().numpy()


def kappa_mix(z, p, field, alpha=0.2):
    """kappa(p) = ||sum_i p_i g_i||^2 / E_i ||g_i||^2.
    Numerator uses the ACTUAL training mixture p over hidden conditions;
    denominator references uniform condition energy."""
    gs = np.stack([expected_grad(z, i, field, alpha) for i in range(N_COND)])
    m = np.asarray(p) @ gs
    es = float(m @ m)
    e_tot = float((gs ** 2).sum(axis=1).mean())
    return es / max(e_tot, 1e-300)


def kappa_expq_closed(z, p):
    pi = np.exp(np.asarray(z) - np.max(z))
    pi = pi / pi.sum()
    dot = float(p @ pi)
    num = float(np.sum(pi ** 2 * (np.asarray(p) - dot) ** 2))
    den_parts = []
    for i in range(N_COND):
        d = np.eye(N_COND)[i] - pi[i]
        den_parts.append(float(np.sum(pi ** 2 * d ** 2)))
    den = float(np.mean(den_parts))
    return num / max(den, 1e-300)


def kappa_softq_closed(z, p, alpha):
    """Two-channel closed form.
    Per-condition: g_i[b] = pi_b [ alpha(l_b - lbar) + 2R(delta_ib - pi_i) ]
    Mixture:       g_p[b] = pi_b [ alpha(l_b - lbar) + 2R(p_b - <p,pi>) ]
    (sign convention irrelevant). The alpha-channel (policy-intrinsic,
    direction l - <pi,l>) and the mixture-channel (direction p - <p,pi>)
    live in the SAME 2D tangent space => they interfere; the observed
    non-monotone kappa(alpha) is their destructive alignment."""
    pi = np.exp(np.asarray(z) - np.max(z))
    pi = pi / pi.sum()
    ell = np.log(pi)
    lbar = float(pi @ ell)
    dotp = float(np.asarray(p) @ pi)
    # softq loss = alpha<pi,logpi> - <pi,Q>: mixture channel enters with MINUS
    num_vec = alpha * (ell - lbar) - 2 * R * (np.asarray(p) - dotp)
    num = float(np.sum(pi ** 2 * num_vec ** 2))
    den_parts = []
    for i in range(N_COND):
        dvec = alpha * (ell - lbar) - 2 * R * (np.eye(N_COND)[i] - pi[i])
        den_parts.append(float(np.sum(pi ** 2 * dvec ** 2)))
    return num / max(float(np.mean(den_parts)), 1e-300)


def main():
    out = {'setup': 'S3 3-condition matching bandit, canonical expected grads',
           'validation': {}, 'symmetric_design': {}, 'direction_test': {},
           'softq_scan': {}, 'heatmap': {}}

    rng = np.random.default_rng(41)

    # --- 1. closed-form vs autograd (machine precision) ---
    worst = 0.0
    for _ in range(200):
        z = rng.normal(size=N_COND) * 1.5
        p = rng.dirichlet(np.ones(N_COND))
        k1 = kappa_expq_closed(z, p)
        k2 = kappa_mix(z, p, 'expq')
        worst = max(worst, abs(k1 - k2))
    out['validation']['expq_closed_vs_autograd_max_abs_err'] = worst
    print(f'expq closed-vs-autograd max|err| = {worst:.3e}')

    # --- 2. symmetric design: odd fields must die exactly ---
    z_test = [0.5, -0.2, -0.3]
    p_unif = np.ones(N_COND) / N_COND
    row = {}
    for f in ['reinforce', 'expq']:
        row[f] = kappa_mix(z_test, p_unif, f)
    out['symmetric_design'] = {'z': z_test, 'p': p_unif.tolist(), **row}
    print(f'uniform mixture: reinforce κ={row["reinforce"]:.2e} '
          f'expq κ={row["expq"]:.2e} (expect ~0)')

    # --- 3. THE DIRECTION TEST (death clause) ---
    # Fix L1 distance from uniform; sweep directions on the 2-simplex.
    pi_fix = np.exp(np.array([0.5, -0.2, -0.3]) -
                    np.max([0.5, -0.2, -0.3]))
    pi_fix = pi_fix / pi_fix.sum()

    def simplex_path(dist, n_dir=24):
        """Points p on the simplex with fixed L1 distance `dist` from center."""
        pts = []
        c = np.ones(N_COND) / N_COND
        for t in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
            v = np.array([np.cos(t), np.sin(t)])
            B = np.array([[1.0, 0.0],
                          [-0.5, np.sqrt(3) / 2],
                          [-0.5, -np.sqrt(3) / 2]])  # columns: sum-zero basis
            dvec = B @ v
            scale = dist / np.abs(dvec).sum()  # L1(c, c+s*dvec) = dist
            pts.append(np.clip(c + scale * dvec, 1e-6, None))
        return np.array(pts)

    dirs = {}
    for dist in [0.15, 0.3, 0.45]:
        pts = simplex_path(dist)
        rows = {}
        for f in ['reinforce', 'expq', 'softq']:
            ks = [kappa_mix(z_test, p, f) for p in pts]
            rows[f] = {'kappa_mean': float(np.mean(ks)),
                       'kappa_std': float(np.std(ks)),
                       'kappa_min': float(np.min(ks)),
                       'kappa_max': float(np.max(ks)),
                       'relative_spread': float(np.std(ks) / max(np.mean(ks), 1e-12))}
        dirs[f'dist={dist}'] = rows
        print(f'L1 dist={dist}: ' + '  '.join(
            f'{f} κ={rows[f]["kappa_mean"]:.4f}±{rows[f]["kappa_std"]:.4f}'
            f' [{rows[f]["kappa_min"]:.4f},{rows[f]["kappa_max"]:.4f}]'
            for f in ['reinforce', 'expq', 'softq']))
    out['direction_test'] = {
        'pi': pi_fix.tolist(), 'note': ('fixed-L1-direction sweep on the '
        'condition simplex; S3-only structure exists iff relative_spread is '
        'materially nonzero (Z2 has a single direction, spread undefined)'),
        'results': dirs}

    # --- 4. softq alpha scan at a fixed asymmetric mixture ---
    p_asym = np.array([0.7, 0.2, 0.1])
    alphas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    scan = {}
    worst_sq = 0.0
    for a in alphas:
        k_auto = kappa_mix(z_test, p_asym, 'softq', alpha=a)
        k_closed = kappa_softq_closed(z_test, p_asym, a)
        scan[str(a)] = k_auto
        worst_sq = max(worst_sq, abs(k_auto - k_closed))
    out['softq_scan'] = {'p': p_asym.tolist(), 'alphas': alphas, 'kappa': scan,
                         'two_channel_closed_vs_autograd_max_abs_err': worst_sq,
                         'note': ('non-monotone kappa(alpha) = destructive '
                                  'interference between the mixture channel '
                                  '(2R(p-<p,pi>)) and the entropy channel '
                                  '(alpha(l-<pi,l>)); impossible in Z2 where '
                                  'the tangent space is 1D')}
    print('softq α scan @ p=(0.7,0.2,0.1):',
          {k: round(v, 4) for k, v in scan.items()})
    print(f'softq two-channel closed-vs-autograd max|err| = {worst_sq:.3e}')

    # --- 5. heatmap over the simplex (for the survival-landscape figure) ---
    n_grid = 25
    grid = []
    for ia in range(n_grid):
        rowv = []
        for ib in range(n_grid):
            pa = 0.02 + 0.96 * ia / (n_grid - 1)
            pb = 0.02 + 0.96 * ib / (n_grid - 1)
            pc = 1.0 - pa - pb
            if pc < 0.01:
                rowv.append(None)
                continue
            p = np.array([pa, pb, pc])
            rowv.append({
                'expq': kappa_expq_closed(z_test, p),
                'softq': kappa_mix(z_test, p, 'softq'),
            })
        grid.append(rowv)
    out['heatmap'] = {'z': z_test, 'n_grid': n_grid,
                      'note': 'grid[ia][ib]; pa, pb linspace 0.02..0.98; pc=1-pa-pb',
                      'grid': grid}

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
