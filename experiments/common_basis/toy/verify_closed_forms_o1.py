"""O1: closed-form kappa under the CANONICAL measurement definition.

Canonical definition (matches src/iigc/metrics/fields.py in expectation):

  Policy-gradient-style fields (reinforce / awr / softmaxq):
      g_r = E_{a~pi}[ grad_theta( -log pi_theta(a) * w_r(a; theta) ) ]
  Full-objective fields (expq / softq / gibbs_expq):
      g_r = grad_theta L_r(theta)

Mirror bandit: Q_A = (+r, -r), Q_B = (-r, +r); pi = softmax(z), p = pi_0,
delta = z_0 - z_1, every gradient is parallel to u = (1, -1), so with
g_r = c_r * u:   kappa = (c_A + c_B)^2 / ((c_A + c_B)^2 + (c_A - c_B)^2).

Closed forms derived by hand (see notes, Prop. O1a/O1b/O1c):

  reinforce (no baseline, w = Q_r)     : c_r = -+ 2 r p q                -> kappa = 0
  expq                                 : c_r = -+ 2 r p q                -> kappa = 0
  softmaxq / awr-nobase / gibbs_expq   : elementwise theta-free weight   -> kappa = 0
  softq  (O1a): c_r = p q ( alpha ln(p/q) -+ 2 r )
      kappa = A^2 / (A^2 + 4 r^2),  A = alpha ln(p/(1-p))
  awr-baseline (O1b): with x = 2 r / tau,
      A0 = e^{x q}, A1 = e^{-x p},  M_A = p ln p A0 + q ln q A1
      B0 = e^{-x q}, B1 = e^{x p},  M_B = p ln p B0 + q ln q B1
      c_A = p q (A1 - A0 + x M_A),  c_B = p q (B1 - B0 - x M_B)
      (kappa = 0 exactly at p = 1/2 for every tau; interior max in tau otherwise)

Validation: three independent routes must agree.
  (1) closed form (numpy, double)
  (2) autograd of the explicit expected loss, outer pi weight stop-gradiented
      (this is exactly E[per-sample grad])
  (3) Monte Carlo of the fields.py-style sampled loss (large N; statistical tol)

Output: data/kappa/toy_fields/o1_closed_forms.json
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'data', 'kappa', 'toy_fields', 'o1_closed_forms.json')

Z0 = [0.3716251850128174, -0.23901528120040894]  # stored init from theory_toy.json
Z_PEAKED = [2.0, -2.0]
ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
TAUS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


# ----------------------------------------------------------------------------
# (1) closed forms
# ----------------------------------------------------------------------------

def _pq(delta):
    p = 1.0 / (1.0 + np.exp(-delta))
    return p, 1.0 - p


def closed_c(field, delta, r, tau=1.0, alpha=1.0):
    """Return (c_A, c_B): scalar coefficients along u = (1, -1)."""
    p, q = _pq(delta)
    if field in ('reinforce', 'expq'):
        return -2 * r * p * q, 2 * r * p * q
    if field == 'softq':
        common = p * q * alpha * np.log(p / q)
        return common - 2 * r * p * q, common + 2 * r * p * q
    if field == 'awr':
        x = 2.0 * r / tau
        A0, A1 = np.exp(x * q), np.exp(-x * p)
        B0, B1 = np.exp(-x * q), np.exp(x * p)
        MA = p * np.log(p) * A0 + q * np.log(q) * A1
        MB = p * np.log(p) * B0 + q * np.log(q) * B1
        cA = p * q * (A1 - A0 + x * MA)
        cB = p * q * (B1 - B0 - x * MB)
        return cA, cB
    raise ValueError(field)


def closed_kappa(field, delta, r, tau=1.0, alpha=1.0):
    cA, cB = closed_c(field, delta, r, tau, alpha)
    s, d = cA + cB, cA - cB
    denom = s * s + d * d
    return (s * s / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------------
# (2) autograd of the explicit expected per-sample gradient
# ----------------------------------------------------------------------------

def autograd_expected_grad(z, r, field, tau=1.0, alpha=1.0, relation='A'):
    """g_r = sum_a pi_a * grad[-log pi(a) * w_a]  (pi_a detached),
    or full-objective grad for expq/softq."""
    q_vec = np.array([r, -r]) if relation == 'A' else np.array([-r, r])
    zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
    pi = torch.softmax(zt, dim=0)
    qt = torch.tensor(q_vec, dtype=torch.double)

    if field == 'expq':
        loss = -(pi * qt).sum()
        loss.backward()
        return zt.grad.detach().numpy()
    if field == 'softq':
        lp = torch.log(torch.clamp(pi, min=1e-300))
        loss = (pi * (alpha * lp - qt)).sum()
        loss.backward()
        return zt.grad.detach().numpy()
    if field == 'awr':
        v = (pi * qt).sum()
        ws = torch.exp((qt - v) / tau)          # differentiable through V
    elif field == 'reinforce':
        ws = qt.detach().clone()
    else:
        raise ValueError(field)

    lp = torch.log(torch.clamp(pi, min=1e-300))
    g = np.zeros(2)
    pi_d = pi.detach()
    for a in range(2):
        if zt.grad is not None:
            zt.grad.zero_()
        loss_a = (-lp[a] * ws[a])
        loss_a.backward(retain_graph=True)
        g += float(pi_d[a]) * zt.grad.detach().numpy()
    return g


def autograd_expected_kappa(z, r, field, tau=1.0, alpha=1.0):
    ga = autograd_expected_grad(z, r, field, tau, alpha, 'A')
    gb = autograd_expected_grad(z, r, field, tau, alpha, 'B')
    m = (ga + gb) / 2.0
    d = (ga - gb) / 2.0
    es, ec = float(m @ m), float(d @ d)
    return es / max(es + ec, 1e-300)


# ----------------------------------------------------------------------------
# (3) Monte Carlo of fields.py-style sampled loss (statistical anchor)
# ----------------------------------------------------------------------------

def mc_kappa(delta, r, field, n=400_000, tau=1.0, alpha=1.0, seed=0):
    """Sample relations/actions; per-sample grad of -log pi(a) w(a) with full
    autograd (w differentiable through V). Grouped-count evaluation of the
    plain MC estimator (identical in law, far cheaper)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + np.exp(-delta))
    rels = rng.integers(0, 2, size=n)
    acts = rng.random(n) < p          # True -> action 0
    grads = {0: np.zeros((2,)), 1: np.zeros((2,))}
    counts = {0: 0, 1: 0}
    for rel in (0, 1):
        sel = rels == rel
        counts[rel] = int(sel.sum())
        for a in (False, True):
            m = sel & (acts == a)
            ai = 0 if a else 1
            if m.sum() == 0:
                continue
            qv = np.array([r, -r]) if rel == 0 else np.array([-r, r])
            zt = torch.tensor([delta / 2.0, -delta / 2.0], dtype=torch.double,
                              requires_grad=True)
            pit = torch.softmax(zt, dim=0)
            qt = torch.tensor(qv, dtype=torch.double)
            if field == 'reinforce':
                rew = qv[ai]                       # one-step return = Q(a)
                w = torch.tensor(rew, dtype=torch.double)
            elif field == 'awr':
                w = torch.exp((qt[ai] - (pit * qt).sum()) / tau)
            elif field == 'softq':
                lp_all = torch.log(torch.clamp(pit, min=1e-300))
                loss_full = (pit * (alpha * lp_all - qt)).sum()
                loss_full.backward()
                grads[rel] += zt.grad.detach().numpy() * int(m.sum())
                zt.grad.zero_()
                continue
            elif field == 'expq':
                loss_full = -(pit * qt).sum()
                loss_full.backward()
                grads[rel] += zt.grad.detach().numpy() * int(m.sum())
                continue
            else:
                raise ValueError(field)
            lp = torch.log(torch.clamp(pit[ai], min=1e-300))
            loss = -lp * w
            loss.backward()
            grads[rel] += zt.grad.detach().numpy() * int(m.sum())
    ga = grads[0] / max(counts[0], 1)
    gb = grads[1] / max(counts[1], 1)
    mm, dd = (ga + gb) / 2.0, (ga - gb) / 2.0
    es, ec = float(mm @ mm), float(dd @ dd)
    return es / max(es + ec, 1e-300)


# ----------------------------------------------------------------------------
def main():
    out = {'definition': 'canonical expected-sampled grad (matches fields.py)',
           'validation': {}, 'softq': {}, 'awr': {}, 'cancellation': {}}

    # --- route agreement: closed vs autograd, machine precision ---
    worst = 0.0
    checks = []
    for delta in [0.0, 0.6106412941506083, 4.0]:
        for r in [1.0, 4.0]:
            for tau in TAUS:
                k1 = closed_kappa('awr', delta, r, tau)
                k2 = autograd_expected_kappa([delta / 2, -delta / 2], r, 'awr', tau)
                err = abs(k1 - k2)
                worst = max(worst, err)
                checks.append({'delta': delta, 'r': r, 'tau': tau,
                               'closed': k1, 'autograd': k2, 'abs_err': err})
    out['validation']['awr_closed_vs_autograd_max_abs_err'] = worst
    out['validation']['awr_checks'] = checks
    print(f'awr closed-vs-autograd max |err| = {worst:.3e}')

    worst_sq = 0.0
    for delta in [0.6106412941506083, 4.0]:
        for r in [1.0, 4.0]:
            for a in ALPHAS:
                k1 = closed_kappa('softq', delta, r, alpha=a)
                k2 = autograd_expected_kappa([delta / 2, -delta / 2], r, 'softq',
                                             alpha=a)
                worst_sq = max(worst_sq, abs(k1 - k2))
    out['validation']['softq_closed_vs_autograd_max_abs_err'] = worst_sq
    print(f'softq closed-vs-autograd max |err| = {worst_sq:.3e}')

    # --- MC anchor (statistical) ---
    mc_rows = []
    for delta, r, tau in [(0.6106412941506083, 1.0, 1.0), (0.6106412941506083, 1.0, 0.25),
                          (4.0, 1.0, 1.0), (0.0, 1.0, 1.0)]:
        kc = closed_kappa('awr', delta, r, tau)
        km = mc_kappa(delta, r, 'awr', n=600_000, tau=tau, seed=41)
        mc_rows.append({'delta': delta, 'r': r, 'tau': tau,
                        'closed': kc, 'mc': km, 'abs_err': abs(kc - km)})
        print(f'MC awr delta={delta:.2f} tau={tau}: closed={kc:.5f} mc={km:.5f}')
    ks = closed_kappa('softq', 4.0, 1.0, alpha=1.0)
    kms = mc_kappa(4.0, 1.0, 'softq', n=200_000, alpha=1.0, seed=42)
    mc_rows.append({'field': 'softq', 'delta': 4.0, 'closed': ks, 'mc': kms})
    print(f'MC softq delta=4.00 alpha=1: closed={ks:.5f} mc={kms:.5f}')
    out['validation']['mc_anchor'] = mc_rows

    # --- O1a: softq closed form vs STORED T2 values (theory_toy.json grid) ---
    def delta_of(z):
        return z[0] - z[1]

    d_z0 = delta_of(Z0)
    stored_z0 = [0.000, 0.001, 0.023, 0.085, 0.272, 0.700, 0.903]
    stored_pk = [0.000, 0.039, 0.500, 0.800, 0.941, 0.990, 0.998]
    got_z0 = [round(closed_kappa('softq', d_z0, 1.0, alpha=a), 3) for a in ALPHAS]
    got_pk = [round(closed_kappa('softq', delta_of(Z_PEAKED), 1.0, alpha=a), 3)
              for a in ALPHAS]
    out['softq'] = {
        'formula': 'kappa = A^2/(A^2 + 4 r^2), A = alpha*ln(p/q)',
        'alphas': ALPHAS,
        'z0_closed_rounded': got_z0, 'z0_stored': stored_z0,
        'z0_match': got_z0 == stored_z0,
        'peaked_closed_rounded': got_pk, 'peaked_stored': stored_pk,
        'peaked_match': got_pk == stored_pk,
    }
    print(f"softq z0 match={got_z0 == stored_z0}: {got_z0}")
    print(f"softq peaked match={got_pk == stored_pk}: {got_pk}")

    # --- O1b: awr-baseline kappa(tau) curves (canonical track) ---
    awr_curves = {}
    for name, z in [('uniform', [0.0, 0.0]), ('z0_near_uniform', Z0),
                    ('peaked', Z_PEAKED)]:
        d = delta_of(z)
        awr_curves[name] = {
            'taus': TAUS,
            'kappa': [closed_kappa('awr', d, 1.0, tau=t) for t in TAUS],
        }
    out['awr'] = {
        'note': ('canonical track: kappa(uniform p) = 0 exactly for all tau; '
                 'interior max in tau for p != 1/2'),
        'curves': awr_curves,
    }

    # --- O1c: exact cancellation family ---
    for name, z in [('uniform', [0.0, 0.0]), ('z0', Z0), ('peaked', Z_PEAKED)]:
        d = delta_of(z)
        out['cancellation'][name] = {
            f: closed_kappa(f, d, 1.0) for f in ['reinforce', 'expq']
        }
        out['cancellation'][name]['awr_nobase'] = 0.0   # elementwise theta-free
        out['cancellation'][name]['softmaxq'] = 0.0
        out['cancellation'][name]['gibbs_expq'] = 0.0
        out['cancellation'][name]['awr_baseline_tau1'] = closed_kappa(
            'awr', d, 1.0, tau=1.0)
        out['cancellation'][name]['softq_alpha1'] = closed_kappa(
            'softq', d, 1.0, alpha=1.0)

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
