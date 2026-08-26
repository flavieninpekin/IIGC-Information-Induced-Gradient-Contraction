"""Exact closed-form kappa for the two-action relational bandit (T1-T3 verdicts).

Rebuild of the lost verify_theory_toy.py. Cross-validated against the stored
theory_toy.json: the old calculator's "exact" track used the CE-fit objective
g_r = grad_theta sum_a log pi(a) w_r(a) (no outer pi weight; w differentiable
through the baseline V), which reproduces ALL stored values exactly.

Also implements the framework (C.1) pi-weighted form g_r = sum_a pi(a)
grad log pi(a) w_r(a) for comparison.

Fields (weight functions w on the two actions):
  - reinforce : w = Q - V (baseline) or w = Q (no baseline)
  - awr       : w = exp((Q - V)/tau) (baseline) or exp(Q/tau) (no baseline)
  - softmaxq  : w = softmax(Q/tau)          [framework "gibbs" dial]
  - softq     : w = alpha log pi - Q        [SAC-style, alpha-weighted]
  - expq      : w = Q
  - gibbs_expq: objective E_{pi_tau}[Q], pi_tau = softmax(logits/tau) [fields.py]

Mirror structure: Q_A = (+r, -r), Q_B = (-r, +r).

Outputs:
  data/kappa/toy_fields/theory_toy2.json
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'data', 'kappa', 'toy_fields', 'theory_toy2.json')

TAUS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
Z0 = [0.3716251850128174, -0.23901528120040894]  # from theory_toy.json
Z_UNIFORM = [0.0, 0.0]
Z_PEAKED = [2.0, -2.0]

TAUS3 = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # T3 grid (stored)


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def _obj_grad(z, q, field, tau, alpha, baseline, ce_weighted):
    """grad_theta of the field objective for relation with Q vector `q`.

    ce_weighted=True  : g = grad sum_a log pi(a) w(a)          (old exact)
    ce_weighted=False : g = grad sum_a pi(a) log pi(a) w(a)    (pi-weighted)
    """
    zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
    pi = torch.softmax(zt, dim=0)
    qt = torch.tensor(q, dtype=torch.double)
    if field == 'reinforce':
        w = qt.detach().clone()
        if baseline:
            w = w - (pi * qt).sum()
        w = w.detach()  # Q/V are data for the REINFORCE estimator
    elif field == 'expq':
        w = qt.detach().clone()
    elif field == 'awr':
        if baseline:
            w = torch.exp((qt - (pi * qt).sum()) / tau)
        else:
            w = torch.exp(qt / tau)
    elif field == 'softmaxq':
        w = torch.exp(qt / tau)
        w = w / w.sum()
    elif field == 'softq':
        w = alpha * torch.log(torch.clamp(pi, min=1e-30)) - qt
    elif field == 'gibbs_expq':
        pi_tau = torch.softmax(zt / tau, dim=0)
        loss = -(pi_tau * qt).sum()
        loss.backward()
        return zt.grad.detach().numpy()
    else:
        raise ValueError(field)

    lp = torch.log(torch.clamp(pi, min=1e-30))
    if ce_weighted:
        loss = (lp * w).sum()
    else:
        loss = (pi * lp * w).sum()
    loss.backward()
    return zt.grad.detach().numpy()


def exact_grad(z, r, field, tau=1.0, alpha=1.0, baseline=True, ce_weighted=True,
               relation='A'):
    q = np.array([r, -r]) if relation == 'A' else np.array([-r, r])
    return _obj_grad(z, q, field, tau, alpha, baseline, ce_weighted)


def kappa_from_grads(gA, gB):
    gA = np.asarray(gA, dtype=float)
    gB = np.asarray(gB, dtype=float)
    m = (gA + gB) / 2.0
    e_shared = float(m @ m)
    e_contrast = float(((gA - gB) / 2.0) @ ((gA - gB) / 2.0))
    denom = max(e_shared + e_contrast, 1e-30)
    return e_shared / denom, e_shared, e_contrast


def exact_kappa(z, r, field, tau=1.0, alpha=1.0, baseline=True, ce_weighted=True):
    gA = exact_grad(z, r, field, tau, alpha, baseline, ce_weighted, 'A')
    gB = exact_grad(z, r, field, tau, alpha, baseline, ce_weighted, 'B')
    return kappa_from_grads(gA, gB)


def closed_form_kappa(z, r, field, tau=1.0, alpha=1.0, baseline=True):
    """Framework (C.1) closed form: g_r = sum_a pi(a)(e_a - pi) w_r(a)."""
    pi = softmax(z)
    e = np.eye(2)

    def g(q):
        if field == 'reinforce':
            w = q.copy()
            if baseline:
                w = w - pi @ q
        elif field == 'expq':
            w = q.copy()
        elif field == 'awr':
            w = np.exp(q / tau)
            if baseline:
                w = np.exp((q - pi @ q) / tau)
        elif field == 'softmaxq':
            w = np.exp(q / tau)
            w = w / w.sum()
        elif field == 'softq':
            w = alpha * np.log(np.maximum(pi, 1e-30)) - q
        else:
            raise ValueError(field)
        return np.sum([pi[a] * (e[a] - pi) * w[a] for a in range(2)], axis=0)

    gA = g(np.array([r, -r]))
    gB = g(np.array([-r, r]))
    return kappa_from_grads(gA, gB)


def sampled_kappa(z, r, field, n_eps=200, tau=1.0, alpha=1.0, baseline=True, seed=0):
    """Sampled track: per-episode gradient of the fields.py-style loss.

    One episode = one step; action sampled from pi(z); reward = +/-r.
    Per-episode gradient: grad[-log pi(a) w(a)] with w depending on pi (V).
    """
    rng = np.random.default_rng(seed)
    pi = softmax(z)
    grads = {0: [], 1: []}
    for _ in range(n_eps):
        rel = rng.integers(0, 2)
        a = rng.choice(2, p=pi)
        q = np.array([r, -r]) if rel == 0 else np.array([-r, r])
        zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
        pit = torch.softmax(zt, dim=0)
        qt = torch.tensor(q, dtype=torch.double)
        if field == 'reinforce':
            rew = r if a == rel else -r
            w = torch.tensor(rew, dtype=torch.double)
        elif field == 'awr':
            w = torch.exp((qt[a] - (pit * qt).sum()) / tau) if baseline else torch.exp(qt[a] / tau)
        elif field == 'softmaxq':
            wv = torch.exp(qt / tau)
            w = (wv / wv.sum())[a]
        elif field == 'expq':
            w = qt[a]
        elif field == 'softq':
            w = alpha * torch.log(torch.clamp(pit[a], min=1e-30)) - qt[a]
        else:
            raise ValueError(field)
        loss = -torch.log(torch.clamp(pit[a], min=1e-30)) * w
        loss.backward()
        grads[int(rel)].append(zt.grad.detach().numpy())

    gA = np.mean(grads[0], axis=0) if grads[0] else np.zeros(2)
    gB = np.mean(grads[1], axis=0) if grads[1] else np.zeros(2)
    return kappa_from_grads(gA, gB)


def run():
    out = {}

    # ---- validation: old "exact" = CE-fit form must reproduce stored values ----
    stored_base = [0.4988, 0.4582, 0.3591, 0.2533, 0.2393, 0.3827, 0.663]
    stored_nobase = [0.0806, 0.0807, 0.0862, 0.1314, 0.2912, 0.5939, 0.8501]
    v_base = [round(exact_kappa(Z0, 1.0, 'awr', tau=t, baseline=True)[0], 4) for t in TAUS]
    v_nobase = [round(exact_kappa(Z0, 1.0, 'awr', tau=t, baseline=False)[0], 4) for t in TAUS]
    out['validation'] = {
        'awr_base_reproduced': v_base,
        'awr_nobase_reproduced': v_nobase,
        'match': v_base == stored_base and v_nobase == stored_nobase,
    }
    print('validation awr baseline match:', v_base == stored_base)
    print('validation awr nobase  match:', v_nobase == stored_nobase)

    # ---- T1: awr kappa(tau) as function of (pi, baseline), CE-fit form ----
    t1 = {}
    for name, z in [('z0_near_uniform', Z0), ('uniform', Z_UNIFORM), ('peaked', Z_PEAKED)]:
        row = {'z': z}
        for baseline in [True, False]:
            row[f'baseline={baseline}'] = [
                round(exact_kappa(z, 1.0, 'awr', tau=tau, baseline=baseline)[0], 4)
                for tau in TAUS
            ]
        t1[name] = row
    out['T1_awr_tau'] = {'r': 1.0, 'taus': TAUS, 'grid': t1}

    # ---- T1c: framework closed-form (pi-weighted) ----
    t1c = {}
    for name, z in [('z0_near_uniform', Z0), ('uniform', Z_UNIFORM), ('peaked', Z_PEAKED)]:
        row = {'z': z}
        for baseline in [True, False]:
            row[f'baseline={baseline}'] = [
                round(closed_form_kappa(z, 1.0, 'awr', tau=tau, baseline=baseline)[0], 4)
                for tau in TAUS
            ]
        t1c[name] = row
    out['T1c_awr_tau_pi_weighted'] = {'r': 1.0, 'taus': TAUS, 'grid': t1c}

    # ---- T2: softq alpha scan ----
    t2 = {}
    for name, z in [('z0_near_uniform', Z0), ('peaked', Z_PEAKED)]:
        t2[name] = [round(exact_kappa(z, 1.0, 'softq', alpha=a)[0], 4) for a in ALPHAS]
    out['T2_softq_alpha'] = {'r': 1.0, 'alphas': ALPHAS, 'grid': t2}

    # ---- T3: gibbs definition split ----
    t3 = {}
    for name, z in [('z0_near_uniform', Z0), ('uniform', Z_UNIFORM), ('peaked', Z_PEAKED)]:
        t3[name] = {
            'softmaxq': [round(exact_kappa(z, 1.0, 'softmaxq', tau=tau)[0], 4) for tau in TAUS3],
            'gibbs_expq': [round(exact_kappa(z, 1.0, 'gibbs_expq', tau=tau)[0], 4) for tau in TAUS3],
            'softmaxq_pi_weighted': [round(closed_form_kappa(z, 1.0, 'softmaxq', tau=tau)[0], 4)
                                     for tau in TAUS3],
        }
    out['T3_gibbs_tau'] = {'r': 1.0, 'taus': TAUS3, 'grid': t3}

    # ---- T1b: kappa(pi) profile at fixed tau ----
    pgrid = np.linspace(0.05, 0.95, 19)
    tau_fixed = 1.0
    prof_ce = []
    prof_fw = []
    for p in pgrid:
        z = np.array([np.log(p / (1 - p)), 0.0])
        prof_ce.append(round(exact_kappa(z, 1.0, 'awr', tau=tau_fixed, baseline=True)[0], 4))
        prof_fw.append(round(closed_form_kappa(z, 1.0, 'awr', tau=tau_fixed, baseline=True)[0], 4))
    out['T1b_kappa_pi_profile'] = {'r': 1.0, 'tau': tau_fixed, 'p': pgrid.tolist(),
                                   'kappa_ce_fit': prof_ce, 'kappa_pi_weighted': prof_fw}

    # ---- exact cancellation for odd fields ----
    canc = {}
    for name, z in [('z0', Z0), ('uniform', Z_UNIFORM), ('peaked', Z_PEAKED)]:
        canc[name] = {
            'reinforce_base': exact_kappa(z, 1.0, 'reinforce', baseline=True)[0],
            'reinforce_nobase': exact_kappa(z, 1.0, 'reinforce', baseline=False)[0],
            'expq': exact_kappa(z, 1.0, 'expq')[0],
            'gibbs_expq': exact_kappa(z, 1.0, 'gibbs_expq', tau=1.0)[0],
        }
    out['exact_cancellation'] = canc

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('saved:', OUT)

    print('\n--- T1 awr kappa(tau), CE-fit (old exact) ---')
    for name, row in t1.items():
        print(f'  {name}: base={row["baseline=True"]} nobase={row["baseline=False"]}')
    print('\n--- T1c awr kappa(tau), pi-weighted framework form ---')
    for name, row in t1c.items():
        print(f'  {name}: base={row["baseline=True"]} nobase={row["baseline=False"]}')
    print('\n--- T2 softq kappa(alpha) ---')
    for name, vals in t2.items():
        print(f'  {name}: {vals}')
    print('\n--- T3 gibbs split ---')
    for name, row in t3.items():
        print(f'  {name}: softmaxq={row["softmaxq"]}')
        print(f'  {name}: softmaxq_pw={row["softmaxq_pi_weighted"]}')
        print(f'  {name}: gibbs_expq={row["gibbs_expq"]}')
    print('\n--- exact cancellation ---')
    for name, row in canc.items():
        print(f'  {name}: {row}')


if __name__ == '__main__':
    run()
