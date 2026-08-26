"""C.6: supervised-domain minimal transfer (fixed-model mirror protocol).

Same input x, relation = label sign flip (A: y=+1, B: y=-1). Same model,
same data, only the loss changes -> kappa is a property of the loss geometry.
Fixed random model (NOT trained at mixture p: training adapts the model and
washes out the signature, the supervised analog of the E2 confound).

Theory (2026-08-22):
  g_A = sum_i L'(f_i) * x_i          (y=+1: grad of L(f))
  g_B = sum_i L'(-f_i) * (-1) * x_i  (y=-1: grad of L(-f))
  g_B = -g_A  <=>  L' even.
  - hinge:  L'(u) = -I(u<1), L'(-u) = -I(u>-1)  -> at f~0, g_A = -sum x,
    g_B = +sum x -> anti-parallel -> kappa = 0 EXACTLY (brittle end).
  - BCE:    L'(u) = -sigma(-u), not even -> shared term
    ||sum tanh(f/2) x||^2 survives -> kappa ~ 1 on zero-mean data (dull end).
  - quadratic: same as BCE family (dull).
  - perceptron: margin-0 gating -> half-space bias (sum_{f>0}x = c w_hat) ->
    kappa ~ 1 (artifact; flagged, not a retention claim).

Prediction: the RL "brittle/dull" dichotomy transfers: hinge = brittle
(kappa 0), BCE/quadratic = dull (kappa high but useless for either relation).

Outputs:
  data/kappa/supervised_transfer/fixed_model.json
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'supervised_transfer')
os.makedirs(OUT_DIR, exist_ok=True)

LOSSES = ['bce', 'hinge', 'perceptron', 'quadratic']
D = 8
N = 4000
SCALES = [0.01, 0.1, 1.0]
SEEDS = 5


def kappa_from_grads(gA, gB):
    gA = np.asarray(gA, dtype=float)
    gB = np.asarray(gB, dtype=float)
    m = (gA + gB) / 2.0
    e_sh = float(m @ m)
    e_co = float(((gA - gB) / 2.0) @ ((gA - gB) / 2.0))
    return e_sh / max(e_sh + e_co, 1e-30), e_sh, e_co


def grads(x, w0, loss):
    def gf(flip):
        w = torch.nn.Parameter(w0.clone())
        f = (x @ w).squeeze(-1) * flip   # f already signed by the relation
        if loss == 'quadratic':
            y = torch.full_like(f, float(flip))
            L = ((f - y) ** 2).mean()
        elif loss == 'bce':
            y = (torch.ones_like(f) * float(flip) + 1) / 2
            L = F.binary_cross_entropy_with_logits(f, y)
        elif loss == 'hinge':
            L = torch.clamp(1 - f, min=0.0).mean()
        elif loss == 'perceptron':
            L = torch.clamp(-f, min=0.0).mean()
        return torch.autograd.grad(L, w)[0].detach().flatten()
    return gf(1.0), gf(-1.0)


def run():
    results = {}
    rng = np.random.default_rng(0)
    for mean_shift in [0.0, 1.0]:
        tag = f'mean_{mean_shift}'
        results[tag] = {}
        for scale in SCALES:
            rows = []
            for loss in LOSSES:
                ks = []
                for s in range(SEEDS):
                    x = torch.FloatTensor(rng.normal(
                        size=(N, D)) + mean_shift)  # zero-mean or shifted
                    w0 = torch.randn(D, 1) * scale
                    gA, gB = grads(x, w0, loss)
                    k, e_sh, e_co = kappa_from_grads(gA, gB)
                    ks.append(k)
                rows.append({'loss': loss, 'scale': scale,
                             'kappa_mean': float(np.mean(ks)),
                             'kappa_std': float(np.std(ks))})
                print(f'{tag} scale={scale} {loss:10}: '
                      f'kappa={np.mean(ks):.4f} +/- {np.std(ks):.4f}')
            results[tag][str(scale)] = rows

    with open(os.path.join(OUT_DIR, 'fixed_model.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nsaved: {os.path.join(OUT_DIR, "fixed_model.json")}')


if __name__ == '__main__':
    run()
