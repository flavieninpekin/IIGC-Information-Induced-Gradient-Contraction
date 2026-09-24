"""Regression tests for the canonical two-action closed forms.

These mirror the derivations in paper/resources/canonical_theory.md and the
verification script experiments/common_basis/toy/verify_closed_forms_o1.py.
"""

import numpy as np
import pytest
import torch

from iigc.metrics.kappa import condition_decomposition


def _expected_condition_gradients(z, r, field, alpha=1.0):
    """Exact expected logit gradients for the mirror bandit per relation."""
    gradients = []
    for q in (np.array([r, -r]), np.array([-r, r])):
        zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
        pi = torch.softmax(zt, dim=0)
        qt = torch.tensor(q, dtype=torch.double)
        if field == "softq":
            loss = (pi * (alpha * torch.log(pi) - qt)).sum()
        elif field == "expq":
            loss = -(pi * qt).sum()
        else:
            raise ValueError(field)
        gradients.append(torch.autograd.grad(loss, zt)[0].detach().numpy())
    return gradients


def test_softq_closed_form_matches_autograd():
    for delta in (0.0, 0.61, 4.0):
        for alpha in (0.1, 1.0, 5.0):
            z = np.array([delta / 2.0, -delta / 2.0])
            p = 1.0 / (1.0 + np.exp(-delta))
            a = alpha * np.log(p / (1.0 - p))
            closed = a ** 2 / (a ** 2 + 4.0)

            grads = _expected_condition_gradients(z, 1.0, "softq", alpha)
            computed = condition_decomposition(grads, [0.5, 0.5])["kappa_mix"]

            assert computed == pytest.approx(closed, abs=1e-10)


def test_expq_cancels_exactly_under_mirror_mixture():
    grads = _expected_condition_gradients([0.5, -0.2], 1.0, "expq")
    result = condition_decomposition(grads, [0.5, 0.5])

    assert result["kappa_mix"] == pytest.approx(0.0, abs=1e-24)
    assert result["E_shared"] == pytest.approx(0.0, abs=1e-24)
    assert result["E_mixture"] > 0.0


def _awr_closed_kappa(delta, r=1.0, tau=1.0):
    """Appendix closed form: kappa of the differentiable-baseline AWR field."""
    p = 1.0 / (1.0 + np.exp(-delta))
    q = 1.0 - p
    x = 2.0 * r / tau
    a0, a1 = np.exp(x * q), np.exp(-x * p)
    b0, b1 = np.exp(-x * q), np.exp(x * p)
    ma = p * np.log(p) * a0 + q * np.log(q) * a1
    mb = p * np.log(p) * b0 + q * np.log(q) * b1
    ca = p * q * (a1 - a0 + x * ma)
    cb = p * q * (b1 - b0 - x * mb)
    s, d = ca + cb, ca - cb
    den = s * s + d * d
    return (s * s / den) if den > 0 else 0.0


def test_awr_cancels_at_uniform_and_is_monotone_decreasing_in_tau():
    for tau in (0.05, 0.5, 5.0):
        assert _awr_closed_kappa(0.0, tau=tau) == pytest.approx(0.0, abs=1e-20)

    delta = 0.3716251850128174 - (-0.23901528120040894)
    taus = np.logspace(-1.5, 1.7, 40)
    kappas = np.asarray([_awr_closed_kappa(delta, tau=t) for t in taus])
    assert (np.diff(kappas) <= 1e-12).all()

    stored = {
        0.1: 0.49787806253674555,
        0.25: 0.42332590418943067,
        0.5: 0.2533187192560543,
        1.0: 0.09823311615045022,
        2.0: 0.028658409369042914,
        4.0: 0.007482294729905181,
        8.0: 0.0018916502942411036,
    }
    for tau, value in stored.items():
        assert _awr_closed_kappa(delta, tau=tau) == pytest.approx(value, abs=1e-12)


def test_awr_autograd_matches_closed_form():
    def expected_grad(z, r, tau, relation):
        q_vec = np.array([r, -r]) if relation == "A" else np.array([-r, r])
        zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
        pi = torch.softmax(zt, dim=0)
        qt = torch.tensor(q_vec, dtype=torch.double)
        v = (pi * qt).sum()
        ws = torch.exp((qt - v) / tau)
        log_pi = torch.log(torch.clamp(pi, min=1e-300))
        grad = np.zeros(2)
        pi_d = pi.detach()
        for a in range(2):
            if zt.grad is not None:
                zt.grad.zero_()
            (-log_pi[a] * ws[a]).backward(retain_graph=True)
            grad += float(pi_d[a]) * zt.grad.detach().numpy()
        return grad

    delta = 0.6106
    z = np.array([delta / 2.0, -delta / 2.0])
    for tau in (0.25, 1.0, 4.0):
        grads = [expected_grad(z, 1.0, tau, relation) for relation in ("A", "B")]
        kappa = condition_decomposition(grads, [0.5, 0.5])["kappa_mix"]
        assert kappa == pytest.approx(_awr_closed_kappa(delta, tau=tau), abs=1e-12)
