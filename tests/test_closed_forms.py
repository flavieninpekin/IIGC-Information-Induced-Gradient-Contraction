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
