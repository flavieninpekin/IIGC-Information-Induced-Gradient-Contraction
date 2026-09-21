"""Regression tests for the differentiable-baseline AWR weight helper.

These cover the shape contract and the float64/float32 underflow diagnostics
that made the first Overcooked field-axis rerun invalid (a ``[T]`` return
vector minus a ``[T, 1]`` value head broadcast to ``[T, T]`` weights).
"""

import pytest
import torch

from iigc.metrics.fields import awr_weight_stats
from iigc.metrics.kappa import condition_decomposition


def test_column_value_shape_is_flattened():
    returns = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([[0.5], [1.0], [1.5]])

    weights, _ = awr_weight_stats(returns, values, tau=1.0, shift=0.0)

    assert weights.shape == returns.shape
    expected = torch.exp(torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64))
    assert torch.allclose(weights, expected)


def test_shape_mismatch_raises_instead_of_broadcasting():
    with pytest.raises(ValueError):
        awr_weight_stats(torch.zeros(4), torch.zeros(3), shift=0.0)
    with pytest.raises(ValueError):
        awr_weight_stats(torch.zeros(4), torch.zeros(4, 4), shift=0.0)


def test_shift_multiplies_weights_and_preserves_kappa():
    torch.manual_seed(0)
    n = 32
    raw_a = torch.randn(n)
    raw_b = torch.randn(n)
    wa0, _ = awr_weight_stats(torch.zeros(n), raw_a, tau=0.5, shift=0.0)
    wb0, _ = awr_weight_stats(torch.zeros(n), raw_b, tau=0.5, shift=0.0)
    wa1, _ = awr_weight_stats(torch.zeros(n), raw_a, tau=0.5, shift=20.0)
    wb1, _ = awr_weight_stats(torch.zeros(n), raw_b, tau=0.5, shift=20.0)

    ratio = wa1 / wa0
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-12)

    kappa0 = condition_decomposition([wa0, wb0], [0.5, 0.5])["kappa_mix"]
    kappa1 = condition_decomposition([wa1, wb1], [0.5, 0.5])["kappa_mix"]
    assert kappa1 == pytest.approx(kappa0, abs=1e-12)


def test_default_shift_sets_max_weight_to_one():
    returns = torch.tensor([-1.0, 0.0, 4.0])
    values = torch.zeros(3)

    _, diag = awr_weight_stats(returns, values, tau=1.0)

    assert diag["awr_shift"] == pytest.approx(4.0)
    assert diag["awr_adv_max_post_shift"] == pytest.approx(0.0)
    assert diag["awr_weight_max"] == pytest.approx(1.0)


def test_float32_underflow_is_quantified():
    returns = torch.tensor([0.0, -300.0])
    values = torch.zeros(2)

    weights, diag = awr_weight_stats(returns, values, tau=1.0)

    assert diag["awr_zero_weight_fraction"] == 0.0
    assert diag["awr_zero_weight_fraction_float32"] == pytest.approx(0.5)
    assert float(weights[1]) > 0.0


def test_values_keep_the_autograd_graph():
    values = torch.randn(5, requires_grad=True)
    returns = torch.zeros(5)

    weights, _ = awr_weight_stats(returns, values, tau=1.0, shift=0.0)
    weights.sum().backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
