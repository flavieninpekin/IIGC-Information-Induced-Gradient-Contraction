import pytest
import torch

from iigc.metrics.fields import kappa_and_energy, kappa_from_gradients
from iigc.metrics.kappa import episode_decomposition, measurement_metadata


def test_mixture_reference_kappa_is_bounded_and_weighted():
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    kappa, energy = kappa_from_gradients(gradients, weights=[0.8, 0.2])

    assert energy == pytest.approx(1.0)
    assert kappa == pytest.approx(0.68)
    assert 0.0 <= kappa <= 1.0


def test_two_condition_wrapper_matches_uniform_mixture():
    gradients = [torch.tensor([2.0, 0.0]), torch.tensor([-1.0, 0.0])]
    expected = kappa_from_gradients(gradients, weights=[0.5, 0.5])
    scaled = kappa_from_gradients([g * 1e-8 for g in gradients],
                                  weights=[0.5, 0.5])

    assert kappa_and_energy(*gradients) == pytest.approx(expected)
    assert scaled[0] == pytest.approx(expected[0])


def test_invalid_mixture_weights_are_rejected():
    gradients = [torch.ones(2), torch.ones(2)]

    with pytest.raises(ValueError):
        kappa_from_gradients(gradients, weights=[1.0])
    with pytest.raises(ValueError):
        kappa_from_gradients(gradients, weights=[1.0, -1.0])
    with pytest.raises(ValueError):
        kappa_from_gradients(gradients, weights=[0.0, 0.0])


def test_episode_decomposition_separates_structure_from_noise():
    groups = [
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]),
    ]
    result = episode_decomposition(groups)

    assert result["kappa_mix"] == pytest.approx(0.0)
    assert result["kappa_ep"] == pytest.approx(0.0)
    assert result["sigma2"] == pytest.approx(0.0)


def test_measurement_metadata_records_protocol_contract():
    metadata = measurement_metadata(
        "expected_logit_gradient", [0.5, 0.5], "stochastic_policy", "euclidean"
    )

    assert metadata["kappa_schema"] == "mixture-v1"
    assert metadata["condition_weights"] == [0.5, 0.5]
    assert metadata["rollout_protocol"] == "stochastic_policy"
