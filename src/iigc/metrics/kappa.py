"""Canonical kappa definitions and condition-gradient decompositions.

The structural metric is the mixture-reference ratio

    ||sum_i p_i mu_i||^2 / sum_i p_i ||mu_i||^2,

where ``mu_i`` is the population gradient for condition ``i``. Episode noise
is reported separately and must not be silently folded into this ratio.
"""

import numpy as np

KAPPA_SCHEMA_VERSION = "mixture-v1"


def measurement_metadata(gradient_definition, condition_weights,
                         rollout_protocol, parameter_space="euclidean",
                         noise_treatment="separate"):
    """Build the required metadata block for a kappa result file."""
    return {
        "kappa_schema": KAPPA_SCHEMA_VERSION,
        "gradient_definition": str(gradient_definition),
        "condition_weights": (
            str(condition_weights)
            if isinstance(condition_weights, str)
            else np.asarray(condition_weights, dtype=float).tolist()
        ),
        "rollout_protocol": str(rollout_protocol),
        "parameter_space": str(parameter_space),
        "noise_treatment": str(noise_treatment),
    }

def _as_gradient_array(gradients):
    rows = []
    for gradient in gradients:
        if hasattr(gradient, "detach"):
            gradient = gradient.detach().cpu().numpy()
        rows.append(np.asarray(gradient, dtype=float).reshape(-1))
    if not rows:
        raise ValueError("gradients must not be empty")
    lengths = {row.size for row in rows}
    if len(lengths) != 1:
        raise ValueError("all gradients must have the same size")
    return np.stack(rows)


def _normalized_weights(n, weights):
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    result = np.asarray(weights, dtype=float).reshape(-1)
    if result.size != n:
        raise ValueError("weights and gradients must have the same length")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("weights must be finite and nonnegative")
    total = result.sum()
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return result / total


def condition_decomposition(gradients, weights=None):
    """Return structural shared, contrast, and mixture-reference quantities."""
    rows = _as_gradient_array(gradients)
    p = _normalized_weights(rows.shape[0], weights)
    mixed = p @ rows
    e_shared = float(mixed @ mixed)
    e_mixture = float(np.sum(p[:, None] * rows * rows))
    e_contrast = float(np.sum(p[:, None] * (rows - mixed) ** 2))
    kappa = e_shared / e_mixture if e_mixture > 0 else 0.0
    return {
        "E_shared": e_shared,
        "E_contrast": e_contrast,
        "E_mixture": e_mixture,
        "kappa_mix": kappa,
        "weights": p.tolist(),
    }


def kappa_mix(gradients, weights=None):
    """Return only the bounded structural mixture-reference kappa."""
    return condition_decomposition(gradients, weights)["kappa_mix"]


def episode_decomposition(groups, weights=None):
    """Decompose per-condition episode gradients without conflating noise.

    ``groups[i]`` contains episode gradients for condition ``i``. The returned
    ``kappa_mix`` uses condition means only. ``kappa_ep`` adds within-condition
    episode variance to the denominator for a one-episode diagnostic.
    ``kappa_mean_surrogate`` adds the variance of the weighted sample mean; it
    is a calibration quantity, not the expectation of a ratio estimator.
    """
    arrays = [_as_gradient_array(group) for group in groups]
    if not arrays:
        raise ValueError("groups must not be empty")
    dim = arrays[0].shape[1]
    if any(array.shape[1] != dim for array in arrays):
        raise ValueError("all episode gradients must have the same size")

    p = _normalized_weights(len(arrays), weights)
    means = np.stack([array.mean(axis=0) for array in arrays])
    structural = condition_decomposition(means, p)
    within = np.asarray([
        np.mean(np.sum((array - mean) ** 2, axis=1))
        for array, mean in zip(arrays, means)
    ])
    sigma2 = float(p @ within)
    mean_noise = float(np.sum((p ** 2) * within /
                              np.asarray([array.shape[0] for array in arrays])))
    e_total_ep = structural["E_mixture"] + sigma2
    e_total_mean = structural["E_mixture"] + mean_noise
    structural.update({
        "sigma2": sigma2,
        "mean_noise_energy": mean_noise,
        "kappa_ep": structural["E_shared"] / e_total_ep
        if e_total_ep > 0 else 0.0,
        "kappa_mean_surrogate": structural["E_shared"] / e_total_mean
        if e_total_mean > 0 else 0.0,
        "n_per_condition": [int(array.shape[0]) for array in arrays],
    })
    return structural
