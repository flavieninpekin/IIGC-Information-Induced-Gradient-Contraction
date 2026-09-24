# Canonical Theory Notes

Status: canonical working theory for the ICLR 2027 revision.

This file intentionally contains only claims that match the current code and
the mixture-reference kappa definition. Historical CE-fit, S3-only, and
softq-dynamics claims remain in the archived drafts and are not part of this
theory.

## 1. Hidden-Condition Gradient Mixture

Let `i` be a hidden condition with distribution `p`, and let
`g_i(theta) = grad_theta L_i(theta)` be the population gradient under a fixed
measurement protocol. Define

```text
g_bar = sum_i p_i g_i
E_shared = ||g_bar||^2
E_mix = sum_i p_i ||g_i||^2
E_contrast = sum_i p_i ||g_i - g_bar||^2
kappa_mix = E_shared / E_mix
```

The variance identity is

```text
E_mix = E_shared + E_contrast.
```

Because squared norm is convex,
`0 <= kappa_mix <= 1`. This is the primary structural metric. The historical
uniform-reference denominator `mean_i ||g_i||^2` is a different diagnostic and
is not bounded when `p` is non-uniform.

## 2. Symmetric Cancellation

Consider the two-action mirror bandit

```text
Q_A = ( r, -r )
Q_B = ( -r, r ).
```

For an elementwise weight that is independent of policy parameters, swapping
the two conditions swaps the two action coordinates. The policy score vectors
also swap with opposite sign in the two-dimensional logit tangent direction.
Therefore

```text
g_B = -g_A.
```

Under the equal hidden mixture, `g_bar=0` and `kappa_mix=0`. This theorem is
about the stated mirror model and expected gradients. It is not a claim that
every policy-gradient field in a neural environment is exactly zero.

For the K-way matching bandit `Q_i(b)=2 R delta(i,b)`, the expected expq field
is

```text
g_i[b]    = -2 R pi_b (delta(i,b) - pi_i)
g_bar[b] = -2 R pi_b (p_b - sum_i p_i pi_i).
```

At the uniform condition mixture, `sum_i g_i=0` for every `K`. The cancellation
is a normalization result, not an S3 or non-abelian result.

## 3. Two-Action softq Closed Form

Let `rho=pi(action 0)`, `q=1-rho`, and define

```text
L_i = sum_a pi_a (alpha log(pi_a) - Q_i(a)).
```

With `u=(1,-1)` the two condition gradients are `g_i=c_i u`, where

```text
Delta = alpha log(rho / q)
c_A = rho*q*(Delta - 2*r)
c_B = rho*q*(Delta + 2*r).
```

For an equal hidden mixture,

```text
kappa_mix = Delta^2 / (Delta^2 + 4*r^2).
```

This is a fixed-policy Euclidean logit-gradient identity. It does not imply a
monotone training trajectory, a universal algorithm ranking, or better return.
The limiting cases are `alpha=0 -> kappa_mix=0` and large `alpha` or a highly
non-uniform policy giving a larger shared fraction.

## 4. K-Way Direction and Interference

For the K-way matching bandit, define

```text
a_b = log(pi_b) - sum_c pi_c log(pi_c)
b_b = p_b - sum_i p_i pi_i
c_i,b = delta(i,b) - pi_i.
```

The softq mixture gradient is

```text
g_bar[b] = pi_b * (alpha*a_b - 2*R*b_b).
```

The entropy direction `a` and the hidden-mixture direction `b` can have a
negative inner product in the `pi_b^2` weighted geometry. This is the precise
meaning of channel interference. A direction-dependent score first becomes
possible when the condition simplex has dimension at least two, so `K>=3` is a
minimal example. It is not evidence of a special S3 representation.

For the actual condition mixture, the denominator is

```text
E_mix = sum_i p_i sum_b pi_b^2 *
        (alpha*a_b - 2*R*c_i,b)^2.
```

Any alpha scan must report this denominator. On the 1001-point scans the
numerator-energy minimum and the kappa-ratio minimum coincide on the same grid
point (`alpha_scan_minima` in `kway_geometry.json`), so the scan reports a
single minimum location rather than two distinct cancellation points.

## 5. Softq Optimization Direction

In the symmetric hidden mirror case, the hidden Q terms cancel after averaging,
leaving

```text
L_mix = alpha * sum_a pi_a log(pi_a) = -alpha * entropy(pi).
```

Gradient descent minimizes negative entropy and therefore moves toward higher
entropy, namely the uniform policy. The earlier claim that this term is
necessarily mode-seeking or drives a peaked policy has the wrong optimization
sign for the current implementation and is excluded from the paper.

This static result must not be extrapolated to arbitrary actor-critic training,
where the Q field, parameterization, replay distribution, and optimizer add
other terms.

## 6. Fisher/KL Variant

The default kappa is Euclidean in parameter space. A separate Fisher-weighted
variant can be defined for categorical logits:

```text
F(pi) = diag(pi) - pi*pi^T
kappa_F = (g_bar^T F g_bar) / sum_i p_i (g_i^T F g_i).
```

For a logit perturbation `v`,

```text
KL(pi(z) || pi(z + epsilon*v))
  = (epsilon^2/2) * v^T F v + O(epsilon^3).
```

Thus `kappa_F` has an infinitesimal KL interpretation when the perturbation is
made in logit space. The rank-one term `-pi*pi^T` must be retained. No claim is
made that the Euclidean kappa used elsewhere equals `kappa_F`.

## 7. Episode Noise

For episode gradients `G_{ij}` in condition `i`, let `mu_i` be the sample mean
and let

```text
sigma_i^2 = mean_j ||G_ij - mu_i||^2.
```

The structural quantity is computed from the `mu_i`. A one-episode noise
diagnostic is

```text
kappa_ep = E_shared / (E_mix + sum_i p_i*sigma_i^2).
```

For a weighted sample mean with `n_i` episodes per condition, the noise energy
in the mixture mean is

```text
sum_i p_i^2 * sigma_i^2 / n_i.
```

The resulting denominator can be reported as a calibration surrogate, but it is
not a high-probability guarantee for a ratio estimator. Such a guarantee would
require an explicit distributional assumption and a separate proof.
