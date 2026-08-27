"""Prop 9: derive KL projection FROM the hidden mixture (friend's question).

Claim: for softq objectives under hidden conditions with mixture p,
    L_r(pi) = alpha * KL(pi || pi_r*) + const,   pi_r*(a) = softmax(Q_r(a)/alpha)
    sum_r p_r L_r  =  alpha * KL(pi || pi_tilde) + const
    pi_tilde(a)    = softmax( sum_r p_r Q_r(a) / alpha )     [geometric mean]
    => g_hat = alpha * grad_z KL(pi || softmax(Qbar_p / alpha)),
       Qbar_p = sum_r p_r Q_r  (mixture-averaged value)
Corollaries:
  - mirror Q, uniform p: Qbar = 0 => pi_tilde = uniform => g_hat = entropy channel
  - asymmetric p: survival direction = geometry of the projection target

Verifications (machine precision):
  V1. grad L_r == alpha * grad KL(pi||softmax(Q_r/alpha))   for all r
  V2. grad(sum p_r L_r) == alpha * grad KL(pi||softmax(Qbar_p/alpha))
  V3. mirror case: pi_tilde == uniform exactly; g_hat == entropy channel
"""
import numpy as np
import torch

N = 3


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def qvec(r, R=1.0):
    return 2.0 * R * np.eye(N)[r]


def grad_softq(z, q, alpha):
    zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
    pi = torch.softmax(zt, 0)
    qt = torch.tensor(q, dtype=torch.double)
    lp = torch.log(torch.clamp(pi, min=1e-300))
    loss = (pi * (alpha * lp - qt)).sum()
    loss.backward()
    return zt.grad.detach().numpy()


def grad_kl_to(z, pi_star, alpha):
    """alpha * grad_z KL(pi || pi_star)."""
    zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
    pi = torch.softmax(zt, 0)
    pst = torch.tensor(pi_star, dtype=torch.double)
    loss = alpha * (pi * (torch.log(pi) - torch.log(pst))).sum()
    loss.backward()
    return zt.grad.detach().numpy()


def main():
    rng = np.random.default_rng(3)
    worst1 = worst2 = 0.0
    for _ in range(30):
        z = rng.normal(size=N) * 1.0
        alpha = float(rng.uniform(0.1, 2.0))
        qs = [qvec(r) * float(rng.uniform(0.3, 1.5)) for r in range(N)]
        p = rng.dirichlet(np.ones(N))

        # V1: per-condition softq == alpha*KL to softmax(Q_r/alpha)
        for r in range(N):
            g1 = grad_softq(z, qs[r], alpha)
            pi_star = softmax(qs[r] / alpha)
            g2 = grad_kl_to(z, pi_star, alpha)
            worst1 = max(worst1, np.abs(g1 - g2).max())

        # V2: mixed gradient == alpha*KL to softmax(Qbar/alpha)
        g_mix_auto = np.mean([p[r] * grad_softq(z, qs[r], alpha)
                              for r in range(N)], axis=0) * N
        qbar = np.sum([p[r] * qs[r] for r in range(N)], axis=0)
        g_mix_proj = grad_kl_to(z, softmax(qbar / alpha), alpha)
        worst2 = max(worst2, np.abs(g_mix_auto - g_mix_proj).max())
    print(f'V1 per-condition softq == alpha*KL(pi||softmax(Q_r/alpha)): '
          f'max err {worst1:.2e}')
    print(f'V2 mixed grad == alpha*KL(pi||softmax(Qbar_p/alpha)): '
          f'max err {worst2:.2e}')

    # V3: mirror conditions, uniform mixture -> projection target = uniform
    z = np.array([1.0, -0.5, -0.5])
    alpha = 1.0
    qA = qvec(0)          # (2,0,0)
    qB = qvec(0)[[2, 0, 1]]  # (0,2,0) permutation
    qC = qvec(0)[[1, 2, 0]]  # (0,0,2) permutation
    qbar = (qA + qB + qC) / 3.0
    print(f'V3 mirror mixture average Qbar = {qbar} (uniform => 2/3 each)')
    pi_tilde = softmax(qbar / alpha)
    print(f'   pi_tilde = {np.round(pi_tilde, 6)} (uniform? '
          f'{np.allclose(pi_tilde, np.ones(N)/N, atol=1e-12)})')
    g_mix = np.mean([grad_softq(z, q, alpha) for q in [qA, qB, qC]], axis=0)
    pi = softmax(z)
    ell = np.log(pi); lbar = pi @ ell
    entropy_channel = alpha * pi * (ell - lbar)
    print(f'   mixed grad == entropy channel? '
          f'{np.allclose(g_mix, entropy_channel, atol=1e-10)}')

    # V4: the chain end-to-end with asymmetric mixture
    p = np.array([0.7, 0.2, 0.1])
    qbar_p = np.sum([p[r] * q for r, q in enumerate([qA, qB, qC])], axis=0)
    g_mix_p = np.sum([p[r] * grad_softq(z, q, alpha)
                      for r, q in enumerate([qA, qB, qC])], axis=0)
    g_proj_p = grad_kl_to(z, softmax(qbar_p / alpha), alpha)
    print(f'V4 asymmetric mixture chain: max err '
          f'{np.abs(g_mix_p - g_proj_p).max():.2e}')


if __name__ == '__main__':
    main()