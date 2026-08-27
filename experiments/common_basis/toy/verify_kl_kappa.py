"""KL formalization of kappa (Prop 6-8).

Claims verified to machine precision (V1/V2/V4) or convergent (V3):

V1. Reverse-KL field: g_r = grad_z KL(pi || pi_r*) equals the C.1 form with
    log-ratio weights w_r(a) = log(pi(a)/pi_r*(a)).
V2. Mixture = geometric-mean projection (KL Pythagorean identity):
    E_p[g_r] = grad_z KL(pi || pi_tilde),  pi_tilde = norm(exp(Sum p_r log pi_r*)).
    In particular for mirror targets + uniform mixture: pi_tilde = uniform,
    and the mixed gradient equals the entropy channel pi_b(ell_b - lbar).
V3. kappa as Fisher-metric / infinitesimal-KL ratio:
    kappa_F = (g_hat^T F g_hat) / E[(g_r^T F g_r)],  F = diag(pi) - pi pi^T;
    on the tangent space F g = diag(pi) g, and
    kappa_F = lim_{eps->0} KL(pi || pi + eps*g_hat) / E[KL(pi || pi + eps*g_r)].
V4. Forward-KL field: g_r = grad_z KL(pi_r* || pi) = pi - pi_r*; mixture
    gradient = pi - p-weighted arithmetic mean; mirror closed form.
"""
import numpy as np
import torch


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def grad_kl_rev_autograd(z, pi_star):
    """grad_z KL(pi || pi_star), autograd ground truth."""
    zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
    pi = torch.softmax(zt, 0)
    pst = torch.tensor(pi_star, dtype=torch.double)
    loss = (pi * (torch.log(pi) - torch.log(pst))).sum()
    loss.backward()
    return zt.grad.detach().numpy()


def grad_kl_rev_closed(z, pi_star):
    """C.1 form with w(a) = log(pi_a / pi_star_a); g_j = sum_a pi_a (delta_aj - pi_j) w_a."""
    pi = softmax(z)
    w = np.log(pi / pi_star)
    return np.array([np.sum(pi * (np.eye(3)[j] - pi[j]) * w) for j in range(3)])


def grad_kl_fwd(z, pi_star):
    """grad_z KL(pi_star || pi) = pi - pi_star (exact for categoricals)."""
    pi = softmax(z)
    return pi - pi_star


def fisher_kappa(gs, p, pi):
    """kappa_F = (g_hat^T F g_hat) / E_r[(g_r^T F g_r)]; F g = diag(pi) g on tangent."""
    gs = np.asarray(gs)
    ghat = np.asarray(p) @ gs
    num = np.sum(pi * ghat ** 2)
    den = np.mean([np.sum(pi * g ** 2) for g in gs])
    return num / den if den > 1e-300 else 0.0


def kl(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sum(a * np.log(a / b)))


def main():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(50):
        z = rng.normal(size=3) * 1.2
        ps = rng.dirichlet(np.ones(3))
        g1 = grad_kl_rev_autograd(z, ps)
        g2 = grad_kl_rev_closed(z, ps)
        worst = max(worst, np.abs(g1 - g2).max())
    print(f'V1 reverse-KL field (autograd vs C.1 closed): max err {worst:.2e}')

    # V2 geometric-mean projection
    worst2 = 0.0
    for _ in range(50):
        z = rng.normal(size=3) * 1.2
        ps = [rng.dirichlet(np.ones(3)) for _ in range(3)]
        p = rng.dirichlet(np.ones(3))
        ghat_auto = np.mean([p[i] * grad_kl_rev_autograd(z, ps[i])
                             for i in range(3)], axis=0) * 3
        log_tilde = np.sum([p[i] * np.log(ps[i]) for i in range(3)], axis=0)
        tilde = np.exp(log_tilde - log_tilde.max())
        tilde = tilde / tilde.sum()
        ghat_closed = grad_kl_rev_autograd(z, tilde)
        worst2 = max(worst2, np.abs(ghat_auto - ghat_closed).max())
    print(f'V2 mixture = KL-to-geometric-mean: max err {worst2:.2e}')

    # V2b mirror targets, uniform mixture -> uniform geometric mean -> entropy channel
    z = np.array([1.5, -0.5, -1.0])
    ps0 = np.array([0.8, 0.15, 0.05])
    ps1 = ps0[[2, 0, 1]]  # cyclic permutation (S3-style)
    ps2 = ps0[[1, 2, 0]]
    ghat = np.mean([grad_kl_rev_autograd(z, ps) for ps in [ps0, ps1, ps2]],
                   axis=0)
    pi = softmax(z)
    ell = np.log(pi); lbar = pi @ ell
    entropy_channel = pi * (ell - lbar)
    print('V2b mirror geometric mean -> uniform; mixed grad vs entropy channel:')
    print(f'   equal? = {np.allclose(ghat, entropy_channel, atol=1e-10)}')

    # V3 kappa_F = infinitesimal-KL ratio (converges as eps -> 0)
    ps = [np.array([0.8, 0.15, 0.05]), np.array([0.05, 0.8, 0.15]),
          np.array([0.15, 0.05, 0.8])]
    p_mix = np.array([0.5, 0.3, 0.2])
    gs = [grad_kl_rev_autograd(z, s) for s in ps]
    kF = fisher_kappa(gs, p_mix, pi)
    ghat = np.asarray(p_mix) @ np.asarray(gs)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        kl_hat = kl(pi, softmax(z + eps * ghat))
        kl_r = np.mean([kl(pi, softmax(z + eps * g)) for g in gs])
        ratio = kl_hat / kl_r
        print(f'V3 eps={eps:.0e}: inf-KL ratio={ratio:.6f} '
              f'(rel err {abs(kF - ratio) / kF:.2e})')

    # V4 forward-KL field: g_r = pi - pi_star; mixture -> p-weighted arithmetic mean
    pi_stars = [np.array([0.8, 0.15, 0.05]), np.array([0.05, 0.8, 0.15]),
                np.array([0.15, 0.05, 0.8])]
    gs_f = [grad_kl_fwd(z, s) for s in pi_stars]
    gbar = pi - np.asarray(p_mix) @ np.asarray(pi_stars)
    ghat_f = np.asarray(p_mix) @ np.asarray(gs_f)
    print(f'V4 forward-KL mixture = pi - p-weighted mean: '
          f'max err {np.abs(ghat_f - gbar).max():.2e}')
    # mirror special case: arithmetic mean = uniform; closed form check
    pi_mir = [np.array([0.7, 0.3]), np.array([0.3, 0.7])]
    z2 = np.array([1.0, -1.0])
    pi2 = softmax(z2)
    gs2 = [grad_kl_fwd(z2, s) for s in pi_mir]
    kF_mirror = fisher_kappa(gs2, np.array([0.5, 0.5]), pi2)
    num = np.sum(pi2 * (pi2 - 0.5) ** 2)
    den = np.mean([np.sum(pi2 * (pi2 - s) ** 2) for s in pi_mir])
    print(f'V4b mirror forward-KL closed form: kappa_F={kF_mirror:.6f} '
          f'closed={num / den:.6f}')


if __name__ == '__main__':
    main()