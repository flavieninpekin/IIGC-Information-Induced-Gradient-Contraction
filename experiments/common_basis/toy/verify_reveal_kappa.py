"""Prop 10: continuous reveal -> kappa is MONOTONE INCREASING (no valley).

Connects the abandoned AAAI reveal experiment (p = reveal probability,
"75% valley" died with n=6) to the new KL form of kappa.

Theory: with partial reveal the policy has three states (hidden / revealed-A /
revealed-B). Condition-conditioned gradients split:
    g_r(p) = (1-p) g_r^hid + p g_r^rev,   and mirror gives g_A^hid = -g_B^hid.
With block-orthogonal parameters (separate logits per state):
    kappa(p) = p^2 ||m_rev||^2 / ( p^2 E_rev + (1-p)^2 E_hid )
        where m_rev = (g_A^rev + g_B^rev)/2, E_rev = avg ||g_r^rev||^2,
              E_hid = ||g_A^hid||^2
    => strictly increasing on (0,1), no peak/valley.  d kappa/dp > 0.

Verifications:
  V1. block-logits closed form vs autograd (machine precision)
  V2. shared-MLP variant: monotonicity checked numerically
  V3. KL reading: kappa(p) = lim_eps KL(pi||pi+eps g_hat)/E_r[KL(pi||pi+eps g_r)]
"""
import numpy as np
import torch
import torch.nn as nn


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# V1: block-logit policy, exact expected gradients
# ---------------------------------------------------------------------------

def grads_block(z_hid, z_a, z_b, q_a, q_b):
    """Expected reinforce/expq gradients for each state, as 6-dim vectors."""
    def per_state(z, q):
        zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
        pi = torch.softmax(zt, 0)
        qt = torch.tensor(q, dtype=torch.double)
        # expq canonical: g = -grad <pi,Q> (sign irrelevant for kappa)
        (- (pi * qt).sum()).backward()
        return zt.grad.numpy()

    ga_h = per_state(z_hid, q_a)
    gb_h = per_state(z_hid, q_b)
    ga_r = per_state(z_a, q_a)
    gb_r = per_state(z_b, q_b)
    return ga_h, gb_h, ga_r, gb_r


def kappa_of(p, ga_h, gb_h, ga_r, gb_r):
    def embed(v, block):
        x = np.zeros(6)
        x[block * 2: block * 2 + 2] = v
        return x

    ga = (1 - p) * embed(ga_h, 0) + p * embed(ga_r, 1)
    gb = (1 - p) * embed(gb_h, 0) + p * embed(gb_r, 2)
    m = (ga + gb) / 2
    return (m @ m) / max((ga @ ga + gb @ gb) / 2, 1e-300)


def main():
    q_a = np.array([1.0, -1.0])
    q_b = np.array([-1.0, 1.0])
    z_hid = np.array([0.3, -0.3])
    z_a = np.array([1.5, -1.5])   # sharp revealed policy for A
    z_b = np.array([-1.5, 1.5])   # sharp revealed policy for B

    ga_h, gb_h, ga_r, gb_r = grads_block(z_hid, z_a, z_b, q_a, q_b)
    print('mirror check g_A^hid = -g_B^hid:',
          np.allclose(ga_h, -gb_h, atol=1e-12))

    # closed form parameters (embed revealed gradients into separate blocks)
    def embed(v, block):
        x = np.zeros(6)
        x[block * 2: block * 2 + 2] = v
        return x

    ga_r6, gb_r6 = embed(ga_r, 1), embed(gb_r, 2)
    A = float(np.sum(((ga_r6 + gb_r6) / 2) ** 2))
    B = float((ga_r6 @ ga_r6 + gb_r6 @ gb_r6) / 2)
    C = float(ga_h @ ga_h)
    print(f'A={A:.4f} B={B:.4f} C={C:.4f}  A/B={A / B:.4f} (kappa(1))')

    print('\nV1 block-logits: kappa(p) closed form vs autograd')
    worst = 0.0
    ps = np.linspace(0, 1, 21)
    ks = []
    for p in ps:
        k_auto = kappa_of(p, ga_h, gb_h, ga_r, gb_r)
        k_closed = p ** 2 * A / (p ** 2 * B + (1 - p) ** 2 * C)
        ks.append(k_auto)
        worst = max(worst, abs(k_auto - k_closed))
    print(f'  max|auto - closed| = {worst:.2e}')
    print('  kappa(p):', [round(k, 3) for k in ks])
    inc = all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
    print(f'  strictly increasing? {inc}   kappa(0)={ks[0]:.2e}  '
          f'kappa(1)={ks[-1]:.4f} (= A/B = {A / B:.4f})')

    # ------------------------------------------------------------------
    # V2: shared-MLP policy (gradients NOT block-orthogonal)
    # ------------------------------------------------------------------
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 2))

        def logits(self, x):
            return self.net(torch.tensor(x, dtype=torch.double))

    torch.manual_seed(0)
    net = Net()
    net.double()
    obs_hid = np.array([1.0, 0, 0]); obs_a = np.array([0, 1.0, 0])
    obs_b = np.array([0, 0, 1.0])

    def grad_state(obs, q):
        zt = torch.tensor(net.net[0].weight.detach().numpy() @ obs,
                          dtype=torch.double)
        # full autograd through the net
        for prm in net.parameters():
            prm.requires_grad = True
        net.zero_grad()
        lg = net.logits(obs)
        pi = torch.softmax(lg, 0)
        qt = torch.tensor(q, dtype=torch.double)
        (-(pi * qt).sum()).backward()
        g = torch.cat([p.grad.detach().flatten()
                       for p in net.parameters() if p.grad is not None])
        return g.numpy()

    net.train()
    ga_h2 = grad_state(obs_hid, q_a)
    gb_h2 = grad_state(obs_hid, q_b)
    ga_r2 = grad_state(obs_a, q_a)
    gb_r2 = grad_state(obs_b, q_b)

    def kappa_shared(p):
        ga = (1 - p) * ga_h2 + p * ga_r2
        gb = (1 - p) * gb_h2 + p * gb_r2
        m = (ga + gb) / 2
        return (m @ m) / max((ga @ ga + gb @ gb) / 2, 1e-300)

    ks2 = [kappa_shared(p) for p in ps]
    inc2 = all(ks2[i] < ks2[i + 1] for i in range(len(ks2) - 1))
    print(f'\nV2 shared-MLP: kappa(p) = {[round(k, 3) for k in ks2]}')
    print(f'  strictly increasing? {inc2}')

    # ------------------------------------------------------------------
    # V3: KL reading at one p (2-dim z_hid space, block policy)
    # ------------------------------------------------------------------
    def kl(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.sum(a * np.log(a / b)))

    p = 0.5
    ga = (1 - p) * ga_h + p * ga_r   # block-embedded? no: use raw 2-dim hid/revealed
    gb = (1 - p) * gb_h + p * gb_r
    ghat = (ga + gb) / 2
    pi0 = softmax(z_hid)
    kF = kappa_of(p, ga_h, gb_h, ga_r, gb_r)
    for eps in [1e-4, 1e-5]:
        kl_h = kl(pi0, softmax(z_hid + eps * ghat))
        kl_r = (kl(pi0, softmax(z_hid + eps * ga)) +
                kl(pi0, softmax(z_hid + eps * gb))) / 2
        print(f'V3 eps={eps:.0e}: kappa={kF:.5f} KL-ratio={kl_h / kl_r:.5f}')


if __name__ == '__main__':
    main()