"""Toy theory validation for theory_program T1-T3 (retention-specificity).

Two complementary computations on the exact two-action relational bandit
(HiddenMatchingEnv, hidden, constant obs -> a single 2-action softmax policy):

  EXACT  : closed-form expected gradients g_A, g_B via autograd on the analytic
           objective, for any field/weight function -> kappa_mean (noise-free).
  SAMPLED: per-episode gradients via rollout -> full components including
           per-episode sigma2 and kappa_ep (the measurement-level numbers).

Fields: reinforce (no baseline) / awr(tau) / softq(alpha) / expq / gibbs(tau).
Sweeps for T1 (awr tau), T2 (softq alpha), T3 (gibbs tau). r = reward scale.

Writes data/kappa/toy_fields/theory_toy.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch._dynamo  # noqa: F401

sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

from iigc.envs._toy.toy_env import HiddenMatchingEnv  # noqa: E402

ROOT = r"C:\Users\Flavi\opencode\IIGC"
OUT_DIR = os.path.join(ROOT, "data", "kappa", "toy_fields")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "theory_toy.json")

N_EPS = 200
N_STEPS = 20
SEED_A, SEED_B = 100, 200
HIDDEN_INITS = 5


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, obs):
        return self.net(obs)


def train_revealed(obs_dim=2, steps=250, lr=1e-2, ent_coef=0.15):
    policy = PolicyNet(obs_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        p = np.random.randint(0, 2)
        obs = torch.FloatTensor([1.0 - p, p]).unsqueeze(0)
        logits = policy(obs)
        probs = F.softmax(logits, dim=-1)
        lp = F.log_softmax(logits, dim=-1)
        ent = -(probs * lp).sum()
        loss = F.cross_entropy(logits, torch.tensor([p])) - ent_coef * ent
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


# --------------------------------------------------------------------------
# EXACT closed-form (autograd on the analytic objective)
# --------------------------------------------------------------------------

def exact_kappa(logits, field, r=1.0, tau=1.0, alpha=1.0):
    """g_A, g_B for a fixed policy (2 logits) under relation A/B. Returns
    components from the exact expected gradients (noise-free kappa_mean)."""
    QA = torch.tensor([r, -r], dtype=torch.float32)   # A: a0 good
    QB = torch.tensor([-r, r], dtype=torch.float32)   # B: a1 good

    def g_of(Q):
        z = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
        pi = F.softmax(z, dim=-1)
        lp = F.log_softmax(z, dim=-1)
        if field == "reinforce":
            loss = -(lp * Q).sum()                    # -E_pi[log pi * Q] (no baseline)
        elif field == "awr":
            v = (pi * Q).sum()                        # baseline V = E_pi[Q]
            adv = Q - v
            w = torch.exp(adv / tau)
            loss = -(lp * w).sum()
        elif field == "awr_nobase":
            w = torch.exp(Q / tau)
            loss = -(lp * w).sum()
        elif field == "softmaxq":                     # framework gibbs: weight softmax(Q/tau)
            w = F.softmax(Q / tau, dim=-1)
            loss = -(lp * w).sum()
        elif field == "softq":
            loss = (pi * (alpha * lp - Q)).sum()      # note: entropy sign as fields.py
        elif field == "expq":
            loss = -(pi * Q).sum()
        elif field == "gibbs":                        # fields.py: E_{pi_tau}[Q]
            pi_t = F.softmax(z / tau, dim=-1)
            loss = -(pi_t * Q).sum()
        else:
            raise ValueError(field)
        loss.backward()
        return z.grad.detach().clone()

    gA = g_of(QA)
    gB = g_of(QB)
    muA, muB = gA, gB
    mu = (muA + muB) / 2.0
    E_shared = mu.norm().pow(2).item()
    E_contrast = ((muA - muB) / 2.0).norm().pow(2).item()
    E_total = E_shared + E_contrast
    kappa = E_shared / E_total if E_total > 0 else 0.0
    return dict(gA=gA.tolist(), gB=gB.tolist(), E_shared=E_shared,
                E_contrast=E_contrast, kappa_mean=kappa)


# --------------------------------------------------------------------------
# SAMPLED measurement (per-episode gradients -> full components)
# --------------------------------------------------------------------------

def rollout(policy, env, partner, n_eps):
    env.set_partner(partner)
    trajs = []
    for ep in range(n_eps):
        torch.manual_seed(SEED_A + ep if partner == 0 else SEED_B + ep)
        np.random.seed(SEED_A + ep if partner == 0 else SEED_B + ep)
        obs, info = env.reset()
        done = False
        traj = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            a = dist.sample()
            lp = dist.log_prob(a)
            obs, r, done, trunc, info = env.step(a.item())
            traj.append((obs_t, a, lp, r))
        trajs.append(traj)
    env.set_partner(None)
    return trajs


def q_fn(revealed, assignment, r=1.0):
    if revealed:
        def qf(obs):
            partner = obs.argmax(dim=1)
            q = torch.full_like(obs, -r)
            q[range(obs.shape[0]), partner] = r
            return q
        return qf
    p = 0 if assignment == "A" else 1

    def qf(obs):
        q = torch.full((obs.shape[0], 2), -r)
        q[:, p] = r
        return q
    return qf


def ep_grad_field(policy, traj, field, qf, tau=1.0, alpha=1.0):
    obs = torch.cat([t[0] for t in traj], dim=0)
    act = torch.tensor([t[1].item() for t in traj])
    rews = [t[3] for t in traj]
    G = np.cumsum(rews[::-1])[::-1].copy()
    logits = policy(obs)
    pi = F.softmax(logits, dim=-1)
    lp = F.log_softmax(logits, dim=-1)
    q = qf(obs)

    policy.zero_grad()
    if field == "reinforce":
        loss = -(lp[range(len(act)), act] * torch.FloatTensor(G)).sum()
    elif field == "awr":
        v = (pi * q).sum(dim=-1)                      # baseline V(s) = E_pi[Q]
        adv = q[range(len(act)), act] - v
        w = torch.exp(adv / tau)
        loss = -(lp[range(len(act)), act] * w).sum()
    elif field == "awr_nobase":
        adv = q[range(len(act)), act]
        w = torch.exp(adv / tau)
        loss = -(lp[range(len(act)), act] * w).sum()
    elif field == "softmaxq":
        w = F.softmax(q / tau, dim=-1)
        loss = -(lp[range(len(act)), act] * w[range(len(act)), act]).sum()
    elif field == "softq":
        loss = (pi * (alpha * lp - q)).sum()
    elif field == "expq":
        loss = -(pi * q).sum()
    elif field == "gibbs":
        pi_t = F.softmax(logits / tau, dim=-1)
        loss = -(pi_t * q).sum()
    else:
        raise ValueError(field)
    loss.backward()
    gv = [p.grad.detach().clone().flatten()
          for p in policy.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


def components(gA, gB):
    muA, muB = gA.mean(0), gB.mean(0)
    mu = (muA + muB) / 2.0
    E_shared = mu.norm().pow(2).item()
    E_contrast = ((muA - muB) / 2.0).norm().pow(2).item()
    varA = (gA - muA).norm(dim=1).pow(2).mean().item()
    varB = (gB - muB).norm(dim=1).pow(2).mean().item()
    sigma2 = (varA + varB) / 2.0
    E_total = E_shared + E_contrast + sigma2
    k_ep = E_shared / E_total if E_total > 0 else 0.0
    k_mean = E_shared / (E_shared + E_contrast) if (E_shared + E_contrast) > 0 else 0.0
    return dict(E_shared=E_shared, E_contrast=E_contrast, sigma2=sigma2,
                E_total=E_total, kappa_ep=k_ep, kappa_mean=k_mean)


def sample_fields(policy, revealed, r=1.0):
    env = HiddenMatchingEnv(revealed=revealed, n_steps=N_STEPS)
    qA = q_fn(revealed, "A", r)
    qB = q_fn(revealed, "B", r)
    eps_a = rollout(policy, env, 0, N_EPS)
    eps_b = rollout(policy, env, 1, N_EPS)
    env.close()
    out = {}
    for field in ["reinforce", "expq", "awr", "softq"]:
        gA = torch.stack([ep_grad_field(policy, t, field, qA)
                          for t in eps_a])
        gB = torch.stack([ep_grad_field(policy, t, field, qB)
                          for t in eps_b])
        out[field] = components(gA, gB)
    # gibbs at tau=1
    for field in ["gibbs"]:
        gA = torch.stack([ep_grad_field(policy, t, field, qA, tau=1.0)
                          for t in eps_a])
        gB = torch.stack([ep_grad_field(policy, t, field, qB, tau=1.0)
                          for t in eps_b])
        out[field] = components(gA, gB)
    return out


def sweep(policy, revealed, field, key, values, r=1.0):
    env = HiddenMatchingEnv(revealed=revealed, n_steps=N_STEPS)
    qA = q_fn(revealed, "A", r)
    qB = q_fn(revealed, "B", r)
    eps_a = rollout(policy, env, 0, N_EPS)
    eps_b = rollout(policy, env, 1, N_EPS)
    env.close()
    rows = []
    for v in values:
        kw = {key: v}
        gA = torch.stack([ep_grad_field(policy, t, field, qA, **kw) for t in eps_a])
        gB = torch.stack([ep_grad_field(policy, t, field, qB, **kw) for t in eps_b])
        c = components(gA, gB)
        rows.append({**{key: v}, **c})
    return rows


def main():
    results = {}

    # ---- HIDDEN: random-init policy (near-uniform pi), 5 inits ----
    hid = []
    exact_logits = []
    for i in range(HIDDEN_INITS):
        torch.manual_seed(i); np.random.seed(i)
        policy = PolicyNet(1)
        # exact closed-form from the policy's logits at the constant hidden obs
        with torch.no_grad():
            z = policy(torch.zeros(1, 1)).squeeze(0).tolist()
        exact_logits.append(z)
        fields = sample_fields(policy, False, r=1.0)
        hid.append(fields)
    results["HIDDEN"] = {
        "fields": {f: {k: float(np.mean([x[f][k] for x in hid]))
                       for k in ["E_shared", "E_contrast", "sigma2",
                                 "kappa_ep", "kappa_mean"]}
                   for f in ["reinforce", "expq", "awr", "softq", "gibbs"]},
        "n_inits": HIDDEN_INITS, "N_EPS": N_EPS,
        "exact_logits": exact_logits,
    }
    print("--- HIDDEN sampled (mean over %d inits, N=%d) ---" % (HIDDEN_INITS, N_EPS))
    for f, v in results["HIDDEN"]["fields"].items():
        print(f"  {f:9s} kappa_ep={v['kappa_ep']:.4f} kappa_mean={v['kappa_mean']:.4f} "
              f"E_shared={v['E_shared']:.2f} E_contrast={v['E_contrast']:.2f} "
              f"sigma2={v['sigma2']:.2f}")

    # ---- REVEALED ----
    torch.manual_seed(0); np.random.seed(0)
    pol_r = train_revealed()
    # soften to match run_toy_fields (tau=5)
    class Scaled(nn.Module):
        def __init__(self, base, t):
            super().__init__(); self.base = base; self.t = t
        def forward(self, obs):
            return self.base(obs) / self.t
    pol_r = Scaled(pol_r, 5.0)
    results["REVEALED"] = sample_fields(pol_r, True, r=1.0)
    print("--- REVEALED sampled ---")
    for f, v in results["REVEALED"].items():
        print(f"  {f:9s} kappa_ep={v['kappa_ep']:.4f} kappa_mean={v['kappa_mean']:.4f}")

    # ---- T1: awr tau sweep (exact + sampled) ----
    # use the SAME init-0 HIDDEN policy for exact and sampled; also a peaked one
    torch.manual_seed(0); np.random.seed(0)
    pol0 = PolicyNet(1)
    with torch.no_grad():
        z0 = pol0(torch.zeros(1, 1)).squeeze(0).tolist()

    class FixedLogits(nn.Module):
        def __init__(self, z):
            super().__init__()
            self.z = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        def forward(self, obs):
            return self.z.expand(obs.shape[0], -1)

    peaked = FixedLogits([2.0, -2.0])

    taus = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    ex = [exact_kappa(z0, "awr", r=1.0, tau=t)["kappa_mean"] for t in taus]
    ex_nb = [exact_kappa(z0, "awr_nobase", r=1.0, tau=t)["kappa_mean"] for t in taus]
    sam = sweep(pol0, False, "awr", "tau", taus, r=1.0)
    results["T1_awr_tau"] = {
        "r": 1.0, "taus": taus, "z0": z0,
        "exact_kappa_baseline": [round(x, 4) for x in ex],
        "exact_kappa_nobase": [round(x, 4) for x in ex_nb],
        "sampled_kappa_ep": [round(x["kappa_ep"], 4) for x in sam],
        "sampled_kappa_mean": [round(x["kappa_mean"], 4) for x in sam],
    }
    ex_peak = [exact_kappa([2.0, -2.0], "awr", r=1.0, tau=t)["kappa_mean"] for t in taus]
    ex_peak_nb = [exact_kappa([2.0, -2.0], "awr_nobase", r=1.0, tau=t)["kappa_mean"]
                  for t in taus]
    results["T1_awr_tau_peaked"] = {
        "r": 1.0, "taus": taus,
        "exact_kappa_baseline": [round(x, 4) for x in ex_peak],
        "exact_kappa_nobase": [round(x, 4) for x in ex_peak_nb],
    }
    print("--- T1 awr tau sweep (r=1) ---")
    print("  tau:", taus)
    print("  exact baseline:", results["T1_awr_tau"]["exact_kappa_baseline"])
    print("  exact nobase  :", results["T1_awr_tau"]["exact_kappa_nobase"])
    print("  sampled ep    :", results["T1_awr_tau"]["sampled_kappa_ep"])
    print("  exact baseline peak:", results["T1_awr_tau_peaked"]["exact_kappa_baseline"])
    print("  exact nobase peak  :", results["T1_awr_tau_peaked"]["exact_kappa_nobase"])

    # ---- T2: softq alpha sweep (exact + sampled) ----
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ex2 = [exact_kappa(z0, "softq", r=1.0, alpha=a)["kappa_mean"] for a in alphas]
    sam2 = sweep(pol0, False, "softq", "alpha", alphas, r=1.0)
    ex2_peak = [exact_kappa([2.0, -2.0], "softq", r=1.0, alpha=a)["kappa_mean"]
                for a in alphas]
    results["T2_softq_alpha"] = {
        "r": 1.0, "alphas": alphas,
        "exact_kappa": [round(x, 4) for x in ex2],
        "exact_kappa_peaked": [round(x, 4) for x in ex2_peak],
        "sampled_kappa_ep": [round(x["kappa_ep"], 4) for x in sam2],
        "sampled_kappa_mean": [round(x["kappa_mean"], 4) for x in sam2],
    }
    print("--- T2 softq alpha sweep (r=1) ---")
    print("  alpha:", alphas)
    print("  exact      :", results["T2_softq_alpha"]["exact_kappa"])
    print("  exact peak :", results["T2_softq_alpha"]["exact_kappa_peaked"])
    print("  sampled ep :", results["T2_softq_alpha"]["sampled_kappa_ep"])

    # ---- T3: gibbs tau sweep. Two meanings:
    #   gibbs (fields.py: grad E_{pi_tau}[Q]) -> exact kappa = 0 for all tau
    #   softmaxq (framework C.1: weight softmax(Q/tau)) -> monotone tau trend
    gtaus = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    ex3 = [exact_kappa(z0, "gibbs", r=1.0, tau=t)["kappa_mean"] for t in gtaus]
    ex_smq = [exact_kappa(z0, "softmaxq", r=1.0, tau=t)["kappa_mean"] for t in gtaus]
    sam3 = sweep(pol0, False, "softmaxq", "tau", gtaus, r=1.0)
    results["T3_gibbs_tau"] = {
        "r": 1.0, "taus": gtaus,
        "exact_gibbs_expq": [round(x, 4) for x in ex3],
        "exact_softmaxq": [round(x, 4) for x in ex_smq],
        "sampled_softmaxq_ep": [round(x["kappa_ep"], 4) for x in sam3],
    }
    print("--- T3 gibbs tau sweep (r=1) ---")
    print("  tau:", gtaus)
    print("  exact gibbs(E_{pi_tau}[Q]):", results["T3_gibbs_tau"]["exact_gibbs_expq"])
    print("  exact softmax(Q/tau)      :", results["T3_gibbs_tau"]["exact_softmaxq"])
    print("  sampled softmaxq ep       :", results["T3_gibbs_tau"]["sampled_softmaxq_ep"])

    # ---- r=4 variant for awr closed-form (notes use r=4) ----
    ex4 = [exact_kappa(z0, "awr", r=4.0, tau=t)["kappa_mean"] for t in taus]
    results["T1_awr_tau_r4"] = {
        "r": 4.0, "taus": taus, "exact_kappa_baseline": [round(x, 4) for x in ex4],
    }
    print("--- T1 awr r=4 (exact baseline) ---")
    print("  exact:", results["T1_awr_tau_r4"]["exact_kappa_baseline"])

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
