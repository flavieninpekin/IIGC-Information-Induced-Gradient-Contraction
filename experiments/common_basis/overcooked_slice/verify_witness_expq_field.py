"""Expected-Q field on Overcooked witness states.

For each witness state s and each role r, compute per-primitive-action Q_r(s,a)
by: first primitive action a, then the role's best-option script (deliver under
chef, cook under waiter) for the remaining horizon, under the role-credit reward.
Then the expected-Q (expq, mean-seeking) gradient field is

    g_r(s) = sum_a pi(a) grad_log_pi(a) * Q_r(s,a)

Aggregated over witness states, we compare g_chef vs g_waiter -> kappa, and
report the value field (obs-only, role-invisible) for contrast.

This avoids the sparse-reward degeneration of rollout reinforce (random policy
never completes tasks -> zero gradients) because Q_r is scripted-completed.

Usage: python verify_witness_expq_field.py [--horizon 40]
Results: data/kappa/overcooked_slice/witness_expq_field.json
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch._dynamo  # noqa: F401

from overcooked_ai_py.mdp.actions import Action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_conflict_witness as V  # noqa: E402
from verify_witness_field_axis import (  # noqa: E402
    PolicyNet, partner_for, role_credit, obs_of, components,
    rebuild_witness_states,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "kappa", "overcooked_slice")
OUT = os.path.join(OUT_DIR, "witness_expq_field.json")
WITNESSES = os.path.join(OUT_DIR, "conflict_witnesses.json")

N_ACTIONS = 6


def dense_credit(infos, role):
    """Densified role-credit: reward progress toward the role's goal so the
    first primitive action changes the return."""
    ev = infos["event_infos"]
    c = 0.0
    if role == "chef":
        c += 1.0 if ev["useful_dish_pickup"][0] else 0.0
        c += 2.0 if ev["soup_pickup"][0] else 0.0
        c += 20.0 if ev["soup_delivery"][0] else 0.0
    else:
        c += 1.0 if ev["useful_onion_pickup"][0] else 0.0
        c += 3.0 if ev["potting_onion"][0] else 0.0
    return c


def q_of_primitive_stochastic(rc, state, role, a, H, K=8):
    """Q_r(s,a) = mean densified-credit return of: primitive `a`, then a FIXED
    seeded-random continuation (role-agnostic). The imperfect continuation makes
    the first action matter (no compensation)."""
    partner = partner_for(role, rc.env.mlam, rc.mdp)
    acc = 0.0
    for k in range(K):
        rng = np.random.default_rng(1000 + k)
        s = copy.deepcopy(state)
        tot = 0.0
        for t in range(H):
            if t == 0:
                a_opt = Action.ALL_ACTIONS[a]
            else:
                a_opt = Action.ALL_ACTIONS[int(rng.integers(0, N_ACTIONS))]
            p_act = partner.action(s)[0]
            ns, infos = rc.mdp.get_state_transition(
                s, [a_opt, p_act], display_phi=False, motion_planner=rc.env.mp)
            tot += dense_credit(infos, role)
            s = ns
            if rc.mdp.is_terminal(s):
                break
        acc += tot
    return acc / K


def field_grad(model, obs, Q, mode="expq"):
    """Expected-Q (expq) or argmax-committed (hard) gradient at one obs."""
    ot = torch.FloatTensor(obs).unsqueeze(0)
    logits = model.logits(ot)
    pi = F.softmax(logits, dim=-1).squeeze(0)
    lp = F.log_softmax(logits, dim=-1).squeeze(0)
    grad_logp = torch.zeros(N_ACTIONS, N_ACTIONS)
    for a in range(N_ACTIONS):
        e = torch.zeros(N_ACTIONS)
        e[a] = 1.0
        grad_logp[a] = e - pi  # d/d logits log pi(a) = e_a - pi
    if mode == "expq":
        g = sum(pi[a].item() * grad_logp[a] * Q[a] for a in range(N_ACTIONS))
    else:  # hard: commit to argmax action
        a_star = int(np.argmax(Q))
        g = grad_logp[a_star] * Q[a_star]
    return g.detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--layout", default="asymmetric_advantages")
    args = ap.parse_args()

    meta = json.load(open(WITNESSES))
    rc = V.RoleCreditEnv(layout=args.layout, horizon=args.horizon)
    states = rebuild_witness_states(rc, meta["witnesses"])
    print(f"rebuilt {len(states)} witness states", flush=True)

    # random-init policy (96-dim obs) -> pi distribution per state
    model = PolicyNet(obs_dim=96)
    model.eval()

    per_state = {}
    agg = {"expq": {"gA": [], "gB": []}, "hard": {"gA": [], "gB": []},
           "value": {"gA": [], "gB": []}}

    for si, st in enumerate(states):
        obs = obs_of(rc.env, st)
        ot = torch.FloatTensor(obs).unsqueeze(0)
        Qs, gs = {}, {}
        for role in ("chef", "waiter"):
            Q = np.array([q_of_primitive_stochastic(rc, st, role, a, args.horizon)
                          for a in range(N_ACTIONS)])
            Qs[role] = Q.tolist()
            gs[("expq", role)] = field_grad(model, obs, Q, "expq")
            gs[("hard", role)] = field_grad(model, obs, Q, "hard")
        # value field (obs-only): gradient of V(s)
        model.value.zero_grad()
        v = model.v(ot)
        v.sum().backward()
        g_val = torch.cat([p.grad.detach().clone().flatten()
                           for p in model.value.parameters() if p.grad is not None])
        g_val = g_val.detach().cpu().numpy()

        row = {"pos": list(st.players[0].position),
               "ori": list(st.players[0].orientation),
               "Q": Qs,
               "expq_angle_deg": float(np.degrees(np.arccos(np.clip(
                   np.dot(gs[("expq", "chef")], gs[("expq", "waiter")]) /
                   (np.linalg.norm(gs[("expq", "chef")]) *
                    np.linalg.norm(gs[("expq", "waiter")]) + 1e-12), -1, 1)))),
               }
        per_state[si] = row
        for name, key in (("expq", "expq"), ("hard", "hard"), ("value", "value")):
            gA = gs[(key, "chef")] if key != "value" else g_val
            gB = gs[(key, "waiter")] if key != "value" else g_val
            agg[name]["gA"].append(np.asarray(gA, dtype=float))
            agg[name]["gB"].append(np.asarray(gB, dtype=float))
        print(f"  s{si} {st.players[0].position}: Q_chef={np.round(Qs['chef'])} "
              f"Q_waiter={np.round(Qs['waiter'])} "
              f"expq_angle={row['expq_angle_deg']:.0f}deg", flush=True)

    def agg_kappa(store):
        gA = np.stack([g for g in store["gA"] if g.size == store["gA"][0].size])
        gB = np.stack([g for g in store["gB"] if g.size == store["gB"][0].size])
        muA, muB = gA.mean(0), gB.mean(0)
        mu = (muA + muB) / 2.0
        E_shared = float(np.linalg.norm(mu) ** 2)
        E_contrast = float(np.linalg.norm((muA - muB) / 2.0) ** 2)
        k_mean = E_shared / (E_shared + E_contrast) if (E_shared + E_contrast) > 0 else 0.0
        return {"kappa_mean": k_mean, "E_shared": E_shared, "E_contrast": E_contrast}

    result = {}
    for name in ("expq", "hard", "value"):
        result[name] = agg_kappa(agg[name])
    print("=== aggregated across witness states ===", flush=True)
    for name in ("expq", "hard", "value"):
        r = result[name]
        print(f"  {name:6s} kappa_mean={r['kappa_mean']:.3f} "
              f"E_shared={r['E_shared']:.2f} E_contrast={r['E_contrast']:.2f}", flush=True)

    out = {"config": {"layout": args.layout, "horizon": args.horizon,
                      "n_witness_states": len(states)},
           "per_state": per_state, "aggregated": result}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
