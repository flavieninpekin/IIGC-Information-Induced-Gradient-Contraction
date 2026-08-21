"""Field axis on Overcooked witness states.

Loads the conflict witnesses found by verify_conflict_witness.py, then for each
witness state computes, under the SAME policy and SAME hidden observation, the
relation-conditioned gradients for role=chef vs role=waiter, for two fields:

  reinforce : grad sum_t log pi(a_t) * r_t(role-credit)   [hard policy-gradient]
  value     : grad sum_t V(s_t)                            [mean-seeking / TD-like]

Role only changes the partner behavior and the credit reward; obs is identical
(hidden). Prediction: reinforce field CONFLICTS (low kappa) at witness states,
value field ALIGNS (kappa ~ 1) because the value function cannot see the role.

Usage: python verify_witness_field_axis.py [--horizon 40] [--n-seed 40]
Results: data/kappa/overcooked_slice/witness_field_axis.json
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

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedState,
    PlayerState,
)

from iigc.envs._overcooked.partner_agents import ChefAgent, WaiterAgent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_conflict_witness as V  # noqa: E402  (DetChefAgent/DetWaiterAgent, RoleCreditEnv)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "kappa", "overcooked_slice")
OUT = os.path.join(OUT_DIR, "witness_field_axis.json")
WITNESSES = os.path.join(OUT_DIR, "conflict_witnesses.json")

N_ACTIONS = 6


class PolicyNet(nn.Module):
    """Small actor+value net over the 96-dim hidden obs."""

    def __init__(self, obs_dim=96, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, N_ACTIONS))
        self.value = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 1))

    def logits(self, obs):
        return self.actor(obs)

    def v(self, obs):
        return self.value(obs)


def a_probs(action):
    p = {}
    for a in Action.ALL_ACTIONS:
        p[a] = 1.0 if a == action else 0.0
    return {"action_probs": p}


def partner_for(role, mlam, mdp):
    cls = V.DetChefAgent if role == "chef" else V.DetWaiterAgent
    a = cls(mlam)
    a.set_agent_index(1)
    a.set_mdp(mdp)
    return a


def role_credit(infos, role):
    ev = infos["event_infos"]
    if role == "chef":
        return float(infos["sparse_reward_by_agent"][0])
    return 1.0 if ev["potting_onion"][0] else 0.0


def obs_of(env, state):
    return env.mdp.featurize_state(state, env.mlam)[0].astype(np.float32)


def rollout_grads(model, env, state, partner, role, H, seed):
    """One seeded episode: reinforce gradient (policy params) + value gradient
    (value params) under role-credit reward. Returns (g_reinforce, g_value)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = copy.deepcopy(state)
    olist = []
    g_reinf = None
    tot = 0.0
    for t in range(H):
        obs = obs_of(env, s)
        olist.append(obs)
        ot = torch.FloatTensor(obs).unsqueeze(0)
        logits = model.logits(ot)
        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        a = dist.sample()
        # episode reinforce gradient (accumulated once at end)
        p_act = partner.action(s)[0]
        ns, infos = env.mdp.get_state_transition(
            s, [Action.ALL_ACTIONS[a.item()], p_act],
            display_phi=False, motion_planner=env.mp)
        r = role_credit(infos, role)
        tot += r
        # per-step reinforce grad: grad log pi(a) * r
        model.actor.zero_grad()
        lp = dist.log_prob(a)
        (lp * r).backward(retain_graph=True)
        gv = torch.cat([p.grad.detach().clone().flatten()
                        for p in model.actor.parameters() if p.grad is not None])
        g_reinf = gv if g_reinf is None else g_reinf + gv
        s = ns
        if env.mdp.is_terminal(s):
            break
    # value gradient: grad sum_t V(s_t)
    model.value.zero_grad()
    obs_b = torch.FloatTensor(np.array(olist))
    loss_v = model.v(obs_b).sum()
    loss_v.backward()
    g_val = torch.cat([p.grad.detach().clone().flatten()
                       for p in model.value.parameters() if p.grad is not None])
    return (g_reinf if g_reinf is not None else torch.zeros(1)), g_val


def rebuild_witness_states(rc, witness_meta):
    """Reconstruct witness OvercookedState from saved (pos, ori)."""
    by_key = {}
    for i, st in enumerate(rc.candidate_states()):
        by_key[(tuple(st.players[0].position), tuple(st.players[0].orientation))] = st
    out = []
    for w in witness_meta:
        key = (tuple(w["player0_pos"]), tuple(w["player0_ori"]))
        if key in by_key:
            out.append(by_key[key])
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="asymmetric_advantages")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--n-seed", type=int, default=40)
    args = ap.parse_args()

    meta = json.load(open(WITNESSES))
    witnesses = meta["witnesses"]
    rc = V.RoleCreditEnv(layout=args.layout, horizon=args.horizon)
    states = rebuild_witness_states(rc, witnesses)
    print(f"rebuilt {len(states)} witness states", flush=True)

    model = PolicyNet(obs_dim=96)
    model.eval()

    per_state = {}
    all_rf = {"gA": [], "gB": []}
    all_val = {"gA": [], "gB": []}

    for si, st in enumerate(states):
        rf = {"chef": [], "waiter": []}
        val = {"chef": [], "waiter": []}
        for role in ("chef", "waiter"):
            partner = partner_for(role, rc.env.mlam, rc.mdp)
            for seed in range(args.n_seed):
                g_r, g_v = rollout_grads(model, rc.env, st, partner, role,
                                         args.horizon, seed)
                rf[role].append(g_r)
                val[role].append(g_v)
        gA = torch.stack(rf["chef"])
        gB = torch.stack(rf["waiter"])
        vA = torch.stack(val["chef"])
        vB = torch.stack(val["waiter"])
        per_state[si] = {
            "pos": list(st.players[0].position),
            "ori": list(st.players[0].orientation),
            "reinforce": components(gA, gB),
            "value": components(vA, vB),
        }
        all_rf["gA"].append(gA.mean(0))
        all_rf["gB"].append(gB.mean(0))
        all_val["gA"].append(vA.mean(0))
        all_val["gB"].append(vB.mean(0))
        c = per_state[si]
        print(f"  s{si} {st.players[0].position}: "
              f"reinforce kappa={c['reinforce']['kappa_mean']:.3f} "
              f"(ep={c['reinforce']['kappa_ep']:.3f}) | "
              f"value kappa={c['value']['kappa_mean']:.3f} "
              f"(ep={c['value']['kappa_ep']:.3f})", flush=True)

    # aggregate: stack per-state condition-mean gradients -> overall kappa
    agg = {}
    for name, store in (("reinforce", all_rf), ("value", all_val)):
        gA = torch.stack(store["gA"])
        gB = torch.stack(store["gB"])
        agg[name] = components(gA, gB)
    print("=== aggregated across witness states ===", flush=True)
    for name in ("reinforce", "value"):
        c = agg[name]
        print(f"  {name:9s} kappa_mean={c['kappa_mean']:.3f} "
              f"kappa_ep={c['kappa_ep']:.3f} "
              f"E_shared={c['E_shared']:.1f} E_contrast={c['E_contrast']:.1f} "
              f"sigma2={c['sigma2']:.1f}", flush=True)

    results = {
        "config": {"layout": args.layout, "horizon": args.horizon,
                   "n_seed": args.n_seed, "n_witness_states": len(states)},
        "per_state": per_state,
        "aggregated": agg,
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
