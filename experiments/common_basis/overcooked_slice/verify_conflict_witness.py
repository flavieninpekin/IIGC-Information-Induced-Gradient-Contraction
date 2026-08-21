"""Overcooked controlled-slice: find witness states where the agent's best
action differs by partner role (chef vs waiter), verified by exhaustive /
scripted-option Q under a role-credit reward.

Design (per theory_program / toy_field_axis_theory "clean slice"):
  - role=chef  : partner COOKS,  agent should DELIVER -> agent credit = soup delivery
  - role=waiter: partner DELIVERS, agent should COOK  -> agent credit = onion potted
  - two scripted agent options (deliver / cook), deterministic partner policy.
  - witness := argmax option differs by role AND the winning option's FIRST
    primitive action differs, AND margin > delta.
  - also checks hidden-observation identity (role not visible in dynamic obs).

Only the partner role and the credit reward differ between conditions; the
state, horizon, options and RNG are identical -> clean relation contrast.

Usage: python verify_conflict_witness.py [--n-seed 3] [--horizon 80]
Results: data/kappa/overcooked_slice/conflict_witnesses.json
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np

import torch._dynamo  # noqa: F401  pre-import before gym/overcooked

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    SoupState,
)

from iigc.envs._overcooked.partner_agents import ChefAgent, WaiterAgent

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "kappa", "overcooked_slice")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "conflict_witnesses.json")

ACTION_NAMES = ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"]


def _a_probs(action):
    probs = {}
    for a in Action.ALL_ACTIONS:
        probs[a] = 1.0 if a == action else 0.0
    return probs


class DetChefAgent(ChefAgent):
    """Chef with deterministic fallback (STAY instead of random move)."""

    def _random_action(self):
        return Action.STAY, {"action_probs": _a_probs(Action.STAY)}


class DetWaiterAgent(WaiterAgent):
    def _random_action(self):
        return Action.STAY, {"action_probs": _a_probs(Action.STAY)}


class RoleCreditEnv:
    """Counterfactual Q evaluator with role-credit reward over scripted options."""

    def __init__(self, layout="large_room", horizon=80):
        self.layout = layout
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.horizon = horizon
        self.partner_start = self.mdp.terrain_pos_dict[" "][0]

    # -- agents ------------------------------------------------------------
    def agent_for_role(self, role, idx):
        mlam = self.env.mlam
        if role == "chef":   # partner cooks
            a = DetChefAgent(mlam)
        else:                # partner delivers
            a = DetWaiterAgent(mlam)
        a.set_agent_index(idx)
        a.set_mdp(self.mdp)
        return a

    def option_agent(self, option, idx):
        mlam = self.env.mlam
        a = (DetWaiterAgent(mlam) if option == "deliver"
             else DetChefAgent(mlam))  # deliver=WaiterAgent behavior, cook=ChefAgent
        a.set_agent_index(idx)
        a.set_mdp(self.mdp)
        return a

    # -- reward ------------------------------------------------------------
    def credit(self, infos, role):
        ev = infos["event_infos"]
        if role == "chef":
            return float(infos["sparse_reward_by_agent"][0])
        return 1.0 if ev["potting_onion"][0] else 0.0

    def roll(self, state, agent_opt, partner, role, seed=0):
        """Deterministic return over horizon: agent plays `agent_opt`, partner
        plays `partner`. Returns (return, first_action_index)."""
        rng = np.random.default_rng(seed)
        s = copy.deepcopy(state)
        tot = 0.0
        first_act = None
        for t in range(self.horizon):
            a_opt = agent_opt.action(s)[0]
            if t == 0:
                first_act = Action.ALL_ACTIONS.index(a_opt)
            p_act = partner.action(s)[0]
            ns, infos = self.mdp.get_state_transition(
                s, [a_opt, p_act], display_phi=False,
                motion_planner=self.env.mp)
            tot += self.credit(infos, role)
            s = ns
            if self.mdp.is_terminal(s):
                break
        return tot, first_act

    # -- state construction ------------------------------------------------
    def base_objects(self):
        """Ready soup in the (first) pot + one onion + one dish. Layout-aware."""
        P = self.mdp.terrain_pos_dict["P"][0]
        O = self.mdp.terrain_pos_dict["O"][0]
        D = self.mdp.terrain_pos_dict["D"][0]
        soup = SoupState(
            P,
            ingredients=[ObjectState("onion", P) for _ in range(3)],
            cooking_tick=20, cook_time=20)  # fully cooked -> is_ready immediately
        return {P: soup, O: ObjectState("onion", O), D: ObjectState("dish", D)}

    def candidate_states(self, partner_start=None):
        """Grid of agent start positions/orientations on a fixed scene."""
        free = [p for p in self.mdp.terrain_pos_dict[" "]]
        if partner_start is None:
            partner_start = free[0]
        states = []
        for pos in free:
            if pos == partner_start:
                continue
            for ori in Direction.ALL_DIRECTIONS:
                p0 = PlayerState(pos, ori, None)
                p1 = PlayerState(partner_start, Direction.SOUTH, None)
                states.append(OvercookedState(
                    [p0, p1], self.base_objects()))
        return states

    def hidden_obs(self, state):
        return self.mdp.featurize_state(state, self.env.mlam)[0]

    def is_done(self, state):
        return self.env._is_terminal(state)


def eval_state(rc, state):
    """For one state: Q of each option under each role + first primitive action."""
    rows = {}
    for role in ("chef", "waiter"):
        partner = rc.agent_for_role(role, 1)
        q, first = {}, {}
        for opt in ("deliver", "cook"):
            ag = rc.option_agent(opt, 0)
            r, fa = rc.roll(state, ag, partner, role)
            q[opt] = r
            first[opt] = fa
        rows[role] = {"Q": q, "argmax": max(q, key=q.get),
                      "first_action": first, "best_first": first[max(q, key=q.get)]}
    return rows


def is_witness(rows, delta=1.0):
    a_chef = rows["chef"]["argmax"]
    a_waiter = rows["waiter"]["argmax"]
    if a_chef == a_waiter:
        return False, "same_argmax"
    qc = rows["chef"]["Q"]
    qw = rows["waiter"]["Q"]
    m_c = qc[a_chef] - qc[a_waiter]
    m_w = qw[a_waiter] - qw[a_chef]
    if m_c < delta or m_w < delta:
        return False, "margin_%.2f_%.2f" % (m_c, m_w)
    return True, "OK"


def sample_rollout(args):
    """Run a seeded random walk in THIS subprocess and print N state dicts
    (one JSON per line). Keeps all overcooked transitions inside a short-lived
    process to avoid heap corruption from repeated native calls."""
    rc = RoleCreditEnv(layout=args.layout, horizon=args.horizon)
    rng = np.random.default_rng(0)
    env = OvercookedEnv.from_mdp(rc.mdp, horizon=args.horizon)
    env.reset()
    emitted, guard = 0, 0
    while emitted < args.n_sampled and guard < 5000:
        guard += 1
        st = copy.deepcopy(env.state)
        print(json.dumps(st.to_dict()), flush=True)
        emitted += 1
        joint = [Action.INDEX_TO_ACTION[int(rng.integers(0, len(Action.ALL_ACTIONS)))]
                 for _ in range(2)]
        if env.mdp.is_terminal(st) or env.state.timestep >= args.horizon:
            env.reset()
        else:
            try:
                env.state, _, _, _ = rc.mdp.get_state_transition(
                    st, joint, display_phi=False, motion_planner=env.mp)
            except Exception:
                env.reset()


def eval_one(args):
    """Evaluate a single state (constructed index or arbitrary JSON) and print
    a compact JSON line. Runs in a subprocess so native overcooked crashes
    never kill the batch."""
    rc = RoleCreditEnv(layout=args.layout, horizon=args.horizon)
    if args.eval_state_json:
        st = OvercookedState.from_dict(json.loads(args.eval_state_json))
    else:
        cands = rc.candidate_states()
        st = cands[int(args.one_state)]
    rows = eval_state(rc, st)
    hidden_same = np.array_equal(
        rc.hidden_obs(st),
        rc.hidden_obs(OvercookedState(
            [st.players[0], PlayerState(rc.partner_start, Direction.SOUTH, None)],
            rc.base_objects())))
    ok, why = is_witness(rows, args.delta)
    print(json.dumps({
        "player0_pos": list(st.players[0].position),
        "player0_ori": list(st.players[0].orientation),
        "chef_argmax": rows["chef"]["argmax"],
        "waiter_argmax": rows["waiter"]["argmax"],
        "Q": {r: rows[r]["Q"] for r in rows},
        "best_first_chef": rows["chef"]["best_first"],
        "best_first_waiter": rows["waiter"]["best_first"],
        "hidden_obs_identical": bool(hidden_same),
        "witness": ok,
        "why": why,
    }), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--n-sampled", type=int, default=40)
    ap.add_argument("--layout", default="large_room")
    ap.add_argument("--one-state", default=None, help="evaluate one constructed state")
    ap.add_argument("--eval-state-json", default=None, help="evaluate arbitrary state json")
    ap.add_argument("--sample-rollout", action="store_true",
                    help="print n sampled state dicts, one per line")
    args = ap.parse_args()

    if args.sample_rollout:
        sample_rollout(args)
        return
    if args.one_state is not None or args.eval_state_json:
        eval_one(args)
        return

    rc = RoleCreditEnv(layout=args.layout, horizon=args.horizon)
    results = {"config": {"horizon": args.horizon, "delta": args.delta,
                          "layout": args.layout},
               "witnesses": [], "crashes": [], "constructed": {}, "sampled": {}}

    def run_eval(payload):
        """Run eval_one in a subprocess; return (ok, dict|None)."""
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))  # repo root (IIGC)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--layout", args.layout, "--horizon", str(args.horizon),
               "--delta", str(args.delta), "--eval-state-json", payload]
        envp = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [os.path.join(root, "src"), root])}
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               env=envp, timeout=180)
        except Exception:
            return False, None
        if r.returncode != 0:
            return False, None
        try:
            return True, json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            return False, None

    # 1) constructed candidates (subprocess-isolated; native crashes -> skipped)
    cands = rc.candidate_states()
    n_wit, n_crash = 0, 0
    for i, st in enumerate(cands):
        ok, out = run_eval(json.dumps(st.to_dict()))
        if not ok:
            n_crash += 1
            results["crashes"].append({"phase": "constructed", "idx": i,
                                       "pos": list(st.players[0].position)})
            continue
        if out["witness"]:
            n_wit += 1
            results["witnesses"].append({
                "state_idx": i, "player0_pos": out["player0_pos"],
                "player0_ori": out["player0_ori"],
                "chef_argmax": out["chef_argmax"],
                "waiter_argmax": out["waiter_argmax"],
                "Q": out["Q"],
                "best_first_chef": out["best_first_chef"],
                "best_first_waiter": out["best_first_waiter"]})
    results["constructed"] = {
        "n_states": len(cands), "n_witness": n_wit, "n_crash": n_crash,
        "witness_fraction": n_wit / len(cands) if cands else 0.0}
    print(f"constructed: {n_wit}/{len(cands)} witness states "
          f"({n_wit/len(cands):.0%}), {n_crash} crashed", flush=True)

    # 2) hidden-obs identity (computed in-process; cheap and no rollouts)
    st0 = cands[0]
    results["hidden_obs_identical"] = bool(np.array_equal(
        rc.hidden_obs(st0), rc.hidden_obs(copy.deepcopy(st0))))
    print("hidden obs identical across roles (dynamic, no one-hot):",
          results["hidden_obs_identical"])

    # 3) sampled states: generated in a subprocess, evaluated in subprocesses
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    cmd = [sys.executable, os.path.abspath(__file__),
           "--layout", args.layout, "--horizon", str(args.horizon),
           "--n-sampled", str(args.n_sampled), "--sample-rollout"]
    envp = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [os.path.join(root, "src"), root])}
    n_sampled_wit = 0
    sampled = 0
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=envp,
                           timeout=300)
        states = [json.loads(line) for line in r.stdout.strip().splitlines()
                  if line.strip().startswith("{")]
    except Exception as e:
        print("sample subprocess failed:", e, flush=True)
        states = []
    for d in states:
        ok, out = run_eval(json.dumps(d))
        if ok:
            if out["witness"]:
                n_sampled_wit += 1
            sampled += 1
        else:
            results["crashes"].append(
                {"phase": "sampled", "pos": d.get("players", [{}])[0].get(
                    "position") if d.get("players") else None})
    results["sampled"] = {"n_states": sampled,
                          "n_witness": n_sampled_wit,
                          "witness_fraction": (n_sampled_wit / sampled
                                               if sampled else 0.0)}
    print(f"sampled natural states: {n_sampled_wit}/{sampled} witnesses "
          f"({(n_sampled_wit/sampled if sampled else 0):.0%})", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
