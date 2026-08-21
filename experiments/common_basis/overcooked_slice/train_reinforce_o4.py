"""O4: training-level performance consequence of the retention-specificity
trade-off on Overcooked.

Claim: a HARD policy-gradient field (REINFORCE, return-weighted, no baseline =
odd field) can train when the relation is VISIBLE but stalls when it is HIDDEN,
while a value-assisted field (PPO, advantage/mixed) learns under both.

Design (common basis):
  - same network (obs_dim = 97: 96 featurized + 1 role slot)
  - dynamic-hidden : partner role switches mid-episode, role slot = 0 (hidden)
  - static-visible : partner role fixed,      role slot = real role (visible)
  - shaped reward (per-step) so REINFORCE is tractable
  - run REINFORCE on both, compare reward curves; reference PPO learns dynamic.

Usage: python train_reinforce_o4.py <mode> <seed> [--steps 150000]
Results: data/kappa/overcooked_slice/o4_reinforce_<mode>_s<seed>.json
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
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_conflict_witness import DetChefAgent, DetWaiterAgent  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "kappa", "overcooked_slice")
os.makedirs(OUT_DIR, exist_ok=True)

N_ACTIONS = 6
OBS_DIM = 97


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_ACTIONS))

    def forward(self, obs):
        return self.net(obs)


class OCEnv:
    """Overcooked with per-step shaped reward + optional role visibility."""

    def __init__(self, mode, layout="cramped_room", horizon=400, switch=30,
                 seed=0):
        self.mode = mode  # 'dynamic-hidden' / 'dynamic-visible' / 'static-*'
        self.switch_on = "dynamic" in mode
        self.visible = "visible" in mode
        self.horizon = horizon
        self.switch = switch
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.pool = {"chef": DetChefAgent(self.env.mlam),
                     "waiter": DetWaiterAgent(self.env.mlam)}
        for a in self.pool.values():
            a.set_agent_index(1)
            a.set_mdp(self.mdp)
        self.partner_idx = 0
        self.switch_timer = 0
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        base = self.env.mdp.featurize_state(self.env.state,
                                            self.env.mlam)[0].astype(np.float32)
        role = np.zeros(1, dtype=np.float32)
        if self.visible:
            role[0] = float(self.partner_idx)
        return np.concatenate([base, role])

    def reset(self):
        self.env.reset()
        self.partner_idx = int(self.rng.integers(2))
        self.switch_timer = 0
        self.steps = 0
        return self._obs()

    def step(self, action):
        ptype = ("chef", "waiter")[self.partner_idx]
        p_act = self.pool[ptype].action(self.env.state)[0]
        a_act = Action.ALL_ACTIONS[int(action)]
        ns, infos = self.mdp.get_state_transition(
            self.env.state, [a_act, p_act], display_phi=False,
            motion_planner=self.env.mp)
        self.env.state = ns
        shaped = (infos["shaped_reward_by_agent"][0] +
                  infos["sparse_reward_by_agent"][0])
        self.steps += 1
        if self.switch_on:
            self.switch_timer += 1
            if self.switch_timer >= self.switch:
                self.partner_idx = 1 - self.partner_idx
                self.switch_timer = 0
        done = self.mdp.is_terminal(ns) or self.steps >= self.horizon
        return self._obs(), shaped, done


def train(mode, seed, steps, log_every=5000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = OCEnv(mode, seed=seed)
    net = PolicyNet().to("cuda")
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    curve = []
    t, ep = 0, 0
    best_rew = -1e9
    while t < steps:
        obs = env.reset()
        done = False
        obs_l, act_l, rew_l = [], [], []
        while not done and len(obs_l) < env.horizon:
            ot = torch.FloatTensor(obs).unsqueeze(0).to("cuda")
            logits = net(ot)
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            a = dist.sample()
            obs_l.append(obs)
            act_l.append(a.item())
            nxt, r, done = env.step(a.item())
            rew_l.append(r)
            obs = nxt
            t += 1
            if t % log_every == 0:
                curve.append({"t": t, "mean_recent": None})
        ep += 1
        G = np.cumsum(rew_l[::-1])[::-1].copy()
        obs_b = torch.FloatTensor(np.array(obs_l)).to("cuda")
        act_b = torch.tensor(act_l).to("cuda")
        Gt = torch.FloatTensor(G).to("cuda")
        logits = net(obs_b)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        lp = log_probs[torch.arange(len(act_b), device="cuda"), act_b]
        loss = -(lp * Gt).mean()
        ent = (probs * log_probs).sum(-1).mean()
        loss = loss - 0.01 * ent
        opt.zero_grad()
        loss.backward()
        opt.step()
        ep_rew = float(sum(rew_l))
        best_rew = max(best_rew, ep_rew)
        if curve and curve[-1]["t"] is None:
            curve[-1]["mean_recent"] = None
        if curve:
            curve[-1]["episode"] = ep
            curve[-1]["ep_rew"] = ep_rew
            curve[-1]["best_rew"] = best_rew
        if ep % 20 == 0:
            print(f"[{mode} s{seed}] t={t} ep={ep} last_ep_rew={ep_rew:.1f} "
                  f"best={best_rew:.1f}", flush=True)
    return {"mode": mode, "seed": seed, "steps": t, "episodes": ep,
            "best_reward": best_rew, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode",
                    choices=["dynamic-hidden", "dynamic-visible",
                             "static-hidden", "static-visible"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--steps", type=int, default=150000)
    args = ap.parse_args()

    res = train(args.mode, args.seed, args.steps)
    out = os.path.join(OUT_DIR, f"o4_reinforce_{args.mode}_s{args.seed}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"SAVED {out} best={res['best_reward']:.1f}", flush=True)


if __name__ == "__main__":
    main()
