"""510K field-axis: reinforce-field vs value-field kappa on ppo_reveal models.

Uses the existing team-conditioned protocol (run_510k_reveal.py): full-info eval
(mode='obvious'), per-episode gradients, grouped by latent team (player 0 in red
team A or not) -> variance decomposition -> kappa_ep = mu2/(mu2+var_total).

For each model we compute BOTH fields on the SAME episodes:
  reinforce : per-step r-weighted PG gradient (hard, mode-seeking)
  value     : grad sum_t V(s_t) (mean-seeking / TD-like)

Field-axis prediction (from Overcooked): value field shows higher kappa than
reinforce, most clearly on hidden-adapted (reveal=0) models.
"""
import os
import sys

import numpy as np
import torch

import torch._dynamo  # noqa: F401  pre-import

sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

from sb3_contrib import MaskablePPO  # noqa: E402
from iigc.envs._510k.env import FiveTenKEnv  # noqa: E402

MODEL_DIR = r"C:\Users\Flavi\opencode\IIGC\data\models_reveal"
OUT = r"C:\Users\Flavi\opencode\IIGC\data\kappa\server_tasks\results\510k_field_axis.json"
N_EPS = 60
LEVELS = [0.0, 0.5, 1.0]
SEEDS = [41, 42, 43, 44, 45, 46]


def model_path(level, seed):
    fp = os.path.join(MODEL_DIR, f"ppo_reveal_{level:.2f}_s{seed}.zip")
    if not os.path.exists(fp):
        return None
    import zipfile
    try:
        z = zipfile.ZipFile(fp)
        ok = True
        for n in z.namelist():
            try:
                z.read(n)
            except Exception:
                ok = False
                break
        z.close()
        return fp if ok else None
    except Exception:
        return None


def ep_grad_rf(model, env):
    """One episode's REINFORCE gradient (per-step r-weighted) + team + reward."""
    obs, info = env.reset()
    game = env.unwrapped.game
    team = 1 if (game.red_a_team and 0 in game.red_a_team) else 0
    total_r = 0.0
    g_rf = None
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        d = model.policy.get_distribution(ot)
        a = d.get_actions().item()
        next_obs, r, done, trunc, info = env.step(a)
        total_r += r
        d2 = model.policy.get_distribution(torch.FloatTensor(obs).unsqueeze(0))
        lp = d2.log_prob(torch.tensor([a]))
        model.policy.zero_grad()
        (-lp * r).backward()
        gv = torch.cat([p.grad.detach().clone().flatten()
                        for p in model.policy.parameters() if p.grad is not None])
        g_rf = gv if g_rf is None else g_rf + gv
        obs = next_obs
    return (g_rf if g_rf is not None else torch.zeros(1)), team, total_r


def ep_grad_value(model, env):
    """One episode's value-field gradient: grad sum_t V(s_t) + team + reward."""
    obs, info = env.reset()
    game = env.unwrapped.game
    team = 1 if (game.red_a_team and 0 in game.red_a_team) else 0
    total_r = 0.0
    g = None
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        with torch.enable_grad():
            v = model.policy.predict_values(ot)
            model.policy.zero_grad()
            v.backward()
        gv = torch.cat([p.grad.detach().clone().flatten()
                        for p in model.policy.parameters()
                        if p.grad is not None])
        g = gv if g is None else g + gv
        obs, r, done, trunc, info = env.step(
            model.policy.get_distribution(
                torch.FloatTensor(obs).unsqueeze(0)).get_actions().item())
        total_r += r
    return (g if g is not None else torch.zeros(1)), team, total_r


def variance_decomp(ep_grads, ep_teams):
    g_by_team = {}
    for g, t in zip(ep_grads, ep_teams):
        g_by_team.setdefault(int(t), []).append(g)
    if len(g_by_team) < 2:
        return {}
    team_mus, team_vars, n_t = {}, {}, {}
    for t, gs in g_by_team.items():
        gs_t = torch.stack(gs)
        team_mus[t] = gs_t.mean(0)
        team_vars[t] = (gs_t - team_mus[t]).norm(dim=1).pow(2).mean().item()
        n_t[t] = len(gs)
    total_n = sum(n_t.values())
    mu_global = sum(mu * n_t[t] / total_n for t, mu in team_mus.items())
    var_between = sum((mu - mu_global).norm().pow(2).item() * n_t[t] / total_n
                      for t, mu in team_mus.items())
    var_within = sum(v * n_t[t] / total_n for t, v in team_vars.items())
    allg = torch.cat([torch.stack(g_by_team[t]) for t in g_by_team])
    var_total = (allg - mu_global).norm(dim=1).pow(2).mean().item()
    mu2 = mu_global.norm().pow(2).item()
    k_ep = mu2 / (mu2 + var_total) if (mu2 + var_total) > 0 else float("nan")
    return {"mu2": mu2, "var_between": var_between, "var_within": var_within,
            "var_total": var_total, "kappa_ep": k_ep, "n_team": n_t}


def measure(model, n_eps=N_EPS):
    env = FiveTenKEnv(mode="obvious")
    rf_g, v_g, teams, rews = [], [], [], []
    for _ in range(n_eps):
        torch.manual_seed(np.random.randint(0, 2**31))
        np.random.seed(np.random.randint(0, 2**31))
        g_rf, team, r = ep_grad_rf(model, env)
        rf_g.append(g_rf); teams.append(team); rews.append(r)
        torch.manual_seed(np.random.randint(0, 2**31))
        np.random.seed(np.random.randint(0, 2**31))
        g_v, team2, r2 = ep_grad_value(model, env)
        v_g.append(g_v)
    env.close()
    d_rf = variance_decomp(rf_g, teams)
    d_v = variance_decomp(v_g, teams)
    return {
        "reinforce": d_rf,
        "value": d_v,
        "avg_reward": float(np.mean(rews)),
    }


def main():
    out = {}
    print(f'{"lvl seed":>10} {"rf_kep":>7} {"val_kep":>7} {"rf_varw":>10} {"val_varw":>10} {"r":>7}')
    for level in LEVELS:
        for seed in SEEDS:
            fp = model_path(level, seed)
            if fp is None:
                continue
            model = MaskablePPO.load(fp, device="cpu")
            model.policy.eval()
            m = measure(model)
            out.setdefault(level, {})[f"s{seed}"] = m
            rf = m["reinforce"].get("kappa_ep", float("nan"))
            vl = m["value"].get("kappa_ep", float("nan"))
            rfw = m["reinforce"].get("var_within", 0)
            valw = m["value"].get("var_within", 0)
            print(f'{level:>5.2f} s{seed:>2d}  {rf:>7.4f} {vl:>7.4f} '
                  f'{rfw:>10.2e} {valw:>10.2e} {m["avg_reward"]:>7.2f}')
            del model
    print("\n=== SUMMARY (mean over seeds) ===")
    for level in LEVELS:
        rows = out.get(level, {})
        if not rows:
            continue
        rf = [r["reinforce"]["kappa_ep"] for r in rows.values() if "kappa_ep" in r["reinforce"]]
        vl = [r["value"]["kappa_ep"] for r in rows.values() if "kappa_ep" in r["value"]]
        print(f"  p={level:.2f} (n={len(rf)}): reinforce kappa_ep={np.mean(rf):.4f}+-{np.std(rf):.4f}  "
              f"value kappa_ep={np.mean(vl):.4f}+-{np.std(vl):.4f}")
    import json
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved", OUT)


if __name__ == "__main__":
    main()
