"""510K field-axis measurements on ppo_reveal checkpoints.

The protocol uses full-information evaluation and groups per-episode gradients
by latent team. ``value`` is the field ``grad sum_t V(s_t)``; it is a value
diagnostic, not a generic TD-residual or mean-seeking theorem.

Both fields are computed on the SAME episode batch (paired protocol): one
stochastic rollout per episode yields the reinforce and value gradients for
identical observations, actions, and team labels. Paired results are stored
under keys ``s<seed>_paired`` so that legacy unpaired entries remain readable
but are never mixed into the paired summary.
"""
import json
import os
import sys

import numpy as np
import torch

import torch._dynamo  # noqa: F401  pre-import

sys.path.insert(0, r"C:\Users\Flavi\opencode\IIGC\src")

from sb3_contrib import MaskablePPO  # noqa: E402
from iigc.envs._510k.env import FiveTenKEnv  # noqa: E402
from iigc.metrics.kappa import episode_decomposition, measurement_metadata  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.environ.get(
    "IIGC_510K_MODELS", os.path.join(ROOT, "data", "models_reveal"))
OUT = os.path.join(ROOT, "data", "kappa", "server_tasks", "results",
                   "510k_field_axis.json")
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


def _flat_policy_grad(model):
    parts = []
    for p in model.policy.parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        parts.append(g.detach().clone().flatten())
    return torch.cat(parts)


def episode_gradients_both(model, env):
    """Paired per-episode gradients from ONE stochastic trajectory.

    Returns (reinforce_grad, value_grad, team, total_reward). Both fields see
    the identical observation/action sequence and team label.
    """
    obs, info = env.reset()
    game = env.unwrapped.game
    team = 1 if (game.red_a_team and 0 in game.red_a_team) else 0
    total_r = 0.0
    g_rf = None
    g_val = None
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        d = model.policy.get_distribution(ot)
        a = d.get_actions().item()
        next_obs, r, done, trunc, info = env.step(a)
        total_r += r

        d2 = model.policy.get_distribution(ot)
        lp = d2.log_prob(torch.tensor([a]))
        model.policy.zero_grad()
        (-lp * r).backward()
        gv = _flat_policy_grad(model)
        g_rf = gv if g_rf is None else g_rf + gv

        with torch.enable_grad():
            v = model.policy.predict_values(ot)
            model.policy.zero_grad()
            v.backward()
        gv2 = _flat_policy_grad(model)
        g_val = gv2 if g_val is None else g_val + gv2

        obs = next_obs
    zero = torch.zeros(1)
    return (g_rf if g_rf is not None else zero), \
        (g_val if g_val is not None else zero), team, total_r


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
    ordered = sorted(g_by_team)
    groups = [torch.stack(g_by_team[t]) for t in ordered]
    weights = [n_t[t] / total_n for t in ordered]
    canonical = episode_decomposition(groups, weights)
    return {
        "mu2": canonical["E_shared"],
        "var_between": canonical["E_contrast"],
        "var_within": canonical["sigma2"],
        "var_total": canonical["E_contrast"] + canonical["sigma2"],
        "E_shared": canonical["E_shared"],
        "E_contrast": canonical["E_contrast"],
        "E_mixture": canonical["E_mixture"],
        "sigma2": canonical["sigma2"],
        "mean_noise_energy": canonical["mean_noise_energy"],
        "kappa_mix": canonical["kappa_mix"],
        "kappa_mean_surrogate": canonical["kappa_mean_surrogate"],
        "kappa_ep": canonical["kappa_ep"],
        "n_team": n_t,
    }


def measure(model, n_eps=N_EPS, base_seed=20260916):
    """Paired measurement: both fields share the same episode batch.

    The global torch/numpy RNG is reset per episode from a deterministic seed
    schedule, so the run is reproducible and each environment draw is shared
    by the two fields.
    """
    env = FiveTenKEnv(mode="obvious")
    rf_g, v_g, teams, rews = [], [], [], []
    for i in range(n_eps):
        torch.manual_seed(base_seed + i)
        np.random.seed(base_seed + i)
        g_rf, g_v, team, r = episode_gradients_both(model, env)
        rf_g.append(g_rf)
        v_g.append(g_v)
        teams.append(team)
        rews.append(r)
    env.close()
    d_rf = variance_decomp(rf_g, teams)
    d_v = variance_decomp(v_g, teams)
    return {
        "reinforce": d_rf,
        "value": d_v,
        "avg_reward": float(np.mean(rews)),
        "protocol": "paired_same_episode",
        "base_seed": base_seed,
    }


def main():
    out = {
        "_metadata": measurement_metadata(
            "reinforce/value per-episode field gradient",
            "empirical_team_frequency", "stochastic_full_info_eval_paired",
            "euclidean", "episode_noise_separate")
    }
    if os.path.exists(OUT):
        try:
            previous = json.load(open(OUT))
            for key, value in previous.items():
                if key != "_metadata":
                    out.setdefault(key, value)
        except (OSError, ValueError):
            pass
    print(f'{"lvl seed":>10} {"rf_mix":>7} {"val_mix":>7} {"rf_varw":>10} {"val_varw":>10} {"r":>7}')
    for level in LEVELS:
        level_key = str(level)
        for seed in SEEDS:
            key = f"s{seed}_paired"
            if key in out.get(level_key, {}):
                print(f"{level:.2f} s{seed}: cached paired result")
                continue
            fp = model_path(level, seed)
            if fp is None:
                continue
            model = MaskablePPO.load(fp, device="cpu")
            model.policy.eval()
            m = measure(model)
            out.setdefault(level_key, {})[key] = m
            rf = m["reinforce"].get("kappa_mix", float("nan"))
            vl = m["value"].get("kappa_mix", float("nan"))
            rfw = m["reinforce"].get("var_within", 0)
            valw = m["value"].get("var_within", 0)
            print(f'{level:>5.2f} s{seed:>2d}  {rf:>7.4f} {vl:>7.4f} '
                  f'{rfw:>10.2e} {valw:>10.2e} {m["avg_reward"]:>7.2f}')
            del model
            with open(OUT, "w") as f:
                json.dump(out, f, indent=2, default=float)
    print("\n=== SUMMARY (paired runs, mean over seeds) ===")
    for level in LEVELS:
        rows = [v for k, v in out.get(str(level), {}).items()
                if k.endswith("_paired")]
        if not rows:
            continue
        rf = [r["reinforce"]["kappa_mix"] for r in rows if "kappa_mix" in r["reinforce"]]
        vl = [r["value"]["kappa_mix"] for r in rows if "kappa_mix" in r["value"]]
        print(f"  p={level:.2f} (n={len(rf)}): reinforce kappa_mix={np.mean(rf):.4f}+-{np.std(rf):.4f}  "
              f"value kappa_mix={np.mean(vl):.4f}+-{np.std(vl):.4f}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("saved", OUT)


if __name__ == "__main__":
    main()
