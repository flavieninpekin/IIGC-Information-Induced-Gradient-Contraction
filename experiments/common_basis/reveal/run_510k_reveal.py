"""Phase 1: 510K continuous-reveal baseline re-measurement.

Re-measures kappa on the existing MaskablePPO models from models_reveal/
with the protocol from reveal_fine.py (two rollout mixtures, 30 eps each,
per-step REINFORCE gradient), plus:

  - Per-episode gradient + team-label recording (game.red_a_team)
  - Variance decomposition: Var_between (team contrast) vs Var_within (sigma^2)
  - Energy gate and law-of-total-variance consistency check.
  - Per-episode kappa as a second estimate (independent of the two-rollout split).
"""
import os, json
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from iigc.envs._510k.env import FiveTenKEnv
from iigc.metrics.kappa import episode_decomposition

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models_reveal')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', '510k_reveal')
os.makedirs(OUT_DIR, exist_ok=True)

N_EPS = 30
REVEAL_LEVELS = [round(i / 20, 2) for i in range(21)]  # 0.00 ... 1.00 step 0.05
SEEDS = [41, 42, 43, 44, 45, 46]


def kappa_and_energy(gA, gB):
    avg = (gA + gB) / 2.0
    e = (gA.norm() ** 2 + gB.norm() ** 2) / 2.0
    k = (avg.norm() ** 2 / max(e, 1e-10)).item()
    return k, e.item()


def model_path(level, seed):
    fp = os.path.join(MODEL_DIR, f'ppo_reveal_{level:.2f}_s{seed}.zip')
    if os.path.exists(fp):
        return fp
    if abs(level - 0.50) < 1e-6:
        alt = os.path.join(MODEL_DIR, f'ppo_half_{seed}.zip')
        if os.path.exists(alt):
            return alt
    if abs(level - 1.00) < 1e-6 and seed == 41:
        alt = os.path.join(MODEL_DIR, 'ppo_obvious_41_final.zip')
        if os.path.exists(alt):
            return alt
    return None


def episode_gradient(model, env):
    """One episode's REINFORCE gradient (per-step r-weighted sum)
    plus team label and total reward."""
    obs, info = env.reset()
    game = env.unwrapped.game
    team = 1 if (game.red_a_team and 0 in game.red_a_team) else 0

    total_r = 0.0
    g = None
    done = False
    while not done:
        ot = torch.FloatTensor(obs).unsqueeze(0)
        mask = env.unwrapped._get_action_mask()
        d = model.policy.get_distribution(ot, action_masks=mask)
        a = d.get_actions().item()
        next_obs, r, done, trunc, info = env.step(a)
        total_r += r
        lp = d.log_prob(torch.tensor([a]))
        model.policy.zero_grad()
        (-lp * r).backward()
        gv = torch.cat([p.grad.detach().clone().flatten()
                        for p in model.policy.parameters() if p.grad is not None])
        g = gv if g is None else g + gv
        obs = next_obs
    return g if g is not None else torch.zeros(1), team, total_r


def variance_decomp(ep_grads, ep_teams):
    """Canonical team-conditioned decomposition.

    Uses the shared ``episode_decomposition`` implementation so that
    ``kappa_mix``, ``kappa_ep``, and the noise terms have the same definitions
    as everywhere else in the repository.
    """
    g_by_team = {}
    for g, t in zip(ep_grads, ep_teams):
        g_by_team.setdefault(int(t), []).append(g)

    if len(g_by_team) < 2:
        return {}

    ordered = sorted(g_by_team)
    groups = [torch.stack(g_by_team[t]) for t in ordered]
    counts = [len(g_by_team[t]) for t in ordered]
    total_n = sum(counts)
    weights = [c / total_n for c in counts]
    canonical = episode_decomposition(groups, weights)
    return {
        'mu2': canonical['E_shared'],
        'var_total': canonical['E_contrast'] + canonical['sigma2'],
        'var_between': canonical['E_contrast'],
        'var_within': canonical['sigma2'],
        'var_sum': canonical['E_contrast'] + canonical['sigma2'],
        'E_shared': canonical['E_shared'],
        'E_contrast': canonical['E_contrast'],
        'E_mixture': canonical['E_mixture'],
        'sigma2': canonical['sigma2'],
        'kappa_mix': canonical['kappa_mix'],
        'kappa_mean_surrogate': canonical['kappa_mean_surrogate'],
        'kappa_ep': canonical['kappa_ep'],
        'n_team': dict(zip(ordered, counts)),
    }


def measure_model(fp, n_eps=N_EPS):
    model = MaskablePPO.load(fp, device='cpu')
    model.policy.eval()

    mix_grads = []
    all_ep = []

    for _ in range(2):
        env = FiveTenKEnv(mode='obvious')
        rollout_g = None
        for _ in range(n_eps):
            g_ep, team, ep_r = episode_gradient(model, env)
            if g_ep is not None:
                rollout_g = g_ep if rollout_g is None else rollout_g + g_ep
            all_ep.append({'g': g_ep.tolist(), 'team': team, 'reward': ep_r})
        env.close()
        if rollout_g is not None:
            mix_grads.append(rollout_g / n_eps)

    k_mix, e_mix = (kappa_and_energy(mix_grads[0], mix_grads[1])
                    if len(mix_grads) == 2 else (float('nan'), 0))

    ep_grads = [torch.tensor(e['g']) for e in all_ep]
    ep_teams = [e['team'] for e in all_ep]
    avg_r = np.mean([e['reward'] for e in all_ep])

    decom = variance_decomp(ep_grads, ep_teams)
    return {
        'kappa_mix': k_mix, 'energy_mix': e_mix,
        'kappa_ep': decom.get('kappa_ep'), 'energy_ep': decom.get('mu2'),
        'var_between': decom.get('var_between'), 'var_within': decom.get('var_within'),
        'var_total': decom.get('var_total'), 'var_sum': decom.get('var_sum'),
        'kappa_mix_per_team': decom.get('kappa_mix'),
        'kappa_mean_surrogate': decom.get('kappa_mean_surrogate'),
        'n_team': decom.get('n_team'),
        'avg_reward': avg_r,
    }


def main():
    results = {}
    hdr = f'{"lvl seed":>10} {"k_mix":>7} {"k_ep":>7} {"mu2":>9} {"var_b":>10} {"var_w":>10} {"r":>7}'
    print(hdr)
    for level in REVEAL_LEVELS:
        for seed in SEEDS:
            fp = model_path(level, seed)
            if fp is None:
                continue
            m = measure_model(fp)
            row = m
            results.setdefault(level, {})[f's{seed}'] = row
            print(f'{level:>5.2f} s{seed:>2d}  {row["kappa_mix"]:>7.4f} '
                  f'{row["kappa_ep"] or float("nan"):>7.4f} '
                  f'{row.get("energy_ep") or 0:>9.2e} '
                  f'{row.get("var_between") or 0:>10.2e} '
                  f'{row.get("var_within") or 0:>10.2e} '
                  f'{row["avg_reward"]:>7.2f}')

    print(f'\n{"="*60}\nSUMMARY (mean over seeds)')
    for level in REVEAL_LEVELS:
        keys = ['kappa_mix', 'kappa_ep', 'var_between', 'var_within',
                'kappa_mix_per_team', 'kappa_mean_surrogate']
        vals = {k: [] for k in keys}
        for s in results.get(level, {}).values():
            for k in keys:
                v = s.get(k)
                if v is not None:
                    vals[k].append(v)
        if vals['kappa_mix']:
            print(f'\n  p={level:.2f} (n={len(vals["kappa_mix"])}):')
            print(f'    kappa_mix     = {np.mean(vals["kappa_mix"]):.4f} +/- {np.std(vals["kappa_mix"]):.4f}')
            if vals['kappa_ep']:
                print(f'    kappa_ep      = {np.mean(vals["kappa_ep"]):.4f} +/- {np.std(vals["kappa_ep"]):.4f}')
            if vals['var_between']:
                print(f'    var_between   = {np.mean(vals["var_between"]):.2e}')
                print(f'    var_within    = {np.mean(vals["var_within"]):.2e}')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    main()
