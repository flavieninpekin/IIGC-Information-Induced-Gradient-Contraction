"""Experiment 2: interpolation spectrum on a common measurement basis.

Same actor parameters theta, same rollouts, same relations — only the
objective that defines the gradient changes.

Fields (mode-seeking -> mean-seeking):
  - reinforce : return-weighted grad log pi(a_taken)        [hard mode-seeking]
  - awr       : advantage-weighted grad log pi(a_taken)     [intermediate]
  - softq     : SAC actor loss grad                          [soft, alpha-tuned]
  - expq      : expected-Q grad sum_a pi(a) Q(a)             [mean-seeking]
  - gibbs sweep: pi_tau ~ softmax(logits/tau); grad of E[Q]; tau -> inf is
    mean-seeking (uniform over actions), tau -> 0 is mode-seeking (argmax).

Prediction: kappa grows as the field becomes more mean-seeking (averages over
actions) and contracts as it becomes mode-seeking (commits to the best action
per relation assignment). The SINGLE vs DYNAMIC gap should shrink with tau.
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F

from iigc.envs._510k.dqn_wrapper import FiveTenKMaskedEnv, MASK_DIM, MAX_ACTIONS
from iigc.envs._510k.discrete_sac import DiscreteSAC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_DIR = os.path.join(ROOT, 'data', 'models', '510k_sac')
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'common_basis_interp')
os.makedirs(OUT_DIR, exist_ok=True)

MODES = ['single', 'dynamic']
SEEDS = [41, 42, 43, 44]
N_EPS = 30
SEED_A, SEED_B = 1000, 2000
TAUS = [0.2, 0.5, 1.0, 2.0, 5.0]


def kappa_and_energy(gA, gB):
    avg = (gA + gB) / 2.0
    e = (torch.norm(gA) ** 2 + torch.norm(gB) ** 2) / 2.0
    k = (torch.norm(avg) ** 2 / max(e, 1e-10)).item()
    return k, e.item()


def rollout_episodes(model, env, n_eps=N_EPS, base_seed=0):
    episodes = []
    for ep in range(n_eps):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        traj = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                act, _ = model.actor.get_action(obs_t, deterministic=True)
            action = act.item()
            next_obs, reward, done, trunc, info = env.step(action)
            traj.append((obs, action, reward, next_obs, done))
            obs = next_obs
        episodes.append(traj)
    return episodes


def _flatten(episodes):
    return [t for traj in episodes for t in traj]


def _probs_logits_q(model, obs_batch):
    obs = torch.FloatTensor(np.array(obs_batch)).to(model.device)
    logits = model.actor(obs)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        q = torch.min(model.critic1(obs), model.critic2(obs))
    return obs, probs, log_probs, q


def compute_grad(model, loss):
    model.actor.zero_grad()
    loss.backward()
    gv = [p.grad.detach().clone().flatten() for p in model.actor.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


# --- objectives (all w.r.t. actor params) ---

def loss_reinforce(model, episodes):
    total = torch.zeros(1, device=model.device); count = 0
    for traj in episodes:
        G = sum(t[2] for t in traj)
        _, _, log_probs, _ = _probs_logits_q(model, [t[0] for t in traj])
        act = torch.tensor([t[1] for t in traj]).to(model.device)
        lp = log_probs[range(len(act)), act]
        total = total + (-lp * G).sum()
        count += len(traj)
    return total / max(count, 1)


def loss_awr(model, episodes, tau=1.0):
    trans = _flatten(episodes)
    obs_b = torch.FloatTensor(np.array([t[0] for t in trans])).to(model.device)
    act_b = torch.tensor([t[1] for t in trans]).to(model.device)
    _, probs, log_probs, q = _probs_logits_q(model, [t[0] for t in trans])
    v = (probs * q).sum(dim=-1)
    adv = q[range(len(act_b)), act_b] - v
    w = torch.exp(adv / tau)
    lp = log_probs[range(len(act_b)), act_b]
    return (-lp * w).mean()


def loss_softq(model, episodes):
    trans = _flatten(episodes)
    _, probs, log_probs, q = _probs_logits_q(model, [t[0] for t in trans])
    alpha = model.log_alpha.exp().detach()
    return (probs * (alpha * log_probs - q)).sum(dim=-1).mean()


def loss_expq(model, episodes):
    trans = _flatten(episodes)
    _, probs, _, q = _probs_logits_q(model, [t[0] for t in trans])
    return -(probs * q).sum(dim=-1).mean()


def loss_gibbs_expq(model, episodes, tau):
    trans = _flatten(episodes)
    obs = torch.FloatTensor(np.array([t[0] for t in trans])).to(model.device)
    logits = model.actor(obs)
    pi = F.softmax(logits / tau, dim=-1)
    with torch.no_grad():
        q = torch.min(model.critic1(obs), model.critic2(obs))
    return -(pi * q).sum(dim=-1).mean()


FIELDS = ['reinforce', 'awr', 'softq', 'expq']


def field_loss(model, episodes, name):
    if name == 'reinforce':
        return loss_reinforce(model, episodes)
    if name == 'awr':
        return loss_awr(model, episodes)
    if name == 'softq':
        return loss_softq(model, episodes)
    if name == 'expq':
        return loss_expq(model, episodes)
    raise ValueError(name)


def run():
    results = {}
    for mode in MODES:
        results[mode] = {}
        for seed in SEEDS:
            fp = os.path.join(MODEL_DIR, f'510k_sac_{mode}_seed{seed}.pt')
            if not os.path.exists(fp):
                continue
            sac = DiscreteSAC(112 + MASK_DIM, MASK_DIM, MAX_ACTIONS, device='cpu')
            sac.load(fp)
            sac.actor.eval()

            env_a = FiveTenKMaskedEnv(mode=mode)
            eps_a = rollout_episodes(sac, env_a, base_seed=SEED_A); env_a.close()
            env_b = FiveTenKMaskedEnv(mode=mode)
            eps_b = rollout_episodes(sac, env_b, base_seed=SEED_B); env_b.close()

            ra = np.mean([sum(t[2] for t in traj) for traj in eps_a])
            rb = np.mean([sum(t[2] for t in traj) for traj in eps_b])

            row = {'rA': ra, 'rB': rb, 'fields': {}, 'gibbs': {}}
            for name in FIELDS:
                gA = compute_grad(sac, field_loss(sac, eps_a, name))
                gB = compute_grad(sac, field_loss(sac, eps_b, name))
                k, e = kappa_and_energy(gA, gB)
                row['fields'][name] = {'kappa': k, 'energy': e}
                print(f'{mode} s{seed} {name:10}: κ={k:.4f} (E={e:.1e})')

            for tau in TAUS:
                gA = compute_grad(sac, loss_gibbs_expq(sac, eps_a, tau))
                gB = compute_grad(sac, loss_gibbs_expq(sac, eps_b, tau))
                k, e = kappa_and_energy(gA, gB)
                row['gibbs'][str(tau)] = {'kappa': k, 'energy': e}
                print(f'{mode} s{seed} gibbs τ={tau:<4}: κ={k:.4f} (E={e:.1e})')

            results[mode][f'seed{seed}'] = row
            print()

    print(f'\n{"="*70}\nSUMMARY  (mean over seeds)')
    for mode in MODES:
        print(f'\n--- {mode} ---')
        for name in FIELDS:
            vals = [results[mode][s]['fields'][name]['kappa'] for s in results[mode]]
            if vals:
                print(f'  {name:10}: κ={np.mean(vals):.4f} ± {np.std(vals):.4f}')
        print('  gibbs τ sweep:')
        for tau in TAUS:
            vals = [results[mode][s]['gibbs'][str(tau)]['kappa'] for s in results[mode]]
            if vals:
                print(f'    τ={tau:<4}: κ={np.mean(vals):.4f} ± {np.std(vals):.4f}')

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "results.json")}')


if __name__ == '__main__':
    run()
