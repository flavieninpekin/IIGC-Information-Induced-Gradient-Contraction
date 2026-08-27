"""O4 performance-consequence experiment (S2-P1, preregistered design).

Four objective arms trained FROM SCRATCH on a common basis (same trunk size,
same rollout protocol, same step budget, same seeds) — only the objective
differs:

  reinforce : -log pi(a) * G_hat                (G_hat = episode return / n_steps)
  awr       : -log pi(a) * exp(clip(G_hat - V(s)) / tau),  V differentiable
              (matches fields.py loss_awr semantics: autograd through baseline)
  softq     : actor grad of sum_a pi(a)(alpha*log pi(a) - Q(s,a)), Q detached;
              Q trained by TD(0) on the SAME buffer   (matches fields.py loss_softq)
  td        : eps-greedy argmax Q; Q trained by TD(0)   (value-field reference)

Env: AdaptiveHiddenMatchingEnv (modes static/switch x hidden/revealed).
See notes/o4_performance_design.md for the preregistered predictions P1-P4
and falsifiers F1-F3.

Usage:
  python run_toy_adaptive.py --smoke           # tiny pipeline check
  python run_toy_adaptive.py                   # full grid (multiprocess)
  python run_toy_adaptive.py --aggregate       # summarize results dir

Output: data/kappa/o4_adaptive/<mode>_<arm>_s<seed>.json + aggregate.json
"""
import argparse
import json
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iigc.envs._toy import AdaptiveHiddenMatchingEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(ROOT, 'data', 'kappa', 'o4_adaptive')

ARMS = ['reinforce', 'awr', 'softq', 'td']
MODES = ['static_hidden', 'switch_hidden', 'switch_revealed']

N_STEPS = 40
T_FLIP = 20
TOTAL_STEPS = 150_000
EPS_PER_ITER = 8            # episodes collected per iteration (8*40 = 320 steps)
BUFFER_EPISODES = 64        # rolling window (near-on-policy for PG arms)
UPDATES_PER_ITER = 4
BATCH_EPISODES = 32
LR = 3e-3
GAMMA = 0.95
TAU_AWR = 1.0
ALPHA_SOFTQ = 0.2
EPS_MIX = 0.1               # prob of uniform action during collection
EPS_GREEDY_TD = 0.15        # td-arm exploration
EVAL_EVERY = 10_000
KAPPA_EVERY = 20_000
N_EVAL = 200
DEPLOY_STEPS = 30_000       # static_hidden -> static_revealed fine-tune
DEPLOY_EVAL_EVERY = 6_000
DEPLOY_TARGET = 32.0        # adaptation threshold (return) for speed metric


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(mode):
    env_mode = {'static_hidden': 'static',
                'switch_hidden': 'switch'}.get(mode, mode)
    return AdaptiveHiddenMatchingEnv(mode=env_mode, n_steps=N_STEPS,
                                     t_flip=T_FLIP)


class Trunk(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.trunk = Trunk(obs_dim)
        self.head = nn.Linear(64, 2)

    def logits(self, obs):
        return self.head(self.trunk(torch.as_tensor(obs, dtype=torch.float32)))

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.logits(obs))


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.trunk = Trunk(obs_dim)
        self.head = nn.Linear(64, 1)

    def forward(self, obs):
        return self.head(self.trunk(torch.as_tensor(obs, dtype=torch.float32))).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.trunk = Trunk(obs_dim)
        self.head = nn.Linear(64, 2)

    def forward(self, obs):
        return self.head(self.trunk(torch.as_tensor(obs, dtype=torch.float32)))


# ----------------------------------------------------------------------------
# rollout / eval
# ----------------------------------------------------------------------------

def act(arm, nets, obs, rng, explore):
    """explore=True adds mixing/greedy exploration; False = eval-stochastic."""
    if arm == 'td':
        if explore and rng.random() < EPS_GREEDY_TD:
            return int(rng.integers(0, 2))
        with torch.no_grad():
            q = nets['q'](obs[None]).numpy()[0]
        return int(np.argmax(q))
    probs = nets['pi'].dist(obs[None]).probs.detach().numpy()[0]
    if explore and rng.random() < EPS_MIX:
        return int(rng.integers(0, 2))
    return int(rng.choice(2, p=probs))


def run_episode(env, arm, nets, rng, explore=True):
    obs, _ = env.reset()
    traj = []
    total = 0.0
    for _ in range(N_STEPS):
        a = act(arm, nets, obs, rng, explore)
        nobs, r, done, _, _ = env.step(a)
        traj.append((obs, a, r, nobs, done))
        total += r
        obs = nobs
    return traj, total


def evaluate(env, arm, nets, rng, n_eps=N_EVAL):
    rets, post, pre = [], [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        total, post_r, pre_r = 0.0, [], []
        for t in range(N_STEPS):
            a = act(arm, nets, obs, rng, explore=False)
            nobs, r, done, _, _ = env.step(a)
            total += r
            if env.switching and 20 <= t < 28:
                post_r.append(r)
            if env.switching and 11 <= t < 20:
                pre_r.append(r)
            obs = nobs
        rets.append(total)
        post.append(np.mean(post_r) if post_r else np.nan)
        pre.append(np.mean(pre_r) if pre_r else np.nan)
    out = {'return': float(np.mean(rets)), 'return_std': float(np.std(rets))}
    if env.switching:
        out['post_flip_reward'] = float(np.nanmean(post))
        out['pre_flip_reward'] = float(np.nanmean(pre))
    return out


# ----------------------------------------------------------------------------
# objectives
# ----------------------------------------------------------------------------

def _batch(buffer, rng, n_ep=BATCH_EPISODES):
    idx = rng.choice(len(buffer), size=min(n_ep, len(buffer)), replace=False)
    trans = [t for i in idx for t in buffer[i]]
    obs = torch.tensor(np.array([t[0] for t in trans]), dtype=torch.float32)
    acts = torch.tensor([t[1] for t in trans], dtype=torch.long)
    rews = torch.tensor([t[2] for t in trans], dtype=torch.float32)
    nobs = torch.tensor(np.array([t[3] for t in trans]), dtype=torch.float32)
    dones = torch.tensor([t[4] for t in trans], dtype=torch.float32)
    ghat = torch.tensor([t[5] for t in trans], dtype=torch.float32)
    return obs, acts, rews, nobs, dones, ghat


def td_step(qnet, opt_q, batch):
    obs, acts, rews, nobs, dones, _ = batch
    with torch.no_grad():
        target = rews + GAMMA * (1 - dones) * qnet(nobs).max(dim=-1).values
    loss = F.mse_loss(qnet(obs).gather(1, acts[:, None]).squeeze(1), target)
    opt_q.zero_grad()
    loss.backward()
    opt_q.step()
    return float(loss.item())


def reinforce_step(pi, opt_pi, batch):
    obs, acts, _, _, _, ghat = batch
    logp = pi.dist(obs).log_prob(acts)
    loss = -(logp * ghat).mean()
    opt_pi.zero_grad()
    loss.backward()
    opt_pi.step()


def awr_step(pi, vnet, opt_pi, opt_v, batch):
    obs, acts, _, _, _, ghat = batch
    v = vnet(obs)
    opt_v.zero_grad()
    v_loss = F.mse_loss(v, ghat)
    v_loss.backward()
    opt_v.step()

    adv = torch.clamp(ghat - v.detach(), -4.0, 4.0)
    logp = pi.dist(obs).log_prob(acts)
    # NOTE: fields.py differentiates awr's weight through the *critic-in-graph*
    # baseline; here V lives in a separate head, so the actor-side graph equals
    # the canonical sampled estimator with the baseline evaluated at theta.
    loss = -(logp * torch.exp(adv / TAU_AWR)).mean()
    opt_pi.zero_grad()
    loss.backward()
    opt_pi.step()


def softq_step(pi, qnet, opt_pi, batch):
    obs, acts, _, _, _, _ = batch
    with torch.no_grad():
        q = qnet(obs)
    probs = pi.dist(obs).probs
    logp_all = torch.log_softmax(pi.logits(obs), dim=-1)
    loss = (probs * (ALPHA_SOFTQ * logp_all - q)).sum(dim=-1).mean()
    opt_pi.zero_grad()
    loss.backward()
    opt_pi.step()


# ----------------------------------------------------------------------------
# kappa on the training basis (forced initial partner)
# ----------------------------------------------------------------------------

def flat_grad(model, loss):
    model.zero_grad()
    loss.backward()
    gs = [p.grad.detach().flatten() for p in model.parameters()
          if p.grad is not None]
    return torch.cat(gs) if gs else torch.zeros(1)


def measure_kappa(mode, arm, nets, master_seed):
    """Per-relation gradients under the arm's own objective; returns the
    energy decomposition (no per-episode sigma2 here — deterministic batching)."""
    out = {}
    for rel in (0, 1):
        env = make_env(mode)
        env.set_partner(rel)
        rng = np.random.default_rng(10_000 + master_seed)
        eps = [run_episode(env, arm, nets, rng)[0] for _ in range(40)]
        trans = [t for ep in eps for t in trans_with_g(ep)]
        batch = _to_batch(trans)
        if arm == 'reinforce':
            logp = nets['pi'].dist(batch[0]).log_prob(batch[1])
            loss = -(logp * batch[5]).mean()
            g = flat_grad(nets['pi'], loss)
        elif arm == 'awr':
            v = nets['v'](batch[0])
            adv = torch.clamp(batch[5] - v, -4.0, 4.0)
            logp = nets['pi'].dist(batch[0]).log_prob(batch[1])
            loss = -(logp * torch.exp(adv / TAU_AWR)).mean()
            g = flat_grad(nets['pi'], loss)
        elif arm == 'softq':
            with torch.no_grad():
                q = nets['q'](batch[0])
            probs = nets['pi'].dist(batch[0]).probs
            logp_all = torch.log_softmax(nets['pi'].logits(batch[0]), dim=-1)
            loss = (probs * (ALPHA_SOFTQ * logp_all - q)).sum(dim=-1).mean()
            g = flat_grad(nets['pi'], loss)
        else:  # td
            obs, acts, rews, nobs, dones, _ = batch
            with torch.no_grad():
                target = rews + GAMMA * (1 - dones) * nets['q'](nobs).max(-1).values
            loss = F.mse_loss(nets['q'](obs).gather(1, acts[:, None]).squeeze(1),
                              target)
            g = flat_grad(nets['q'], loss)
        out[rel] = g.numpy()
        env.close()
    ga, gb = out[0], out[1]
    m, d = (ga + gb) / 2.0, (ga - gb) / 2.0
    es, ec = float(m @ m), float(d @ d)
    k = es / max(es + ec, 1e-12)
    return {'kappa': k, 'e_shared': es, 'e_contrast': ec}


def trans_with_g(traj):
    g = sum(t[2] for t in traj) / N_STEPS
    return [(t[0], t[1], t[2], t[3], t[4], g) for t in traj]


def _to_batch(trans):
    obs = torch.tensor(np.array([t[0] for t in trans]), dtype=torch.float32)
    acts = torch.tensor([t[1] for t in trans], dtype=torch.long)
    rews = torch.tensor([t[2] for t in trans], dtype=torch.float32)
    nobs = torch.tensor(np.array([t[3] for t in trans]), dtype=torch.float32)
    dones = torch.tensor([t[4] for t in trans], dtype=torch.float32)
    ghat = torch.tensor([t[5] for t in trans], dtype=torch.float32)
    return obs, acts, rews, nobs, dones, ghat


# ----------------------------------------------------------------------------
# main training loop for one config
# ----------------------------------------------------------------------------

def _updates(arm, nets, opts, buffer, rng):
    for _ in range(UPDATES_PER_ITER):
        batch = _batch(buffer, rng)
        if arm == 'reinforce':
            reinforce_step(nets['pi'], opts['pi'], batch)
        elif arm == 'awr':
            awr_step(nets['pi'], nets['v'], opts['pi'], opts['v'], batch)
        elif arm == 'softq':
            td_step(nets['q'], opts['q'], batch)
            softq_step(nets['pi'], nets['q'], opts['pi'], batch)
        else:
            td_step(nets['q'], opts['q'], batch)


def collect(buffer, env, arm, nets, rng, n_ep=EPS_PER_ITER):
    for _ in range(n_ep):
        traj, _ret = run_episode(env, arm, nets, rng, explore=True)
        buffer.append(trans_with_g(traj))


def deploy_adaptation(mode, arm, nets, opts, buffer, rng, seed):
    """Fine-tune the static_hidden-trained nets on static_revealed and record
    the adaptation curve (identity-critical deployment, prereg §8)."""
    deploy_env = make_env('static_revealed')
    curve = []
    steps = 0
    while steps < DEPLOY_STEPS:
        collect(buffer, deploy_env, arm, nets, rng)
        steps += EPS_PER_ITER * N_STEPS
        _updates(arm, nets, opts, buffer, rng)
        if (steps % DEPLOY_EVAL_EVERY < EPS_PER_ITER * N_STEPS
                or steps >= DEPLOY_STEPS):
            ev = evaluate(make_env('static_revealed'), arm, nets,
                          np.random.default_rng(77_000 + seed), n_eps=100)
            curve.append({'steps': steps, **ev})
    rets = [c['return'] for c in curve]
    t_target = next((c['steps'] for c in curve if c['return'] >= DEPLOY_TARGET),
                    None)
    out = {'curve': curve, 'auc': float(np.mean(rets)),
           'steps_to_32': t_target,
           'early_window_return': float(np.mean(rets[:2])),
           'final_return': rets[-1] if rets else None}
    deploy_env.close()
    return out


def train_config(mode, arm, seed, total_steps, smoke=False):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(mode)
    obs_dim = env.observation_space.shape[0]

    nets = {'pi': PolicyNet(obs_dim)} if arm != 'td' else {}
    if arm == 'awr':
        nets['v'] = ValueNet(obs_dim)
    if arm in ('softq', 'td'):
        nets['q'] = QNet(obs_dim)
    opts = {k: torch.optim.Adam(v.parameters(), lr=LR) for k, v in nets.items()}

    buffer = deque(maxlen=BUFFER_EPISODES)
    history = {'steps': [], 'eval': [], 'kappa': []}
    steps = 0
    it = 0
    while steps < total_steps:
        for _ in range(EPS_PER_ITER):
            traj, _ret = run_episode(env, arm, nets, rng, explore=True)
            buffer.append(trans_with_g(traj))
            steps += N_STEPS
        it += 1

        for _ in range(UPDATES_PER_ITER):
            batch = _batch(buffer, rng)
            if arm == 'reinforce':
                reinforce_step(nets['pi'], opts['pi'], batch)
            elif arm == 'awr':
                awr_step(nets['pi'], nets['v'], opts['pi'], opts['v'], batch)
            elif arm == 'softq':
                td_step(nets['q'], opts['q'], batch)
                softq_step(nets['pi'], nets['q'], opts['pi'], batch)
            else:
                td_step(nets['q'], opts['q'], batch)

        if steps % EVAL_EVERY < EPS_PER_ITER * N_STEPS or steps >= total_steps:
            ev = evaluate(make_env(mode), arm, nets,
                          np.random.default_rng(99_000 + seed))
            ev['steps'] = steps
            history['eval'].append(ev)
            history['steps'].append(steps)

        if (not smoke) and (steps % KAPPA_EVERY < EPS_PER_ITER * N_STEPS
                            or steps >= total_steps):
            kap = measure_kappa(mode, arm, nets, seed)
            kap['steps'] = steps
            history['kappa'].append(kap)

    adapt = None
    if mode == 'static_hidden' and not smoke:
        adapt = deploy_adaptation(mode, arm, nets, opts, buffer, rng, seed)

    final_eval = evaluate(make_env(mode), arm, nets,
                          np.random.default_rng(99_000 + seed))
    result = {
        'mode': mode, 'arm': arm, 'seed': seed, 'total_steps': total_steps,
        'final_eval': final_eval, 'history': history, 'adapt': adapt,
    }
    env.close()
    return result


def run_one(args_tuple):
    mode, arm, seed, total_steps, smoke = args_tuple
    tag = f'{mode}_{arm}_s{seed}'
    out_path = os.path.join(OUT_DIR, tag + '.json')
    if os.path.exists(out_path):
        return tag + ' (cached)'
    res = train_config(mode, arm, seed, total_steps, smoke)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1, default=float)
    return f"{tag}: return={res['final_eval']['return']:.2f}"


def aggregate(results_dir=OUT_DIR):
    rows = []
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith('.json') or fn == 'aggregate.json':
            continue
        with open(os.path.join(results_dir, fn)) as f:
            d = json.load(f)
        row = {'mode': d['mode'], 'arm': d['arm'], 'seed': d['seed'],
               **{f'eval_{k}': v for k, v in d['final_eval'].items()}}
        if d['history']['kappa']:
            row['final_kappa'] = d['history']['kappa'][-1]['kappa']
        if d.get('adapt'):
            row['adapt_auc'] = d['adapt']['auc']
            row['adapt_early'] = d['adapt']['early_window_return']
            row['adapt_final'] = d['adapt']['final_return']
            row['adapt_steps_to_32'] = d['adapt']['steps_to_32']
        rows.append(row)
    agg = {}
    for row in rows:
        key = (row['mode'], row['arm'])
        agg.setdefault(key, []).append(row)
    summary = {}
    for (mode, arm), rs in sorted(agg.items()):
        def ms(field):
            vals = [r[field] for r in rs
                    if r.get(field) is not None and r[field] == r[field]]
            return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None
        summary[f'{mode}|{arm}'] = {
            'n_seeds': len(rs),
            'return': ms('eval_return'),
            'post_flip_reward': ms('eval_post_flip_reward'),
            'pre_flip_reward': ms('eval_pre_flip_reward'),
            'final_kappa': ms('final_kappa'),
            'adapt_auc': ms('adapt_auc'),
            'adapt_early': ms('adapt_early'),
            'adapt_final': ms('adapt_final'),
            'adapt_steps_to_32': ms('adapt_steps_to_32'),
        }
    out = {'n_runs': len(rows), 'summary': summary}
    with open(os.path.join(results_dir, 'aggregate.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    print(f"{'mode|arm':34} {'n':>2} {'return':>16} {'post20-28':>16} "
          f"{'kappa':>16} {'adapt_auc':>16} {'adapt@12k':>16}")
    for k, v in summary.items():
        def fmt(t):
            return '-' if t is None else f'{t[0]:7.2f}±{t[1]:5.2f}({t[2]})'
        print(f'{k:34} {v["n_seeds"]:2d} {fmt(v["return"]):>16} '
              f'{fmt(v["post_flip_reward"]):>16} {fmt(v["final_kappa"]):>16} '
              f'{fmt(v["adapt_auc"]):>16} {fmt(v["adapt_early"]):>16}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--aggregate', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.aggregate:
        aggregate()
        return

    seeds = [41, 42] if args.smoke else list(range(41, 49))
    total = 24_000 if args.smoke else TOTAL_STEPS
    jobs = [(m, a, s, total, args.smoke)
            for m in MODES for a in ARMS for s in seeds]

    if args.smoke:
        for j in jobs:
            print(run_one(j))
        aggregate()
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            done += 1
            print(f'[{done}/{len(jobs)}] {fut.result()}', flush=True)
    aggregate()


if __name__ == '__main__':
    main()
