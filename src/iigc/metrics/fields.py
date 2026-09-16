"""Gradient fields and kappa measurement protocols.

Each field is a gradient of a specified scalar objective with respect to the
actor parameters. The fields are measurement definitions, not a universal
mode-seeking/mean-seeking ordering. In particular, ``awr`` deliberately keeps
the policy-dependent baseline in the autograd graph; detached AWR is a
different field and should not share this name in a paper.

``softmaxq`` uses a Q-dependent, elementwise weight. ``gibbs_expq`` is kept
separate because it differentiates E[Q] under a temperature-scaled policy and
is not the same objective as ``softmaxq``.
"""
import numpy as np
import torch
import torch.nn.functional as F

from iigc.metrics.kappa import condition_decomposition

FIELDS = ['reinforce', 'awr', 'softq', 'expq', 'softmaxq']


def kappa_and_energy(gA, gB):
    """Compatibility wrapper for two equally weighted condition gradients."""
    return kappa_from_gradients([gA, gB])


def kappa_from_gradients(gradients, weights=None):
    """Compute bounded mixture-reference kappa from condition gradients.

    For condition gradients ``g_i`` and normalized nonnegative weights ``p_i``
    this returns ``||sum_i p_i g_i||^2 / sum_i p_i ||g_i||^2``.
    Episode noise must be handled before calling this function; it is not
    folded into the structural denominator.
    """
    if not gradients:
        raise ValueError("gradients must not be empty")

    result = condition_decomposition(gradients, weights)
    return result["kappa_mix"], result["E_mixture"]


def rollout_episodes(model, env, n_eps=30, base_seed=0, stochastic=False):
    """Collect episodes of (obs, action, reward, next_obs, done).

    stochastic=True samples actions from pi (pi-weighted expectation, the
    correct protocol for kappa); False takes argmax (deterministic). NOTE:
    deterministic rollouts make relation gradients parallel on the mirror
    bandit and measure a weight ratio, NOT gradient retention (see
    notes/rollout_protocol_artifact.md).
    """
    episodes = []
    for ep in range(n_eps):
        obs, info = env.reset(seed=base_seed + ep)
        done = False
        traj = []
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            probs = F.softmax(model.actor(obs_t), dim=-1)
            if stochastic:
                act = torch.distributions.Categorical(probs).sample().item()
            else:
                act = probs.argmax(dim=-1).item()
            next_obs, reward, done, trunc, info = env.step(act)
            traj.append((obs, act, reward, next_obs, done))
            obs = next_obs
        episodes.append(traj)
    return episodes


def _flatten(episodes):
    return [t for traj in episodes for t in traj]


def _probs_logits_q(model, obs_batch, q_fn=None):
    obs = torch.FloatTensor(np.array(obs_batch)).to(model.device)
    logits = model.actor(obs)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        if q_fn is not None:
            q = q_fn(obs)
        else:
            q = torch.min(model.critic1(obs), model.critic2(obs))
    return obs, probs, log_probs, q


def compute_grad(model, loss):
    model.actor.zero_grad()
    loss.backward()
    gv = [p.grad.detach().clone().flatten() for p in model.actor.parameters() if p.grad is not None]
    return torch.cat(gv) if gv else torch.zeros(1)


def loss_reinforce(model, episodes):
    total = torch.zeros(1, device=model.device); count = 0
    for traj in episodes:
        G = sum(t[2] for t in traj)
        obs = torch.FloatTensor(np.array([t[0] for t in traj])).to(model.device)
        log_probs = F.log_softmax(model.actor(obs), dim=-1)
        act = torch.tensor([t[1] for t in traj]).to(model.device)
        lp = log_probs[range(len(act)), act]
        total = total + (-lp * G).sum()
        count += len(traj)
    return total / max(count, 1)


def loss_awr(model, episodes, tau=1.0, q_fn=None):
    """AWR-like field with a differentiable policy-dependent baseline."""
    trans = _flatten(episodes)
    act_b = torch.tensor([t[1] for t in trans]).to(model.device)
    _, probs, log_probs, q = _probs_logits_q(model, [t[0] for t in trans], q_fn)
    v = (probs * q).sum(dim=-1)
    adv = q[range(len(act_b)), act_b] - v
    w = torch.exp(adv / tau)
    lp = log_probs[range(len(act_b)), act_b]
    return (-lp * w).mean()


def loss_softq(model, episodes, q_fn=None):
    """SAC actor objective with detached Q values."""
    trans = _flatten(episodes)
    _, probs, log_probs, q = _probs_logits_q(model, [t[0] for t in trans], q_fn)
    alpha = model.log_alpha.exp().detach()
    return (probs * (alpha * log_probs - q)).sum(dim=-1).mean()


def loss_expq(model, episodes, q_fn=None):
    trans = _flatten(episodes)
    _, probs, _, q = _probs_logits_q(model, [t[0] for t in trans], q_fn)
    return -(probs * q).sum(dim=-1).mean()


def loss_softmaxq(model, episodes, tau=1.0, q_fn=None):
    """Framework gibbs dial: weight w = softmax(Q/tau)."""
    trans = _flatten(episodes)
    act_b = torch.tensor([t[1] for t in trans]).to(model.device)
    _, _, log_probs, q = _probs_logits_q(model, [t[0] for t in trans], q_fn)
    w = torch.softmax(q / tau, dim=-1)
    lp = log_probs[range(len(act_b)), act_b]
    return (-lp * w[range(len(act_b)), act_b]).mean()


def loss_gibbs_expq(model, episodes, tau, q_fn=None):
    trans = _flatten(episodes)
    obs = torch.FloatTensor(np.array([t[0] for t in trans])).to(model.device)
    logits = model.actor(obs)
    pi = F.softmax(logits / tau, dim=-1)
    with torch.no_grad():
        if q_fn is not None:
            q = q_fn(obs)
        else:
            q = torch.min(model.critic1(obs), model.critic2(obs))
    return -(pi * q).sum(dim=-1).mean()


def field_loss(model, episodes, name, q_fn=None):
    if name == 'reinforce':
        return loss_reinforce(model, episodes)
    if name == 'awr':
        return loss_awr(model, episodes, q_fn=q_fn)
    if name == 'softq':
        return loss_softq(model, episodes, q_fn=q_fn)
    if name == 'expq':
        return loss_expq(model, episodes, q_fn=q_fn)
    if name == 'softmaxq':
        return loss_softmaxq(model, episodes, q_fn=q_fn)
    raise ValueError(name)
