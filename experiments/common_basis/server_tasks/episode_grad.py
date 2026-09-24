"""Per-episode gradient collectors for the measurement scripts.

Vendored from the old server-task engine (``engine._ep_grad`` /
``engine._ep_grad_rew``) so the measurement scripts are self-contained and do
not depend on an external code checkout. Callers seed ``torch`` and ``numpy``
before each episode (see ``run_switch_kappa.py``).
"""
import torch


def ep_grad(model, env):
    """Return the per-episode flat REINFORCE gradient of the policy."""
    return ep_grad_rew(model, env)[0]


def ep_grad_rew(model, env):
    """Run one episode with stochastic policy actions; return (grad, reward)."""
    obs, info = env.reset()
    grad = None
    total_reward = 0.0
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        dist = model.policy.get_distribution(obs_t)
        action = dist.get_actions().item()
        next_obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        dist_logp = model.policy.get_distribution(torch.FloatTensor(obs).unsqueeze(0))
        logp = dist_logp.log_prob(torch.tensor([action]))
        model.policy.zero_grad()
        (-logp * reward).backward()
        grads = torch.cat([p.grad.detach().clone().flatten()
                           for p in model.policy.parameters() if p.grad is not None])
        grad = grads if grad is None else grad + grads
        obs = next_obs
    return (grad if grad is not None else torch.zeros(1)), total_reward
