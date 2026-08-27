"""
Analytic Toy: Hidden Matching Environment.

A 2-action contextual bandit where the true partner (B or C) is hidden.
Action 0 = cooperate with B, Action 1 = cooperate with C.
When hidden, gradients from the two partners cancel → no learning.
When revealed, agent learns to match.

This is the simplest environment exhibiting Information-Induced Gradient Contraction.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HiddenMatchingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, revealed=False, n_steps=10):
        super().__init__()
        self.revealed = revealed
        self.n_steps = n_steps

        if revealed:
            # 2-dim observation: [is_partner_B, is_partner_C]
            self.observation_space = spaces.Box(0, 1, (2,), dtype=np.float32)
        else:
            # 1-dim constant observation (no partner info)
            self.observation_space = spaces.Box(0, 1, (1,), dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self._forced_partner = -1
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._forced_partner >= 0:
            self.partner = self._forced_partner
        else:
            self.partner = np.random.randint(0, 2)
        self.step_count = 0
        return self._obs(), {}

    def _obs(self):
        if self.revealed:
            return np.array([1 - self.partner, self.partner], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        # Action 0 → cooperate with B, Action 1 → cooperate with C
        reward = 1.0 if int(action) == self.partner else -1.0
        self.step_count += 1
        done = self.step_count >= self.n_steps
        return self._obs(), reward, done, False, {}

    def set_partner(self, partner):
        """Force partner for gradient measurement."""
        self._forced_partner = partner
        self.partner = partner
        self.step_count = 0


class AdaptiveHiddenMatchingEnv(gym.Env):
    """O4 environment: hidden matching with a mid-episode partner switch.

    Relation-adaptive task for the performance-consequence experiment
    (notes/o4_performance_design.md). The partner flips at `t_flip`
    (static mode: never). The relation itself is never in the observation,
    but is inferable from the reward history (last K (action, reward) pairs
    are stacked into the obs), so a memoryless-on-raw-obs policy cannot
    track it while a history-conditioned policy can.

    Analytic references: oracle return = n_steps; chance = 0; best fixed
    policy = 0 (any positive return demonstrates tracking).

    Modes:
      switch / static  ×  hidden / revealed.
    All modes share ONE observation layout (constant + history + 2-dim
    partner slot); hidden modes zero the partner slot so a single network
    shape transfers from hidden training to revealed deployment.
    """
    K = 8

    def __init__(self, mode='switch', n_steps=40, t_flip=20):
        assert mode in ('switch', 'static', 'switch_revealed', 'static_revealed')
        super().__init__()
        self.mode = mode
        self.n_steps = n_steps
        self.t_flip = t_flip
        self.revealed = mode.endswith('_revealed')
        self.switching = mode.startswith('switch')
        obs_dim = 1 + 3 * self.K + 2
        self.observation_space = spaces.Box(-1, 1, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._forced_partner = -1
        self.history = []
        self.reset()

    @property
    def _partner_now(self):
        if not self.switching or self.step_count < self.t_flip:
            return self.partner
        return 1 - self.partner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._forced_partner >= 0:
            self.partner = self._forced_partner
        else:
            self.partner = np.random.randint(0, 2)
        self.step_count = 0
        self.history = [(0, 0.0)] * self.K
        return self._obs(), {}

    def _obs(self):
        feats = [1.0]
        for act, rew in self.history:
            feats.extend([1.0 if act == 0 else -1.0, 1.0 if act == 1 else -1.0,
                          float(rew)])
        if self.revealed:
            p = self._partner_now
            feats.extend([1.0 - p, p])
        else:
            feats.extend([0.0, 0.0])
        return np.array(feats, dtype=np.float32)

    def step(self, action):
        reward = 1.0 if int(action) == self._partner_now else -1.0
        self.step_count += 1
        self.history.pop(0)
        self.history.append((int(action), reward))
        done = self.step_count >= self.n_steps
        return self._obs(), reward, done, False, {}

    def set_partner(self, partner):
        """Force the INITIAL partner for gradient measurement."""
        self._forced_partner = partner
        self.partner = partner
        self.step_count = 0
