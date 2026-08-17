"""Overcooked V3 + memory: stack the last k observations so the policy can
infer the hidden partner role from behavior over time (an intervention that is
NOT "reveal the role" = static).

Dynamic mode: role hidden (96-dim obs), switches every switch_interval steps.
With memory the policy sees obs[t-k:t] and can infer the current role.
"""
import numpy as np
from collections import deque

from .overcooked_v3_env import OvercookedV3Env, PARTNER_TYPES


class OvercookedMemoryEnv(OvercookedV3Env):
    """Dynamic-mode env with a k-frame observation history stack."""

    def __init__(self, layout_name='cramped_room', mode='dynamic',
                 horizon=400, switch_interval=30, memory=4, seed=None):
        self.memory_k = memory
        self._history = deque(maxlen=memory)
        super().__init__(layout_name=layout_name, mode=mode,
                         horizon=horizon, switch_interval=switch_interval,
                         seed=seed)

    def _get_obs(self):
        base = super()._get_obs()
        self._history.append(base)
        # pad history up to k with the first frame
        while len(self._history) < self.memory_k:
            self._history.append(base)
        return np.concatenate(list(self._history)).astype(np.float32)

    def reset(self, seed=None, options=None):
        self._history.clear()
        return super().reset(seed=seed, options=options)


def memory_obs_dim(memory=4):
    """96-dim dynamic obs x memory frames."""
    return 96 * memory
