"""Regression tests for the 510K action-mask contract.

The env must not silently replace actions: the pass bit of the mask is
``can_pass``, every mask-legal index is executable verbatim, and an illegal
action raises unless the env is explicitly built with
``allow_illegal_action=True``.
"""

import numpy as np
import pytest

from iigc.envs._510k.env import FiveTenKEnv


def _fresh(allow_illegal_action=False):
    env = FiveTenKEnv(mode="obvious", allow_illegal_action=allow_illegal_action)
    env.reset()
    return env


def test_pass_bit_matches_can_pass():
    env = _fresh()
    try:
        for _ in range(20):
            mask = env._get_action_mask()
            assert int(mask[0]) == int(env.game.can_pass(env.agent_id))
            action = int(np.flatnonzero(mask)[0])
            _, _, done, trunc, _ = env.step(action)
            if done or trunc:
                env.reset()
    finally:
        env.close()


def test_illegal_action_raises_by_default():
    env = _fresh()
    try:
        mask = env._get_action_mask()
        illegal = int(np.flatnonzero(mask == 0)[0])
        with pytest.raises(ValueError):
            env.step(illegal)
    finally:
        env.close()


def test_lenient_mode_replaces_illegal_action():
    env = _fresh(allow_illegal_action=True)
    try:
        mask = env._get_action_mask()
        illegal = int(np.flatnonzero(mask == 0)[0])
        can_pass = env.game.can_pass(env.agent_id)
        valid = env.game.get_valid_actions(env.agent_id)
        n_before = len(env.game.actions_log)
        env.step(illegal)
        entry = env.game.actions_log[n_before]
        assert entry["player"] == env.agent_id
        if entry["action"] == "pass":
            assert can_pass
        else:
            assert entry["action"] == "play"
            assert entry["cards"] in [p.cards for p in valid]
    finally:
        env.close()


def test_legal_actions_are_executed_verbatim():
    env = _fresh()
    rng = np.random.default_rng(0)
    try:
        for _ in range(30):
            mask = env._get_action_mask()
            action = int(rng.choice(np.flatnonzero(mask)))
            valid = env.game.get_valid_actions(env.agent_id)
            n_before = len(env.game.actions_log)
            _, _, done, trunc, _ = env.step(action)
            entry = env.game.actions_log[n_before]
            assert entry["player"] == env.agent_id
            if action == 0:
                assert entry["action"] == "pass"
            else:
                assert entry["action"] == "play"
                assert entry["cards"] == valid[action - 1].cards
            if done or trunc:
                env.reset()
    finally:
        env.close()
