"""Environment contract tests: observation spaces and reset/info semantics."""

import numpy as np
import pytest

from iigc.envs._overcooked.overcooked_v3_env import OvercookedV3Env
from iigc.envs._510k.env import FiveTenKEnv


@pytest.mark.parametrize("mode", ["single", "obvious"])
def test_510k_observation_space_contains_observations(mode):
    env = FiveTenKEnv(mode=mode)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(30):
            mask = env._get_action_mask()
            action = int(np.flatnonzero(mask)[0])
            obs, _, done, trunc, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if done or trunc:
                obs, _ = env.reset(seed=0)
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_overcooked_observation_space_contains_observations(mode):
    env = OvercookedV3Env(mode=mode, horizon=20)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for _ in range(10):
            obs, _, done, trunc, _ = env.step(0)
            assert env.observation_space.contains(obs)
            if done or trunc:
                break
    finally:
        env.close()


def test_overcooked_reset_seed_controls_partner():
    env = OvercookedV3Env(mode="dynamic", horizon=20)
    try:
        env.reset(seed=7)
        first = env._partner_idx
        env.reset(seed=7)
        assert env._partner_idx == first
    finally:
        env.close()


def test_overcooked_partner_type_is_pre_switch():
    env = OvercookedV3Env(mode="dynamic", switch_interval=1, horizon=20)
    try:
        env.reset(seed=0)
        partner = env.ptype
        _, _, _, _, info = env.step(0)
        assert info["partner_type"] == partner
        assert env.ptype != partner
    finally:
        env.close()
