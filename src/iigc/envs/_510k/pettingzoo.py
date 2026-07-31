from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .game import Game, GameMode
from .scorer import Scorer
from .obs_utils import obs_for_player, action_mask_for_player
from .env import MAX_ACTIONS

try:
    from pettingzoo.utils.env import AECEnv
    from pettingzoo.utils import wrappers as pz_wrappers
    _HAS_PETTINGZOO = True
except ImportError:
    _HAS_PETTINGZOO = False
    AECEnv = object


class FiveTenKMultiEnv(AECEnv):
    """510K multi-agent environment (PettingZoo AEC API).

    All 4 players are controllable independently. Turn order follows the
    game's natural sequence (the player holding 3D starts).

    Parameters
    ----------
    mode : str
        ``'single'``, ``'static'``, ``'dynamic'``, or ``'obvious'``.
    num_players : int
        3 or 4. Default 4.
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "five_ten_k_v0"}

    def __init__(self, mode: str = "single", num_players: int = 4,
                 render_mode: Optional[str] = None):
        super().__init__()
        if not _HAS_PETTINGZOO:
            raise ImportError(
                "pettingzoo is required. Install with: pip install env_510k[pettingzoo]"
            )

        self._mode_str = mode
        self.render_mode = render_mode
        game_mode = GameMode(mode) if mode != "3p" else GameMode.SINGLE
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        n_cards = 54 if num_players == 3 else 52
        obs_dim = n_cards * 2 + 1 + 4 + 1 + 1 + 1
        if game_mode == GameMode.OBVIOUS:
            obs_dim += 4

        self._observation_spaces = {
            a: spaces.Dict({
                "observation": spaces.Box(0, 1, (obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (MAX_ACTIONS,), dtype=np.int8),
            }) for a in self.possible_agents
        }
        self._action_spaces = {
            a: spaces.Discrete(MAX_ACTIONS) for a in self.possible_agents
        }

        self.game: Optional[Game] = None
        self._clear_rewards()

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if not _HAS_PETTINGZOO:
            return

        self.game = Game(
            mode=GameMode(self._mode_str) if self._mode_str != "3p" else GameMode.SINGLE,
            num_players=len(self.possible_agents),
        )
        self.agents = self.possible_agents[:]
        self.agent_selection = f"player_{self.game.current_player}"

        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def step(self, action):
        if not _HAS_PETTINGZOO:
            return

        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        pid = int(agent.split("_")[1])
        self._cumulative_rewards[agent] = 0

        if action == 0:
            self.game.pass_turn(pid)
        else:
            patterns = self.game.get_valid_actions(pid)
            if 1 <= action <= len(patterns):
                self.game.play_cards(pid, patterns[action - 1].cards)
            else:
                if patterns:
                    self.game.play_cards(pid, patterns[0].cards)
                else:
                    self.game.pass_turn(pid)

        if self.game.is_over:
            scorer = Scorer(self.game)
            all_rewards = scorer.compute_rewards()
            for a in self.agents:
                pid = int(a.split("_")[1])
                self.rewards[a] = all_rewards.get(pid, 0.0)
            self.terminations = {a: True for a in self.agents}

        self._accumulate_rewards()
        # Turn advances dynamically via Game.play_cards/pass_turn
        self.agent_selection = f"player_{self.game.current_player}"

    def observe(self, agent):
        if not self.game:
            return self.observation_space(agent).low.copy()
        pid = int(agent.split("_")[1])
        return {
            "observation": obs_for_player(self.game, pid).astype(np.float32),
            "action_mask": action_mask_for_player(self.game, pid).astype(np.int8),
        }

    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            if not self.game:
                return "Game not started"
            lines = [f"Turn: P{self.game.current_player}"]
            for i, p in enumerate(self.game.players):
                marker = " <<<" if i == self.game.current_player else ""
                status = " FINISHED" if p.finished else ""
                lines.append(f"P{i}: {len(p.hand)} cards{marker}{status}")
            if self.game.last_trick:
                cards = " ".join(str(c) for c in self.game.last_trick.cards)
                lines.append(f"Last play (P{self.game.last_trick.player}): {cards}")
            return "\n".join(lines)
        return ""

    def close(self):
        pass


def env(**kwargs):
    """PettingZoo environment factory for 510K.

    Returns
    -------
    AECEnv
        Wrapped with ``TerminateIllegalWrapper``,
        ``AssertOutOfBoundsWrapper`` and ``OrderEnforcingWrapper``.
    """
    if not _HAS_PETTINGZOO:
        raise ImportError(
            "pettingzoo is required. Install with: pip install env_510k[pettingzoo]"
        )
    env = FiveTenKMultiEnv(**kwargs)
    env = pz_wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = pz_wrappers.AssertOutOfBoundsWrapper(env)
    env = pz_wrappers.OrderEnforcingWrapper(env)
    return env


five_ten_k_v0 = env
