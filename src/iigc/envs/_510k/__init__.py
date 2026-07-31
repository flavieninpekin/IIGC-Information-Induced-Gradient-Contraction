"""510K environment package (single-agent Gymnasium + multi-agent PettingZoo).

Merged from:
- upstream `flavieninpekin/510k_env` (rule engine, cleaned `env.py`, PettingZoo)
- the original paper's `env` package (masked-DQN wrapper, discrete SAC, MAPPO)
"""
from .card import Card, Rank, Suit, card_to_id
from .env import FiveTenKEnv, MAX_ACTIONS
from .game import Game, GameMode
from .patterns import Pattern, PatternType, detect_pattern, get_valid_plays, can_beat
from .scorer import Scorer

from .dqn_wrapper import (
    FiveTenKMaskedEnv,
    MaskedQNetwork,
    MaskedDQNPolicy,
    MASK_DIM,
)
from .discrete_sac import DiscreteSAC, Actor

try:
    from .pettingzoo import five_ten_k_v0, FiveTenKMultiEnv
    _HAS_PETTINGZOO = True
except ImportError:  # pettingzoo not installed
    _HAS_PETTINGZOO = False

__all__ = [
    'Card', 'Rank', 'Suit', 'card_to_id',
    'FiveTenKEnv', 'MAX_ACTIONS',
    'Game', 'GameMode',
    'Pattern', 'PatternType', 'detect_pattern', 'get_valid_plays', 'can_beat',
    'Scorer',
    'FiveTenKMaskedEnv', 'MaskedQNetwork', 'MaskedDQNPolicy', 'MASK_DIM',
    'DiscreteSAC', 'Actor',
]
if _HAS_PETTINGZOO:
    __all__ += ['five_ten_k_v0', 'FiveTenKMultiEnv']
