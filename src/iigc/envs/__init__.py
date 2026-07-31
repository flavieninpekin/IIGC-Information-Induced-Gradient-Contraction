"""Environments for IIGC. Each environment lives in its own sub-package.

- ``_510k``    : 510K card game (SINGLE/STATIC/DYNAMIC/OBVIOUS) + masked-DQN,
                 discrete-SAC, MAPPO wrappers, and a PettingZoo multi-agent API.
- ``_toy``     : Hidden Matching bandit (HIDDEN/REVEALED).
- ``_partner`` : Hidden Partner with safe fallback (HIDDEN/REVEALED).

Add new environments (e.g. ``_overcooked``) as separate sub-packages.
"""
from ._510k import (
    FiveTenKEnv,
    MAX_ACTIONS,
    FiveTenKMaskedEnv,
    MaskedQNetwork,
    MaskedDQNPolicy,
    DiscreteSAC,
    Actor,
    Game,
    GameMode,
    Scorer,
)
from ._toy import HiddenMatchingEnv
from ._partner import HiddenPartnerEnv

__all__ = [
    'FiveTenKEnv', 'MAX_ACTIONS',
    'FiveTenKMaskedEnv', 'MaskedQNetwork', 'MaskedDQNPolicy',
    'DiscreteSAC', 'Actor',
    'Game', 'GameMode', 'Scorer',
    'HiddenMatchingEnv',
    'HiddenPartnerEnv',
]
