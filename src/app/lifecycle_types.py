"""Shared lifecycle data types for application runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class GameState(Enum):
    """Application state machine."""

    MENU = auto()
    TRAINING = auto()
    PAUSED = auto()
    PLAY_MODE = auto()
    HUMAN_MODE = auto()


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""

    score: int
    reward: float
    steps: int
    epsilon: float
    bricks_hit: int
    won: bool
