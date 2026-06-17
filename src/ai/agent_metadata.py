"""Serializable metadata types for agent checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class TrainingHistory:
    """
    Training history for dashboard visualization persistence.

    Stores arrays of metrics that allow the dashboard to restore
    charts and statistics when resuming training.
    """

    # Episode-level metrics (one per episode)
    scores: List[int]
    rewards: List[float]
    steps: List[int]
    epsilons: List[float]
    bricks: List[int]
    wins: List[bool]

    # Running averages (computed, not stored per-episode)
    losses: List[float]  # Recent losses for averaging
    q_values: List[float]  # Average Q-values per episode for chart

    # Dashboard state metrics (cumulative counters)
    exploration_actions: int = 0
    exploitation_actions: int = 0
    target_updates: int = 0
    best_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingHistory":
        # Handle backwards compatibility - older saves may not have all fields
        return cls(
            scores=data.get("scores", []),
            rewards=data.get("rewards", []),
            steps=data.get("steps", []),
            epsilons=data.get("epsilons", []),
            bricks=data.get("bricks", []),
            wins=data.get("wins", []),
            losses=data.get("losses", []),
            q_values=data.get("q_values", []),
            exploration_actions=data.get("exploration_actions", 0),
            exploitation_actions=data.get("exploitation_actions", 0),
            target_updates=data.get("target_updates", 0),
            best_score=data.get("best_score", 0),
        )

    @classmethod
    def empty(cls) -> "TrainingHistory":
        """Create empty history."""
        return cls(
            scores=[],
            rewards=[],
            steps=[],
            epsilons=[],
            bricks=[],
            wins=[],
            losses=[],
            q_values=[],
            exploration_actions=0,
            exploitation_actions=0,
            target_updates=0,
            best_score=0,
        )


@dataclass
class SaveMetadata:
    """Rich metadata stored with each model checkpoint."""

    # Timing
    timestamp: str
    save_reason: str  # 'best', 'periodic', 'manual', 'final', 'interrupted'
    total_training_time_seconds: float

    # Training progress
    episode: int
    total_steps: int
    epsilon: float

    # Performance metrics
    best_score: int
    avg_score_last_100: float
    avg_loss: float
    win_rate: float
    memory_buffer_size: int

    # Config snapshot
    learning_rate: float
    gamma: float
    batch_size: int
    hidden_layers: List[int]
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    use_dueling: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SaveMetadata":
        return cls(**data)
