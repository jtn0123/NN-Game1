"""Small contracts for opt-in neural-network training extensions.

The core DQN loss should stay simple. Extra route, demo, correction, or
architecture experiments can report weighted loss contributions through these
objects so Agent.learn does not grow one custom branch per idea.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class AuxiliaryMetric:
    """A scalar metric emitted by an auxiliary loss.

    ``history`` is the deque attribute on the agent that receives the value,
    for example ``demo_action_losses``.
    """

    history: str
    value: float


@dataclass(frozen=True)
class AuxiliaryLossContribution:
    """One optional differentiable loss term added to the DQN loss."""

    name: str
    loss: torch.Tensor
    weight: float
    metrics: tuple[AuxiliaryMetric, ...] = ()

    def weighted_loss(self) -> torch.Tensor:
        return self.loss * float(self.weight)


class AuxiliaryLossProvider(Protocol):
    """Protocol for future mixins/modules that contribute extra training losses."""

    def auxiliary_loss_contributions(
        self, states: torch.Tensor
    ) -> tuple[AuxiliaryLossContribution, ...]:
        """Return active auxiliary losses for the current learner batch."""
