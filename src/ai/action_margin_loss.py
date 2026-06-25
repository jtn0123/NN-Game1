"""Reusable supervised action-margin losses for DQN extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class QNetwork(Protocol):
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action."""


@dataclass(frozen=True)
class ActionMarginLossResult:
    """Loss values from one sampled supervised action-margin batch."""

    loss: torch.Tensor
    accuracy: float
    conservative_loss: torch.Tensor | None = None


def sample_action_margin_loss(
    policy_net: QNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int,
    margin: float,
    conservative_temperature: float | None = None,
) -> ActionMarginLossResult | None:
    """Sample states/actions and compute DQfD-style action-margin loss.

    The helper is deliberately config-free so demo, close-zone, correction, and future
    auxiliary providers can share the same math without depending on ``Agent`` internals.
    """
    count = int(actions.shape[0])
    if count <= 0 or batch_size <= 0:
        return None

    batch_size = min(batch_size, count)
    idx = torch.randint(0, count, (batch_size,), device=states.device)
    batch_states = states.index_select(0, idx)
    batch_actions = actions.index_select(0, idx)
    q_values = policy_net(batch_states)
    action_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
    q_with_margin = q_values + float(margin)
    q_with_margin.scatter_(1, batch_actions.unsqueeze(1), action_q.unsqueeze(1))
    loss = (q_with_margin.max(dim=1).values - action_q).mean()

    conservative_loss: torch.Tensor | None = None
    if conservative_temperature is not None:
        temperature = max(float(conservative_temperature), 1e-6)
        conservative_loss = (
            temperature * torch.logsumexp(q_values / temperature, dim=1) - action_q
        ).mean()

    with torch.no_grad():
        accuracy = float((q_values.argmax(dim=1) == batch_actions).float().mean().item())
    return ActionMarginLossResult(loss=loss, accuracy=accuracy, conservative_loss=conservative_loss)
