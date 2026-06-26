# ruff: noqa: F401,F403,F405,I001
"""Frozen-teacher policy anchoring for Crystal Caves fine-tunes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.ai.extension_contracts import AuxiliaryLossContribution, AuxiliaryMetric


@dataclass
class PolicyAnchorSummary:
    """Small serializable summary for anchored fine-tune artifacts."""

    enabled: bool
    weight: float
    temperature: float
    min_target_distance_norm: float | None = None
    target_distance_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "temperature": self.temperature,
            "min_target_distance_norm": self.min_target_distance_norm,
            "target_distance_index": self.target_distance_index,
        }


class FrozenPolicyAnchorProvider:
    """Distill current Q-action preferences toward a frozen teacher policy."""

    def __init__(
        self,
        *,
        policy_net: torch.nn.Module,
        teacher_net: torch.nn.Module,
        weight: float,
        temperature: float,
        min_target_distance_norm: float | None = None,
        target_distance_index: int | None = None,
    ) -> None:
        if weight < 0:
            raise ValueError("policy anchor weight must be non-negative")
        if temperature <= 0:
            raise ValueError("policy anchor temperature must be positive")
        if min_target_distance_norm is not None and min_target_distance_norm < 0:
            raise ValueError("minimum target distance must be non-negative")
        self.policy_net = policy_net
        self.teacher_net = teacher_net
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.min_target_distance_norm = min_target_distance_norm
        self.target_distance_index = target_distance_index

    def auxiliary_loss_contributions(
        self, states: torch.Tensor
    ) -> tuple[AuxiliaryLossContribution, ...]:
        if self.weight <= 0:
            return ()

        anchor_states = self._anchor_states(states)
        if anchor_states.shape[0] == 0:
            return ()

        current_q = self.policy_net(anchor_states)
        with torch.no_grad():
            teacher_q = self.teacher_net(anchor_states)

        temperature = self.temperature
        teacher_probs = F.softmax(teacher_q / temperature, dim=1)
        current_log_probs = F.log_softmax(current_q / temperature, dim=1)
        loss = F.kl_div(current_log_probs, teacher_probs, reduction="batchmean") * (
            temperature * temperature
        )
        with torch.no_grad():
            accuracy = float(
                (current_q.argmax(dim=1) == teacher_q.argmax(dim=1)).float().mean().item()
            )

        return (
            AuxiliaryLossContribution(
                name="policy_anchor",
                loss=loss,
                weight=self.weight,
                metrics=(
                    AuxiliaryMetric("policy_anchor_losses", float(loss.detach().item())),
                    AuxiliaryMetric("policy_anchor_accuracies", accuracy),
                ),
            ),
        )

    def _anchor_states(self, states: torch.Tensor) -> torch.Tensor:
        threshold = self.min_target_distance_norm
        index = self.target_distance_index
        if threshold is None or threshold <= 0 or index is None:
            return states
        if index < 0 or index >= states.shape[1]:
            return states
        distances = states[:, index]
        mask = torch.isfinite(distances) & (distances > float(threshold))
        return states[mask]


def install_policy_anchor_provider(
    agent: Any,
    *,
    weight: float,
    temperature: float,
    min_target_distance_norm: float | None = None,
) -> dict[str, Any]:
    """Register a frozen copy of the current policy as an auxiliary anchor."""

    if weight < 0:
        raise ValueError("policy anchor weight must be non-negative")
    if temperature <= 0:
        raise ValueError("policy anchor temperature must be positive")
    if min_target_distance_norm is not None and min_target_distance_norm < 0:
        raise ValueError("minimum target distance must be non-negative")
    target_distance_index = _target_distance_index(agent)
    if weight <= 0:
        return PolicyAnchorSummary(
            enabled=False,
            weight=0.0,
            temperature=temperature,
            min_target_distance_norm=min_target_distance_norm,
            target_distance_index=target_distance_index,
        ).as_dict()

    teacher_net = copy.deepcopy(agent.policy_net)
    teacher_net.to(agent.device)
    teacher_net.eval()
    for parameter in teacher_net.parameters():
        parameter.requires_grad_(False)

    provider = FrozenPolicyAnchorProvider(
        policy_net=agent.policy_net,
        teacher_net=teacher_net,
        weight=weight,
        temperature=temperature,
        min_target_distance_norm=min_target_distance_norm,
        target_distance_index=target_distance_index,
    )
    agent.register_auxiliary_loss_provider(provider)
    return PolicyAnchorSummary(
        enabled=True,
        weight=weight,
        temperature=temperature,
        min_target_distance_norm=min_target_distance_norm,
        target_distance_index=target_distance_index,
    ).as_dict()


def _target_distance_index(agent: Any) -> int | None:
    layout = getattr(getattr(agent, "config", None), "STATE_LAYOUT", None)
    if not isinstance(layout, dict):
        return None
    window_rows, window_cols = layout.get("window", (0, 0))
    gmap_rows, gmap_cols = layout.get("gmap", (0, 0))
    meta_size = int(layout.get("meta", 0) or 0)
    if meta_size < 18:
        return None
    meta_start = int(window_rows) * int(window_cols) + int(gmap_rows) * int(gmap_cols)
    return meta_start + 17
