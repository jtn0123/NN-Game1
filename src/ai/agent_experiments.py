"""Optional experiment-only losses for the DQN agent.

These hooks support Crystal Caves route diagnostics and B3-series experiments.
They are kept separate from ``agent.py`` so the core DQN implementation remains
readable while still exposing the same methods to experiment runners.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .action_margin_loss import sample_action_margin_loss
from .extension_contracts import AuxiliaryLossContribution, AuxiliaryMetric


class AgentExperimentMixin:
    """Mixin for opt-in auxiliary/demo losses used by experiment runners."""

    def register_auxiliary_loss_provider(self: Any, provider: Any) -> None:
        """Register an external auxiliary-loss provider for experiment code."""
        if not callable(getattr(provider, "auxiliary_loss_contributions", None)):
            raise TypeError("provider must define auxiliary_loss_contributions(states)")
        self._extra_auxiliary_loss_providers.append(provider)

    def _auxiliary_loss_contributions(
        self: Any, states: torch.Tensor
    ) -> tuple[AuxiliaryLossContribution, ...]:
        """Collect optional Crystal Caves loss terms in a uniform shape."""
        contributions: list[AuxiliaryLossContribution] = []

        route_aux = self._route_auxiliary_loss(states)
        if route_aux is not None:
            route_aux_loss, route_aux_accuracy = route_aux
            contributions.append(
                AuxiliaryLossContribution(
                    name="route_aux",
                    loss=route_aux_loss,
                    weight=float(getattr(self.config, "CRYSTAL_CAVES_ROUTE_AUX_WEIGHT", 0.05)),
                    metrics=(
                        AuxiliaryMetric(
                            "route_aux_losses",
                            float(route_aux_loss.detach().item()),
                        ),
                        AuxiliaryMetric("route_aux_accuracies", route_aux_accuracy),
                    ),
                )
            )

        demo_action = self._demo_action_supervised_loss()
        if demo_action is not None:
            demo_action_loss, demo_action_accuracy, demo_conservative_loss = demo_action
            contributions.append(
                AuxiliaryLossContribution(
                    name="demo_action",
                    loss=demo_action_loss,
                    weight=float(getattr(self.config, "CRYSTAL_CAVES_DEMO_ACTION_WEIGHT", 0.05)),
                    metrics=(
                        AuxiliaryMetric(
                            "demo_action_losses",
                            float(demo_action_loss.detach().item()),
                        ),
                        AuxiliaryMetric("demo_action_accuracies", demo_action_accuracy),
                    ),
                )
            )
            if demo_conservative_loss is not None:
                contributions.append(
                    AuxiliaryLossContribution(
                        name="demo_conservative",
                        loss=demo_conservative_loss,
                        weight=float(
                            getattr(
                                self.config,
                                "CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT",
                                0.0,
                            )
                        ),
                        metrics=(
                            AuxiliaryMetric(
                                "demo_conservative_losses",
                                float(demo_conservative_loss.detach().item()),
                            ),
                        ),
                    )
                )

        close_zone_demo_action = self._close_zone_demo_action_supervised_loss()
        if close_zone_demo_action is not None:
            close_zone_loss, close_zone_demo_action_accuracy = close_zone_demo_action
            contributions.append(
                AuxiliaryLossContribution(
                    name="close_zone_demo_action",
                    loss=close_zone_loss,
                    weight=float(
                        getattr(
                            self.config,
                            "CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT",
                            0.03,
                        )
                    ),
                    metrics=(
                        AuxiliaryMetric(
                            "close_zone_demo_action_losses",
                            float(close_zone_loss.detach().item()),
                        ),
                        AuxiliaryMetric(
                            "close_zone_demo_action_accuracies",
                            close_zone_demo_action_accuracy,
                        ),
                    ),
                )
            )

        correction_action = self._correction_action_supervised_loss()
        if correction_action is not None:
            correction_loss, correction_accuracy = correction_action
            contributions.append(
                AuxiliaryLossContribution(
                    name="correction_action",
                    loss=correction_loss,
                    weight=float(
                        getattr(
                            self.config,
                            "CRYSTAL_CAVES_CORRECTION_ACTION_WEIGHT",
                            0.02,
                        )
                    ),
                    metrics=(
                        AuxiliaryMetric(
                            "correction_action_losses",
                            float(correction_loss.detach().item()),
                        ),
                        AuxiliaryMetric("correction_action_accuracies", correction_accuracy),
                    ),
                )
            )

        contact_action = self._contact_action_supervised_loss()
        if contact_action is not None:
            contact_loss, contact_accuracy = contact_action
            contributions.append(
                AuxiliaryLossContribution(
                    name="contact_action",
                    loss=contact_loss,
                    weight=float(
                        getattr(
                            self.config,
                            "CRYSTAL_CAVES_CONTACT_ACTION_WEIGHT",
                            0.02,
                        )
                    ),
                    metrics=(
                        AuxiliaryMetric(
                            "contact_action_losses",
                            float(contact_loss.detach().item()),
                        ),
                        AuxiliaryMetric("contact_action_accuracies", contact_accuracy),
                    ),
                )
            )

        for provider in tuple(getattr(self, "_extra_auxiliary_loss_providers", ())):
            contributions.extend(provider.auxiliary_loss_contributions(states))

        return tuple(contributions)

    def _prepare_demo_action_tensors(
        self: Any, states: np.ndarray, actions: np.ndarray, *, label: str
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int64)
        if states_np.ndim != 2:
            raise ValueError(f"{label} states must be a 2D array")
        if states_np.shape[1] != self.state_size:
            raise ValueError(
                f"{label} state size mismatch: expected {self.state_size}, got {states_np.shape[1]}"
            )
        if actions_np.ndim != 1 or len(actions_np) != len(states_np):
            raise ValueError(f"{label} actions must be a 1D array matching demo states")
        if len(actions_np) and (
            int(actions_np.min()) < 0 or int(actions_np.max()) >= self.action_size
        ):
            raise ValueError(f"{label} actions contain values outside the action space")

        if len(states_np) == 0:
            return None, None, 0

        states_tensor = torch.as_tensor(states_np.copy(), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions_np.copy(), dtype=torch.int64, device=self.device)
        return states_tensor, actions_tensor, int(len(actions_np))

    def set_demo_action_dataset(
        self: Any, states: np.ndarray, actions: np.ndarray
    ) -> dict[str, int]:
        """Install demonstration states/actions for optional supervised action loss."""
        states_tensor, actions_tensor, count = self._prepare_demo_action_tensors(
            states, actions, label="demo"
        )
        self._demo_action_states = states_tensor
        self._demo_action_actions = actions_tensor
        return {"demo_action_transitions": count}

    def set_close_zone_demo_action_dataset(
        self: Any, states: np.ndarray, actions: np.ndarray
    ) -> dict[str, int]:
        """Install close-zone states/actions for optional final-contact action loss."""
        states_tensor, actions_tensor, count = self._prepare_demo_action_tensors(
            states, actions, label="close-zone demo"
        )
        self._close_zone_demo_action_states = states_tensor
        self._close_zone_demo_action_actions = actions_tensor
        return {"close_zone_demo_action_transitions": count}

    def set_correction_action_dataset(
        self: Any, states: np.ndarray, actions: np.ndarray
    ) -> dict[str, int]:
        """Install policy-visited correction states/actions for optional action loss."""
        states_tensor, actions_tensor, count = self._prepare_demo_action_tensors(
            states, actions, label="correction"
        )
        self._correction_action_states = states_tensor
        self._correction_action_actions = actions_tensor
        return {"correction_action_transitions": count}

    def set_contact_action_dataset(
        self: Any, states: np.ndarray, actions: np.ndarray
    ) -> dict[str, int]:
        """Install close-zone states/actions for an opt-in detached contact head."""
        states_tensor, actions_tensor, count = self._prepare_demo_action_tensors(
            states, actions, label="contact action"
        )
        self._contact_action_states = states_tensor
        self._contact_action_actions = actions_tensor
        return {"contact_action_transitions": count}

    def _demo_action_supervised_loss(
        self: Any,
    ) -> Optional[Tuple[torch.Tensor, float, torch.Tensor | None]]:
        """Optional DQfD/CQL-style losses on successful demonstration states."""
        if not getattr(self.config, "CRYSTAL_CAVES_DEMO_ACTION_LOSS", False):
            return None
        return self._demo_action_loss_for_dataset(
            self._demo_action_states,
            self._demo_action_actions,
            batch_size=int(getattr(self.config, "CRYSTAL_CAVES_DEMO_ACTION_BATCH_SIZE", 64)),
            include_conservative=True,
        )

    def _close_zone_demo_action_supervised_loss(
        self: Any,
    ) -> Optional[Tuple[torch.Tensor, float]]:
        """Optional low-weight action-margin loss on close-zone demonstration states."""
        if not getattr(self.config, "CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS", False):
            return None
        result = self._demo_action_loss_for_dataset(
            self._close_zone_demo_action_states,
            self._close_zone_demo_action_actions,
            batch_size=int(
                getattr(self.config, "CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE", 64)
            ),
            include_conservative=False,
        )
        if result is None:
            return None
        demo_loss, accuracy, _conservative_loss = result
        return demo_loss, accuracy

    def _correction_action_supervised_loss(
        self: Any,
    ) -> Optional[Tuple[torch.Tensor, float]]:
        """Optional low-weight action-margin loss on policy-visited correction states."""
        if not getattr(self.config, "CRYSTAL_CAVES_CORRECTION_ACTION_LOSS", False):
            return None
        result = self._demo_action_loss_for_dataset(
            self._correction_action_states,
            self._correction_action_actions,
            batch_size=int(getattr(self.config, "CRYSTAL_CAVES_CORRECTION_ACTION_BATCH_SIZE", 64)),
            include_conservative=False,
            margin=float(getattr(self.config, "CRYSTAL_CAVES_CORRECTION_ACTION_MARGIN", 0.6)),
        )
        if result is None:
            return None
        correction_loss, accuracy, _conservative_loss = result
        return correction_loss, accuracy

    def _contact_action_supervised_loss(
        self: Any,
    ) -> Optional[Tuple[torch.Tensor, float]]:
        """Optional cross-entropy loss for a detached close-zone contact head."""
        if not getattr(self.config, "CRYSTAL_CAVES_CONTACT_ACTION_HEAD", False):
            return None
        if self._contact_action_states is None or self._contact_action_actions is None:
            return None
        logits_fn = getattr(self.policy_net, "contact_action_logits", None)
        if not callable(logits_fn):
            return None

        count = int(self._contact_action_actions.shape[0])
        if count <= 0:
            return None
        batch_size = min(
            int(getattr(self.config, "CRYSTAL_CAVES_CONTACT_ACTION_BATCH_SIZE", 64)),
            count,
        )
        indices = torch.randint(count, (batch_size,), device=self.device)
        states = self._contact_action_states.index_select(0, indices)
        actions = self._contact_action_actions.index_select(0, indices)
        logits = logits_fn(states)
        loss = F.cross_entropy(logits, actions)
        with torch.no_grad():
            accuracy = float((logits.argmax(dim=1) == actions).float().mean().item())
        return loss, accuracy

    def _demo_action_loss_for_dataset(
        self: Any,
        demo_states: torch.Tensor | None,
        demo_actions: torch.Tensor | None,
        *,
        batch_size: int,
        include_conservative: bool,
        margin: float | None = None,
    ) -> Optional[Tuple[torch.Tensor, float, torch.Tensor | None]]:
        if demo_states is None or demo_actions is None:
            return None
        if margin is None:
            margin = float(getattr(self.config, "CRYSTAL_CAVES_DEMO_ACTION_MARGIN", 0.8))
        conservative_temperature = None
        if include_conservative:
            conservative_weight = float(
                getattr(self.config, "CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT", 0.0)
            )
            if conservative_weight > 0:
                conservative_temperature = float(
                    getattr(
                        self.config,
                        "CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE",
                        1.0,
                    )
                )
        result = sample_action_margin_loss(
            self.policy_net,
            demo_states,
            demo_actions,
            batch_size=batch_size,
            margin=float(margin),
            conservative_temperature=conservative_temperature,
        )
        if result is None:
            return None
        return result.loss, result.accuracy, result.conservative_loss

    def _route_auxiliary_targets(
        self: Any, states: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Build 9-way objective-direction labels from Crystal Caves metadata.

        Op 2 (geodesic): when enabled, the label is the GEODESIC next-step direction carried
        in the trailing route-label slots (sliced off the policy input) — the real route
        around walls, which the net must learn to predict. Otherwise falls back to the legacy
        euclidean target bearing read from meta 15/16 (which the net can already see)."""
        layout = getattr(self.config, "STATE_LAYOUT", None)
        if not layout:
            return None
        route_label = int(layout.get("route_label", 0))
        if getattr(self.config, "CRYSTAL_CAVES_ROUTE_AUX_GEODESIC", False) and route_label >= 2:
            if states.shape[1] < route_label:
                return None
            lab = states[:, -route_label:]
            dx, dy = lab[:, 0], lab[:, 1]
            reachable = lab[:, 2] if route_label > 2 else torch.ones_like(dx)
            mask = (
                torch.isfinite(dx)
                & torch.isfinite(dy)
                & (reachable > 0.5)
                & ((dx != 0) | (dy != 0))  # a real directional step exists
            )
            if not bool(mask.any().item()):
                return None
            sx = dx.clamp(-1.0, 1.0).round().long()
            sy = dy.clamp(-1.0, 1.0).round().long()
            labels = (sy + 1) * 3 + (sx + 1)
            return labels.long(), mask
        wr, wc = layout["window"]
        gr, gc = layout.get("gmap", (0, 0))
        meta_size = int(layout.get("meta", 0))
        if meta_size < 19:
            return None

        meta_start = int(wr) * int(wc) + int(gr) * int(gc)
        if states.shape[1] < meta_start + meta_size:
            return None

        meta = states[:, meta_start : meta_start + meta_size]
        dx = meta[:, 15] * 2.0 - 1.0
        dy = meta[:, 16] * 2.0 - 1.0
        distance = meta[:, 17]
        kind = meta[:, 18]
        mask = (
            torch.isfinite(dx)
            & torch.isfinite(dy)
            & torch.isfinite(distance)
            & (kind > 0.0)
            & (distance < 0.999)
        )
        if not bool(mask.any().item()):
            return None

        deadband = float(getattr(self.config, "CRYSTAL_CAVES_ROUTE_AUX_DEADBAND", 0.01))
        sx = torch.zeros_like(dx, dtype=torch.long)
        sy = torch.zeros_like(dy, dtype=torch.long)
        sx = torch.where(dx > deadband, torch.ones_like(sx), sx)
        sx = torch.where(dx < -deadband, -torch.ones_like(sx), sx)
        sy = torch.where(dy > deadband, torch.ones_like(sy), sy)
        sy = torch.where(dy < -deadband, -torch.ones_like(sy), sy)
        labels = (sy + 1) * 3 + (sx + 1)
        return labels.long(), mask

    def _route_auxiliary_loss(
        self: Any, states: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, float]]:
        """Optional supervised objective-direction loss for spatial Crystal Caves runs."""
        if not getattr(self.config, "CRYSTAL_CAVES_ROUTE_AUX_LOSS", False):
            return None
        logits_fn = getattr(self.policy_net, "route_aux_logits", None)
        if not callable(logits_fn):
            return None
        targets = self._route_auxiliary_targets(states)
        if targets is None:
            return None

        labels, mask = targets
        logits = logits_fn(states)
        masked_logits = logits[mask]
        masked_labels = labels[mask]
        aux_loss = F.cross_entropy(masked_logits, masked_labels)
        with torch.no_grad():
            accuracy = float((masked_logits.argmax(dim=1) == masked_labels).float().mean().item())
        return aux_loss, accuracy

    def _mean_recent(self: Any, values, n: int) -> float:
        with self._losses_lock:
            if not values:
                return 0.0
            count = min(n, len(values))
            recent = list(values)[-count:]
            return sum(recent) / count if count > 0 else 0.0

    def get_average_route_aux_loss(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.route_aux_losses, n)

    def get_average_route_aux_accuracy(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.route_aux_accuracies, n)

    def get_average_demo_action_loss(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.demo_action_losses, n)

    def get_average_demo_conservative_loss(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.demo_conservative_losses, n)

    def get_average_demo_action_accuracy(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.demo_action_accuracies, n)

    def get_average_close_zone_demo_action_loss(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.close_zone_demo_action_losses, n)

    def get_average_close_zone_demo_action_accuracy(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.close_zone_demo_action_accuracies, n)

    def get_average_correction_action_loss(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.correction_action_losses, n)

    def get_average_correction_action_accuracy(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.correction_action_accuracies, n)

    def get_average_contact_action_loss(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.contact_action_losses, n)

    def get_average_contact_action_accuracy(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.contact_action_accuracies, n)

    def get_average_policy_anchor_loss(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.policy_anchor_losses, n)

    def get_average_policy_anchor_accuracy(self: Any, n: int = 100) -> float:
        return self._mean_recent(self.policy_anchor_accuracies, n)

    def get_policy_anchor_metric_count(self: Any, n: int = 100) -> int:
        with self._losses_lock:
            return min(n, len(self.policy_anchor_losses))

    def get_correction_action_transition_count(self: Any) -> int:
        actions = getattr(self, "_correction_action_actions", None)
        return int(actions.shape[0]) if actions is not None else 0

    def get_correction_action_metric_count(self: Any, n: int = 100) -> int:
        with self._losses_lock:
            return min(n, len(self.correction_action_losses))

    def get_contact_action_transition_count(self: Any) -> int:
        actions = getattr(self, "_contact_action_actions", None)
        return int(actions.shape[0]) if actions is not None else 0

    def get_contact_action_metric_count(self: Any, n: int = 100) -> int:
        with self._losses_lock:
            return min(n, len(self.contact_action_losses))
