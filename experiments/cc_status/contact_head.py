# ruff: noqa: F401,F403,F405,I001
"""Learned close-zone contact-action selector for Crystal Caves probes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .common import *
from .evals import _action_label, objective_snapshot


def new_contact_action_head_stats() -> dict[str, Any]:
    return {
        "head_actions": 0,
        "policy_actions": 0,
        "unavailable_actions": 0,
        "confidence_rejected_actions": 0,
        "head_confidence_total": 0.0,
        "head_action_counts": Counter(),
        "policy_action_counts": Counter(),
        "rejected_head_action_counts": Counter(),
    }


def contact_action_head_action(
    agent: Any,
    state: np.ndarray,
    game: CrystalCaves,
    *,
    action_labels: list[str],
    close_zone_distance_tiles: float,
    min_confidence: float = 0.0,
    jump_min_confidence: float = 0.0,
    action_min_confidences: Mapping[str, float] | None = None,
) -> tuple[int, dict[str, Any]]:
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if jump_min_confidence < 0.0 or jump_min_confidence > 1.0:
        raise ValueError("jump_min_confidence must be in [0, 1]")
    action_thresholds = normalized_contact_head_action_thresholds(action_min_confidences)
    objective = objective_snapshot(game)
    target_distance = objective.get("target_distance_tiles")
    if target_distance is None or float(target_distance) > close_zone_distance_tiles:
        action = int(agent.select_action(state, training=False))
        return action, {"source": "policy", "target_distance_tiles": target_distance}

    logits_fn = getattr(agent.policy_net, "contact_action_logits", None)
    if not callable(logits_fn):
        action = int(agent.select_action(state, training=False))
        return action, {
            "source": "policy",
            "target_distance_tiles": target_distance,
            "contact_head_unavailable": True,
        }

    was_training = bool(getattr(agent.policy_net, "training", False))
    if was_training and hasattr(agent.policy_net, "eval"):
        agent.policy_net.eval()
    try:
        with torch.inference_mode():
            tensor = torch.as_tensor(
                state.reshape(1, -1),
                dtype=torch.float32,
                device=agent.device,
            )
            logits = logits_fn(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, action_tensor = torch.max(probs, dim=1)
            head_action = int(action_tensor.item())
            confidence_value = float(confidence.item())
    finally:
        if was_training and hasattr(agent.policy_net, "train"):
            agent.policy_net.train()

    head_action_label = _action_label(action_labels, head_action)
    required_confidence = _required_contact_head_confidence(
        action_label=head_action_label,
        min_confidence=min_confidence,
        jump_min_confidence=jump_min_confidence,
        action_min_confidences=action_thresholds,
    )
    if confidence_value < required_confidence:
        policy_action = int(agent.select_action(state, training=False))
        return policy_action, {
            "source": "policy",
            "target_distance_tiles": target_distance,
            "confidence": confidence_value,
            "head_confidence_rejected": True,
            "head_action": head_action,
            "head_action_label": head_action_label,
            "required_confidence": required_confidence,
            "base_min_confidence": min_confidence,
            "jump_min_confidence": jump_min_confidence,
            "action_min_confidences": action_thresholds,
        }

    action = head_action
    return action, {
        "source": "contact_action_head",
        "target_distance_tiles": target_distance,
        "confidence": confidence_value,
        "required_confidence": required_confidence,
        "base_min_confidence": min_confidence,
        "jump_min_confidence": jump_min_confidence,
        "action_min_confidences": action_thresholds,
        "action_label": head_action_label,
    }


def record_contact_action_head_decision(
    stats: dict[str, Any],
    decision: dict[str, Any],
    action: int,
    action_labels: list[str],
) -> None:
    source = str(decision.get("source", "") or "")
    label = _action_label(action_labels, action)
    if source == "contact_action_head":
        stats["head_actions"] += 1
        stats["head_confidence_total"] += float(decision.get("confidence", 0.0) or 0.0)
        stats["head_action_counts"][label] += 1
        return
    stats["policy_actions"] += 1
    stats["policy_action_counts"][label] += 1
    if decision.get("contact_head_unavailable"):
        stats["unavailable_actions"] += 1
    if decision.get("head_confidence_rejected"):
        stats["confidence_rejected_actions"] += 1
        rejected_label = str(decision.get("head_action_label", "") or "")
        if rejected_label:
            stats["rejected_head_action_counts"][rejected_label] += 1


def contact_action_head_stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    head_actions = int(stats.get("head_actions", 0) or 0)
    policy_actions = int(stats.get("policy_actions", 0) or 0)
    return {
        "head_actions": head_actions,
        "policy_actions": policy_actions,
        "unavailable_actions": int(stats.get("unavailable_actions", 0) or 0),
        "confidence_rejected_actions": int(stats.get("confidence_rejected_actions", 0) or 0),
        "head_action_rate": float(head_actions / max(1, head_actions + policy_actions)),
        "mean_head_confidence": float(
            stats.get("head_confidence_total", 0.0) / max(1, head_actions)
        ),
        "head_action_counts": dict(stats.get("head_action_counts", {})),
        "policy_action_counts": dict(stats.get("policy_action_counts", {})),
        "rejected_head_action_counts": dict(stats.get("rejected_head_action_counts", {})),
    }


def _required_contact_head_confidence(
    *,
    action_label: str,
    min_confidence: float,
    jump_min_confidence: float,
    action_min_confidences: Mapping[str, float] | None = None,
) -> float:
    threshold = min_confidence
    if jump_min_confidence > 0.0 and action_label in DEMO_JUMP_ACTIONS:
        threshold = max(threshold, jump_min_confidence)
    if action_min_confidences:
        threshold = max(
            threshold,
            float(action_min_confidences.get(_normalized_action_label(action_label), 0.0)),
        )
    return threshold


def parse_contact_head_action_thresholds(raw: str | None) -> dict[str, float]:
    """Parse ACTION:confidence pairs used for class-specific selector gates."""

    if raw is None or not raw.strip():
        return {}
    thresholds: dict[str, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("contact-head action thresholds must use ACTION:threshold pairs")
        action_label, raw_threshold = item.split(":", 1)
        label = _normalized_action_label(action_label)
        if not label:
            raise ValueError("contact-head action threshold label must be non-empty")
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError(
                f"invalid contact-head threshold for {label}: {raw_threshold!r}"
            ) from exc
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"contact-head threshold for {label} must be in [0, 1]")
        thresholds[label] = threshold
    return thresholds


def normalized_contact_head_action_thresholds(
    thresholds: Mapping[str, float] | None,
) -> dict[str, float]:
    if not thresholds:
        return {}
    normalized: dict[str, float] = {}
    for action_label, threshold in thresholds.items():
        label = _normalized_action_label(action_label)
        if not label:
            raise ValueError("contact-head action threshold label must be non-empty")
        threshold_value = float(threshold)
        if threshold_value < 0.0 or threshold_value > 1.0:
            raise ValueError(f"contact-head threshold for {label} must be in [0, 1]")
        normalized[label] = threshold_value
    return normalized


def validate_contact_head_action_thresholds(
    thresholds: Mapping[str, float] | None,
    action_labels: list[str],
) -> dict[str, float]:
    normalized = normalized_contact_head_action_thresholds(thresholds)
    if not normalized:
        return {}
    known_labels = {_normalized_action_label(label) for label in action_labels}
    unknown_labels = sorted(label for label in normalized if label not in known_labels)
    if unknown_labels:
        known_text = ", ".join(sorted(known_labels))
        unknown_text = ", ".join(unknown_labels)
        raise ValueError(
            f"unknown contact-head action threshold label(s): {unknown_text}; "
            f"known labels: {known_text}"
        )
    return normalized


def _normalized_action_label(action_label: str) -> str:
    return action_label.strip().upper().replace("-", "_")


def contact_action_head_report_lines(run: dict[str, Any]) -> list[str]:
    head_training = run.get("contact_action_head_training") or {}
    if not head_training:
        return []
    lines = [
        (
            f"- Contact action head: {head_training.get('contact_action_transitions', 0)} "
            f"states, {_contact_head_training_mode_text(head_training)}, "
            f"batch {head_training.get('batch_size', 0)}, "
            f"active <= {head_training.get('distance_tiles', 0):.1f} target tiles, "
            f"dataset `{head_training.get('dataset_path', '')}`"
        ),
        (
            f"- Contact head metrics avg100: "
            f"loss {run.get('avg_contact_action_loss_100', 0):.4f}, "
            f"accuracy {100 * run.get('avg_contact_action_accuracy_100', 0):.1f}%, "
            f"samples {run.get('contact_action_samples_100', 0)}"
        ),
    ]
    offline_training = head_training.get("offline_training") or {}
    if offline_training:
        lines.append(
            f"- Contact head offline fit: steps {offline_training.get('steps', 0)}, "
            f"lr {offline_training.get('learning_rate', 0):.4f}, "
            f"balanced {offline_training.get('balance_classes', False)}, "
            f"dataset accuracy {100 * (offline_training.get('dataset_eval') or {}).get('accuracy', 0):.1f}%, "
            f"route delta {offline_training.get('route_max_abs_delta', 0):.2e}, "
            f"head delta {offline_training.get('head_max_abs_delta', 0):.2e}"
        )
    action_counts = (head_training.get("dataset_stats") or {}).get("action_counts") or {}
    if action_counts:
        lines.append(f"- Contact head dataset action counts: {action_counts}")
    selector = run.get("contact_action_head_eval") or {}
    if selector:
        lines.append(
            f"- Contact head eval selector: {selector.get('head_actions', 0)} head actions, "
            f"{selector.get('policy_actions', 0)} policy actions, "
            f"{selector.get('unavailable_actions', 0)} unavailable fallbacks, "
            f"{selector.get('confidence_rejected_actions', 0)} confidence rejects, "
            f"mean confidence {selector.get('mean_head_confidence', 0):.3f}"
        )
        if selector.get("head_action_counts"):
            lines.append(f"- Contact head eval action counts: {selector.get('head_action_counts')}")
    calibration = run.get("contact_action_head_calibration") or {}
    if calibration:
        eval_payload = calibration.get("calibration_eval") or {}
        decision = calibration.get("decision") or {}
        reasons = decision.get("reasons") or []
        reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "none"
        lines.append(
            f"- Contact head calibration: {calibration.get('train_examples', 0)} train / "
            f"{calibration.get('calibration_examples', 0)} held-out labels, "
            f"accuracy {100 * eval_payload.get('accuracy', 0):.1f}%, "
            f"mean confidence {eval_payload.get('mean_confidence', 0):.3f}, "
            f"decision `{decision.get('verdict', 'unknown')}` ({reason_text})"
        )
        if eval_payload.get("per_class_accuracy"):
            lines.append(
                f"- Contact head calibration per-class accuracy: "
                f"{eval_payload.get('per_class_accuracy')}"
            )
        if eval_payload.get("per_class_mean_confidence"):
            lines.append(
                f"- Contact head calibration per-class confidence: "
                f"{eval_payload.get('per_class_mean_confidence')}"
            )
    return lines


def _contact_head_training_mode_text(head_training: dict[str, Any]) -> str:
    if head_training.get("mode") == "offline_head_only":
        jump_threshold = float(head_training.get("jump_confidence_threshold", 0.0) or 0.0)
        jump_text = f", jump >= {jump_threshold:.2f}" if jump_threshold > 0.0 else ""
        action_thresholds = head_training.get("action_confidence_thresholds") or {}
        action_text = (
            f", action gates {_format_action_thresholds(action_thresholds)}"
            if action_thresholds
            else ""
        )
        return (
            f"offline/head-only, lr {head_training.get('learning_rate', 0):.4f}, "
            f"steps {head_training.get('offline_steps', 0)}, "
            f"confidence >= {head_training.get('confidence_threshold', 0):.2f}"
            f"{jump_text}"
            f"{action_text}"
        )
    return f"weight {head_training.get('weight', 0):.3f}"


def _format_action_thresholds(thresholds: Mapping[str, Any]) -> str:
    parts = [
        f"{str(label)} >= {float(threshold):.2f}" for label, threshold in sorted(thresholds.items())
    ]
    return "{" + ", ".join(parts) + "}"


def contact_action_dataset_stats(
    actions: np.ndarray | torch.Tensor,
    *,
    action_labels: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy().astype(np.int64)
    else:
        actions_np = np.asarray(actions, dtype=np.int64)
    counts = Counter(int(action) for action in actions_np.tolist())
    labelled_counts: dict[str, int] = {}
    for action, count in sorted(counts.items()):
        label = _action_label(action_labels or [], action)
        labelled_counts[label] = int(count)
    max_count = max(counts.values(), default=0)
    min_count = min(counts.values(), default=0)
    return {
        "classes": len(counts),
        "count": int(actions_np.size),
        "action_counts": labelled_counts,
        "max_class_count": int(max_count),
        "min_class_count": int(min_count),
        "imbalance_ratio": float(max_count / max(1, min_count)) if counts else 0.0,
    }


def train_contact_action_head_offline(
    agent: Any,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    balance_classes: bool,
    action_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Train only the detached contact-action head on fixed labels."""
    if steps <= 0:
        raise ValueError("contact head offline steps must be positive")
    if batch_size <= 0:
        raise ValueError("contact head offline batch size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("contact head learning rate must be positive")

    states_np = np.asarray(states, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.int64)
    if states_np.ndim != 2:
        raise ValueError("contact action states must be a 2D array")
    if actions_np.ndim != 1 or len(actions_np) != len(states_np):
        raise ValueError("contact action labels must be a 1D array matching states")
    if len(actions_np) <= 0:
        raise ValueError("contact head offline training requires at least one label")

    logits_fn = getattr(agent.policy_net, "contact_action_logits", None)
    head = getattr(agent.policy_net, "contact_action_head", None)
    if not callable(logits_fn) or head is None:
        raise RuntimeError("contact action head is not enabled on the policy network")

    states_tensor = torch.as_tensor(states_np.copy(), dtype=torch.float32, device=agent.device)
    actions_tensor = torch.as_tensor(actions_np.copy(), dtype=torch.int64, device=agent.device)
    class_indices = _contact_action_class_indices(actions_tensor)

    original_requires_grad = {
        name: bool(param.requires_grad) for name, param in agent.policy_net.named_parameters()
    }
    before_params = {
        name: param.detach().cpu().clone() for name, param in agent.policy_net.named_parameters()
    }
    head_params: list[torch.nn.Parameter] = []
    for name, param in agent.policy_net.named_parameters():
        is_head_param = name.startswith("contact_action_head.")
        param.requires_grad_(is_head_param)
        if is_head_param:
            head_params.append(param)
    if not head_params:
        raise RuntimeError("contact action head has no trainable parameters")

    optimizer = torch.optim.Adam(head_params, lr=float(learning_rate))
    agent.policy_net.train()
    losses: list[float] = []
    accuracies: list[float] = []
    for _step in range(steps):
        indices = _sample_contact_action_indices(
            actions_tensor,
            class_indices,
            batch_size=batch_size,
            balance_classes=balance_classes,
        )
        batch_states = states_tensor.index_select(0, indices)
        batch_actions = actions_tensor.index_select(0, indices)
        optimizer.zero_grad()
        logits = logits_fn(batch_states)
        loss = F.cross_entropy(logits, batch_actions)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracy = float((logits.argmax(dim=1) == batch_actions).float().mean().item())
            loss_value = float(loss.detach().item())
        losses.append(loss_value)
        accuracies.append(accuracy)
        with agent._losses_lock:
            agent.contact_action_losses.append(loss_value)
            agent.contact_action_accuracies.append(accuracy)

    eval_payload = evaluate_contact_action_head_dataset(
        agent,
        states_np,
        actions_np,
        action_labels=action_labels,
    )
    route_max_abs_delta = 0.0
    head_max_abs_delta = 0.0
    for name, param in agent.policy_net.named_parameters():
        before = before_params[name].to(param.device)
        delta = float((param.detach() - before).abs().max().item())
        if name.startswith("contact_action_head."):
            head_max_abs_delta = max(head_max_abs_delta, delta)
        else:
            route_max_abs_delta = max(route_max_abs_delta, delta)
        param.requires_grad_(original_requires_grad[name])

    return {
        "mode": "offline_head_only",
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "balance_classes": bool(balance_classes),
        "loss": float(losses[-1]) if losses else 0.0,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracies[-1]) if accuracies else 0.0,
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "route_max_abs_delta": route_max_abs_delta,
        "head_max_abs_delta": head_max_abs_delta,
        "dataset_eval": eval_payload,
    }


def evaluate_contact_action_head_dataset(
    agent: Any,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    action_labels: list[str] | None = None,
) -> dict[str, Any]:
    states_tensor = torch.as_tensor(
        np.asarray(states, dtype=np.float32).copy(),
        dtype=torch.float32,
        device=agent.device,
    )
    actions_np = np.asarray(actions, dtype=np.int64)
    actions_tensor = torch.as_tensor(actions_np.copy(), dtype=torch.int64, device=agent.device)
    logits_fn = getattr(agent.policy_net, "contact_action_logits", None)
    if not callable(logits_fn):
        raise RuntimeError("contact action head is not enabled on the policy network")
    was_training = bool(getattr(agent.policy_net, "training", False))
    if was_training and hasattr(agent.policy_net, "eval"):
        agent.policy_net.eval()
    try:
        with torch.inference_mode():
            logits = logits_fn(states_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predictions = torch.max(probs, dim=1)
            correct = predictions == actions_tensor
    finally:
        if was_training and hasattr(agent.policy_net, "train"):
            agent.policy_net.train()

    predicted_np = predictions.detach().cpu().numpy().astype(np.int64)
    correct_np = correct.detach().cpu().numpy().astype(bool)
    confidence_np = confidence.detach().cpu().numpy().astype(np.float32)
    per_class_accuracy: dict[str, float] = {}
    per_class_mean_confidence: dict[str, float] = {}
    for action in sorted(set(actions_np.tolist())):
        mask = actions_np == int(action)
        label = _action_label(action_labels or [], int(action))
        per_class_accuracy[label] = float(correct_np[mask].mean()) if mask.any() else 0.0
        per_class_mean_confidence[label] = float(confidence_np[mask].mean()) if mask.any() else 0.0
    return {
        "accuracy": float(correct.float().mean().item()) if len(actions_np) else 0.0,
        "mean_confidence": float(confidence.mean().item()) if len(actions_np) else 0.0,
        "label_counts": contact_action_dataset_stats(
            actions_np,
            action_labels=action_labels,
        )["action_counts"],
        "prediction_counts": contact_action_dataset_stats(
            predicted_np,
            action_labels=action_labels,
        )["action_counts"],
        "per_class_accuracy": per_class_accuracy,
        "per_class_mean_confidence": per_class_mean_confidence,
    }


def _contact_action_class_indices(actions: torch.Tensor) -> dict[int, torch.Tensor]:
    classes = torch.unique(actions).detach().cpu().numpy().astype(np.int64).tolist()
    return {
        int(action): torch.nonzero(actions == int(action), as_tuple=False).flatten()
        for action in classes
    }


def _sample_contact_action_indices(
    actions: torch.Tensor,
    class_indices: dict[int, torch.Tensor],
    *,
    batch_size: int,
    balance_classes: bool,
) -> torch.Tensor:
    count = int(actions.shape[0])
    if not balance_classes or not class_indices:
        return torch.randint(count, (batch_size,), device=actions.device)

    per_class = max(1, int(np.ceil(batch_size / max(1, len(class_indices)))))
    chunks: list[torch.Tensor] = []
    for indices in class_indices.values():
        pick = torch.randint(int(indices.shape[0]), (per_class,), device=actions.device)
        chunks.append(indices.index_select(0, pick))
    sampled = torch.cat(chunks, dim=0)
    if int(sampled.shape[0]) > batch_size:
        sampled = sampled.index_select(
            0,
            torch.randperm(int(sampled.shape[0]), device=actions.device)[:batch_size],
        )
    elif int(sampled.shape[0]) < batch_size:
        extra = torch.randint(count, (batch_size - int(sampled.shape[0]),), device=actions.device)
        sampled = torch.cat([sampled, extra], dim=0)
    return sampled
