# ruff: noqa: F401,F403,F405,I001
"""Correction-dataset combination and contact-head calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .common import *
from .config_helpers import *
from .contact_head import (
    contact_action_dataset_stats,
    evaluate_contact_action_head_dataset,
    train_contact_action_head_offline,
)
from .corrections import (
    CORRECTION_DATASET_VERSION,
    CORRECTION_REASON_BITS,
    _validate_checkpoint_shape,
    load_correction_action_dataset,
)
from .io_utils import *
from .reports import load_selected_weight_snapshot, summarize_trainer
from .runs_transfer import config_from_selected_checkpoint
from .training import *


@dataclass(frozen=True)
class CombinedCorrectionDataset:
    states: np.ndarray
    actions: np.ndarray
    source_dataset_indices: np.ndarray
    source_example_indices: np.ndarray
    source_summaries: tuple[dict[str, Any], ...]


def parse_correction_dataset_paths(raw: str | None) -> tuple[Path, ...]:
    if raw is None or not str(raw).strip():
        return ()
    paths = tuple(Path(part.strip()) for part in str(raw).split(",") if part.strip())
    if not paths:
        raise ValueError("--correction-datasets must include at least one .npz path")
    return paths


def combine_correction_action_datasets(paths: Sequence[Path]) -> CombinedCorrectionDataset:
    if not paths:
        raise ValueError("correction dataset combination requires at least one dataset")
    states_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    source_dataset_indices: list[np.ndarray] = []
    source_example_indices: list[np.ndarray] = []
    source_summaries: list[dict[str, Any]] = []
    expected_state_size: int | None = None

    for dataset_index, path in enumerate(paths):
        states, actions = load_correction_action_dataset(path)
        if states.shape[0] <= 0:
            raise ValueError(f"correction dataset is empty: {path}")
        if expected_state_size is None:
            expected_state_size = int(states.shape[1])
        elif int(states.shape[1]) != expected_state_size:
            raise ValueError(
                "correction datasets must have the same state size: "
                f"{path} has {states.shape[1]}, expected {expected_state_size}"
            )
        states_parts.append(states)
        action_parts.append(actions)
        source_dataset_indices.append(np.full((len(actions),), dataset_index, dtype=np.int64))
        source_example_indices.append(np.arange(len(actions), dtype=np.int64))
        source_summaries.append(_load_source_dataset_summary(path, len(actions)))

    return CombinedCorrectionDataset(
        states=np.concatenate(states_parts, axis=0).astype(np.float32, copy=False),
        actions=np.concatenate(action_parts, axis=0).astype(np.int64, copy=False),
        source_dataset_indices=np.concatenate(source_dataset_indices, axis=0),
        source_example_indices=np.concatenate(source_example_indices, axis=0),
        source_summaries=tuple(source_summaries),
    )


def stratified_calibration_split(
    actions: np.ndarray,
    *,
    calibration_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if calibration_frac <= 0.0 or calibration_frac >= 1.0:
        raise ValueError("contact-head calibration fraction must be in (0, 1)")
    actions_np = np.asarray(actions, dtype=np.int64)
    if actions_np.ndim != 1 or len(actions_np) < 2:
        raise ValueError("calibration split requires at least two labels")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    calibration_indices: list[int] = []
    for action in sorted(set(actions_np.tolist())):
        class_indices = np.flatnonzero(actions_np == int(action))
        rng.shuffle(class_indices)
        if len(class_indices) == 1:
            train_indices.extend(int(index) for index in class_indices)
            continue
        calibration_count = int(round(len(class_indices) * calibration_frac))
        calibration_count = min(len(class_indices) - 1, max(1, calibration_count))
        calibration_indices.extend(int(index) for index in class_indices[:calibration_count])
        train_indices.extend(int(index) for index in class_indices[calibration_count:])

    if not train_indices or not calibration_indices:
        raise ValueError("calibration split produced an empty train or calibration set")
    return (
        np.asarray(sorted(train_indices), dtype=np.int64),
        np.asarray(sorted(calibration_indices), dtype=np.int64),
    )


def run_contact_head_calibration(
    out_dir: Path,
    *,
    checkpoint_path: Path,
    correction_dataset_paths: Sequence[Path],
    seed: int,
    log_every: int,
    report_seconds: float,
    contact_action_batch_size: int,
    contact_action_distance_tiles: float,
    contact_head_offline_steps: int,
    contact_head_learning_rate: float,
    contact_head_balance_classes: bool,
    calibration_frac: float,
    calibration_seed: int,
    min_calibration_accuracy: float,
    min_class_examples: int,
    label: str = "contact_head_calibration",
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    combined = combine_correction_action_datasets(correction_dataset_paths)
    train_indices, calibration_indices = stratified_calibration_split(
        combined.actions,
        calibration_frac=calibration_frac,
        seed=calibration_seed,
    )

    snapshot = load_selected_weight_snapshot(checkpoint_path)
    config = config_from_selected_checkpoint(
        run_dir,
        snapshot=snapshot,
        seed=seed,
        log_every=log_every,
        report_seconds=report_seconds,
    )
    config.MAX_EPISODES = 1
    config.EVAL_EVERY = 0
    config.EVAL_EPISODES = 0
    apply_contact_action_head_override(
        config,
        contact_action_weight=0.0,
        contact_action_batch_size=contact_action_batch_size,
        contact_action_distance_tiles=contact_action_distance_tiles,
    )
    trainer = prepare_trainer(
        config,
        episodes=1,
        vec_envs=1,
        transfer_weights=snapshot["weights"],
        strict_transfer=False,
        save_checkpoints=False,
    )
    trainer.agent.epsilon = 0.0
    selected_episode = int(snapshot.get("episode", 0) or 0)
    trainer.current_episode = selected_episode
    _validate_checkpoint_shape(trainer, snapshot, checkpoint_path)
    action_labels = list(getattr(trainer.game, "ACTION_LABELS", [])) or [
        str(index) for index in range(trainer.agent.action_size)
    ]

    dataset_summary = _write_combined_correction_dataset(
        run_dir,
        label=label,
        combined=combined,
        train_indices=train_indices,
        calibration_indices=calibration_indices,
        action_labels=action_labels,
        calibration_frac=calibration_frac,
        calibration_seed=calibration_seed,
    )
    train_states = combined.states[train_indices]
    train_actions = combined.actions[train_indices]
    calibration_states = combined.states[calibration_indices]
    calibration_actions = combined.actions[calibration_indices]

    started = time.time()
    offline_training = train_contact_action_head_offline(
        trainer.agent,
        train_states,
        train_actions,
        steps=contact_head_offline_steps,
        batch_size=contact_action_batch_size,
        learning_rate=contact_head_learning_rate,
        balance_classes=contact_head_balance_classes,
        action_labels=action_labels,
    )
    train_seconds = time.time() - started
    calibration_eval = evaluate_contact_action_head_dataset(
        trainer.agent,
        calibration_states,
        calibration_actions,
        action_labels=action_labels,
    )
    train_stats = contact_action_dataset_stats(train_actions, action_labels=action_labels)
    calibration_stats = contact_action_dataset_stats(
        calibration_actions,
        action_labels=action_labels,
    )
    decision = _contact_head_calibration_decision(
        calibration_eval=calibration_eval,
        train_stats=train_stats,
        min_calibration_accuracy=min_calibration_accuracy,
        min_class_examples=min_class_examples,
        route_max_abs_delta=float(offline_training.get("route_max_abs_delta", 0.0)),
    )
    calibration_payload = {
        "decision": decision,
        "calibration_fraction": float(calibration_frac),
        "calibration_seed": int(calibration_seed),
        "train_examples": int(len(train_indices)),
        "calibration_examples": int(len(calibration_indices)),
        "train_dataset_stats": train_stats,
        "calibration_dataset_stats": calibration_stats,
        "calibration_eval": calibration_eval,
        "min_calibration_accuracy": float(min_calibration_accuracy),
        "min_class_examples": int(min_class_examples),
    }
    write_json(run_dir / "contact_head_calibration.json", calibration_payload)

    return summarize_trainer(
        trainer,
        label=label,
        train_seconds=train_seconds,
        extra={
            "checkpoint": str(checkpoint_path),
            "checkpoint_eval": {
                "kind": SELECTED_WEIGHT_SNAPSHOT_KIND,
                "source_label": snapshot.get("label", ""),
                "source_episode": selected_episode,
                "source_eval": snapshot.get("source_eval") or {},
            },
            "correction_dataset": dataset_summary,
            "contact_action_head_training": {
                "mode": "offline_head_only",
                "dataset_path": str(dataset_summary["train_states_path"]),
                "dataset_states": int(train_states.shape[0]),
                "state_size": int(train_states.shape[1]) if train_states.ndim == 2 else 0,
                "weight": 0.0,
                "batch_size": int(contact_action_batch_size),
                "distance_tiles": float(contact_action_distance_tiles),
                "learning_rate": float(contact_head_learning_rate),
                "offline_steps": int(contact_head_offline_steps),
                "confidence_threshold": 0.0,
                "balance_classes": bool(contact_head_balance_classes),
                "dataset_stats": train_stats,
                "offline_training": offline_training,
                "contact_action_transitions": int(train_states.shape[0]),
            },
            "contact_action_head_calibration": calibration_payload,
        },
    )


def _write_combined_correction_dataset(
    run_dir: Path,
    *,
    label: str,
    combined: CombinedCorrectionDataset,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    action_labels: list[str],
    calibration_frac: float,
    calibration_seed: int,
) -> dict[str, Any]:
    dataset_dir = run_dir / "corrections" / f"{label}_combined"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    states_path = dataset_dir / "correction_examples.npz"
    train_states_path = dataset_dir / "correction_examples_train.npz"
    calibration_states_path = dataset_dir / "correction_examples_calibration.npz"
    rows_path = dataset_dir / "correction_examples.jsonl"
    summary_path = dataset_dir / "summary.json"

    _save_correction_npz(
        states_path,
        combined.states,
        combined.actions,
        source_dataset_indices=combined.source_dataset_indices,
        source_example_indices=combined.source_example_indices,
    )
    _save_correction_npz(
        train_states_path,
        combined.states[train_indices],
        combined.actions[train_indices],
        source_dataset_indices=combined.source_dataset_indices[train_indices],
        source_example_indices=combined.source_example_indices[train_indices],
    )
    _save_correction_npz(
        calibration_states_path,
        combined.states[calibration_indices],
        combined.actions[calibration_indices],
        source_dataset_indices=combined.source_dataset_indices[calibration_indices],
        source_example_indices=combined.source_example_indices[calibration_indices],
    )
    _write_combined_rows(rows_path, combined, action_labels=action_labels)

    summary = _combined_dataset_summary(
        label=label,
        combined=combined,
        train_indices=train_indices,
        calibration_indices=calibration_indices,
        action_labels=action_labels,
        states_path=states_path,
        train_states_path=train_states_path,
        calibration_states_path=calibration_states_path,
        rows_path=rows_path,
        summary_path=summary_path,
        calibration_frac=calibration_frac,
        calibration_seed=calibration_seed,
    )
    write_json(summary_path, summary)
    return summary


def _save_correction_npz(
    path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    source_dataset_indices: np.ndarray,
    source_example_indices: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        source_dataset_indices=np.asarray(source_dataset_indices, dtype=np.int64),
        source_example_indices=np.asarray(source_example_indices, dtype=np.int64),
    )


def _write_combined_rows(
    path: Path,
    combined: CombinedCorrectionDataset,
    *,
    action_labels: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, action in enumerate(combined.actions.tolist()):
            source_index = int(combined.source_dataset_indices[index])
            source = combined.source_summaries[source_index]
            row = {
                "combined_index": index,
                "source_dataset_index": source_index,
                "source_path": source.get("states_path", ""),
                "source_example_index": int(combined.source_example_indices[index]),
                "label_action": int(action),
                "label_action_label": _action_label(action_labels, int(action)),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _combined_dataset_summary(
    *,
    label: str,
    combined: CombinedCorrectionDataset,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    action_labels: list[str],
    states_path: Path,
    train_states_path: Path,
    calibration_states_path: Path,
    rows_path: Path,
    summary_path: Path,
    calibration_frac: float,
    calibration_seed: int,
) -> dict[str, Any]:
    full_stats = contact_action_dataset_stats(combined.actions, action_labels=action_labels)
    train_stats = contact_action_dataset_stats(
        combined.actions[train_indices],
        action_labels=action_labels,
    )
    calibration_stats = contact_action_dataset_stats(
        combined.actions[calibration_indices],
        action_labels=action_labels,
    )
    source_summaries = list(combined.source_summaries)
    games_completed = sum(int(source.get("games_completed", 0) or 0) for source in source_summaries)
    agreements = sum(int(source.get("agreement_states", 0) or 0) for source in source_summaries)
    kept_examples = int(len(combined.actions))
    gate_evaluations = sum(
        int(source.get("gate_evaluations", 0) or 0) for source in source_summaries
    )
    gate_rejections = sum(int(source.get("gate_rejections", 0) or 0) for source in source_summaries)
    mean_advantage = _weighted_mean_source_value(
        source_summaries,
        value_key="mean_gate_option_advantage",
        weight_key="kept_examples",
    )
    return {
        "dataset_version": CORRECTION_DATASET_VERSION,
        "label": f"{label}_combined",
        "label_mode": "combined_contact_head_calibration",
        "source_dataset_count": len(source_summaries),
        "source_datasets": source_summaries,
        "kept_examples": kept_examples,
        "train_examples": int(len(train_indices)),
        "calibration_examples": int(len(calibration_indices)),
        "calibration_fraction": float(calibration_frac),
        "calibration_seed": int(calibration_seed),
        "states_path": str(states_path),
        "train_states_path": str(train_states_path),
        "calibration_states_path": str(calibration_states_path),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "states_shape": [int(item) for item in combined.states.shape],
        "actions_shape": [int(item) for item in combined.actions.shape],
        "games_completed": int(games_completed),
        "agreement_states": int(agreements),
        "disagreement_rate": float(1.0 - agreements / max(1, kept_examples)),
        "trigger_counts": _merge_counter_dicts(source_summaries, "trigger_counts")
        or {"combined": kept_examples},
        "label_action_counts": full_stats["action_counts"],
        "train_action_counts": train_stats["action_counts"],
        "calibration_action_counts": calibration_stats["action_counts"],
        "gate_evaluations": int(gate_evaluations),
        "gate_rejections": int(gate_rejections),
        "gate_rejection_rate": float(gate_rejections / max(1, gate_evaluations)),
        "mean_gate_option_advantage": float(mean_advantage),
    }


def _contact_head_calibration_decision(
    *,
    calibration_eval: dict[str, Any],
    train_stats: dict[str, Any],
    min_calibration_accuracy: float,
    min_class_examples: int,
    route_max_abs_delta: float,
) -> dict[str, Any]:
    accuracy = float(calibration_eval.get("accuracy", 0.0) or 0.0)
    train_counts = train_stats.get("action_counts") or {}
    undercovered = {
        str(label): int(count)
        for label, count in train_counts.items()
        if int(count) < min_class_examples
    }
    reasons: list[str] = []
    if accuracy < min_calibration_accuracy:
        reasons.append(f"calibration accuracy {accuracy:.3f} below {min_calibration_accuracy:.3f}")
    if undercovered:
        reasons.append(f"undercovered train classes: {undercovered}")
    if route_max_abs_delta != 0.0:
        reasons.append(f"route weights changed by {route_max_abs_delta:.2e}")
    return {
        "verdict": "pass" if not reasons else "hold",
        "passed": not reasons,
        "reasons": reasons,
    }


def _load_source_dataset_summary(path: Path, fallback_count: int) -> dict[str, Any]:
    summary_path = path.with_name("summary.json")
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("kept_examples", fallback_count)
                payload.setdefault("states_path", str(path))
                payload.setdefault("summary_path", str(summary_path))
                return payload
        except json.JSONDecodeError:
            pass
    return {
        "dataset_version": CORRECTION_DATASET_VERSION,
        "kept_examples": fallback_count,
        "states_path": str(path),
        "summary_path": str(summary_path),
        "label_action_counts": {},
    }


def _merge_counter_dicts(
    summaries: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for summary in summaries:
        payload = summary.get(key) or {}
        if isinstance(payload, dict):
            for item_key, value in payload.items():
                counts[str(item_key)] += int(value or 0)
    return dict(counts)


def _weighted_mean_source_value(
    summaries: Sequence[dict[str, Any]],
    *,
    value_key: str,
    weight_key: str,
) -> float:
    weighted_total = 0.0
    weight_total = 0
    for summary in summaries:
        weight = int(summary.get(weight_key, 0) or 0)
        weighted_total += float(summary.get(value_key, 0.0) or 0.0) * weight
        weight_total += weight
    return weighted_total / max(1, weight_total)


def _action_label(action_labels: list[str], action: int) -> str:
    if 0 <= action < len(action_labels):
        return action_labels[action]
    return str(action)
