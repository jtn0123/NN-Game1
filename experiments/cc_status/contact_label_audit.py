# ruff: noqa: F401,F403,F405,I001
"""Contact-label quality audit for B10-gated correction datasets."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from .common import *
from .contact_head import contact_action_dataset_stats
from .correction_calibration import combine_correction_action_datasets
from .corrections import CORRECTION_DATASET_VERSION
from .io_utils import *


def run_contact_label_audit(
    out_dir: Path,
    *,
    correction_dataset_paths: Sequence[Path],
    state_round_decimals: int,
    adjacent_step_window: int,
    top_groups: int,
    label: str = "contact_label_audit",
) -> dict[str, Any]:
    if state_round_decimals < 0:
        raise ValueError("contact-label state decimals must be non-negative")
    if adjacent_step_window <= 0:
        raise ValueError("contact-label adjacent step window must be positive")
    if top_groups <= 0:
        raise ValueError("contact-label top groups must be positive")

    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    combined = combine_correction_action_datasets(correction_dataset_paths)
    action_labels = list(CrystalCaves.ACTION_LABELS)
    source_rows = _load_source_rows(correction_dataset_paths)
    enriched_rows = _enriched_audit_rows(
        combined,
        source_rows,
        action_labels=action_labels,
    )
    audit = {
        "source_dataset_count": len(correction_dataset_paths),
        "state_round_decimals": int(state_round_decimals),
        "adjacent_step_window": int(adjacent_step_window),
        "label_action_counts": contact_action_dataset_stats(
            combined.actions,
            action_labels=action_labels,
        )["action_counts"],
        "state_conflicts": _state_conflict_audit(
            combined,
            enriched_rows,
            action_labels=action_labels,
            state_round_decimals=state_round_decimals,
            top_groups=top_groups,
        ),
        "semantic_ambiguity": _semantic_ambiguity_audit(
            enriched_rows,
            top_groups=top_groups,
        ),
        "direction_alignment": _direction_alignment_audit(
            enriched_rows,
            top_groups=top_groups,
        ),
        "adjacent_label_flips": _adjacent_label_flip_audit(
            enriched_rows,
            adjacent_step_window=adjacent_step_window,
            top_groups=top_groups,
        ),
    }
    dataset_summary = _write_audit_dataset(
        run_dir,
        label=label,
        combined=combined,
        rows=enriched_rows,
        action_labels=action_labels,
        audit=audit,
    )
    write_json(run_dir / "contact_label_audit.json", audit)
    return {
        "label": label,
        "episodes": 0,
        "train_seconds": 0.0,
        "steps_per_second": 0.0,
        "device": "offline",
        "avg_score_100": 0.0,
        "avg_progress_100": 0.0,
        "best_progress": 0.0,
        "config": {},
        "eval_history": [],
        "correction_dataset": dataset_summary,
        "contact_label_audit": audit,
    }


def run_contact_label_filter(
    out_dir: Path,
    *,
    correction_dataset_paths: Sequence[Path],
    semantic_majority_threshold: float,
    adjacent_step_window: int,
    label: str = "contact_label_filter",
) -> dict[str, Any]:
    if semantic_majority_threshold < 0.5 or semantic_majority_threshold > 1.0:
        raise ValueError("contact-label filter majority threshold must be in [0.5, 1.0]")
    if adjacent_step_window <= 0:
        raise ValueError("contact-label adjacent step window must be positive")

    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    combined = combine_correction_action_datasets(correction_dataset_paths)
    action_labels = list(CrystalCaves.ACTION_LABELS)
    source_rows = _load_source_rows(correction_dataset_paths)
    enriched_rows = _enriched_audit_rows(
        combined,
        source_rows,
        action_labels=action_labels,
    )
    filter_result = _stable_label_filter_indices(
        enriched_rows,
        semantic_majority_threshold=semantic_majority_threshold,
        adjacent_step_window=adjacent_step_window,
    )
    keep_indices = np.asarray(filter_result["keep_indices"], dtype=np.int64)
    if keep_indices.size <= 0:
        raise ValueError("contact-label filter removed every label")
    filtered_rows = [dict(enriched_rows[int(index)]) for index in keep_indices.tolist()]
    dataset_summary = _write_filtered_dataset(
        run_dir,
        label=label,
        combined=combined,
        keep_indices=keep_indices,
        rows=filtered_rows,
        action_labels=action_labels,
        filter_summary=filter_result["summary"],
    )
    filter_payload = {
        "source_dataset_count": len(correction_dataset_paths),
        "semantic_majority_threshold": float(semantic_majority_threshold),
        "adjacent_step_window": int(adjacent_step_window),
        **filter_result["summary"],
        "filtered_dataset": dataset_summary,
    }
    write_json(run_dir / "contact_label_filter.json", filter_payload)
    return {
        "label": label,
        "episodes": 0,
        "train_seconds": 0.0,
        "steps_per_second": 0.0,
        "device": "offline",
        "avg_score_100": 0.0,
        "avg_progress_100": 0.0,
        "best_progress": 0.0,
        "config": {},
        "eval_history": [],
        "correction_dataset": dataset_summary,
        "contact_label_filter": filter_payload,
    }


def _load_source_rows(paths: Sequence[Path]) -> tuple[tuple[dict[str, Any], ...], ...]:
    rows_by_source: list[tuple[dict[str, Any], ...]] = []
    for path in paths:
        rows_path = path.with_name("correction_examples.jsonl")
        rows: list[dict[str, Any]] = []
        if rows_path.exists():
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
        rows_by_source.append(tuple(rows))
    return tuple(rows_by_source)


def _enriched_audit_rows(
    combined: Any,
    source_rows: tuple[tuple[dict[str, Any], ...], ...],
    *,
    action_labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(combined.actions.tolist()):
        source_index = int(combined.source_dataset_indices[index])
        source_example_index = int(combined.source_example_indices[index])
        source_row = (
            source_rows[source_index][source_example_index]
            if source_index < len(source_rows)
            and source_example_index < len(source_rows[source_index])
            else {}
        )
        label = _action_label(action_labels, int(action))
        objective = source_row.get("objective") if isinstance(source_row, dict) else {}
        if not isinstance(objective, dict):
            objective = {}
        row = {
            "combined_index": index,
            "source_dataset_index": source_index,
            "source_example_index": source_example_index,
            "game_index": _int_or_none(source_row.get("game_index")),
            "step": _int_or_none(source_row.get("step")),
            "tile": source_row.get("tile"),
            "objective": objective,
            "policy_action_label": str(source_row.get("policy_action_label", "")),
            "label_action_label": str(source_row.get("label_action_label") or label),
            "label_action": int(action),
        }
        rows.append(row)
    return rows


def _state_conflict_audit(
    combined: Any,
    rows: list[dict[str, Any]],
    *,
    action_labels: list[str],
    state_round_decimals: int,
    top_groups: int,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    rounded_states = np.round(combined.states, decimals=state_round_decimals)
    for index, state in enumerate(rounded_states):
        digest = hashlib.sha1(state.astype(np.float32, copy=False).tobytes()).hexdigest()[:16]
        label = _action_label(action_labels, int(combined.actions[index]))
        group = groups.setdefault(
            digest,
            {"count": 0, "label_counts": Counter(), "examples": []},
        )
        group["count"] += 1
        group["label_counts"][label] += 1
        if len(group["examples"]) < 5:
            group["examples"].append(_row_reference(rows[index]))
    conflict_groups = [
        {
            "fingerprint": fingerprint,
            "count": int(group["count"]),
            "label_counts": dict(group["label_counts"]),
            "examples": group["examples"],
        }
        for fingerprint, group in groups.items()
        if len(group["label_counts"]) > 1
    ]
    conflict_groups.sort(key=lambda item: (-int(item["count"]), item["fingerprint"]))
    return {
        "round_decimals": int(state_round_decimals),
        "groups": len(groups),
        "conflict_groups": len(conflict_groups),
        "conflict_examples": int(sum(int(group["count"]) for group in conflict_groups)),
        "top_conflicts": conflict_groups[:top_groups],
    }


def _semantic_ambiguity_audit(
    rows: list[dict[str, Any]],
    *,
    top_groups: int,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _semantic_key(row)
        label = str(row.get("label_action_label", ""))
        group = groups.setdefault(
            key,
            {"count": 0, "label_counts": Counter(), "examples": []},
        )
        group["count"] += 1
        group["label_counts"][label] += 1
        if len(group["examples"]) < 5:
            group["examples"].append(_row_reference(row))
    ambiguous = []
    for key, group in groups.items():
        if len(group["label_counts"]) <= 1:
            continue
        entropy = _label_entropy(group["label_counts"])
        ambiguous.append(
            {
                "key": key,
                "count": int(group["count"]),
                "entropy": entropy,
                "label_counts": dict(group["label_counts"]),
                "examples": group["examples"],
            }
        )
    ambiguous.sort(key=lambda item: (-float(item["entropy"]) * int(item["count"]), item["key"]))
    return {
        "groups": len(groups),
        "ambiguous_groups": len(ambiguous),
        "ambiguous_examples": int(sum(int(group["count"]) for group in ambiguous)),
        "top_ambiguous_groups": ambiguous[:top_groups],
    }


def _direction_alignment_audit(
    rows: list[dict[str, Any]],
    *,
    top_groups: int,
) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, Any]] = []
    mismatch_counts: Counter[str] = Counter()
    for row in rows:
        label = str(row.get("label_action_label", ""))
        label_direction = _horizontal_label_direction(label)
        if label_direction == 0:
            continue
        target_dx = _target_dx(row)
        if target_dx is None or target_dx == 0:
            continue
        checked += 1
        target_direction = 1 if target_dx > 0 else -1
        if label_direction != target_direction:
            mismatch_counts[label] += 1
            if len(mismatches) < top_groups:
                reference = _row_reference(row)
                reference["target_dx_tiles"] = target_dx
                mismatches.append(reference)
    return {
        "checked": checked,
        "mismatches": int(sum(mismatch_counts.values())),
        "mismatch_rate": float(sum(mismatch_counts.values()) / max(1, checked)),
        "mismatch_counts": dict(mismatch_counts),
        "examples": mismatches,
    }


def _adjacent_label_flip_audit(
    rows: list[dict[str, Any]],
    *,
    adjacent_step_window: int,
    top_groups: int,
) -> dict[str, Any]:
    by_game: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        source_index = _int_or_none(row.get("source_dataset_index"))
        game_index = _int_or_none(row.get("game_index"))
        if source_index is None or game_index is None:
            continue
        by_game.setdefault((source_index, game_index), []).append(row)

    checked = 0
    flips = 0
    flip_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for game_rows in by_game.values():
        game_rows.sort(key=lambda item: (_int_or_none(item.get("step")) or 0))
        for previous, current in zip(game_rows, game_rows[1:]):
            prev_step = _int_or_none(previous.get("step"))
            curr_step = _int_or_none(current.get("step"))
            if prev_step is None or curr_step is None:
                continue
            if curr_step - prev_step > adjacent_step_window:
                continue
            if _target_tile(previous) != _target_tile(current):
                continue
            checked += 1
            prev_label = str(previous.get("label_action_label", ""))
            curr_label = str(current.get("label_action_label", ""))
            if prev_label == curr_label:
                continue
            flips += 1
            pair = f"{prev_label}->{curr_label}"
            flip_counts[pair] += 1
            if len(examples) < top_groups:
                examples.append(
                    {
                        "from": _row_reference(previous),
                        "to": _row_reference(current),
                        "step_delta": curr_step - prev_step,
                        "pair": pair,
                    }
                )
    return {
        "checked_pairs": checked,
        "flips": flips,
        "flip_rate": float(flips / max(1, checked)),
        "flip_counts": dict(flip_counts),
        "examples": examples,
    }


def _write_audit_dataset(
    run_dir: Path,
    *,
    label: str,
    combined: Any,
    rows: list[dict[str, Any]],
    action_labels: list[str],
    audit: dict[str, Any],
) -> dict[str, Any]:
    dataset_dir = run_dir / "corrections" / f"{label}_audit"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    states_path = dataset_dir / "correction_examples.npz"
    rows_path = dataset_dir / "correction_examples.jsonl"
    summary_path = dataset_dir / "summary.json"
    np.savez_compressed(
        states_path,
        states=np.asarray(combined.states, dtype=np.float32),
        actions=np.asarray(combined.actions, dtype=np.int64),
        source_dataset_indices=np.asarray(combined.source_dataset_indices, dtype=np.int64),
        source_example_indices=np.asarray(combined.source_example_indices, dtype=np.int64),
    )
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_summaries = list(combined.source_summaries)
    kept_examples = int(len(combined.actions))
    agreements = sum(int(source.get("agreement_states", 0) or 0) for source in source_summaries)
    games_completed = sum(int(source.get("games_completed", 0) or 0) for source in source_summaries)
    summary = {
        "dataset_version": CORRECTION_DATASET_VERSION,
        "label": f"{label}_audit",
        "label_mode": "contact_label_quality_audit",
        "source_dataset_count": len(source_summaries),
        "source_datasets": source_summaries,
        "kept_examples": kept_examples,
        "states_path": str(states_path),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "states_shape": [int(item) for item in combined.states.shape],
        "actions_shape": [int(item) for item in combined.actions.shape],
        "games_completed": int(games_completed),
        "agreement_states": int(agreements),
        "disagreement_rate": float(1.0 - agreements / max(1, kept_examples)),
        "trigger_counts": _merge_counter_dicts(source_summaries, "trigger_counts")
        or {"combined": kept_examples},
        "label_action_counts": contact_action_dataset_stats(
            combined.actions,
            action_labels=action_labels,
        )["action_counts"],
        "audit_path": str(run_dir / "contact_label_audit.json"),
        "audit_summary": {
            "state_conflict_groups": audit["state_conflicts"]["conflict_groups"],
            "semantic_ambiguous_groups": audit["semantic_ambiguity"]["ambiguous_groups"],
            "direction_mismatch_rate": audit["direction_alignment"]["mismatch_rate"],
            "adjacent_flip_rate": audit["adjacent_label_flips"]["flip_rate"],
        },
    }
    write_json(summary_path, summary)
    return summary


def _stable_label_filter_indices(
    rows: list[dict[str, Any]],
    *,
    semantic_majority_threshold: float,
    adjacent_step_window: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(_semantic_key(row), []).append(index)

    group_decisions: dict[str, dict[str, Any]] = {}
    stable_groups = 0
    for key, indices in groups.items():
        counts = Counter(str(rows[index].get("label_action_label", "")) for index in indices)
        majority_label, majority_count = counts.most_common(1)[0]
        purity = majority_count / max(1, len(indices))
        stable = purity >= semantic_majority_threshold
        if stable:
            stable_groups += 1
        group_decisions[key] = {
            "majority_label": majority_label,
            "majority_count": int(majority_count),
            "purity": float(purity),
            "stable": stable,
            "label_counts": dict(counts),
        }

    flip_indices = _adjacent_label_flip_indices(
        rows,
        adjacent_step_window=adjacent_step_window,
    )
    keep_indices: list[int] = []
    drop_reasons: Counter[str] = Counter()
    for index, row in enumerate(rows):
        decision = group_decisions[_semantic_key(row)]
        label = str(row.get("label_action_label", ""))
        reasons: list[str] = []
        if not decision["stable"]:
            reasons.append("low_semantic_majority")
        elif label != decision["majority_label"]:
            reasons.append("minority_label")
        if index in flip_indices:
            reasons.append("adjacent_label_flip")
        if reasons:
            for reason in reasons:
                drop_reasons[reason] += 1
            continue
        keep_indices.append(index)

    kept_actions = Counter(str(rows[index].get("label_action_label", "")) for index in keep_indices)
    summary = {
        "source_examples": len(rows),
        "kept_examples": len(keep_indices),
        "dropped_examples": len(rows) - len(keep_indices),
        "semantic_groups": len(groups),
        "stable_semantic_groups": int(stable_groups),
        "unstable_semantic_groups": int(len(groups) - stable_groups),
        "drop_reason_counts": dict(drop_reasons),
        "kept_label_action_counts": dict(kept_actions),
    }
    return {"keep_indices": keep_indices, "summary": summary}


def _adjacent_label_flip_indices(
    rows: list[dict[str, Any]],
    *,
    adjacent_step_window: int,
) -> set[int]:
    by_game: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        source_index = _int_or_none(row.get("source_dataset_index"))
        game_index = _int_or_none(row.get("game_index"))
        if source_index is None or game_index is None:
            continue
        by_game.setdefault((source_index, game_index), []).append(row)

    flip_indices: set[int] = set()
    for game_rows in by_game.values():
        game_rows.sort(key=lambda item: (_int_or_none(item.get("step")) or 0))
        for previous, current in zip(game_rows, game_rows[1:]):
            prev_step = _int_or_none(previous.get("step"))
            curr_step = _int_or_none(current.get("step"))
            if prev_step is None or curr_step is None:
                continue
            if curr_step - prev_step > adjacent_step_window:
                continue
            if _target_tile(previous) != _target_tile(current):
                continue
            if previous.get("label_action_label") == current.get("label_action_label"):
                continue
            flip_indices.add(int(previous["combined_index"]))
            flip_indices.add(int(current["combined_index"]))
    return flip_indices


def _write_filtered_dataset(
    run_dir: Path,
    *,
    label: str,
    combined: Any,
    keep_indices: np.ndarray,
    rows: list[dict[str, Any]],
    action_labels: list[str],
    filter_summary: dict[str, Any],
) -> dict[str, Any]:
    dataset_dir = run_dir / "corrections" / f"{label}_stable"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    states_path = dataset_dir / "correction_examples.npz"
    rows_path = dataset_dir / "correction_examples.jsonl"
    summary_path = dataset_dir / "summary.json"
    filtered_states = np.asarray(combined.states[keep_indices], dtype=np.float32)
    filtered_actions = np.asarray(combined.actions[keep_indices], dtype=np.int64)
    np.savez_compressed(
        states_path,
        states=filtered_states,
        actions=filtered_actions,
        source_dataset_indices=np.zeros((len(keep_indices),), dtype=np.int64),
        source_example_indices=np.arange(len(keep_indices), dtype=np.int64),
    )
    with rows_path.open("w", encoding="utf-8") as handle:
        for new_index, row in enumerate(rows):
            row = dict(row)
            row["source_combined_index"] = row.get("combined_index")
            row["combined_index"] = new_index
            row["original_source_dataset_index"] = row.get("source_dataset_index")
            row["original_source_example_index"] = row.get("source_example_index")
            row["source_dataset_index"] = 0
            row["source_example_index"] = new_index
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    action_stats = contact_action_dataset_stats(
        filtered_actions,
        action_labels=action_labels,
    )
    source_summaries = list(combined.source_summaries)
    source_examples = int(filter_summary.get("source_examples", len(combined.actions)))
    agreements = sum(
        1
        for row in rows
        if str(row.get("policy_action_label", "")) == str(row.get("label_action_label", ""))
    )
    summary = {
        "dataset_version": CORRECTION_DATASET_VERSION,
        "label": f"{label}_stable",
        "label_mode": "stable_contact_label_filter",
        "source_dataset_count": len(source_summaries),
        "source_datasets": source_summaries,
        "source_examples": source_examples,
        "kept_examples": int(len(filtered_actions)),
        "dropped_examples": int(source_examples - len(filtered_actions)),
        "states_path": str(states_path),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "states_shape": [int(item) for item in filtered_states.shape],
        "actions_shape": [int(item) for item in filtered_actions.shape],
        "games_completed": sum(
            int(source.get("games_completed", 0) or 0) for source in source_summaries
        ),
        "agreement_states": int(agreements),
        "disagreement_rate": float(1.0 - agreements / max(1, len(filtered_actions))),
        "trigger_counts": _merge_counter_dicts(source_summaries, "trigger_counts")
        or {"combined": int(len(filtered_actions))},
        "label_action_counts": action_stats["action_counts"],
        "filter_path": str(run_dir / "contact_label_filter.json"),
        "filter_summary": filter_summary,
    }
    write_json(summary_path, summary)
    return summary


def _semantic_key(row: dict[str, Any]) -> str:
    objective = _objective(row)
    player_tile = objective.get("player_tile") or row.get("tile") or [None, None]
    target_tile = objective.get("target_tile") or [None, None]
    dx = _safe_int(target_tile, 0) - _safe_int(player_tile, 0)
    dy = _safe_int(target_tile, 1) - _safe_int(player_tile, 1)
    distance = objective.get("target_distance_tiles")
    distance_bucket = (
        round(float(distance) * 2.0) / 2.0 if isinstance(distance, int | float) else None
    )
    return "|".join(
        str(part)
        for part in (
            objective.get("target_kind", ""),
            dx,
            dy,
            distance_bucket,
            row.get("policy_action_label", ""),
        )
    )


def _row_reference(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "combined_index": row.get("combined_index"),
        "source_dataset_index": row.get("source_dataset_index"),
        "source_example_index": row.get("source_example_index"),
        "game_index": row.get("game_index"),
        "step": row.get("step"),
        "label": row.get("label_action_label"),
        "policy": row.get("policy_action_label"),
        "tile": row.get("tile"),
        "target_tile": _target_tile(row),
    }


def _target_dx(row: dict[str, Any]) -> int | None:
    objective = _objective(row)
    player_tile = objective.get("player_tile") or row.get("tile")
    target_tile = objective.get("target_tile")
    if not isinstance(player_tile, list | tuple) or not isinstance(target_tile, list | tuple):
        return None
    return _safe_int(target_tile, 0) - _safe_int(player_tile, 0)


def _target_tile(row: dict[str, Any]) -> tuple[int | None, int | None]:
    objective = _objective(row)
    target_tile = objective.get("target_tile")
    if not isinstance(target_tile, list | tuple):
        return None, None
    return _int_or_none(target_tile[0]), _int_or_none(target_tile[1])


def _objective(row: dict[str, Any]) -> dict[str, Any]:
    objective = row.get("objective")
    return objective if isinstance(objective, dict) else {}


def _horizontal_label_direction(label: str) -> int:
    if label.startswith("LEFT"):
        return -1
    if label.startswith("RIGHT"):
        return 1
    return 0


def _label_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return float(entropy)


def _merge_counter_dicts(summaries: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for summary in summaries:
        payload = summary.get(key) or {}
        if isinstance(payload, dict):
            for item_key, value in payload.items():
                counts[str(item_key)] += int(value or 0)
    return dict(counts)


def _action_label(action_labels: list[str], action: int) -> str:
    if 0 <= action < len(action_labels):
        return action_labels[action]
    return str(action)


def _safe_int(values: Any, index: int) -> int:
    if not isinstance(values, list | tuple) or index >= len(values):
        return 0
    value = _int_or_none(values[index])
    return int(value or 0)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
