"""Metric audits for Crystal Caves status-session artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

SUCCESS_END_REASONS = frozenset({"first_crystal_goal", "won", "win"})
DEFAULT_DEPTH_GUARDRAIL = 0.57


def build_eval_metric_audit(
    eval_payload: dict[str, Any],
    *,
    label: str = "",
    source_path: str = "",
    depth_guardrail: float = DEFAULT_DEPTH_GUARDRAIL,
) -> dict[str, Any]:
    """Split eval depth by success/failure and end reason."""

    rows = eval_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return {
            "has_rows": False,
            "label": label,
            "source_path": source_path,
            "depth_guardrail": depth_guardrail,
            "flags": ["no per-game rows available for metric audit"],
        }

    row_metrics = [_row_metrics(row) for row in rows if isinstance(row, dict)]
    row_metrics = [row for row in row_metrics if row["has_depth"]]
    if not row_metrics:
        return {
            "has_rows": False,
            "label": label,
            "source_path": source_path,
            "depth_guardrail": depth_guardrail,
            "flags": ["per-game rows contain no final depth values"],
        }

    success_rows = [row for row in row_metrics if row["success"]]
    non_success_rows = [row for row in row_metrics if not row["success"]]
    depth_by_end_reason = _depth_by_end_reason(row_metrics)
    mean_depth = _mean([row["depth_frac"] for row in row_metrics])
    success_depth = _mean([row["depth_frac"] for row in success_rows])
    non_success_depth = _mean([row["depth_frac"] for row in non_success_rows])
    success_rate = len(success_rows) / len(row_metrics)
    flags = _metric_audit_flags(
        mean_depth=mean_depth,
        success_depth=success_depth,
        non_success_depth=non_success_depth,
        success_count=len(success_rows),
        non_success_count=len(non_success_rows),
        success_rate=success_rate,
        depth_guardrail=depth_guardrail,
    )
    return {
        "has_rows": True,
        "label": label or str(eval_payload.get("label", "")),
        "source_path": source_path,
        "depth_guardrail": depth_guardrail,
        "games": len(row_metrics),
        "successes": len(success_rows),
        "non_successes": len(non_success_rows),
        "success_rate": success_rate,
        "mean_depth_frac": mean_depth,
        "success_depth_frac": success_depth,
        "non_success_depth_frac": non_success_depth,
        "success_depth_delta_vs_non_success": (
            success_depth - non_success_depth if success_rows and non_success_rows else None
        ),
        "mean_steps": _mean([row["steps"] for row in row_metrics]),
        "success_steps": _mean([row["steps"] for row in success_rows]),
        "non_success_steps": _mean([row["steps"] for row in non_success_rows]),
        "mean_target_distance_tiles": _mean([row["target_distance_tiles"] for row in row_metrics]),
        "success_target_distance_tiles": _mean(
            [row["target_distance_tiles"] for row in success_rows]
        ),
        "non_success_target_distance_tiles": _mean(
            [row["target_distance_tiles"] for row in non_success_rows]
        ),
        "depth_by_end_reason": depth_by_end_reason,
        "flags": flags,
    }


def metric_audit_from_artifact(
    path: Path,
    *,
    depth_guardrail: float = DEFAULT_DEPTH_GUARDRAIL,
) -> dict[str, Any]:
    """Build a metric audit from a status-session folder or eval summary."""

    payload, source_path = _load_best_eval_payload(path)
    label = str(payload.get("label") or source_path.parent.name)
    return build_eval_metric_audit(
        payload,
        label=label,
        source_path=str(source_path),
        depth_guardrail=depth_guardrail,
    )


def format_metric_audit(audit: dict[str, Any]) -> str:
    """Format a metric audit for terminal output."""

    label = str(audit.get("label") or "unknown")
    source_path = str(audit.get("source_path") or "")
    if not audit.get("has_rows"):
        flags = audit.get("flags") or []
        lines = [f"Metric audit: {label}", f"Source: {source_path}"]
        lines.extend(f"- {flag}" for flag in flags)
        return "\n".join(lines)

    lines = [
        f"Metric audit: {label}",
        f"Source: {source_path}",
        (
            f"Rows: {audit['games']} | successes: {audit['successes']}/{audit['games']} "
            f"({100 * audit['success_rate']:.1f}%)"
        ),
        (
            f"Depth: mean {100 * audit['mean_depth_frac']:.1f}% | "
            f"success {100 * audit['success_depth_frac']:.1f}% | "
            f"non-success {100 * audit['non_success_depth_frac']:.1f}%"
        ),
        (
            f"Steps: mean {audit['mean_steps']:.1f} | "
            f"success {audit['success_steps']:.1f} | "
            f"non-success {audit['non_success_steps']:.1f}"
        ),
        "By end reason:",
    ]
    for reason, metrics in sorted(
        audit["depth_by_end_reason"].items(),
        key=lambda item: (item[0] != "first_crystal_goal", item[0]),
    ):
        lines.append(
            f"- {reason}: {metrics['count']} games, "
            f"{100 * metrics['mean_depth_frac']:.1f}% depth, "
            f"{metrics['mean_steps']:.1f} steps"
        )
    flags = audit.get("flags") or []
    if flags:
        lines.append("Flags:")
        lines.extend(f"- {flag}" for flag in flags)
    return "\n".join(lines)


def metric_audit_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Crystal Caves eval metrics by outcome and end reason."
    )
    parser.add_argument("artifacts", nargs="+", help="Artifact folder or summary.json")
    parser.add_argument(
        "--depth-guardrail",
        type=float,
        default=DEFAULT_DEPTH_GUARDRAIL,
        help="Depth guardrail to test raw and outcome-conditioned depth against.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    opts = parser.parse_args(argv)

    audits = [
        metric_audit_from_artifact(Path(artifact), depth_guardrail=opts.depth_guardrail)
        for artifact in opts.artifacts
    ]
    if opts.json:
        print(json.dumps(audits, indent=2, sort_keys=True))
    else:
        print("\n\n".join(format_metric_audit(audit) for audit in audits))
    return 0 if all(audit.get("has_rows") for audit in audits) else 1


def _load_best_eval_payload(path: Path) -> tuple[dict[str, Any], Path]:
    summary_path = path / "summary.json" if path.is_dir() else path
    payload = _load_json(summary_path)
    direct_payload = _direct_eval_payload(payload)
    if direct_payload is not None:
        return direct_payload, summary_path

    root = summary_path.parent if summary_path.is_file() else path
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    if root.exists():
        for candidate_path in sorted(root.rglob("summary.json")):
            if candidate_path == summary_path:
                continue
            candidate = _direct_eval_payload(_load_json(candidate_path))
            if candidate is not None:
                candidates.append((_payload_score(candidate), candidate_path, candidate))
    if candidates:
        _, candidate_path, candidate = max(candidates, key=lambda item: item[0])
        return candidate, candidate_path
    return payload if isinstance(payload, dict) else {}, summary_path


def _direct_eval_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("rows"), list):
        return payload
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return None
    for run in runs:
        if not isinstance(run, dict):
            continue
        for key in ("selected_checkpoint_eval", "selected_source_eval", "final_eval"):
            eval_payload = run.get(key)
            if isinstance(eval_payload, dict) and isinstance(eval_payload.get("rows"), list):
                return eval_payload
    return None


def _payload_score(payload: dict[str, Any]) -> int:
    rows = payload.get("rows")
    row_count = len(rows) if isinstance(rows, list) else 0
    score = row_count
    if payload.get("wins") is not None:
        score += 10_000
    if payload.get("mean_crystal_frac") is not None:
        score += 1_000
    if payload.get("mean_depth_frac") is not None:
        score += 100
    if payload.get("end_reason_counts") is not None:
        score += 10
    return score


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    end_reason = str(row.get("end_reason") or "unknown")
    depth = _optional_float(row.get("final_depth_frac"))
    target_distance = _target_distance(row)
    return {
        "end_reason": end_reason,
        "success": bool(row.get("won")) or end_reason in SUCCESS_END_REASONS,
        "depth_frac": depth or 0.0,
        "has_depth": depth is not None,
        "steps": _float(row.get("steps")),
        "target_distance_tiles": target_distance,
    }


def _depth_by_end_reason(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["end_reason"])].append(row)
    return {
        reason: {
            "count": len(reason_rows),
            "mean_depth_frac": _mean([row["depth_frac"] for row in reason_rows]),
            "mean_steps": _mean([row["steps"] for row in reason_rows]),
            "mean_target_distance_tiles": _mean(
                [row["target_distance_tiles"] for row in reason_rows]
            ),
        }
        for reason, reason_rows in sorted(grouped.items())
    }


def _metric_audit_flags(
    *,
    mean_depth: float,
    success_depth: float,
    non_success_depth: float,
    success_count: int,
    non_success_count: int,
    success_rate: float,
    depth_guardrail: float,
) -> list[str]:
    flags: list[str] = []
    if mean_depth < depth_guardrail:
        flags.append(f"raw mean depth misses the {100 * depth_guardrail:.1f}% guardrail")
    if non_success_count and non_success_depth >= depth_guardrail:
        flags.append(f"non-success depth clears the {100 * depth_guardrail:.1f}% guardrail")
    if success_count and non_success_count and success_depth + 0.03 < non_success_depth:
        flags.append("success episodes are materially shallower than failures")
    if (
        success_rate >= 0.25
        and mean_depth < depth_guardrail
        and non_success_count
        and non_success_depth >= depth_guardrail
        and success_depth + 0.03 < non_success_depth
    ):
        flags.append(
            "raw depth may understate route depth because successes end early; compare non-success depth to baseline before promoting"
        )
    return flags


def _target_distance(row: dict[str, Any]) -> float:
    objective = row.get("final_objective")
    if isinstance(objective, dict) and objective.get("target_distance_tiles") is not None:
        return _float(objective.get("target_distance_tiles"))
    return _float(row.get("final_target_distance_tiles"))


def _mean(values: Sequence[float]) -> float:
    cleaned = [value for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return parsed if parsed is not None else 0.0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
