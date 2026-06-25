"""Validation helpers for status-session artifact folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactValidationIssue:
    severity: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class ArtifactValidationResult:
    out_dir: str
    errors: tuple[ArtifactValidationIssue, ...]
    warnings: tuple[ArtifactValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def validate_status_session_artifacts(
    out_dir: Path,
    *,
    require_live_metrics: bool,
) -> ArtifactValidationResult:
    """Validate that a status-session folder contains comparable run evidence."""
    errors: list[ArtifactValidationIssue] = []
    warnings: list[ArtifactValidationIssue] = []
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    _require_file(summary_path, errors, "missing root summary.json")
    _require_file(report_path, errors, "missing root report.md")
    if errors:
        return ArtifactValidationResult(str(out_dir), tuple(errors), tuple(warnings))

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(_issue(summary_path, f"summary.json is not valid JSON: {exc}"))
        return ArtifactValidationResult(str(out_dir), tuple(errors), tuple(warnings))

    if not isinstance(payload, dict):
        errors.append(_issue(summary_path, "summary.json root must be an object"))
        return ArtifactValidationResult(str(out_dir), tuple(errors), tuple(warnings))

    for key in ("run_id", "mode", "seed", "created_at", "out_dir"):
        if key not in payload:
            errors.append(_issue(summary_path, f"summary.json missing '{key}'"))
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append(_issue(summary_path, "summary.json must contain at least one run"))
        return ArtifactValidationResult(str(out_dir), tuple(errors), tuple(warnings))

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(_issue(summary_path, f"runs[{index}] must be an object"))
            continue
        _validate_run(out_dir, run, index, require_live_metrics, errors, warnings)

    return ArtifactValidationResult(str(out_dir), tuple(errors), tuple(warnings))


def _validate_run(
    out_dir: Path,
    run: dict[str, Any],
    index: int,
    require_live_metrics: bool,
    errors: list[ArtifactValidationIssue],
    warnings: list[ArtifactValidationIssue],
) -> None:
    label = run.get("label")
    run_path = out_dir / str(label or f"run_{index}")
    if not isinstance(label, str) or not label:
        errors.append(_issue(out_dir / "summary.json", f"runs[{index}] missing label"))
        return

    _require_file(run_path, errors, "missing per-run artifact directory", expect_dir=True)
    for key in ("episodes", "train_seconds", "steps_per_second", "config", "eval_history"):
        if key not in run:
            errors.append(_issue(out_dir / "summary.json", f"run '{label}' missing '{key}'"))

    if require_live_metrics:
        _require_file(run_path / "live_metrics.json", errors, "missing live_metrics.json")
        _require_file(run_path / "live_metrics.jsonl", errors, "missing live_metrics.jsonl")

    correction_training = run.get("correction_training")
    if correction_training is not None:
        _validate_correction_training(
            run_path,
            run,
            correction_training,
            f"run '{label}' correction_training",
            errors,
        )

    correction_dataset = run.get("correction_dataset")
    if correction_dataset is not None:
        _validate_correction_dataset(
            run_path,
            correction_dataset,
            f"run '{label}' correction_dataset",
            errors,
        )
    else:
        config_obj = run.get("config")
        config: dict[str, Any] = config_obj if isinstance(config_obj, dict) else {}
        if config.get("drills"):
            if "drill_eval" not in run:
                errors.append(_issue(out_dir / "summary.json", f"run '{label}' missing drill_eval"))
        elif "final_eval" not in run:
            errors.append(_issue(out_dir / "summary.json", f"run '{label}' missing final_eval"))
        else:
            _validate_eval_payload(run_path, run["final_eval"], f"run '{label}' final_eval", errors)

    selected_games = int(run.get("selected_checkpoint_eval_games", 0) or 0)
    if selected_games > 0:
        for key in (
            "selected_checkpoint_eval",
            "selected_checkpoint_failure_diagnostics",
            "selected_checkpoint_near_miss_eval",
        ):
            if key not in run:
                errors.append(_issue(out_dir / "summary.json", f"run '{label}' missing '{key}'"))
        if "selected_checkpoint_eval" in run:
            _validate_eval_payload(
                run_path,
                run["selected_checkpoint_eval"],
                f"run '{label}' selected_checkpoint_eval",
                errors,
            )
        checkpoint_path = run.get("selected_checkpoint_path")
        if checkpoint_path:
            _require_file(Path(str(checkpoint_path)), errors, "missing selected checkpoint")
        else:
            warnings.append(
                _issue(
                    out_dir / "summary.json",
                    f"run '{label}' has selected eval but no selected checkpoint path",
                    severity="warning",
                )
            )


def _validate_correction_dataset(
    path: Path,
    payload: Any,
    label: str,
    errors: list[ArtifactValidationIssue],
) -> None:
    if not isinstance(payload, dict):
        errors.append(_issue(path, f"{label} must be an object"))
        return
    for key in (
        "dataset_version",
        "kept_examples",
        "states_path",
        "rows_path",
        "summary_path",
        "trigger_counts",
        "label_action_counts",
    ):
        if key not in payload:
            errors.append(_issue(path, f"{label} missing '{key}'"))
    for key in ("states_path", "rows_path", "summary_path"):
        value = payload.get(key)
        if value:
            _require_file(Path(str(value)), errors, f"missing correction artifact '{key}'")


def _validate_correction_training(
    path: Path,
    run: dict[str, Any],
    payload: Any,
    label: str,
    errors: list[ArtifactValidationIssue],
) -> None:
    if not isinstance(payload, dict):
        errors.append(_issue(path, f"{label} must be an object"))
        return
    for key in (
        "dataset_path",
        "dataset_states",
        "state_size",
        "weight",
        "margin",
        "batch_size",
        "correction_action_transitions",
    ):
        if key not in payload:
            errors.append(_issue(path, f"{label} missing '{key}'"))

    dataset_path = payload.get("dataset_path")
    if dataset_path:
        _require_file(Path(str(dataset_path)), errors, "missing correction training dataset")

    _require_positive_number(
        path, payload, "dataset_states", f"{label} dataset_states must be positive", errors
    )
    _require_positive_number(
        path, payload, "state_size", f"{label} state_size must be positive", errors
    )
    _require_positive_number(path, payload, "weight", f"{label} weight must be positive", errors)
    _require_nonnegative_number(
        path, payload, "margin", f"{label} margin must be non-negative", errors
    )
    _require_positive_number(
        path, payload, "batch_size", f"{label} batch_size must be positive", errors
    )
    _require_positive_number(
        path,
        payload,
        "correction_action_transitions",
        f"{label} correction_action_transitions must be positive",
        errors,
    )

    for key in (
        "avg_correction_action_loss_100",
        "avg_correction_action_accuracy_100",
        "correction_action_samples_100",
    ):
        if key not in run:
            errors.append(_issue(path, f"{label} run missing '{key}'"))
    _require_nonnegative_number(
        path,
        run,
        "avg_correction_action_loss_100",
        f"{label} avg_correction_action_loss_100 must be non-negative",
        errors,
    )
    _require_fraction(
        path,
        run,
        "avg_correction_action_accuracy_100",
        f"{label} avg_correction_action_accuracy_100 must be between 0 and 1",
        errors,
    )
    _require_positive_number(
        path,
        run,
        "correction_action_samples_100",
        f"{label} correction_action_samples_100 must be positive",
        errors,
    )


def _validate_eval_payload(
    path: Path,
    payload: Any,
    label: str,
    errors: list[ArtifactValidationIssue],
) -> None:
    if not isinstance(payload, dict):
        errors.append(_issue(path, f"{label} must be an object"))
        return
    for key in (
        "wins",
        "num_games",
        "win_rate",
        "mean_crystal_frac",
        "mean_depth_frac",
        "end_reason_counts",
    ):
        if key not in payload:
            errors.append(_issue(path, f"{label} missing '{key}'"))


def _require_file(
    path: Path,
    errors: list[ArtifactValidationIssue],
    message: str,
    *,
    expect_dir: bool = False,
) -> None:
    exists = path.is_dir() if expect_dir else path.is_file()
    if not exists:
        errors.append(_issue(path, message))


def _require_positive_number(
    path: Path,
    payload: dict[str, Any],
    key: str,
    message: str,
    errors: list[ArtifactValidationIssue],
) -> None:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        errors.append(_issue(path, message))


def _require_nonnegative_number(
    path: Path,
    payload: dict[str, Any],
    key: str,
    message: str,
    errors: list[ArtifactValidationIssue],
) -> None:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        errors.append(_issue(path, message))


def _require_fraction(
    path: Path,
    payload: dict[str, Any],
    key: str,
    message: str,
    errors: list[ArtifactValidationIssue],
) -> None:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        errors.append(_issue(path, message))


def _issue(path: Path, message: str, *, severity: str = "error") -> ArtifactValidationIssue:
    return ArtifactValidationIssue(severity=severity, path=str(path), message=message)
