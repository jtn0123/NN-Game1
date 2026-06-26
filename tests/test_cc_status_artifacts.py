"""Tests for Crystal Caves status-session artifact validation."""

import json
from pathlib import Path

from experiments.cc_status.artifacts import validate_status_session_artifacts


def _eval_payload(games: int = 1) -> dict[str, object]:
    return {
        "wins": 0,
        "num_games": games,
        "win_rate": 0.0,
        "mean_crystal_frac": 0.0,
        "mean_depth_frac": 0.0,
        "end_reason_counts": {"timeout": games},
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_status_session_artifact_validator_accepts_selected_run(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "tutorial_demo_conservative"
    checkpoint = run_dir / "models" / "crystal_caves" / "selected.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(run_dir / "live_metrics.json", {"status": "complete"})
    (run_dir / "live_metrics.jsonl").write_text('{"status":"starting"}\n', encoding="utf-8")
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "tutorial-demo-conservative",
            "seed": 0,
            "created_at": "2026-06-24T00:00:00",
            "out_dir": str(out_dir),
            "runs": [
                {
                    "label": "tutorial_demo_conservative",
                    "episodes": 1,
                    "train_seconds": 0.1,
                    "steps_per_second": 10.0,
                    "config": {"drills": False},
                    "eval_history": [],
                    "final_eval": _eval_payload(),
                    "selected_checkpoint_eval_games": 1,
                    "selected_checkpoint_eval": _eval_payload(),
                    "selected_checkpoint_failure_diagnostics": {"rollup": {}},
                    "selected_checkpoint_near_miss_eval": {"rollup": {}},
                    "selected_checkpoint_path": str(checkpoint),
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=True)

    assert result.ok is True
    assert result.to_dict()["errors"] == []


def test_status_session_artifact_validator_accepts_interrupted_partial_run(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "contact_interleaved"
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "live_metrics.json", {"status": "interrupted"})
    (run_dir / "live_metrics.jsonl").write_text(
        '{"status":"interrupted"}\n',
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "contact-interleaved",
            "seed": 0,
            "created_at": "2026-06-25T00:00:00",
            "out_dir": str(out_dir),
            "interrupted": True,
            "runs": [
                {
                    "label": "contact_interleaved",
                    "partial": True,
                    "interrupted": True,
                    "episodes": 150,
                    "train_seconds": 12.5,
                    "steps_per_second": 1000.0,
                    "config": {},
                    "eval_history": [],
                    "source_stats": {},
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=True)

    assert result.ok is True
    assert result.to_dict()["errors"] == []
    assert "interrupted partial run" in result.to_dict()["warnings"][0]["message"]


def test_status_session_artifact_validator_rejects_missing_selected_evidence(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "tutorial_demo_conservative"
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "live_metrics.json", {"status": "complete"})
    (run_dir / "live_metrics.jsonl").write_text('{"status":"complete"}\n', encoding="utf-8")
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "tutorial-demo-conservative",
            "seed": 0,
            "created_at": "2026-06-24T00:00:00",
            "out_dir": str(out_dir),
            "runs": [
                {
                    "label": "tutorial_demo_conservative",
                    "episodes": 1,
                    "train_seconds": 0.1,
                    "steps_per_second": 10.0,
                    "config": {},
                    "eval_history": [],
                    "final_eval": _eval_payload(),
                    "selected_checkpoint_eval_games": 1,
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=True)

    assert result.ok is False
    messages = [issue.message for issue in result.errors]
    assert "run 'tutorial_demo_conservative' missing 'selected_checkpoint_eval'" in messages
    assert (
        "run 'tutorial_demo_conservative' missing 'selected_checkpoint_failure_diagnostics'"
        in messages
    )
    assert (
        "run 'tutorial_demo_conservative' missing 'selected_checkpoint_near_miss_eval'" in messages
    )


def test_status_session_artifact_validator_allows_drill_without_final_eval(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "drill_pretrain"
    run_dir.mkdir(parents=True)
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "drill",
            "seed": 0,
            "created_at": "2026-06-24T00:00:00",
            "out_dir": str(out_dir),
            "runs": [
                {
                    "label": "drill_pretrain",
                    "episodes": 1,
                    "train_seconds": 0.1,
                    "steps_per_second": 10.0,
                    "config": {"drills": True},
                    "eval_history": [],
                    "drill_eval": {"levels": []},
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=False)

    assert result.ok is True


def test_status_session_artifact_validator_accepts_correction_dataset(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "collect_corrections"
    correction_dir = run_dir / "corrections" / "collect_corrections_heldout"
    correction_dir.mkdir(parents=True)
    states_path = correction_dir / "correction_examples.npz"
    rows_path = correction_dir / "correction_examples.jsonl"
    correction_summary_path = correction_dir / "summary.json"
    states_path.write_bytes(b"npz")
    rows_path.write_text('{"step": 1}\n', encoding="utf-8")
    _write_json(correction_summary_path, {"kept_examples": 1})
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "collect-corrections",
            "seed": 0,
            "created_at": "2026-06-24T00:00:00",
            "out_dir": str(out_dir),
            "runs": [
                {
                    "label": "collect_corrections",
                    "episodes": 0,
                    "train_seconds": 0.0,
                    "steps_per_second": 0.0,
                    "config": {},
                    "eval_history": [],
                    "correction_dataset": {
                        "dataset_version": "cc_policy_corrections_v1",
                        "kept_examples": 1,
                        "states_path": str(states_path),
                        "rows_path": str(rows_path),
                        "summary_path": str(correction_summary_path),
                        "trigger_counts": {"close_zone": 1},
                        "label_action_counts": {"RIGHT": 1},
                    },
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=False)

    assert result.ok is True


def test_status_session_artifact_validator_rejects_missing_correction_arrays(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "collect_corrections"
    run_dir.mkdir(parents=True)
    rows_path = run_dir / "rows.jsonl"
    summary_path = run_dir / "summary.json"
    rows_path.write_text("", encoding="utf-8")
    _write_json(summary_path, {})
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "collect-corrections",
            "seed": 0,
            "created_at": "2026-06-24T00:00:00",
            "out_dir": str(out_dir),
            "runs": [
                {
                    "label": "collect_corrections",
                    "episodes": 0,
                    "train_seconds": 0.0,
                    "steps_per_second": 0.0,
                    "config": {},
                    "eval_history": [],
                    "correction_dataset": {
                        "dataset_version": "cc_policy_corrections_v1",
                        "kept_examples": 0,
                        "states_path": str(run_dir / "missing.npz"),
                        "rows_path": str(rows_path),
                        "summary_path": str(summary_path),
                        "trigger_counts": {},
                        "label_action_counts": {},
                    },
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=False)

    assert result.ok is False
    assert "missing correction artifact 'states_path'" in [issue.message for issue in result.errors]


def test_status_session_artifact_validator_accepts_correction_training(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "correction_finetune"
    correction_dataset = tmp_path / "correction_examples.npz"
    correction_dataset.write_bytes(b"npz")
    run_dir.mkdir(parents=True)
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "correction-finetune",
            "seed": 0,
            "created_at": "2026-06-24T00:00:00",
            "out_dir": str(out_dir),
            "runs": [
                {
                    "label": "correction_finetune",
                    "episodes": 1,
                    "train_seconds": 0.1,
                    "steps_per_second": 10.0,
                    "config": {},
                    "eval_history": [],
                    "final_eval": _eval_payload(),
                    "avg_correction_action_loss_100": 0.0,
                    "avg_correction_action_accuracy_100": 1.0,
                    "correction_action_samples_100": 1,
                    "correction_training": {
                        "dataset_path": str(correction_dataset),
                        "dataset_states": 2,
                        "state_size": 295,
                        "weight": 0.02,
                        "margin": 0.6,
                        "batch_size": 2,
                        "correction_action_transitions": 2,
                    },
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=False)

    assert result.ok is True


def test_status_session_artifact_validator_rejects_unsampled_correction_training(tmp_path):
    out_dir = tmp_path / "session"
    run_dir = out_dir / "correction_finetune"
    correction_dataset = tmp_path / "correction_examples.npz"
    correction_dataset.write_bytes(b"npz")
    run_dir.mkdir(parents=True)
    (out_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(
        out_dir / "summary.json",
        {
            "run_id": "run",
            "mode": "correction-finetune",
            "seed": 0,
            "created_at": "2026-06-24T00:00:00",
            "out_dir": str(out_dir),
            "runs": [
                {
                    "label": "correction_finetune",
                    "episodes": 1,
                    "train_seconds": 0.1,
                    "steps_per_second": 10.0,
                    "config": {},
                    "eval_history": [],
                    "final_eval": _eval_payload(),
                    "avg_correction_action_loss_100": 0.0,
                    "avg_correction_action_accuracy_100": 0.0,
                    "correction_action_samples_100": 0,
                    "correction_training": {
                        "dataset_path": str(correction_dataset),
                        "dataset_states": 0,
                        "state_size": 295,
                        "weight": 0.02,
                        "margin": 0.6,
                        "batch_size": 2,
                        "correction_action_transitions": 0,
                    },
                }
            ],
        },
    )

    result = validate_status_session_artifacts(out_dir, require_live_metrics=False)

    assert result.ok is False
    messages = [issue.message for issue in result.errors]
    assert (
        "run 'correction_finetune' correction_training dataset_states must be positive" in messages
    )
    assert (
        "run 'correction_finetune' correction_training correction_action_transitions must be positive"
        in messages
    )
    assert (
        "run 'correction_finetune' correction_training correction_action_samples_100 must be positive"
        in messages
    )
