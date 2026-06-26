import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cc_status_session import (  # noqa: E402
    build_eval_metric_audit,
    format_metric_audit,
    metric_audit_from_artifact,
)


def test_metric_audit_flags_early_success_depth_pressure():
    audit = build_eval_metric_audit(
        {
            "label": "candidate",
            "rows": [
                _row("first_crystal_goal", depth=0.30, steps=120, won=True),
                _row("first_crystal_goal", depth=0.34, steps=130, won=True),
                _row("stalled", depth=0.70, steps=1000),
                _row("timeout", depth=0.76, steps=3000),
            ],
        },
        depth_guardrail=0.57,
    )

    assert audit["has_rows"] is True
    assert audit["success_rate"] == pytest.approx(0.5)
    assert audit["mean_depth_frac"] == pytest.approx(0.525)
    assert audit["success_depth_frac"] == pytest.approx(0.32)
    assert audit["non_success_depth_frac"] == pytest.approx(0.73)
    assert "raw mean depth misses the 57.0% guardrail" in audit["flags"]
    assert "non-success depth clears the 57.0% guardrail" in audit["flags"]
    assert (
        "raw depth may understate route depth because successes end early; compare non-success depth to baseline before promoting"
        in audit["flags"]
    )


def test_metric_audit_artifact_finds_nested_eval_summary(tmp_path):
    root = tmp_path / "run"
    nested = root / "eval" / "final"
    nested.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps({"runs": [{"label": "root", "final_eval": {"wins": 1}}]}),
        encoding="utf-8",
    )
    nested_summary = nested / "summary.json"
    nested_summary.write_text(
        json.dumps(
            {
                "label": "nested_eval",
                "wins": 1,
                "mean_crystal_frac": 0.5,
                "mean_depth_frac": 0.55,
                "end_reason_counts": {"first_crystal_goal": 1, "stalled": 1},
                "rows": [
                    _row("first_crystal_goal", depth=0.40, steps=100, won=True),
                    _row("stalled", depth=0.70, steps=1000),
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = metric_audit_from_artifact(root)

    assert audit["label"] == "nested_eval"
    assert audit["source_path"] == str(nested_summary)
    assert audit["successes"] == 1
    assert audit["non_success_depth_frac"] == pytest.approx(0.70)


def test_format_metric_audit_includes_outcome_split():
    audit = build_eval_metric_audit(
        {
            "label": "candidate",
            "rows": [
                _row("first_crystal_goal", depth=0.30, steps=120, won=True),
                _row("stalled", depth=0.70, steps=1000),
            ],
        }
    )

    text = format_metric_audit(audit)

    assert "Metric audit: candidate" in text
    assert "success 30.0%" in text
    assert "non-success 70.0%" in text
    assert "- stalled: 1 games" in text


def _row(reason: str, *, depth: float, steps: int, won: bool = False) -> dict[str, object]:
    return {
        "end_reason": reason,
        "final_depth_frac": depth,
        "steps": steps,
        "won": won,
        "final_objective": {"target_distance_tiles": 3.0},
    }
