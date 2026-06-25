"""Tests for Crystal Caves status-session promotion gates."""

import json
from pathlib import Path

from experiments.cc_status.promotion import (
    B3S_PROMOTED_BASELINE,
    DECISION_HOLD,
    DECISION_PROMOTE,
    DECISION_REGRESS,
    PromotionSnapshot,
    compare_promotion_candidate,
    gate_candidate,
    promotion_snapshot_from_artifact,
)


def _snapshot(**overrides: object) -> PromotionSnapshot:
    data = {
        "label": "candidate",
        "artifact": "test-artifact",
        "wins": 10,
        "games": 30,
        "win_rate": 10 / 30,
        "crystal_frac": 0.333,
        "depth_frac": 0.605,
        "near_miss_rate_3": 0.600,
        "near_miss_rate_1_5": 0.400,
        "mean_min_target_distance_tiles": 3.51,
        "close_zone_jump_rate": 0.000,
        "stuck_after_close_rate": 0.200,
        "loop_after_close_rate": 0.333,
    }
    data.update(overrides)
    return PromotionSnapshot(**data)


def _write_summary(path: Path, eval_payload: dict[str, object]) -> None:
    path.mkdir(parents=True)
    payload = {
        "mode": "tutorial-demo-conservative",
        "runs": [
            {
                "label": "candidate_run",
                "selected_checkpoint_eval": eval_payload,
                "selected_checkpoint_near_miss_eval": {
                    "rollup": {
                        "near_miss_rate_3": 0.7,
                        "near_miss_rate_1_5": 0.5,
                        "mean_min_target_distance_tiles": 3.0,
                        "mean_close_zone_jump_rate": 0.2,
                        "stuck_after_close_rate": 0.1,
                        "loop_after_close_rate": 0.2,
                    }
                },
            }
        ],
    }
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_promotion_gate_regresses_when_selected_win_rate_trails_baseline():
    result = gate_candidate(_snapshot(wins=9, win_rate=9 / 30))

    assert result.decision == DECISION_REGRESS
    assert "selected win rate trails baseline" in result.reasons[0]


def test_promotion_gate_holds_when_selected_sample_is_too_small():
    result = gate_candidate(_snapshot(wins=1, games=1, win_rate=1.0))

    assert result.decision == DECISION_HOLD
    assert "selected eval sample is too small" in result.reasons[0]


def test_promotion_gate_holds_promising_candidate_until_validation_runs():
    result = gate_candidate(_snapshot(wins=11, win_rate=11 / 30))

    assert result.decision == DECISION_HOLD
    assert "needs expanded validation" in result.reasons[0]


def test_promotion_gate_regresses_when_expanded_validation_trails_baseline():
    result = gate_candidate(
        _snapshot(
            wins=11,
            win_rate=11 / 30,
            validation_wins=18,
            validation_games=60,
            validation_win_rate=18 / 60,
            validation_crystal_frac=0.34,
            validation_depth_frac=0.61,
        )
    )

    assert result.decision == DECISION_REGRESS
    assert "expanded validation win rate trails baseline" in result.reasons[0]


def test_promotion_gate_holds_when_expanded_validation_sample_is_too_small():
    result = gate_candidate(
        _snapshot(
            wins=11,
            win_rate=11 / 30,
            validation_wins=10,
            validation_games=30,
            validation_win_rate=10 / 30,
            validation_crystal_frac=0.34,
            validation_depth_frac=0.61,
        )
    )

    assert result.decision == DECISION_HOLD
    assert "expanded validation sample is too small" in result.reasons[0]


def test_promotion_gate_promotes_when_selected_and_validation_clear_baseline():
    result = gate_candidate(
        _snapshot(
            wins=11,
            win_rate=11 / 30,
            crystal_frac=0.36,
            depth_frac=0.62,
            validation_wins=20,
            validation_games=60,
            validation_win_rate=20 / 60,
            validation_crystal_frac=0.35,
            validation_depth_frac=0.62,
        )
    )

    assert result.decision == DECISION_PROMOTE
    assert result.baseline == B3S_PROMOTED_BASELINE


def test_promotion_gate_regresses_tied_selected_candidate_without_support_metrics():
    result = gate_candidate(_snapshot())

    assert result.decision == DECISION_REGRESS
    assert "fewer than two support metrics improved" in result.reasons[0]


def test_promotion_gate_extracts_selected_checkpoint_artifact(tmp_path):
    artifact = tmp_path / "candidate"
    _write_summary(
        artifact,
        {
            "wins": 11,
            "num_games": 30,
            "mean_crystal_frac": 0.36,
            "mean_depth_frac": 0.62,
        },
    )

    snapshot = promotion_snapshot_from_artifact(artifact)

    assert snapshot.label == "candidate_run"
    assert snapshot.wins == 11
    assert snapshot.games == 30
    assert snapshot.win_rate == 11 / 30
    assert snapshot.near_miss_rate_3 == 0.7
    assert snapshot.close_zone_jump_rate == 0.2


def test_compare_promotion_candidate_attaches_validation_artifact(tmp_path):
    candidate = tmp_path / "candidate"
    validation = tmp_path / "validation"
    _write_summary(
        candidate,
        {
            "wins": 11,
            "num_games": 30,
            "mean_crystal_frac": 0.36,
            "mean_depth_frac": 0.62,
        },
    )
    _write_summary(
        validation,
        {
            "wins": 20,
            "num_games": 60,
            "mean_crystal_frac": 0.35,
            "mean_depth_frac": 0.62,
        },
    )

    result = compare_promotion_candidate(candidate, validation_artifact=validation)

    assert result.decision == DECISION_PROMOTE
    assert result.candidate.validation_wins == 20
    assert result.candidate.validation_games == 60
