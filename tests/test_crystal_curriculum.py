"""Tests for the Crystal Caves staged curriculum runner."""

from types import SimpleNamespace

import pytest

from src.ai.evaluator import EvalResults
from src.app.crystal_curriculum import (
    DEFAULT_CRYSTAL_CURRICULUM,
    StageGateResult,
    _snapshot_stage_eval_best,
    evaluate_stage_gate,
    planned_stage_episodes,
    stage_epsilon_decay,
)


def _eval_result(
    *,
    crystal: float,
    switch: float = 0.0,
    wins: float = 0.0,
    end_reasons: dict[str, int] | None = None,
) -> EvalResults:
    return EvalResults(
        timestamp="now",
        episode=1,
        num_games=10,
        mean_score=100.0,
        median_score=100.0,
        std_score=0.0,
        min_score=0,
        max_score=100,
        q25_score=0.0,
        q75_score=100.0,
        mean_level=1.0,
        max_level=1,
        level_distribution={1: 10},
        wins=int(wins * 10),
        win_rate=wins,
        mean_steps=100.0,
        max_steps=100,
        mean_crystal_frac=crystal,
        mean_switch_rate=switch,
        mean_depth_frac=0.5,
        end_reason_counts=end_reasons or {"timeout": 5, "won": 5},
    )


def test_planned_stage_episodes_use_defaults_without_total_budget():
    budgets = planned_stage_episodes(
        DEFAULT_CRYSTAL_CURRICULUM,
        total_budget=None,
        per_stage_override=None,
    )

    assert budgets == [300, 500, 750, 900, 900, 1200]


def test_planned_stage_episodes_spread_total_budget_across_stage_weights():
    budgets = planned_stage_episodes(
        DEFAULT_CRYSTAL_CURRICULUM,
        total_budget=2000,
        per_stage_override=None,
    )

    assert len(budgets) == len(DEFAULT_CRYSTAL_CURRICULUM)
    assert sum(budgets) == 2000
    assert budgets[0] < budgets[-1]


def test_planned_stage_episodes_allow_equal_stage_override():
    budgets = planned_stage_episodes(
        DEFAULT_CRYSTAL_CURRICULUM,
        total_budget=2000,
        per_stage_override=25,
    )

    assert budgets == [25, 25, 25, 25, 25, 25]


def test_tutorial_gate_requires_crystal_reliability():
    result = evaluate_stage_gate(
        DEFAULT_CRYSTAL_CURRICULUM[0],
        _eval_result(crystal=0.25, wins=0.0, end_reasons={"timeout": 10}),
    )

    assert isinstance(result, StageGateResult)
    assert result.ready is False
    assert "eval crystals" in result.detail


def test_tutorial_gate_passes_with_crystals_and_some_success_signal():
    result = evaluate_stage_gate(
        DEFAULT_CRYSTAL_CURRICULUM[0],
        _eval_result(crystal=0.9, wins=0.3, end_reasons={"timeout": 7, "won": 3}),
    )

    assert result.ready is True
    assert result.status == "ready"


def test_easy_platform_gate_requires_switch_and_wins():
    easy_platform = next(s for s in DEFAULT_CRYSTAL_CURRICULUM if s.stage_id == "easy_platform")
    result = evaluate_stage_gate(
        easy_platform,
        _eval_result(crystal=0.8, switch=0.1, wins=0.0, end_reasons={"timeout": 9, "won": 1}),
    )

    assert result.ready is False
    assert "eval switch" in result.detail
    assert "eval wins" in result.detail


def test_stage_epsilon_decay_anneals_floor_to_end_over_budget():
    decay = stage_epsilon_decay(0.35, 300, 0.105)
    assert 0.0 < decay < 1.0

    epsilon = 0.35
    for _ in range(300):
        epsilon *= decay
    assert epsilon == pytest.approx(0.105, rel=1e-6)


def test_stage_epsilon_decay_returns_no_decay_for_degenerate_inputs():
    assert stage_epsilon_decay(0.35, 0, 0.1) == 1.0  # no budget
    assert stage_epsilon_decay(0.1, 300, 0.2) == 1.0  # end >= start
    assert stage_epsilon_decay(0.0, 300, 0.1) == 1.0  # non-positive start


def test_snapshot_stage_eval_best_copies_held_out_best(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    eval_best = model_dir / "crystal_caves_eval_best.pth"
    eval_best.write_bytes(b"held-out-best")
    stage = DEFAULT_CRYSTAL_CURRICULUM[0]
    config = SimpleNamespace(GAME_MODEL_DIR=str(model_dir), GAME_NAME="crystal_caves")

    snapshot_path = _snapshot_stage_eval_best(config, stage, 1)

    assert snapshot_path is not None
    assert snapshot_path.endswith(f"stage01_{stage.stage_id}_eval_best.pth")
    assert (model_dir / f"crystal_caves_stage01_{stage.stage_id}_eval_best.pth").read_bytes() == (
        b"held-out-best"
    )


def test_snapshot_stage_eval_best_does_not_fall_back_to_score_best(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "crystal_caves_best.pth").write_bytes(b"training-score-best")
    config = SimpleNamespace(GAME_MODEL_DIR=str(model_dir), GAME_NAME="crystal_caves")

    snapshot_path = _snapshot_stage_eval_best(config, DEFAULT_CRYSTAL_CURRICULUM[0], 1)

    assert snapshot_path is None
