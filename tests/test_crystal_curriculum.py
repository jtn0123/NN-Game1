"""Tests for the Crystal Caves staged curriculum runner."""

from src.ai.evaluator import EvalResults
from src.app.crystal_curriculum import (
    DEFAULT_CRYSTAL_CURRICULUM,
    StageGateResult,
    evaluate_stage_gate,
    planned_stage_episodes,
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

    assert budgets == [300, 750, 900, 900, 1200]


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

    assert budgets == [25, 25, 25, 25, 25]


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
    result = evaluate_stage_gate(
        DEFAULT_CRYSTAL_CURRICULUM[1],
        _eval_result(crystal=0.8, switch=0.1, wins=0.0, end_reasons={"timeout": 9, "won": 1}),
    )

    assert result.ready is False
    assert "eval switch" in result.detail
    assert "eval wins" in result.detail
