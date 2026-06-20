"""Tests for the Crystal Caves staged curriculum runner."""

from src.app.crystal_curriculum import (
    DEFAULT_CRYSTAL_CURRICULUM,
    _default_warm_start_checkpoint,
    planned_stage_episodes,
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


def test_default_warm_start_prefers_existing_eval_best(tmp_path):
    game_dir = tmp_path / "crystal_caves"
    game_dir.mkdir()
    best = game_dir / "crystal_caves_best.pth"
    eval_best = game_dir / "crystal_caves_eval_best.pth"
    best.write_bytes(b"best")
    eval_best.write_bytes(b"eval")

    assert _default_warm_start_checkpoint(str(tmp_path), "crystal_caves") == str(eval_best)
