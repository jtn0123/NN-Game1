"""Route/contact scorecard tests for Crystal Caves status sessions."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cc_status_session import (  # noqa: E402
    build_route_contact_scorecard,
    route_contact_source_snapshot_score,
)


def test_route_contact_scorecard_extracts_selected_checkpoint_metrics():
    run = {
        "config": {"first_crystal_goal": True},
        "selected_checkpoint_eval": {
            "wins": 10,
            "num_games": 30,
            "win_rate": 10 / 30,
            "mean_crystal_frac": 10 / 30,
            "mean_depth_frac": 0.605,
            "end_reason_counts": {"first_crystal_goal": 10, "stalled": 15, "timeout": 5},
        },
        "selected_checkpoint_near_miss_eval": {
            "rollup": {
                "near_miss_rate_3": 0.6,
                "near_miss_rate_1_5": 0.4,
                "mean_min_target_distance_tiles": 3.5,
                "mean_close_zone_jump_rate": 0.0,
                "stuck_after_close_rate": 0.2,
                "loop_after_close_rate": 1 / 3,
            }
        },
    }

    scorecard = build_route_contact_scorecard(run)

    assert scorecard["eval_source"] == "selected_checkpoint_eval"
    assert scorecard["verdict"] == "contact regression"
    assert scorecard["metrics"]["first_crystal_rate"] == pytest.approx(10 / 30)
    assert scorecard["metrics"]["stall_rate"] == pytest.approx(0.5)
    assert scorecard["score"] == pytest.approx(
        3.0 * (10 / 30) + 1.5 * (10 / 30) + 0.605 + 0.5 * 0.6 - (1 / 3) - 0.5 * 0.5
    )


def test_source_snapshot_score_is_route_contact_composite():
    shallow = {
        "episode": 50,
        "source_eval": {
            "wins": 2,
            "num_games": 16,
            "win_rate": 0.125,
            "mean_crystal_frac": 0.125,
            "mean_depth_frac": 0.20,
            "end_reason_counts": {"stalled": 8, "timeout": 8},
        },
    }
    deeper = {
        "episode": 100,
        "source_eval": {
            "wins": 2,
            "num_games": 16,
            "win_rate": 0.125,
            "mean_crystal_frac": 0.125,
            "mean_depth_frac": 0.45,
            "end_reason_counts": {"stalled": 2, "timeout": 14},
        },
    }

    assert route_contact_source_snapshot_score(deeper) > route_contact_source_snapshot_score(
        shallow
    )
