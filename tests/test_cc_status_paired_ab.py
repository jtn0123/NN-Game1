"""Tests for Crystal Caves paired A/B aggregation."""

import pytest

from experiments.cc_status.paired_ab import (
    aggregate_paired_ab,
    interquartile_mean,
    pair_level_rows,
    stratified_bootstrap_ci,
)


def test_interquartile_mean_trims_outer_quartiles():
    assert interquartile_mean([100.0, 1.0, 2.0, 3.0]) == pytest.approx(2.5)


def test_pipeline_mean_does_not_floor_bottom_heavy_deltas():
    # Audit B1 regression: a bottom-heavy lever delta (mostly ~0, a few small positives) has
    # a real small-POSITIVE mean, but the interquartile mean trims to the all-zero core and
    # floors to 0.0 — which erased and even sign-flipped lever verdicts on the A/B path.
    from experiments.cc_status.paired_ab import pipeline_mean

    # 6 of 8 zeros: the middle-50% core (sorted indices 2..5) is all zero, so IQM floors to
    # exactly 0.0, while the true mean is a real positive 0.01.
    deltas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04]
    assert pipeline_mean(deltas) == pytest.approx(0.01)  # honest positive (the fix)
    assert interquartile_mean(deltas) == pytest.approx(0.0)  # the bug it replaced (floored)
    assert pipeline_mean([]) == 0.0  # empty-safe


def test_pair_level_rows_uses_seed_and_level_keys():
    a_rows = [
        {"seed": 0, "level_index": 0, "level_name": "l0", "selection_score": 0.2},
        {"seed": 1, "level_index": 0, "level_name": "l0", "selection_score": 0.4},
    ]
    b_rows = [
        {"seed": 0, "level_index": 0, "level_name": "l0", "selection_score": 0.5},
        {"seed": 1, "level_index": 1, "level_name": "l1", "selection_score": 0.9},
    ]

    paired = pair_level_rows(a_rows, b_rows)

    assert paired == [
        {
            "seed": 0,
            "level_index": 0,
            "level_name": "l0",
            "a_selection_score": 0.2,
            "b_selection_score": 0.5,
            "delta_selection_score": pytest.approx(0.3),
            "a_end_reason": None,
            "b_end_reason": None,
            "a_won": False,
            "b_won": False,
        }
    ]


def test_stratified_bootstrap_ci_is_deterministic_for_seed():
    rows = [
        {"seed": 0, "selection_score": 0.1},
        {"seed": 0, "selection_score": 0.2},
        {"seed": 1, "selection_score": 0.4},
        {"seed": 1, "selection_score": 0.6},
    ]

    first = stratified_bootstrap_ci(
        rows,
        metric="selection_score",
        n_bootstrap=50,
        seed=123,
    )
    second = stratified_bootstrap_ci(
        rows,
        metric="selection_score",
        n_bootstrap=50,
        seed=123,
    )

    assert first == second
    # Audit B1: the estimator is now a PLAIN MEAN (mean([0.1,0.2,0.4,0.6]) = 0.325), not the
    # old interquartile mean (which trimmed to [0.2,0.4] -> 0.3). The "iqm" key is kept for
    # compatibility but now holds the plain mean.
    assert first["iqm"] == pytest.approx(0.325)
    assert first["ci_low"] <= first["iqm"] <= first["ci_high"]


def test_aggregate_paired_ab_reports_paired_delta_iqm():
    a_rows = [
        {"seed": 0, "level_index": 0, "selection_score": 0.2},
        {"seed": 0, "level_index": 1, "selection_score": 0.4},
        {"seed": 1, "level_index": 0, "selection_score": 0.1},
        {"seed": 1, "level_index": 1, "selection_score": 0.3},
    ]
    b_rows = [
        {"seed": 0, "level_index": 0, "selection_score": 0.3},
        {"seed": 0, "level_index": 1, "selection_score": 0.5},
        {"seed": 1, "level_index": 0, "selection_score": 0.3},
        {"seed": 1, "level_index": 1, "selection_score": 0.6},
    ]

    aggregate = aggregate_paired_ab(
        a_rows,
        b_rows,
        n_bootstrap=25,
        bootstrap_seed=0,
    )

    assert aggregate["paired_rows"] == 4
    # Audit B1: plain mean of deltas [0.1,0.1,0.2,0.3] = 0.175 (old IQM trimmed to 0.15).
    assert aggregate["paired_delta_b_minus_a"]["iqm"] == pytest.approx(0.175)
    assert aggregate["arm_b"]["iqm"] > aggregate["arm_a"]["iqm"]
