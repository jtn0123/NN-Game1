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
    assert first["iqm"] == pytest.approx(0.3)
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
    assert aggregate["paired_delta_b_minus_a"]["iqm"] == pytest.approx(0.15)
    assert aggregate["arm_b"]["iqm"] > aggregate["arm_a"]["iqm"]
