"""Audit R2-B regression: rollups expose a TRUE mean_crystal_frac, not the collected-≥1 rate.

The near-miss/contact-head eval payload mislabeled any_crystal_rate (fraction of games
collecting at least one crystal) as mean_crystal_frac, over-reporting collection on
multi-crystal difficulties into promotion/scorecard/reports.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cc_status.evals import (  # noqa: E402
    _mean_crystal_frac,
    first_objective_near_miss_rollup,
    trace_rollup,
)


def _rows(collected, initial=10):
    return [{"crystals_collected": c, "initial_crystals": initial, "won": False} for c in collected]


def test_mean_crystal_frac_helper_is_true_mean():
    rows = _rows([0, 2, 0, 5])  # collected /10 -> [0, .2, 0, .5]
    assert _mean_crystal_frac(rows) == pytest.approx(0.175)
    assert _mean_crystal_frac([]) == 0.0


def test_trace_rollup_distinguishes_mean_from_any_rate():
    roll = trace_rollup(_rows([0, 2, 0, 5]))
    assert roll["mean_crystal_frac"] == pytest.approx(0.175)  # true mean
    assert roll["any_crystal_rate"] == pytest.approx(0.5)  # collected-≥1 rate (distinct)


def test_near_miss_rollup_has_true_mean_crystal_frac():
    roll = first_objective_near_miss_rollup(_rows([0, 2, 0, 5]))
    assert roll["mean_crystal_frac"] == pytest.approx(0.175)
    assert roll["any_crystal_rate"] == pytest.approx(0.5)
