"""Regression tests for the diagnose_gap metric-aggregation fixes.

Each test is constructed so it FAILS on the pre-fix code and PASSES on the fix,
i.e. it actually validates the correction rather than just exercising it.

Bugs fixed (RUN-16 audit):
1. crystal_frac was aggregated as mean(bool(value)) — a "collected >=1" rate — instead
   of a true mean fraction.
2. The continuous surrogates used an interquartile mean that discarded the outer 50%
   and floored bottom-heavy distributions; now a plain mean.
3. _target_distance_progress used the best-ever (min) distance, which saturated to ~1.0
   the instant the agent touched its first objective; now the final-step distance.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cc_status.diagnose_gap import _aggregate, _print_report  # noqa: E402
from experiments.cc_status.paired_ab import _target_distance_progress  # noqa: E402


def _metrics(won, crystal):
    return {
        "won": won,
        "crystal_frac": crystal,
        "depth_frac": 0.3,
        "target_distance_progress": 0.4,
        "exit_unlocked_rate": 0.0,
        "selection_score": 0.1,
    }


def _memorisation_summary(train_split_is_holdout):
    # Train looks solved, held-out fails -> the heuristic would normally say MEMORISATION.
    train, test = _metrics(0.5, 0.6), _metrics(0.0, 0.0)
    return {
        "difficulty": "easy",
        "seeds": [0],
        "episodes": 100,
        "games": 10,
        "train": train,
        "test": test,
        "gap_train_minus_test": {m: round(train[m] - test[m], 4) for m in train},
        "train_split_is_holdout": train_split_is_holdout,
    }


def test_b3_regenerate_suppresses_memorisation_verdict(capsys):
    # Audit B3: under regenerate the "train" split is a 2nd held-out set, so the
    # memorisation/generalisation verdict must NOT be emitted.
    _print_report(_memorisation_summary(train_split_is_holdout=True))
    out = capsys.readouterr().out
    assert "DO NOT APPLY" in out
    assert "MEMORISATION" not in out


def test_b3_normal_run_still_emits_memorisation_verdict(capsys):
    # Control: the same train-high/test-low shape on a NON-regenerate run still fires it.
    _print_report(_memorisation_summary(train_split_is_holdout=False))
    out = capsys.readouterr().out
    assert "MEMORISATION" in out


def _row(crystal_frac=0.0, depth=0.0, target=0.0, selection=0.0, won=False, exit_unlocked=False):
    return {
        "won": won,
        "exit_unlocked_rate": 1.0 if exit_unlocked else 0.0,
        "crystal_frac": crystal_frac,
        "depth_frac": depth,
        "target_distance_progress": target,
        "selection_score": selection,
    }


def test_crystal_frac_is_true_mean_not_collected_any_rate():
    # 3 of 5 levels collected >=1 crystal (the old bool rate = 0.6); the true mean of the
    # fractions is (0 + 0 + 0.25 + 0.5 + 1.0) / 5 = 0.35.
    rows = [_row(crystal_frac=f) for f in (0.0, 0.0, 0.25, 0.5, 1.0)]
    agg = _aggregate(rows)
    assert agg["crystal_frac"] == pytest.approx(0.35)  # pre-fix: 0.6 (mean of bools)
    assert agg["crystal_any_rate"] == pytest.approx(0.6)  # the collected->=1 rate, preserved


def test_surrogates_use_plain_mean_not_interquartile():
    # 8 levels, one at 1.0 and the rest 0.0. Plain mean = 0.125. The old interquartile mean
    # keeps only the middle 4 sorted values [0,0,0,0] -> 0.0 (floored), losing the signal.
    rows = [_row(depth=v) for v in (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
    agg = _aggregate(rows)
    assert agg["depth_frac"] == pytest.approx(0.125)  # pre-fix (IQM): 0.0


def test_target_distance_progress_uses_final_not_best():
    # Reaches its first objective (distance 0), the target then switches and the agent ends
    # FAR from the new objective. Best-ever (min) said 1.0 (fully solved); final says ~0.
    reached_then_died_far = [10.0, 4.0, 0.0, 14.0, 13.0]
    assert _target_distance_progress(reached_then_died_far) == pytest.approx(0.0)  # pre-fix: 1.0
    # Sanity: genuinely ending on the objective still scores 1.0.
    assert _target_distance_progress([10.0, 6.0, 2.0, 0.0]) == pytest.approx(1.0)
    # And ending halfway in scores the honest fraction.
    assert _target_distance_progress([10.0, 5.0]) == pytest.approx(0.5)


def test_aggregate_empty_rows_safe():
    agg = _aggregate([])
    assert agg["crystal_frac"] == 0.0
    assert agg["depth_frac"] == 0.0


def test_b5_report_labels_final_vs_best_checkpoint(capsys):
    # Audit B5: when a best checkpoint exists, the report must flag that the table/GAP are
    # the FINAL net (which can disagree with the best-checkpoint verdict).
    summary = _memorisation_summary(train_split_is_holdout=False)
    summary["best"] = {"episode": 500, "train": _metrics(0.6, 0.7), "test": _metrics(0.1, 0.2)}
    _print_report(summary)
    out = capsys.readouterr().out
    assert "table = FINAL net" in out
