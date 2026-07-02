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


def _ckpt(won, crystal):
    return {
        "won": won,
        "exit_unlocked_rate": 0.0,
        "crystal_frac": crystal,
        "depth_frac": 0.3,
        "target_distance_progress": 0.4,
        "selection_score": 0.1,
    }


def _seed_summary(seed, points, holdout=False):
    return {
        "difficulty": "easy",
        "episodes": 1000,
        "games": 10,
        "pool_size": 24,
        "truncation_bootstrap": False,
        "train_split_is_holdout": holdout,
        "curve": [
            {"seed": seed, "episode": ep, "train": tr, "test": te, "mean_q": 0.0}
            for (ep, tr, te) in points
        ],
    }


def test_r2c_aggregate_guards_ragged_seeds_and_carries_holdout(tmp_path):
    import json

    from experiments.cc_status.aggregate_diag import aggregate

    # seed 0 reaches ep1000 with a lucky-high train; seed 1 only reached ep500.
    s0 = _seed_summary(
        0,
        [(500, _ckpt(0.1, 0.1), _ckpt(0.0, 0.0)), (1000, _ckpt(0.9, 0.9), _ckpt(0.0, 0.0))],
        holdout=True,
    )
    s1 = _seed_summary(1, [(500, _ckpt(0.1, 0.1), _ckpt(0.0, 0.0))])
    p0, p1 = tmp_path / "s0.json", tmp_path / "s1.json"
    p0.write_text(json.dumps(s0))
    p1.write_text(json.dumps(s1))

    agg = aggregate([str(p0), str(p1)])
    # n_seeds guard: best comes from the full 2-seed ep500 bucket, NOT the lucky 1-seed ep1000.
    assert agg["best"]["episode"] == 500  # pre-fix: 1000
    # B3 gate carried into the aggregate so the verdict isn't re-run the buggy way.
    assert agg["train_split_is_holdout"] is True
    assert "gap_train_minus_test_best" in agg


def _death_trace(*, killed, hazard, enemy, stalled, crystals):
    return {
        "n": 40.0,
        "reason_won": 0.0,
        "reason_killed": killed,
        "reason_timeout": 0.0,
        "reason_stalled": stalled,
        "killed_by_hazard": hazard,
        "killed_by_enemy": enemy,
        "killed_by_both": 0.0,
        "killed_by_air": 0.0,
        "crystal_frac_mean": crystals,
        "steps_mean": 1000.0,
    }


def test_aggregate_seed_averages_death_trace(tmp_path, capsys):
    # The per-seed aggregator used to DROP `death_trace`, so the survival-lever-picking
    # trace had to be averaged by hand. It must now seed-average it like leg-2.
    import json

    from experiments.cc_status.aggregate_diag import aggregate

    s0 = _seed_summary(0, [(500, _ckpt(0.0, 0.1), _ckpt(0.0, 0.0))])
    s1 = _seed_summary(1, [(500, _ckpt(0.0, 0.1), _ckpt(0.0, 0.0))])
    s0["death_trace"] = _death_trace(killed=0.4, hazard=0.3, enemy=0.1, stalled=0.6, crystals=0.20)
    s1["death_trace"] = _death_trace(killed=0.2, hazard=0.1, enemy=0.1, stalled=0.8, crystals=0.10)
    p0, p1 = tmp_path / "s0.json", tmp_path / "s1.json"
    p0.write_text(json.dumps(s0))
    p1.write_text(json.dumps(s1))

    agg = aggregate([str(p0), str(p1)])

    assert "death_trace" in agg  # pre-fix: missing entirely
    dt = agg["death_trace"]
    assert dt["reason_killed"] == pytest.approx(0.3)  # (0.4 + 0.2) / 2
    assert dt["killed_by_hazard"] == pytest.approx(0.2)  # (0.3 + 0.1) / 2
    assert dt["reason_stalled"] == pytest.approx(0.7)  # (0.6 + 0.8) / 2
    assert dt["crystal_frac_mean"] == pytest.approx(0.15)  # (0.20 + 0.10) / 2
    assert len(agg["death_trace_per_seed"]) == 2

    # And the aggregator prints the seed-averaged block so M4 doesn't hand-compute it.
    from experiments.cc_status.diagnose_gap import _print_death_trace

    _print_death_trace(agg)
    out = capsys.readouterr().out
    assert "DEATH-TRACE (held-out, greedy play, seed-avg)" in out
    assert "killed=0.30" in out
    assert "hazard=0.20" in out


def test_aggregate_without_death_trace_omits_it(tmp_path):
    # Runs without --death-trace must not gain a phantom death_trace key or crash the print.
    import json

    from experiments.cc_status.aggregate_diag import aggregate
    from experiments.cc_status.diagnose_gap import _print_death_trace

    s0 = _seed_summary(0, [(500, _ckpt(0.0, 0.1), _ckpt(0.0, 0.0))])
    p0 = tmp_path / "s0.json"
    p0.write_text(json.dumps(s0))

    agg = aggregate([str(p0)])
    assert "death_trace" not in agg
    _print_death_trace(agg)  # no-op, must not raise


def _stall_trace(*, trapped, near, far, osc, geo, crystals, rate):
    return {
        "n_games": 40.0,
        "n_stalled": 20.0,
        "stalled_rate": rate,
        "trapped_frac": trapped,
        "near_objective_frac": near,
        "far_from_objective_frac": far,
        "oscillating_frac": osc,
        "geo_dist_mean": geo,
        "crystal_frac_mean": crystals,
    }


def test_aggregate_seed_averages_stall_trace(tmp_path, capsys):
    # The stall trace (RUN-19 stall diagnostic) must seed-average through the per-seed
    # aggregator and print, just like the death trace.
    import json

    from experiments.cc_status.aggregate_diag import aggregate
    from experiments.cc_status.diagnose_gap import _print_stall_trace

    s0 = _seed_summary(0, [(500, _ckpt(0.0, 0.1), _ckpt(0.0, 0.0))])
    s1 = _seed_summary(1, [(500, _ckpt(0.0, 0.1), _ckpt(0.0, 0.0))])
    s0["stall_trace"] = _stall_trace(
        trapped=0.4, near=0.2, far=0.6, osc=0.5, geo=10.0, crystals=0.2, rate=0.5
    )
    s1["stall_trace"] = _stall_trace(
        trapped=0.2, near=0.4, far=0.4, osc=0.7, geo=6.0, crystals=0.1, rate=0.6
    )
    p0, p1 = tmp_path / "s0.json", tmp_path / "s1.json"
    p0.write_text(json.dumps(s0))
    p1.write_text(json.dumps(s1))

    agg = aggregate([str(p0), str(p1)])

    assert "stall_trace" in agg  # pre-fix: missing entirely
    st = agg["stall_trace"]
    assert st["trapped_frac"] == pytest.approx(0.3)  # (0.4 + 0.2) / 2
    assert st["far_from_objective_frac"] == pytest.approx(0.5)  # (0.6 + 0.4) / 2
    assert st["geo_dist_mean"] == pytest.approx(8.0)  # (10 + 6) / 2
    assert len(agg["stall_trace_per_seed"]) == 2

    _print_stall_trace(agg)
    out = capsys.readouterr().out
    assert "STALL-TRACE (held-out, greedy play, seed-avg)" in out
    assert "trapped" in out


def test_aggregate_without_stall_trace_omits_it(tmp_path):
    import json

    from experiments.cc_status.aggregate_diag import aggregate
    from experiments.cc_status.diagnose_gap import _print_stall_trace

    s0 = _seed_summary(0, [(500, _ckpt(0.0, 0.1), _ckpt(0.0, 0.0))])
    p0 = tmp_path / "s0.json"
    p0.write_text(json.dumps(s0))

    agg = aggregate([str(p0)])
    assert "stall_trace" not in agg
    _print_stall_trace(agg)  # no-op, must not raise


def test_eval_stall_trace_flags_motionless_oscillation():
    # A do-nothing agent never moves -> the no-net-progress stall timer fires -> end_reason
    # 'stalled'. The trace must record it as a stall and flag oscillation (it stayed within
    # <=3 tiles), validating the rollout + bucketing end to end on a real game.
    from types import SimpleNamespace

    from config import Config
    from experiments.cc_status.diagnose_gap import _eval_stall_trace

    cfg = Config()
    cfg.CRYSTAL_CAVES_DIFFICULTY = "normal"
    cfg.EVAL_MAX_STEPS = 1500  # > the 720 stall threshold; keeps the smoke test short
    agent = SimpleNamespace(select_action=lambda state, training=False: 0)  # NOOP every step
    trainer = SimpleNamespace(agent=agent)

    st = _eval_stall_trace(trainer, cfg, games=2)

    assert st["n_stalled"] >= 1.0  # the motionless agent stalls rather than times out
    assert st["stalled_rate"] > 0.0
    assert st["oscillating_frac"] == pytest.approx(1.0)  # never moved -> <=3 unique tiles
    for k in ("trapped_frac", "near_objective_frac", "far_from_objective_frac"):
        assert 0.0 <= st[k] <= 1.0
