"""Round-2 audit regression tests (R2-D batch): CI clustering, determinism, checkpoint
flag restore, train-level de-duplication."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_r2d_paired_row_ci_clusters_by_seed():
    # Between-seed variance only (each seed's levels are identical): the seed-cluster
    # bootstrap CI must be WIDE. The old i.i.d. row resample would collapse to ~[-.01,.01]
    # and falsely exclude 0.
    from experiments.cc_status.lever_ab import paired_row_ci

    a, b = [], []
    for s, delta in [(0, 0.05), (1, 0.0), (2, -0.05)]:
        for lvl in range(10):
            a.append(
                {"seed": s, "level_index": lvl, "level_name": f"l{lvl}", "selection_score": 0.0}
            )
            b.append(
                {"seed": s, "level_index": lvl, "level_name": f"l{lvl}", "selection_score": delta}
            )
    ci = paired_row_ci(a, b, metric="selection_score", n_bootstrap=400, seed=0)
    assert ci["mean"] == pytest.approx(0.0, abs=1e-9)
    assert ci["ci_high"] - ci["ci_low"] > 0.04  # wide; i.i.d. would be ~0.02
    assert ci["n"] == 30
    assert paired_row_ci(a, b, metric="selection_score", n_bootstrap=400, seed=0) == ci  # determ.


def test_r2d_set_seed_enables_torch_determinism():
    import torch

    from experiments.cc_status.config_helpers import set_seed

    prev = torch.are_deterministic_algorithms_enabled()
    try:
        set_seed(123)
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=True)


def test_r2d_history_state_restored_from_checkpoint(tmp_path):
    from experiments.cc_status.runs_transfer import config_from_selected_checkpoint

    snapshot = {"config": {"history_state": True, "history_steps": 3}}
    cfg = config_from_selected_checkpoint(
        tmp_path, snapshot=snapshot, seed=0, log_every=0, report_seconds=0.0
    )
    assert cfg.CRYSTAL_CAVES_HISTORY_STATE is True
    assert cfg.CRYSTAL_CAVES_HISTORY_STEPS == 3


def test_r2d_use_train_levels_caps_distinct_caves_below_games():
    # The condition that triggered the duplication bug: a small training pool yields fewer
    # DISTINCT caves than `games`, so _eval_split must grade only the distinct ones.
    from experiments.cc_status.lever_ab import make_config
    from src.game.crystal_caves import CrystalCaves

    cfg = make_config({"CRYSTAL_CAVES_POOL_SIZE": 6}, difficulty="easy")
    game = CrystalCaves(cfg, headless=True)
    game.use_train_levels(40)  # ask for 40, pool only has a handful
    assert 0 < len(game._eval_caves) < 40  # capped below `games` -> cycling would duplicate
