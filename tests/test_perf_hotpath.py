"""Correctness guards for the profiled hot-path optimizations (PERF_PLAN P5):
the _current_target per-step memo must be invisible except in speed."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402


def _game() -> CrystalCaves:
    config = Config()
    config.CRYSTAL_CAVES_IMPORTED = True
    config.FORCE_CPU = True
    game = CrystalCaves(config, headless=True)
    game.reset()
    return game


def test_memo_returns_identical_result_on_repeat_calls():
    game = _game()
    first = game._current_target()
    second = game._current_target()
    assert first == second
    assert getattr(game, "_current_target_memo", None) is not None


def test_memo_invalidated_by_crystal_collection():
    game = _game()
    before_target, _ = game._current_target()
    assert before_target is not None
    # Simulate collecting the current target crystal without stepping: the memo
    # key includes len(crystals), so the next call must recompute.
    if before_target[0] == "crystal":
        game.crystals.discard((before_target[1], before_target[2]))
        after_target, _ = game._current_target()
        assert after_target != before_target
    else:
        # switch-first layout: consume the switch instead
        game.used_switches.add((before_target[1], before_target[2]))
        after_target, _ = game._current_target()
        assert after_target != before_target


def test_memo_invalidated_by_player_movement():
    game = _game()
    _, d1 = game._current_target()
    game.player_x += 40.0  # teleport without a step (curriculum-style relocation)
    _, d2 = game._current_target()
    assert d1 != d2


def test_memo_invalidated_across_reset():
    game = _game()
    t1 = game._current_target()
    game.reset()
    t2 = game._current_target()  # fresh episode: must recompute, not crash
    assert t2[1] > 0
    assert getattr(game, "_current_target_memo")[1] == t2
    del t1


def test_target_distance_matches_unmemoized_recompute():
    import math

    game = _game()
    target, distance = game._current_target()
    assert target is not None
    px, py = game._player_center()
    tx, ty = game._tile_center((target[1], target[2]))
    assert distance == math.hypot(tx - px, ty - py)
