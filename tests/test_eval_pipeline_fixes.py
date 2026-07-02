"""Regression tests for the eval-pipeline audit fixes.

Pins the contract points of the 2026-07 scoring/eval audit: authored/imported
level sets get a real deterministic eval mode (previously a silent no-op that
left eval sampling randomly with training-only starts un-gated); a same-frame
exit+death is scored as a WIN (consistent with the first-crystal-goal
precedence); metric rows carry unrounded fractions; missing metrics are
excluded from means instead of scoring 0.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from experiments.cc_status.diagnose_gap import _aggregate  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_entities import Enemy  # noqa: E402


def _imported_game(**flags) -> CrystalCaves:
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    for key, value in flags.items():
        setattr(cfg, key, value)
    return CrystalCaves(cfg, headless=True)


class TestAuthoredEvalMode:
    def test_use_eval_levels_cycles_the_fixed_set_in_order(self) -> None:
        game = _imported_game()
        game.use_eval_levels(16)
        assert game._eval_mode, "was a silent no-op for authored sets (audit #1)"
        game.reset_eval_cursor()
        first_pass = []
        for _ in range(16):
            game.reset()
            first_pass.append(game.level.name)
        assert len(set(first_pass)) == 16, "each level exactly once — no replacement"
        game.reset_eval_cursor()
        game.reset()
        assert game.level.name == first_pass[0], "cursor reset restarts the same order"

    def test_eval_mode_gates_training_only_starts(self) -> None:
        game = _imported_game(
            CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM=True,
            CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P=1.0,
        )
        game.use_eval_levels(16)
        game.reset_eval_cursor()
        game.reset()
        # a reverse-exit start would clear all crystals; eval must begin the real task
        assert len(game.crystals) == game.initial_crystals > 0

    def test_switching_eval_sources_rebuilds_the_cache(self) -> None:
        game = _imported_game()
        game.use_train_levels(4)
        train_set = game._eval_caves
        game.use_eval_levels(16)
        assert (
            game._eval_caves is not train_set or len(game._eval_caves) == 16
        ), "stale-cache guard: same-count switches must not keep the wrong caves"


class TestSameFrameExitPrecedence:
    def test_exit_and_fatal_hit_same_frame_is_a_win(self) -> None:
        game = _imported_game()
        game.use_eval_levels(1)
        game.reset()
        # put the player ON the open exit with a lethal enemy overlapping
        game.crystals.clear()
        game.exit_unlocked = True
        ec, er = game.exit_pos
        game.player_x = ec * game.TILE_SIZE + 5
        game.player_y = er * game.TILE_SIZE + 1
        game.health = 1
        game.invuln_timer = 0
        game.enemies = [Enemy(x=game.player_x, y=game.player_y, vx=0.0, kind="flyer")]
        game.step(game.IDLE)
        assert game.won, "same-frame exit+death must score the win (audit #7)"


class TestAggregateMissingKeys:
    def test_missing_metric_rows_are_excluded_not_zeroed(self) -> None:
        rows = [
            {
                "won": True,
                "crystal_frac": 1.0,
                "depth_frac": 1.0,
                "target_distance_progress": 1.0,
                "selection_score": 1.0,
                "exit_unlocked_rate": True,
            },
            {"won": True},  # broken row: most metrics missing
        ]
        out = _aggregate(rows)
        assert out["crystal_frac"] == 1.0, "missing row must not drag the mean to 0.5"
        assert out["missing_crystal_frac"] == 1.0, "missingness must be surfaced"
        assert out["won"] == 1.0

    def test_movement_telemetry_is_aggregated(self) -> None:
        rows = [
            {
                "won": False,
                "crystal_frac": 0.25,
                "depth_frac": 0.5,
                "target_distance_progress": 0.75,
                "selection_score": 0.4,
                "exit_unlocked_rate": False,
                "damage_taken": 2,
                "tiles_visited": 10,
                "idle_frac": 0.2,
                "end_reason": "killed",
                "last_damage_source": "enemy",
            },
            {
                "won": True,
                "crystal_frac": 0.75,
                "depth_frac": 0.9,
                "target_distance_progress": 0.25,
                "selection_score": 0.8,
                "exit_unlocked_rate": True,
                "damage_taken": 4,
                "tiles_visited": 20,
                "idle_frac": 0.4,
                "end_reason": "won",
                "last_damage_source": "none",
            },
        ]
        out = _aggregate(rows)
        assert out["damage_taken"] == 3.0
        assert out["tiles_visited"] == 15.0
        assert abs(out["idle_frac"] - 0.3) < 1e-12
        assert out["end_reason_counts"] == {"killed": 1, "won": 1}
        assert out["kill_source_counts"] == {"enemy": 1}


class TestUnroundedFractions:
    def test_crystal_frac_not_pre_rounded(self) -> None:
        game = _imported_game()
        game.use_eval_levels(1)
        game.reset()
        # a level with 30+ crystals: collecting 1 gives a fraction needing >3 decimals
        one = next(iter(game.crystals))
        game.crystals.discard(one)
        _phi, parts = game._progress_potential()
        expected = 1 / game.initial_crystals
        assert (
            abs(parts["crystal_frac"] - expected) < 1e-12
        ), "per-episode rounding before averaging biased aggregates (audit #5)"


class TestRewardCalibrationKnobs:
    def test_penalties_are_configurable(self) -> None:
        game = _imported_game(CRYSTAL_CAVES_DEATH_PENALTY=-30.0, CRYSTAL_CAVES_HIT_PENALTY=-1.5)
        game.use_eval_levels(1)
        game.reset()
        game.invuln_timer = 0
        game.health = 2
        assert game._damage_player("enemy") == -1.5
        game.invuln_timer = 0
        assert game._damage_player("enemy") == -30.0
        assert game._end_reason == "killed"

    def test_movement_telemetry_in_info(self) -> None:
        game = _imported_game()
        game.use_eval_levels(1)
        game.reset()
        game.invuln_timer = 0
        game._damage_player("hazard")
        for _ in range(12):
            _s, _r, _d, info = game.step(game.RIGHT)
        assert info["damage_taken"] == 1
        assert info["tiles_visited"] >= 2, "walking must accumulate distinct tiles"
        assert 0.0 <= info["idle_frac"] < 1.0
