"""Win-at-K training tier: the exit opens at K crystals during TRAINING only,
eval keeps the real all-crystals rule, and the full-clear bonus still pays out
exactly once (including under reverse-exit curriculum starts)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402


def _game(**cfg_overrides) -> CrystalCaves:
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    game = CrystalCaves(cfg, headless=True)
    game.reset()
    return game


def _collect_one_crystal(game: CrystalCaves) -> float:
    col, row = next(iter(game.crystals))
    game.player_x = col * game.TILE_SIZE + 5
    game.player_y = row * game.TILE_SIZE + 1
    return game._collect_pickups()


def test_exit_opens_at_k_in_training():
    game = _game(CRYSTAL_CAVES_WIN_AT_K=1)
    assert not game.exit_unlocked
    _collect_one_crystal(game)
    assert game.exit_unlocked, "training exit must open once K crystals are held"
    assert game.crystals, "remaining crystals stay collectible"
    assert not game._all_crystals_bonus_given, "full-clear bonus must remain unearned"


def test_k_tier_does_not_apply_in_eval():
    game = _game(CRYSTAL_CAVES_WIN_AT_K=1)
    game.use_eval_levels(1)
    game.reset_eval_cursor()
    game.reset()
    _collect_one_crystal(game)
    assert not game.exit_unlocked, "eval keeps the real all-crystals rule"


def test_full_clear_bonus_still_pays_once_under_k_tier():
    game = _game(CRYSTAL_CAVES_WIN_AT_K=1)
    _collect_one_crystal(game)  # unlocks the exit via the K tier
    # sweep the rest of the crystals one by one; the last must pay the bonus
    last_reward = 0.0
    while game.crystals:
        last_reward = _collect_one_crystal(game)
    assert game._all_crystals_bonus_given
    assert last_reward >= game.ALL_CRYSTALS_COLLECTED_BONUS
    # a second sweep with no crystals left must not pay again
    assert game._collect_pickups() < game.ALL_CRYSTALS_COLLECTED_BONUS


def test_reverse_exit_curriculum_start_does_not_gift_the_bonus():
    game = _game(
        CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM=True,
        CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_P=1.0,
    )
    for _ in range(10):
        game.reset()
        if not game.crystals:  # the curriculum start actually fired
            assert game._all_crystals_bonus_given, (
                "pre-cleared start must mark the full-clear bonus as spent, "
                "otherwise the first pickup sweep pays it for free"
            )
            assert game._collect_pickups() < game.ALL_CRYSTALS_COLLECTED_BONUS
            return
    raise AssertionError("reverse-exit curriculum start never fired at p=1.0")
