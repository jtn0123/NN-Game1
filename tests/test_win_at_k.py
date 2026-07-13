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


# --- backward demo curriculum (Salimans & Chen) -------------------------------


def _backward_game(offset_map=None):
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    cfg.CRYSTAL_CAVES_DEMO_RESET_P = 1.0
    cfg.CRYSTAL_CAVES_DEMO_BACKWARD = True
    CrystalCaves._BC_SHARED_OFFSET.clear()  # ladder is process-shared; isolate tests
    CrystalCaves._BC_SHARED_WINS.clear()
    game = CrystalCaves(cfg, headless=True)
    # inject a fake demo registry: 400 no-op actions for every level
    game._demo_prefixes = {i: [[0] * 400] for i in range(len(game.CAVES))}
    if offset_map is not None:
        CrystalCaves._BC_SHARED_OFFSET.update(offset_map)
    return game


def test_backward_start_cuts_near_the_win():
    game = _backward_game()
    game.reset()
    level = game.level_index % max(1, len(game.CAVES))
    # first rung: offset = DEMO_BACKWARD_START_OFFSET from the end
    assert CrystalCaves._BC_SHARED_OFFSET[level] == game.DEMO_BACKWARD_START_OFFSET
    assert game._bc_started_level == level


def test_backward_rung_retreats_after_enough_wins():
    """Rungs must advance through the REAL path: a won episode followed by
    reset() (which wipes self.won early — the RUN-38 bug this guards)."""
    game = _backward_game()
    game.reset()
    level = game.level_index % max(1, len(game.CAVES))
    start_offset = CrystalCaves._BC_SHARED_OFFSET[level]
    for _ in range(game.DEMO_BACKWARD_WINS_PER_RUNG):
        game._bc_started_level = level
        game.won = True  # episode ended in a win...
        game.reset()  # ...and the NEXT reset must bank it despite clearing won
        game._bc_started_level = level  # re-arm for the loop (reset may not roll backward)
    assert CrystalCaves._BC_SHARED_OFFSET[level] == start_offset + game.DEMO_BACKWARD_RETREAT_STEP
    assert CrystalCaves._BC_SHARED_WINS[level] == 0  # counter reset at the new rung


def test_backward_cut_never_triggers_win_and_clamps():
    game = _backward_game(offset_map=None)
    game.reset()
    level = game.level_index % max(1, len(game.CAVES))
    # a huge offset must clamp to a plain from-spawn start (cut 0), not negative
    CrystalCaves._BC_SHARED_OFFSET[level] = 10_000
    game.won = False
    game._bc_started_level = None
    game.reset()
    assert not game.game_over


def test_backward_ladder_pace_config_overrides():
    """Config knobs must override the class-default ladder pace."""
    game = _backward_game()
    game.config.CRYSTAL_CAVES_DEMO_BACKWARD_RETREAT = 60
    game.config.CRYSTAL_CAVES_DEMO_BACKWARD_WINS = 2
    game.reset()
    level = game.level_index % max(1, len(game.CAVES))
    start = CrystalCaves._BC_SHARED_OFFSET[level]
    for _ in range(2):  # only 2 wins needed now
        game._bc_started_level = level
        game.won = True
        game.reset()
        game._bc_started_level = level
    assert CrystalCaves._BC_SHARED_OFFSET[level] == start + 60


def test_backward_ladder_shared_across_instances():
    """Vectorized envs must pool rung wins: two instances contributing wins to
    the same level advance ONE shared ladder (8x speedup in the real trainer)."""
    game_a = _backward_game()
    cfg = game_a.config
    game_b = CrystalCaves(cfg, headless=True)
    game_b._demo_prefixes = game_a._demo_prefixes
    game_a.reset()
    level = game_a.level_index % max(1, len(game_a.CAVES))
    start = CrystalCaves._BC_SHARED_OFFSET[level]
    # one win banked by each instance = WINS_PER_RUNG(3)? use 3 alternating
    contributors = [game_a, game_b, game_a]
    for g in contributors:
        g._bc_started_level = level
        g.won = True
        g.reset()
        g._bc_started_level = None
    assert CrystalCaves._BC_SHARED_OFFSET[level] == start + CrystalCaves.DEMO_BACKWARD_RETREAT_STEP


def test_demo_level_bias_resamples_to_demoed_levels():
    """With bias=1.0 every training reset must land on a demoed level."""
    import numpy as np

    game = _backward_game()
    game.config.CRYSTAL_CAVES_DEMO_LEVEL_BIAS = 1.0
    game._demo_prefixes = {2: [[0] * 400], 5: [[0] * 400]}
    np.random.seed(3)
    for _ in range(12):
        game.reset()
        game._bc_started_level = None
        assert game.level_index % max(1, len(game.CAVES)) in (2, 5)


def test_backward_window_rehearses_but_only_frontier_banks():
    """Windowed starts must sample within [frontier-window, frontier] and only
    exact-frontier attempts may bank rung credit."""
    import numpy as np

    game = _backward_game(offset_map=None)
    game.config.CRYSTAL_CAVES_DEMO_BACKWARD_WINDOW = 120
    game.reset()
    level = game.level_index % max(1, len(game.CAVES))
    CrystalCaves._BC_SHARED_OFFSET[level] = 300
    start = 300
    # a rehearsal (non-frontier) win must NOT advance the rung
    game._bc_started_level = level
    game._bc_frontier_attempt = False
    game.won = True
    game.reset()
    game._bc_started_level = None
    assert CrystalCaves._BC_SHARED_OFFSET[level] == start
    # frontier wins still advance after WINS_PER_RUNG
    for _ in range(CrystalCaves.DEMO_BACKWARD_WINS_PER_RUNG):
        game._bc_started_level = level
        game._bc_frontier_attempt = True
        game.won = True
        game.reset()
        game._bc_started_level = None
    assert CrystalCaves._BC_SHARED_OFFSET[level] == start + CrystalCaves.DEMO_BACKWARD_RETREAT_STEP
