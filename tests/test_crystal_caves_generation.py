"""Generation and solvability tests for Crystal Caves levels."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_entities import CAVES
from src.game.crystal_caves_gen import cave_reachable as _cave_reachable
from src.game.crystal_caves_gen import generate_cave as _generate_cave


def _find(layout, ch):
    return [(c, r) for r, row in enumerate(layout) for c, x in enumerate(row) if x == ch]


@pytest.mark.parametrize("cave_index", range(len(CAVES)))
def test_every_cave_is_solvable(cave_index):
    """Each authored cave must be winnable."""

    layout = CAVES[cave_index].layout
    player = _find(layout, "P")[0]
    exit_ = _find(layout, "E")[0]
    crystals = _find(layout, "*")
    switches = _find(layout, "s")

    reach_closed = _cave_reachable(layout, player, doors_open=False)
    reach_open = _cave_reachable(layout, player, doors_open=True)

    assert all(s in reach_closed for s in switches), "switch unreachable"
    assert all(c in reach_open for c in crystals), "a crystal is unreachable"
    assert exit_ in reach_open, "exit unreachable"


@pytest.mark.parametrize("cave_index", range(len(CAVES)))
def test_every_cave_is_dense(cave_index):
    """Authored caves are carved rooms, not sparse platforms over void."""

    layout = CAVES[cave_index].layout
    total = sum(len(row) for row in layout)
    solid = sum(row.count("#") for row in layout)
    assert 0.45 <= solid / total <= 0.85


def _loop_window(game, player_col, player_row):
    """Reference window built the slow per-cell way, for equivalence testing."""

    half_c, half_r = game.WINDOW_COLS // 2, game.WINDOW_ROWS // 2
    win = np.empty((game.WINDOW_ROWS, game.WINDOW_COLS), dtype=np.float32)
    i = 0
    for row in range(player_row - half_r, player_row + half_r + 1):
        for col in range(player_col - half_c, player_col + half_c + 1):
            win.flat[i] = game._tile_code(col, row)
            i += 1
    return win


@pytest.mark.parametrize("difficulty", ["easy", "normal"])
def test_vectorized_window_matches_tile_code_loop(difficulty):
    """The fast vectorized perception window must stay bit-identical."""

    cfg = Config()
    cfg.CRYSTAL_CAVES_PROCEDURAL = True
    cfg.CRYSTAL_CAVES_DIFFICULTY = difficulty
    cfg.CRYSTAL_CAVES_FAMILIES = "platform_network"
    np.random.seed(7)
    game = CrystalCaves(cfg, headless=True)
    game.reset()

    for _ in range(400):
        pc = int((game.player_x + game.PLAYER_WIDTH / 2) // game.TILE_SIZE)
        pr = int((game.player_y + game.PLAYER_HEIGHT / 2) // game.TILE_SIZE)
        expected = _loop_window(game, pc, pr)
        actual = game._fill_window(pc, pr)
        assert np.array_equal(expected, actual)
        _, _, done, _ = game.step(np.random.randint(0, game.action_size))
        if done:
            game.reset()


def test_training_samples_diverse_caves_and_eval_is_held_out():
    """Training samples varied caves; eval uses fixed, held-out caves."""

    cfg = Config()
    cfg.CRYSTAL_CAVES_PROCEDURAL = True
    cfg.CRYSTAL_CAVES_DIFFICULTY = "easy"
    cfg.CRYSTAL_CAVES_FAMILIES = "platform_network"
    np.random.seed(0)
    game = CrystalCaves(cfg, headless=True)

    assert len(game.CAVES) == cfg.CRYSTAL_CAVES_POOL_SIZE
    seen = set()
    for _ in range(40):
        game.reset()
        seen.add(tuple(game.level.layout))
    assert len(seen) > 5

    game.use_eval_levels(20)

    def eval_pass():
        game.reset_eval_cursor()
        seq = []
        for _ in range(20):
            game.reset()
            seq.append(tuple(game.level.layout))
        return seq

    pass1 = eval_pass()
    pass2 = eval_pass()
    assert len(set(pass1)) == 20
    assert pass1 == pass2
    train_layouts = {tuple(c.layout) for c in game.CAVES}
    assert not (set(pass1) & train_layouts)


def test_route_floor_places_single_nearby_walkable_crystal():
    """The first-objective floor should teach real cave routing, not a drill map."""

    spec = _generate_cave(0, "blue_rock", "platform_network", difficulty="route_floor")
    layout = spec.layout
    player = _find(layout, "P")[0]
    crystals = _find(layout, "*")
    exit_ = _find(layout, "E")[0]

    assert len(crystals) == 1
    assert not _find(layout, "s")
    assert not _find(layout, "S")
    assert abs(crystals[0][0] - player[0]) + abs(crystals[0][1] - player[1]) <= 6
    walk_reach = _cave_reachable(layout, player, doors_open=True, jump=0)
    assert crystals[0] in walk_reach
    assert exit_ in walk_reach


def test_route_catch_places_crystal_on_shaft_catch_ledge():
    """The stricter route scaffold should avoid the fall-past timing problem."""

    spec = _generate_cave(0, "blue_rock", "platform_network", difficulty="route_catch")
    layout = spec.layout
    player = _find(layout, "P")[0]
    crystals = _find(layout, "*")
    exit_ = _find(layout, "E")[0]
    shaft_cols = [col for col, ch in enumerate(layout[spec.sky_rows]) if ch == "."]

    assert len(crystals) == 1
    assert not _find(layout, "s")
    assert not _find(layout, "S")
    assert crystals[0][0] in shaft_cols
    assert crystals[0][1] == spec.sky_rows + 2
    walk_reach = _cave_reachable(layout, player, doors_open=True, jump=0)
    assert crystals[0] in walk_reach
    assert exit_ in walk_reach


def test_route_offset_places_crystal_near_but_off_shaft():
    """The middle route scaffold should require lateral movement after descent."""

    spec = _generate_cave(0, "blue_rock", "platform_network", difficulty="route_offset")
    layout = spec.layout
    player = _find(layout, "P")[0]
    crystals = _find(layout, "*")
    exit_ = _find(layout, "E")[0]
    shaft_cols = {col for col, ch in enumerate(layout[spec.sky_rows]) if ch == "."}
    shaft = min(shaft_cols)
    crystal = crystals[0]

    assert len(crystals) == 1
    assert not _find(layout, "s")
    assert not _find(layout, "S")
    assert crystal[0] not in shaft_cols
    assert 2 <= abs(crystal[0] - shaft) + abs(crystal[1] - (spec.sky_rows + 2)) <= 10
    walk_reach = _cave_reachable(layout, player, doors_open=True, jump=0)
    assert crystal in walk_reach
    assert exit_ in walk_reach
