"""Tests for the hand-authored single-skill drill levels."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_drills import DRILL_BY_SKILL, DRILL_CAVES
from src.game.crystal_caves_gen import _find, cave_reachable


@pytest.mark.parametrize("spec", DRILL_CAVES, ids=lambda s: s.name)
def test_drill_is_well_formed(spec):
    """Every drill is a valid 18x44 grid with exactly one player, one exit, >=1 crystal."""
    rows = spec.layout
    assert len(rows) == 18
    assert all(len(r) == 44 for r in rows)
    assert len(_find(rows, "P")) == 1
    assert len(_find(rows, "E")) == 1
    assert len(_find(rows, "*")) >= 1


@pytest.mark.parametrize("spec", DRILL_CAVES, ids=lambda s: s.name)
def test_drill_is_solvable(spec):
    """Every crystal and the exit must be reachable from the player under jump-aware
    physics — a drill the agent literally cannot finish would teach nothing."""
    rows = spec.layout
    player = _find(rows, "P")[0]
    reach = cave_reachable(rows, player, doors_open=True)
    for crystal in _find(rows, "*"):
        assert crystal in reach, f"{spec.name}: crystal {crystal} unreachable"
    assert _find(rows, "E")[0] in reach, f"{spec.name}: exit unreachable"


def test_walk_drill_needs_no_jump_but_jump_drills_do():
    """The walk drill is reachable on foot; the jump drills are not — confirming each
    isolates its skill."""

    def walk_solves(spec):
        rows = spec.layout
        walk = cave_reachable(rows, _find(rows, "P")[0], doors_open=True, jump=0)
        objectives = _find(rows, "*") + [_find(rows, "E")[0]]
        return all(o in walk for o in objectives)

    assert walk_solves(DRILL_BY_SKILL["walk"])
    # jump_up / staircase / reach_exit genuinely require vertical jumps.
    assert not walk_solves(DRILL_BY_SKILL["jump_up"])
    assert not walk_solves(DRILL_BY_SKILL["staircase"])
    assert not walk_solves(DRILL_BY_SKILL["reach_exit"])


def test_game_loads_drills_in_drill_mode():
    config = Config()
    config.GAME_NAME = "crystal_caves"
    config.CRYSTAL_CAVES_DRILLS = True
    game = CrystalCaves(config, headless=True)
    assert game.CAVES == DRILL_CAVES
    # The drill loads and produces a valid state, and the player is placed in it.
    state = game.reset()
    assert state.shape == (game.state_size,)
    assert game.crystals  # the active drill has at least one crystal to collect
