"""Automatic level-quality gate: every authored level must pass the validator's
independent runs (win-requirements, gems, enemies, ladders/elevators, hazard
budget). Runs on the fast tile oracle so the whole battery takes seconds; the
frame-exact physics walkthrough lives in test_crystal_caves_handcrafted.py.

Any NEW level added to HANDCRAFTED_LEVELS is validated automatically by these
parameterized tests — junk geometry fails CI the moment it is committed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cc_status.level_validator import validate_level  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402

# Pre-existing findings awaiting an owner decision — documented, not hidden.
# Cavern of Echoes ships a flyer sealed in a 1x1 rock box at (24, 8): it can
# never move or threaten the player (decorative junk the validator correctly
# flags). Remove the entry once the level is repaired or the enemy relocated.
KNOWN_UNENCOUNTERABLE = {
    "Cavern of Echoes": {(24, 8)},
}

_LEVELS = [pytest.param(lv, id=lv.name.replace(" ", "_")) for lv in HANDCRAFTED_LEVELS]


@pytest.fixture(scope="module")
def reports():
    return {lv.name: validate_level(lv.layout) for lv in HANDCRAFTED_LEVELS}


@pytest.mark.parametrize("lv", _LEVELS)
def test_win_requirements_run(lv, reports):
    """Real lock ordering: every crystal, lever and the exit reachable when doors
    start locked and each lever opens only its own colour."""
    missing = reports[lv.name]["win_requirements_missing"]
    assert not missing, f"{lv.name}: unreachable under lock ordering: {missing}"


@pytest.mark.parametrize("lv", _LEVELS)
def test_gems_run(lv, reports):
    """Gates open, every gem reachable — hazards never wall a gem off (they are
    passable at an HP cost), so this is pure geometry."""
    missing = reports[lv.name]["gems_missing"]
    assert not missing, f"{lv.name}: gems unreachable: {missing}"


@pytest.mark.parametrize("lv", _LEVELS)
def test_enemies_run(lv, reports):
    """Every enemy spawns in open space and patrols where the player can actually
    meet it — a threat the player can never encounter is junk decoration."""
    rep = reports[lv.name]
    assert not rep["enemies_in_walls"], f"{lv.name}: enemies in walls: {rep['enemies_in_walls']}"
    allowed = KNOWN_UNENCOUNTERABLE.get(lv.name, set())
    unexpected = set(rep["enemies_unencounterable"]) - allowed
    assert not unexpected, f"{lv.name}: unencounterable enemies: {sorted(unexpected)}"


@pytest.mark.parametrize("lv", _LEVELS)
def test_ladders_and_elevators_run(lv, reports):
    rep = reports[lv.name]
    assert not rep["ladders_unreachable"], f"{lv.name}: {rep['ladders_unreachable']}"
    assert not rep["elevators_unreachable"], f"{lv.name}: {rep['elevators_unreachable']}"


@pytest.mark.parametrize("lv", _LEVELS)
def test_hazard_budget_run(lv, reports):
    """No objective may require more hazard hits than a full-health player can
    survive. (Current authored set: every objective has a ZERO-damage route at
    tile granularity — hazards are a precision tax, never a wall.)"""
    rep = reports[lv.name]
    assert not rep["hazard_unaffordable"], f"{lv.name}: {rep['hazard_unaffordable']}"
    assert all(hits <= 2 for hits in rep["hazard_taxed_objectives"].values())


@pytest.mark.parametrize("lv", _LEVELS)
def test_harness_clock_run(lv, reports):
    """A perfect walking player must fit the training clock (3000 steps) and no
    single leg between consecutive objectives may exceed the 720-step stall
    window. These are HARNESS budgets, not 1991 rules — the original has no
    timers — but a level that busts them is unwinnable in training."""
    clock = reports[lv.name]["clock"]
    assert clock["budget_frac"] < 1.0, f"{lv.name}: best-case tour busts the episode clock"
    assert clock["stall_frac"] < 1.0, f"{lv.name}: a route leg busts the stall window"


@pytest.mark.parametrize("lv", _LEVELS)
def test_fair_spawn_run(lv, reports):
    """The drop from spawn to first footing crosses no hazard. (Enemies patrolling
    the landing zone are reported as warnings, not failures — 4 authored levels
    ship spawn-adjacent patrols pending an owner ruling.)"""
    assert not reports[lv.name]["spawn_drop_hazards"], f"{lv.name}: hazard in the spawn drop"


@pytest.mark.parametrize("lv", _LEVELS)
def test_no_trap_run(lv, reports):
    """From every reachable standing cell, every objective and the exit stay
    reachable — no one-way drop may strand the player (an eternal softlock in a
    game with no timer)."""
    assert not reports[lv.name]["trapped_cells"], f"{lv.name}: {reports[lv.name]['trapped_cells']}"


@pytest.mark.parametrize("lv", _LEVELS)
def test_ammo_economy_run(lv, reports):
    """True to 1991 (rocket gun starts with 5, pickups grant more): the accessible
    arsenal must cover the enemies guarding objectives."""
    ammo = reports[lv.name]["ammo"]
    assert ammo["arsenal"] >= len(ammo["guards"]), f"{lv.name}: {ammo}"


def test_engine_smoke_loads_every_level():
    """Every authored level boots in the REAL engine: loads, produces the right
    state size, and survives a step — a typo'd tile fails here."""
    from config import Config
    from src.game.crystal_caves import CrystalCaves

    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    game = CrystalCaves(cfg, headless=True)
    expected = game.state_size
    for index in range(len(HANDCRAFTED_LEVELS)):
        game.level_index = index
        state = game.reset()
        assert len(state) == expected, f"level {index}: state size drifted"
        state, _r, _d, _info = game.step(game.RIGHT)
        assert len(state) == expected


def test_engine_determinism_guard():
    """The demo system (recording, verification, mid-route starts) requires the
    engine to be a pure function of (level, actions): the same action script
    must produce identical trajectories twice."""
    from config import Config
    from src.game.crystal_caves import CrystalCaves

    actions = [2, 2, 5, 5, 1, 3, 2, 0, 2, 5] * 30

    def trajectory(level_index):
        cfg = Config()
        cfg.CRYSTAL_CAVES_IMPORTED = True
        game = CrystalCaves(cfg, headless=True)
        game.CAVES = (HANDCRAFTED_LEVELS[level_index],)
        game._randomize_levels = False
        game.use_eval_levels(1)
        game.reset_eval_cursor()
        game.reset()
        trace = []
        for action in actions:
            _s, _r, done, _i = game.step(action)
            trace.append((round(game.player_x, 3), round(game.player_y, 3), game.health))
            if done:
                break
        return trace

    for level_index in (0, 7, 15):
        assert trajectory(level_index) == trajectory(level_index), f"level {level_index}"


def test_validator_flags_junk_levels():
    """The validator must actually catch junk: a gem sealed in rock, an enemy in
    a wall, and an unreachable ladder must all fail."""
    junk = (
        "##########",
        "#P...#*#.#",
        "####.###H#",
        "#....#M#H#",
        "#.E.######",
        "##########",
    )
    res = validate_level(junk)
    assert res["gems_missing"], "sealed gem must be flagged"
    assert res["ladders_unreachable"], "sealed ladder must be flagged"
    assert res["enemies_unencounterable"] or res["enemies_in_walls"]
    assert not res["ok"]
