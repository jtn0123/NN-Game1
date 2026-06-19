"""Tests for the procedural Crystal Caves level generator.

Generated levels are held to the same bar as authored caves: every one must be
fully solvable under the jump-aware reachability flood, sit in the platform-model
density band, start at the top, and be a valid drop-in for the engine.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.crystal_caves_gen import (
    ACID,
    CRYSTAL,
    FAMILY_NAMES,
    SPIKE,
    THEME_NAMES,
    _find,
    cave_reachable,
    generate_cave,
    grade_cave,
)

SEEDS = [0, 1, 2, 3, 7, 11]


@pytest.mark.parametrize("theme", THEME_NAMES)
@pytest.mark.parametrize("seed", SEEDS)
def test_generated_cave_is_solvable_and_well_formed(seed, theme):
    spec = generate_cave(seed, theme)
    report = grade_cave(spec)

    assert report["solvable"], f"{theme}/{seed} not solvable"
    assert report["fully_connected"], "some open space is unreachable"
    assert report["top_entrance"], "player must spawn at the top"
    assert report["exit_near_bottom"], "exit must be near the bottom"
    assert report["switch_gates_crystal"], "the switch-gated door must isolate a crystal"
    assert 0.22 <= report["density"] <= 0.82, f"density {report['density']} off-model"
    assert report["score"] >= 85
    assert report["crystals"] >= 8
    assert report["switches"] >= 1

    flat = "".join(spec.layout)
    assert flat.count("P") == 1
    assert flat.count("E") == 1
    assert "D" in flat, "level must have a switch-controlled door"
    assert spec.sky_rows == 3
    assert len(spec.layout) == 18
    assert all(len(row) == 44 for row in spec.layout)


def test_generated_cave_loads_and_produces_state():
    """A generated CaveSpec must be a valid drop-in for the real engine."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from config import Config
    from src.game.crystal_caves import CrystalCaves

    spec = generate_cave(5, "rust")
    game = CrystalCaves(Config(), headless=True)
    game.level = spec
    game._load_level(spec)

    state = game.get_state()
    assert state.shape == (game.state_size,)
    # the player and the exit were parsed out of the layout
    assert game.exit_pos is not None


def test_generator_is_deterministic():
    """Same seed + theme always yields the identical layout (reproducibility)."""
    a = generate_cave(42, "gray_tech")
    b = generate_cave(42, "gray_tech")
    assert a.layout == b.layout


@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("seed", [0, 1, 5, 9])
def test_every_family_generates_solvable_levels(seed, family):
    """Each level family (platform / snake / terrain / maze) must produce
    solvable, fully-connected caves."""
    spec = generate_cave(seed, "blue_rock", family)
    report = grade_cave(spec)
    assert report["solvable"], f"{family}/{seed} unsolvable"
    assert report["fully_connected"], f"{family}/{seed} not fully connected"
    assert report["score"] >= 80


@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("seed", [0, 1, 5, 9])
def test_easy_difficulty_is_solvable_and_minimal(seed, family):
    """The 'easy' curriculum floor must stay solvable while placing only a few
    crystals and no hazards/enemies, so a fresh agent can earn its first wins."""
    spec = generate_cave(seed, "blue_rock", family, difficulty="easy")
    assert grade_cave(spec)["solvable"], f"{family}/{seed} easy unsolvable"
    crystals = _find(spec.layout, CRYSTAL)
    hazards = _find(spec.layout, SPIKE) + _find(spec.layout, ACID)
    assert 1 <= len(crystals) <= 4, f"{family}/{seed} easy crystals={len(crystals)}"
    assert not hazards, f"{family}/{seed} easy has hazards"


def test_normal_difficulty_keeps_full_objective_budget():
    """'normal' (the default) keeps the full game's crystal + hazard load."""
    spec = generate_cave(3, "rust", "platform_network", difficulty="normal")
    crystals = _find(spec.layout, CRYSTAL)
    hazards = _find(spec.layout, SPIKE) + _find(spec.layout, ACID)
    assert len(crystals) >= 8
    assert len(hazards) >= 1


def test_color_keyed_two_lock_levels_are_solvable():
    """When a normal cave gets a second colour-keyed lock, every crystal sits
    behind its own colour and the keyed fixpoint flood proves it solvable."""
    from src.game.crystal_caves_gen import DOOR2, SWITCH2

    found = 0
    for seed in range(60):
        spec = generate_cave(seed, "blue_rock", "platform_network", difficulty="normal")
        flat = "".join(spec.layout)
        if DOOR2 in flat:
            found += 1
            assert SWITCH2 in flat, f"blue door without a blue lever (seed {seed})"
            assert grade_cave(spec)["solvable"], f"two-lock level {seed} unsolvable"
            assert grade_cave(spec)["switch_gates_crystal"]
    assert found >= 3, f"expected two-lock levels; found {found}"


def test_elevator_levels_stay_solvable():
    """Elevators appear and never break solvability — the flood treats an
    ELEVATOR run as rideable, a superset of jumpable."""
    from src.game.crystal_caves_gen import ELEVATOR

    found = 0
    for seed in range(40):
        spec = generate_cave(seed, "blue_rock", "platform_network")
        if ELEVATOR in "".join(spec.layout):
            found += 1
            assert grade_cave(spec)["solvable"], f"elevator level {seed} unsolvable"
    assert found >= 3, f"expected elevators to appear; found {found}"


def test_procedural_config_replaces_caves_and_is_playable():
    """CRYSTAL_CAVES_PROCEDURAL swaps the authored caves for generated ones that
    theme correctly, stay solvable, and drive the game without error."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from config import Config
    from src.game.crystal_caves import CrystalCaves

    cfg = Config()
    cfg.CRYSTAL_CAVES_PROCEDURAL = True
    cfg.CRYSTAL_CAVES_SEED = 1
    game = CrystalCaves(cfg, headless=True)

    assert len(game.CAVES) == len(FAMILY_NAMES)
    assert all(spec.name.startswith("Generated") for spec in game.CAVES)
    assert all(spec.sky_rows == 3 for spec in game.CAVES)
    assert all(grade_cave(spec)["solvable"] for spec in game.CAVES)

    game.reset()
    for _ in range(20):
        game.step(2)  # RIGHT — walk along the surface toward the shaft
    assert not game.game_over


def test_reachability_oracle_matches_solvable_levels():
    """The exported flood agrees with the grader on a generated level."""
    spec = generate_cave(3, "blue_rock")
    player = next(
        (c, r)
        for r, row in enumerate(spec.layout)
        for c, ch in enumerate(row)
        if ch == "P"
    )
    reach = cave_reachable(spec.layout, player, doors_open=True)
    crystals = [
        (c, r)
        for r, row in enumerate(spec.layout)
        for c, ch in enumerate(row)
        if ch == "*"
    ]
    assert all(crystal in reach for crystal in crystals)
