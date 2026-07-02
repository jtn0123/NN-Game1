"""The hand-crafted Crystal-Caves-style levels load, are well-formed, and are
certified winnable by the physics-faithful reachability oracle.

Every level in HANDCRAFTED_LEVELS must have all objectives (crystals, switches,
exit) physically reachable from the player start using only walk/jump/climb — so
these tests fail loudly if a future edit strands an objective behind a wall.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from experiments.cc_status.level_reach import analyze, analyze_gated, door_value  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402

_LEGEND = set(". # * E D d s S = H P ^ ~ M F A $ O p g z".split())


def test_levels_present_and_well_formed():
    # At least the 16 of the original Episode 1.
    assert len(HANDCRAFTED_LEVELS) >= 16
    for lv in HANDCRAFTED_LEVELS:
        flat = "".join(lv.layout)
        assert all(len(row) == 40 for row in lv.layout), lv.name
        assert flat.count("P") == 1, f"{lv.name}: exactly one player start"
        assert flat.count("E") == 1, f"{lv.name}: exactly one exit"
        assert flat.count("*") >= 1, f"{lv.name}: has crystals"
        assert set(flat) <= _LEGEND, f"{lv.name}: only legend tiles"


def test_every_level_is_winnable():
    for lv in HANDCRAFTED_LEVELS:
        res = analyze(lv.layout)
        assert res["winnable"], (
            f"{lv.name}: not winnable — "
            f"missing crystals {res['missing_crystals']}, "
            f"missing switches {res['missing_switches']}, "
            f"exit reachable={res['exit']}"
        )


def test_every_level_is_gated_winnable():
    """Doors start CLOSED and only open when their switch is physically reached —
    the real in-game lock ordering. Catches switch-behind-its-own-door deadlocks."""
    for lv in HANDCRAFTED_LEVELS:
        res = analyze_gated(lv.layout)
        assert res["gated_winnable"], (
            f"{lv.name}: not solvable under lock ordering — "
            f"open order {res['door_open_order']}, never opened {res['doors_never_opened']}, "
            f"missing crystals {res['missing_crystals']}"
        )


def test_every_door_actually_gates_something():
    """No decorative doors: closing a colour must block at least one objective,
    otherwise the switch puzzle is theatre the player can just walk around."""
    for lv in HANDCRAFTED_LEVELS:
        for colour, gates in door_value(lv.layout).items():
            assert gates, f"{lv.name}: door {colour!r} is decorative (routable around)"


def test_oracle_climbs_ladders():
    """cave_reachable must model ladder climbing: before the fix it treated 'H'
    as plain air, so anything above a ladder shaft was falsely 'unreachable' —
    which inflated the trapped-stall diagnosis to ~0.35 on the ladder-heavy set."""
    from src.game.crystal_caves_gen import cave_reachable

    rows = (
        "#####",
        "#..*#",
        "#..H#",
        "#..H#",
        "#P.H#",
        "#####",
    )
    reach = cave_reachable(rows, (1, 4), True)
    assert (3, 1) in reach, "crystal above the ladder must be reachable by climbing"


def test_oracle_reaches_every_crystal_from_spawn():
    """The live trapped detector (game._oracle_reachable) must agree with the
    physics oracle: on the certified-winnable set, every crystal is reachable
    from spawn with doors open. Fails if the tile oracle regresses (e.g. loses
    ladder support again)."""
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    game = CrystalCaves(cfg, headless=True)
    game.use_eval_levels(len(HANDCRAFTED_LEVELS))
    game.reset_eval_cursor()
    for _ in range(len(HANDCRAFTED_LEVELS)):
        game.reset()
        game.open_colors = set(game.door_color.values())  # endgame: all doors open
        reach = game._oracle_reachable(game._player_tile())
        missing = set(game.crystals) - reach
        assert not missing, f"{game.level.name}: oracle says unreachable: {sorted(missing)}"


def test_engine_loads_and_plays_the_levels():
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    game = CrystalCaves(cfg, headless=True)
    assert game.CAVES is HANDCRAFTED_LEVELS
    state = game.reset()
    assert len(state) == game.state_size
    assert len(game.crystals) >= 1
    for action in (game.RIGHT, game.RIGHT_JUMP, game.IDLE):
        state, _reward, _done, _info = game.step(action)
    assert len(state) == game.state_size
