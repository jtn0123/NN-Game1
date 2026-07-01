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
from experiments.cc_status.level_reach import analyze  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402

_LEGEND = set(". # * E D d s S = H P ^ M F A $ O".split())


def test_levels_present_and_well_formed():
    assert len(HANDCRAFTED_LEVELS) >= 5
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
