"""The imported real Crystal Caves Episode 1 levels load and are structurally valid.

NOTE: the tile-byte -> gameplay mapping is a first pass. Geometry, crystals, player and
exit come through, but a few context-dependent codes (some chains vs terrain) are still
approximate pending graphics-based verification. These tests assert the levels LOAD and
are well-formed, not that every tile is perfectly classified.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_cc1_levels import CC1_LEVELS  # noqa: E402

_LEGEND = set(". # * E D d s S = H P".split())


def test_sixteen_levels_present_and_well_formed():
    assert len(CC1_LEVELS) == 16
    for lv in CC1_LEVELS:
        flat = "".join(lv.layout)
        assert all(len(row) == 40 for row in lv.layout), lv.name  # real levels are 40 wide
        assert flat.count("P") == 1, f"{lv.name}: exactly one player start"
        assert flat.count("E") == 1, f"{lv.name}: exactly one exit"
        assert flat.count("*") >= 1, f"{lv.name}: has crystals"
        assert set(flat) <= _LEGEND, f"{lv.name}: only legend tiles"


def test_engine_loads_and_plays_imported_levels():
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    game = CrystalCaves(cfg, headless=True)
    assert game.CAVES is CC1_LEVELS
    state = game.reset()
    assert len(state) == game.state_size
    assert len(game.crystals) >= 1
    # a few steps of real physics run without error
    for action in (game.RIGHT, game.RIGHT_JUMP, game.IDLE):
        state, _reward, _done, _info = game.step(action)
    assert len(state) == game.state_size
