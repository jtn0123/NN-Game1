"""The native result and completion credit must refer to the cave just played."""

import pytest

from src.unity_bridge.session import CaveSession


@pytest.mark.parametrize("level", [0, 8, 15])
@pytest.mark.parametrize("classic", [False, True])
def test_winning_snapshot_keeps_played_cave_identity_until_explicit_reset(level, classic):
    session = CaveSession(level=level, classic_controls=classic)
    game = session.game
    name = game.level.name
    layout = session.terrain_layout()
    # Completion-boundary fixture. Full normal-spawn wins have separate tests.
    game.crystals.clear()
    game.exit_unlocked = True
    col, row = game.exit_pos
    game.player_x, game.player_y = col * 32 + 5, row * 32 + 1
    game.vx = game.vy = 0.0
    result = session.handle({"op": "step", "actions": [0]})
    assert result["won"] and result["done"] and result["human_only"]
    assert game.level_index == (level + 1) % len(game.CAVES)
    assert result["level"] == level
    assert result["level_name"] == name and result["layout"] == layout
    assert session.handle({"op": "snapshot"})["level"] == level
    assert session.handle({"op": "step", "actions": [0]})["level"] == level
    fresh = session.handle({"op": "reset", "level": level})
    assert fresh["level"] == level and not fresh["won"]
    mine = session.handle({"op": "mine", "cleared": [level]})
    assert mine["level"] == -1 and mine["cleared_caves"] == 1
