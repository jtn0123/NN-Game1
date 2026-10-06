"""Reverse-gravity presentation follows real pickups without changing observations."""

import numpy as np
import pytest

from src.unity_bridge.session import CaveSession


def collect_gravity(session):
    game = session.game
    tile = next(tile for tile, power in game.powerups.items() if power == game.GRAVITY_POWER)
    col, row = tile
    game.player_x, game.player_y = col * 32 + 4, row * 32 + 2
    game.vx = game.vy = 0
    snapshot = session.handle({"op": "step", "actions": [0]})
    assert tile not in game.powerups and game.gravity_dir == -1
    assert "gravity" in snapshot["sounds"]
    return snapshot


@pytest.mark.parametrize("classic_controls", [True, False])
def test_real_gravity_pickup_is_exported_without_changing_observations(classic_controls):
    session = CaveSession(level=1, classic_controls=classic_controls)
    normal = session.snapshot()
    assert normal["player"]["gravity_dir"] == 1
    snapshot = collect_gravity(session)
    assert snapshot["player"]["gravity_dir"] == -1
    before = session.game.get_state().copy()
    authoritative = (session.game.gravity_dir, session.game.gravity_timer, session.game.vy)
    for _ in range(3):
        assert session.snapshot()["player"]["gravity_dir"] == -1
    np.testing.assert_array_equal(session.game.get_state(), before)
    assert authoritative == (
        session.game.gravity_dir,
        session.game.gravity_timer,
        session.game.vy,
    )
    # Player presentation does not invert other actors or horizontal facing.
    player = next(entity for entity in snapshot["entities"] if entity["id"] == "player")
    assert player["flip"] is (snapshot["player"]["facing"] < 0)
    assert all("gravity_dir" not in entity for entity in snapshot["entities"])


@pytest.mark.parametrize("restore", ["expiry", "reset", "mine"])
def test_expiry_restart_and_main_mine_restore_normal_gravity_metadata(restore):
    session = CaveSession(level=1)
    collect_gravity(session)
    if restore == "expiry":
        session.game.gravity_timer = 1
        snapshot = session.handle({"op": "step", "actions": [0]})
    elif restore == "reset":
        snapshot = session.handle({"op": "reset", "level": 1})
    else:
        snapshot = session.handle({"op": "mine"})
    assert session.game.gravity_dir == snapshot["player"]["gravity_dir"] == 1
    before = session.game.get_state().copy()
    assert session.snapshot()["player"]["gravity_dir"] == 1
    np.testing.assert_array_equal(session.game.get_state(), before)
