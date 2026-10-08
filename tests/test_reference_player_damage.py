"""Actual damage keeps action poses and exposes its existing protection clock."""

import numpy as np

from src.unity_bridge.session import CaveSession


def test_real_spike_damage_keeps_airborne_motion_pose_and_original_immunity():
    session = CaveSession()
    # Walking into the real trench spike causes damage; jumping clears it.
    for action in [0] * 12 + [2] * 85:
        snapshot = session.handle({"op": "step", "actions": [action]})
    game = session.game
    assert game.steps == 97 and game.health == 2 and game.invuln_timer == 70
    assert game._player_rect().colliderect(game._tile_rect((9, 22)))
    player = snapshot["player"]
    assert player["invulnerable"] and not player["grounded"]
    assert player["sprite"] == "mylo_jump"
    assert player["invulnerability_left"] == 70
    assert player["invulnerability_frames"] == 70
    before = game.get_state().copy()
    mechanics = (
        game.player_x,
        game.player_y,
        game.vx,
        game.vy,
        game.invuln_timer,
        game.health,
        game.score,
    )
    session.snapshot()
    np.testing.assert_array_equal(game.get_state(), before)
    assert mechanics == (
        game.player_x,
        game.player_y,
        game.vx,
        game.vy,
        game.invuln_timer,
        game.health,
        game.score,
    )
    for action in [0] * 12:
        colored = session.handle({"op": "step", "actions": [action]})
    assert colored["player"]["invulnerability_left"] == 58 and game.health == 2
    assert colored["player"]["sprite"] != "mylo_hurt"


def test_damage_metadata_does_not_suppress_a_real_airborne_shot():
    session = CaveSession(2)
    for action in [0] * 12 + [5] * 8 + [6]:
        session.handle({"op": "step", "actions": [action]})
    # Targeted presentation fixture: immunity has no effect on a fired pose.
    session.game.invuln_timer = 58
    player = session.snapshot()["player"]
    assert player["sprite"] == "mylo_shoot_air"
    assert player["invulnerability_left"] == 58 and player["invulnerable"]
    assert player["invulnerability_frames"] == session.game.INVULN_FRAMES
