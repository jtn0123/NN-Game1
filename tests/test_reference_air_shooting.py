"""Real jump/shot inputs preserve airborne legs without changing the simulation."""

import numpy as np
import pytest

from src.unity_bridge.session import CaveSession


def step_sequence(session, actions):
    result = session.snapshot()
    for action in actions:
        result = session.handle({"op": "step", "actions": [action]})
    return result


def airborne_session(falling=False):
    session = CaveSession(level=2)
    step_sequence(session, [0] * 12 + [5] * 8 + [6])
    if falling:
        step_sequence(session, [0] * 25 + [7])
    return session


@pytest.mark.parametrize("falling", [False, True])
def test_real_airborne_shots_keep_bent_legs_and_snapshot_is_read_only(falling):
    session = airborne_session(falling)
    game = session.game
    observation = game.get_state().copy()
    mechanics = (
        game.player_x,
        game.player_y,
        game.vx,
        game.vy,
        game.ammo,
        game.shoot_cooldown,
        game.health,
        game.steps,
        session.total_reward,
    )
    snapshot = session.snapshot()
    player = snapshot["player"]
    entity = next(item for item in snapshot["entities"] if item["id"] == "player")
    assert not player["grounded"] and not player["climbing"]
    assert (player["vy"] > 0) is falling
    assert player["sprite"] == entity["sprite"] == "mylo_shoot_air"
    assert entity["w"] == 24 and entity["h"] == 32
    assert entity["x"] == game.player_x - 1 and entity["y"] == game.player_y - 2
    assert entity["flip"] is falling
    np.testing.assert_array_equal(game.get_state(), observation)
    assert mechanics == (
        game.player_x,
        game.player_y,
        game.vx,
        game.vy,
        game.ammo,
        game.shoot_cooldown,
        game.health,
        game.steps,
        session.total_reward,
    )


def test_ground_shots_damage_action_pose_and_shot_window_remain_intact():
    session = CaveSession(level=2)
    snapshot = step_sequence(session, [0] * 12 + [6])
    assert snapshot["player"]["grounded"]
    assert snapshot["player"]["sprite"] == "mylo_shoot"
    air = airborne_session()
    air.game.invuln_timer = 5
    assert air.snapshot()["player"]["sprite"] == "mylo_shoot_air"
    assert air.snapshot()["player"]["invulnerable"]
    air.game.invuln_timer = 0
    assert air.snapshot()["player"]["sprite"] == "mylo_shoot_air"
    snapshot = step_sequence(air, [0] * 7)
    assert not snapshot["player"]["grounded"]
    assert snapshot["player"]["sprite"] == "mylo_jump"
