"""An uninterrupted human-input win from Ore Shaft's ordinary cave entrance.

The recording retains authored actors, traps, ammunition and health. It opens
and collects the marked cache, avoids the unkillable ordinary-shot rock, and
finishes through the actual unlocked exit. Inputs are fixed, not a state planner.
"""

import pytest

from src.unity_bridge.session import CaveSession

# (frames, horizontal direction, buttons): jump=1, fire=2, use/descend=4.
# Recorded from the entrance; reviewed snapshots and route outcomes are in
# docs/gameplay-review/README.md. This source trace has no audit-file dependency.
ORE_SHAFT_ROUTE = (
    (55, 1, 0),
    (1, 0, 2),
    (20, 1, 0),
    (1, 1, 1),
    (254, 1, 0),
    (1, 0, 2),
    (137, 1, 0),
    (42, 0, 1),
    (13, -1, 0),
    (13, 1, 0),
    (21, 0, 1),
    (27, -1, 0),
    (27, 1, 0),
    (42, 0, 1),
    (11, -1, 1),
    (43, -1, 0),
    (1, 0, 0),
    (34, 0, 4),
    (14, -1, 0),
    (13, 1, 0),
    (42, 0, 1),
    (1, -1, 2),
    (53, -1, 0),
    (68, 1, 0),
    (14, -1, 0),
    (1, 1, 1),
    (54, 1, 0),
    (51, 0, 1),
    (14, 1, 0),
    (55, -1, 0),
    (41, 1, 0),
    (42, 0, 1),
    (292, 0, 4),
    (219, -1, 0),
    (38, 0, 1),
    (28, -1, 0),
    (41, 1, 0),
    (13, -1, 0),
    (42, 0, 1),
    (69, -1, 0),
    (69, 1, 0),
    (122, 0, 4),
    (128, -1, 0),
    (1, -1, 1),
    (49, -1, 0),
    (27, 0, 0),
    (42, -1, 0),
    (42, 0, 1),
    (41, 1, 0),
    (41, -1, 0),
    (42, 0, 1),
    (13, -1, 0),
    (54, 1, 0),
    (42, 0, 1),
    (27, -1, 0),
    (27, 1, 0),
    (12, 1, 1),
    (29, 1, 0),
    (1, 0, 1),
    (65, 0, 0),
    (40, -1, 0),
    (2, 0, 1),
    (12, -1, 1),
    (30, -1, 0),
    (41, 0, 1),
    (1, 1, 2),
    (30, 0, 2),
    (28, -1, 0),
    (34, 1, 0),
    (2, 0, 1),
    (25, 1, 0),
    (30, 0, 0),
    (1, 1, 1),
    (26, 1, 0),
    (35, 0, 0),
    (138, 1, 0),
    (1, 0, 2),
    (27, 1, 0),
    (1, 0, 1),
    (65, 0, 0),
    (14, 1, 0),
    (1, 1, 1),
    (81, 1, 0),
    (55, -1, 0),
    (62, 0, 4),
    (41, -1, 0),
    (55, 1, 0),
    (14, -1, 0),
    (63, 0, 1),
    (17, -1, 0),
)


def replay_ore_shaft_route(return_delay=0):
    session = CaveSession(level=0)
    game = session.game
    assert game.health == 3 and game.ammo == 5 and game.initial_crystals == 32
    assert game.hidden_crystals == {(10, 9)} and not game.exit_unlocked
    assert len(game.enemies) == 6 and all(enemy.alive for enemy in game.enemies)
    revealed = False
    damage_frames = []
    route = list(ORE_SHAFT_ROUTE)
    # Keep total movement and every later input fixed while varying takeoff by
    # two frames across a 28-pixel return approach, not just one exact timing.
    route[42] = (128 + return_delay, -1, 0)
    route[44] = (49 - return_delay, -1, 0)
    for frames, move, buttons in route:
        controls = {
            "move": move,
            "jump": bool(buttons & 1),
            "shoot": bool(buttons & 2),
            "interact": bool(buttons & 4),
        }
        for _ in range(frames):
            health = game.health
            hidden = bool(game.hidden_crystals)
            session.handle({"op": "human_step", "controls": [controls]})
            revealed |= hidden and not game.hidden_crystals
            if game.health < health:
                damage_frames.append(game.steps)
    return session, revealed, damage_frames


@pytest.mark.parametrize("return_delay", [0, 2, 4, 6, 8, 10, 12])
def test_ore_shaft_can_be_won_from_normal_spawn_with_all_32_crystals_and_secret(return_delay):
    session, revealed, damage_frames = replay_ore_shaft_route(return_delay)
    game = session.game
    result = session.snapshot()
    assert revealed and not game.hidden_crystals
    assert not game.crystals and game.initial_crystals == 32
    assert game.exit_unlocked and result["done"] and result["won"]
    assert result["end_reason"] == "won" and result["level"] == 0
    assert game.health == 3 and game.ammo >= 3 and not damage_frames
    assert (12, 9) in game.air_tanks and game._solid_at(10, 7)
    assert sum(enemy.alive for enemy in game.enemies) == 1
    assert next(enemy for enemy in game.enemies if enemy.alive).appearance == "walking_rock"
    assert result["human_only"] and not session.demo_eligible
