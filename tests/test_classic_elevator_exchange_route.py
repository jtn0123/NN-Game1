"""C06's entrance must reach a real lift with ordinary human buttons."""

import pytest

from src.unity_bridge.human_controls import HumanControls
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_scenery import placement_cells
from src.unity_bridge.visuals import dressing_placements


def board_second_lift(delay=0):
    game = CaveSession(level=5).game
    assert game.health == 3 and game.ammo == 5 and len(game.enemies) == 6
    assert len(game.elevators) == 3 and game.initial_crystals == 30
    for _ in range(55):
        game.step_human(HumanControls(1, False, False, False))
    for _ in range(delay):
        game.step_human(HumanControls(0, False, False, False))
    # Wait beside the pad, jump above its edge, then move onto it. Jumping
    # directly under a solid platform correctly bumps its underside.
    for _ in range(750):
        if game._elevator_solid[1].top >= 19 * 32 and game.elevators[1].direction > 0:
            break
        game.step_human(HumanControls(0, False, False, False))
    else:
        pytest.fail("second lift never reaches the entrance boarding height")
    for _ in range(22):
        game.step_human(HumanControls(0, True, False, False))
    for _ in range(14):
        game.step_human(HumanControls(1, False, False, False))
    for _ in range(120):
        game.step_human(HumanControls(0, False, False, False))
        if game._player_rect().bottom == game._elevator_solid[1].top:
            break
    else:
        pytest.fail("entrance jump cannot reach the second lift")
    return game


@pytest.mark.parametrize("delay", [0, 60, 120, 180])
def test_elevator_exchange_second_lift_can_be_boarded_from_normal_spawn(delay):
    game = board_second_lift(delay)
    for _ in range(150):
        game.step_human(HumanControls(0, False, False, False))
        assert game._player_rect().bottom == game._elevator_solid[1].top
        assert not game._is_on_ladder()
        assert not game._rect_collides_solid(game._player_rect())
    assert game.player_y < 18 * 32 and game.health == 3


def test_elevator_exchange_lower_dismount_has_a_landing_above_the_live_rock():
    game = board_second_lift()
    for _ in range(200):
        if game._elevator_solid[1].top <= 16 * 32:
            break
        game.step_human(HumanControls(0, False, False, False))
    game.step_human(HumanControls(1, True, False, False))
    for _ in range(42):
        game.step_human(HumanControls(1, False, False, False))
    assert game._player_rect().bottom == 18 * 32
    # Four seconds to read the rock's patrol. The mobile eye remains a threat;
    # this ledge is not intended to make an indefinite wait invulnerable.
    for _ in range(240):
        game.step_human(HumanControls(0, False, False, False))
        assert game.health == 3 and not game.game_over
    assert game._player_rect().bottom == 18 * 32
    rock = next(enemy for enemy in game.enemies if enemy.appearance == "walking_rock")
    assert rock.alive and rock.health == 1
    assert (13, 18) in game.crystals and game.ammo == 10


def test_elevator_exchange_warns_before_the_rock_dismount_without_hiding_objects():
    game = CaveSession(level=5).game
    placements = dressing_placements(game.level.layout, 5)
    warning = {"sprite": "danger_sign", "col": 14, "row": 17}
    assert warning in placements
    footprint = set(placement_cells(warning))
    assert all(game.level.layout[row][col] == "." for col, row in footprint)
    assert all(
        not footprint.intersection(placement_cells(item)) for item in placements if item != warning
    )


# (frames, horizontal direction, buttons): jump=1, fire=2, use/descend=4.
# Fixed normal-spawn recording; no dependency on ignored audit inputs.
ELEVATOR_EXCHANGE_ROUTE = (
    (54, 1, 0),
    (201, 0, 0),
    (22, 0, 1),
    (14, 1, 0),
    (105, 0, 0),
    (1, 1, 1),
    (42, 1, 0),
    (4, 0, 0),
    (39, 1, 0),
    (4, 0, 0),
    (1, 1, 1),
    (61, 1, 0),
    (1, -1, 1),
    (60, -1, 0),
    (1, 0, 0),
    (1, -1, 1),
    (26, -1, 0),
    (127, 0, 0),
    (1, -1, 1),
    (54, -1, 0),
    (143, 0, 0),
    (27, -1, 0),
    (20, 0, 0),
    (41, -1, 0),
    (53, 1, 0),
    (32, 0, 0),
    (1, 1, 1),
    (178, 1, 0),
    (1, 0, 4),
    (150, 1, 0),
    (259, -1, 0),
    (1, -1, 1),
    (68, -1, 0),
    (160, 0, 0),
    (22, 0, 1),
    (14, 1, 0),
    (189, 0, 0),
    (26, 1, 0),
    (1, -1, 1),
    (11, -1, 0),
    (16, 0, 0),
    (1, -1, 2),
    (30, 0, 0),
    (1, -1, 1),
    (108, -1, 0),
    (54, 1, 0),
    (4, 0, 0),
    (22, 0, 1),
    (14, -1, 0),
    (161, 0, 0),
    (14, 1, 0),
    (11, 0, 0),
    (1, 1, 2),
    (201, 0, 0),
    (1, -1, 2),
    (56, 0, 2),
    (20, 0, 0),
    (123, 1, 0),
    (122, -1, 0),
    (130, 0, 0),
    (14, -1, 0),
    (294, 0, 0),
    (13, -1, 0),
    (30, 0, 0),
    (28, -1, 0),
    (1, 1, 1),
    (53, 1, 0),
    (19, 0, 0),
    (13, -1, 0),
    (145, 0, 0),
    (13, -1, 0),
    (38, 0, 0),
    (53, 1, 0),
    (1, 0, 1),
    (62, 0, 0),
    (1, 1, 1),
    (13, 1, 0),
    (429, 0, 0),
    (1, 1, 1),
    (26, 1, 0),
    (1, 0, 0),
    (110, 1, 0),
    (41, 0, 1),
    (12, -1, 0),
    (12, 1, 0),
    (22, 0, 1),
    (122, -1, 0),
    (32, 0, 0),
    (67, 1, 0),
    (1, 1, 1),
    (54, 1, 0),
    (58, 0, 1),
    (192, 1, 0),
    (18, 0, 0),
    (14, 1, 0),
    (1, 1, 1),
    (13, 1, 0),
    (109, 0, 0),
    (13, -1, 0),
    (11, 0, 0),
    (1, 1, 1),
    (19, 1, 0),
    (41, 0, 0),
    (1, -1, 1),
    (39, -1, 0),
    (22, 0, 0),
    (110, -1, 0),
    (1, -1, 1),
    (92, -1, 0),
    (95, 0, 1),
    (18, 1, 0),
)
