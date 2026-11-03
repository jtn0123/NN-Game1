"""Regional C08 timing proof; authored threats and contact damage stay active."""

import pytest

from src.unity_bridge.human_controls import HumanControls
from src.unity_bridge.session import CaveSession

# From the right shelf, deliberately climb to take the pickup, cross to the
# left bay, then jump before returning through its chain to the upper shelf.
# Each with/without-pickup pair uses exactly these same frame/button counts.
ROUTE = (
    (4, 1, False),
    (20, -1, False),
    (20, 0, True),
    (32, -1, False),
    (4, 0, False),
    (1, -1, True),
    (71, -1, False),
    (21, 0, False),
    (1, 0, True),
    (6, 0, False),
    (38, 1, True),
    (75, 1, False),
)


def regional_checkpoint(present=True):
    session = CaveSession(7)
    game = session.game
    # This is a regional entry checkpoint, not access proven from cave spawn.
    # No actor, damage, timer, gate or invulnerability state is changed.
    game.player_x, game.player_y = 324, 290
    game.vx = game.vy = 0.0
    assert not game._rect_collides_solid(game._player_rect())
    assert len(game.enemies) == 6 and all(enemy.alive for enemy in game.enemies)
    if not present:
        for tile, power in list(game.powerups.items()):
            if power == game.FREEZE_POWER:
                del game.powerups[tile]
    return session


def frames(game, count, move=0, jump=False):
    for _ in range(count):
        game.step_human(HumanControls(move, jump, False, False))


def crossing(present, phase):
    session = regional_checkpoint(present)
    game = session.game
    frames(game, phase)
    for count, move, jump in ROUTE:
        frames(game, count, move, jump)
    assert game.ammo == 5
    assert all(enemy.alive for enemy in game.enemies)
    assert 300 < game.player_x < 315 and game._is_on_ladder()
    return session


def test_freeze_saves_a_heart_in_the_moving_bat_approach_phase():
    without, with_pickup = crossing(False, 70), crossing(True, 70)
    assert without.game.health == 2 and not without.game.game_over
    assert with_pickup.game.health == 3
    assert with_pickup.game.health > without.game.health


@pytest.mark.parametrize("phase", [0, 70, 140])
@pytest.mark.parametrize("present", [False, True])
def test_three_regional_phases_remain_survivable_with_or_without_pickup(phase, present):
    game = crossing(present, phase).game
    assert game.health == (3 if present else {0: 3, 70: 2, 140: 3}[phase])
    assert not game.game_over


def take_pickup(session):
    game = session.game
    initial = {t for t, p in game.powerups.items() if p == game.FREEZE_POWER}
    for count, move, jump in ROUTE[:3]:
        for _ in range(count):
            frames(game, 1, move, jump)
            if game.freeze_timer:
                assert game.freeze_timer == 300
                return initial
    pytest.fail("the shelf approach did not collect its FREEZE pickup")


def test_floor_traversal_can_leave_freeze_for_a_deliberate_chain_climb():
    session = regional_checkpoint()
    game = session.game
    initial = {t for t, p in game.powerups.items() if p == game.FREEZE_POWER}
    for count, move, jump in ROUTE[:2]:
        frames(game, count, move, jump)
    assert game.freeze_timer == 0
    assert {t for t, p in game.powerups.items() if p == game.FREEZE_POWER} == initial
    frames(game, 1, jump=True)
    assert game.freeze_timer == 300
    assert not any(power == game.FREEZE_POWER for power in game.powerups.values())


def test_freeze_holds_on_snapshot_expires_and_reset_rearms_the_pickup():
    session = regional_checkpoint()
    game = session.game
    initial = take_pickup(session)
    positions = [(enemy.x, enemy.y) for enemy in game.enemies]
    assert session.handle({"op": "snapshot"})["freeze_timer"] == 300
    assert game.freeze_timer == 300
    frames(game, 299)
    assert game.freeze_timer == 1
    assert [(enemy.x, enemy.y) for enemy in game.enemies] == positions
    frames(game, 1)
    assert game.freeze_timer == 0
    assert [(enemy.x, enemy.y) for enemy in game.enemies] != positions
    assert game.health == 3
    session.reset(7)
    assert game.freeze_timer == 0 and game.health == 3
    assert {t for t, p in game.powerups.items() if p == game.FREEZE_POWER} == initial


def test_walking_into_the_frozen_bat_still_costs_a_heart():
    session = regional_checkpoint()
    game = session.game
    take_pickup(session)
    for _ in range(150):
        frames(game, 1, -1)
        if game.health < 3:
            break
    assert game.health == 2 and game.freeze_timer > 0
    assert game._last_damage_source == "enemy"
    assert game.ammo == 5 and all(enemy.alive for enemy in game.enemies)
