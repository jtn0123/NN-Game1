"""Reference pass: active traps, real lift rides, and training isolation."""

import pytest

from src.unity_bridge.session import CaveSession


def test_green_thorn_is_hidden_safe_then_rises_and_retracts_with_player_proximity():
    session = CaveSession()
    game = session.game
    thorn = game.thorns[0]
    assert thorn.frame == 0
    assert session.terrain_layout()[thorn.row][thorn.col] != "^"
    game.player_x, game.player_y = thorn.col * 32 + 8, thorn.row * 32 + 1
    game.enemies.clear()
    game.hazards.clear()
    health = game.health
    for _ in range(3):
        game._check_player_danger()
    assert game.health == health  # The retracted trap has no invisible collider.
    for _ in range(14):
        game._check_player_danger()
    assert thorn.frame == 4 and game.health == health - 1
    assert game.audio.events.count("thorn") == 1
    game.player_x -= 96
    game._check_player_danger()
    assert thorn.frame == 0
    game.player_x += 96
    game._check_player_danger()
    assert game.audio.events.count("thorn") == 2


def test_thorn_detection_stops_at_a_solid_ceiling_and_snapshot_matches_collision():
    session = CaveSession()
    game = session.game
    thorn = game.thorns[0]
    game.enemies.clear()
    game.player_x, game.player_y = thorn.col * 32 + 8, (thorn.row - 3) * 32
    game.grid[thorn.row - 1][thorn.col] = "#"
    for _ in range(20):
        game._check_player_danger()
    assert thorn.frame == 0
    game.grid[thorn.row - 1][thorn.col] = "."
    game.player_y = thorn.row * 32 + 1
    for _ in range(16):
        game._check_player_danger()
    entity = next(e for e in session.snapshot()["entities"] if e["id"] == "thorn_0")
    assert entity["sprite"] == "green_thorn_4"
    assert entity["w"] == entity["h"] == 32
    session.reset(0)
    assert session.game.thorns[0].frame == 0
    assert not hasattr(CaveSession(classic_controls=False).game, "thorns")


@pytest.mark.parametrize("direction", [-1, 1])
def test_lift_carries_the_player_in_both_directions_and_jump_detaches(direction):
    session = CaveSession()
    session.reset(2)
    game = session.game
    lift = game.elevators[0]
    lift.pos, lift.direction = 8.0, direction
    game._refresh_elevator_rects()
    game.player_x, game.player_y = lift.col * 32 + 5, 8 * 32 - game.PLAYER_HEIGHT
    game.vy = 0
    for _ in range(30):
        game.step(0)
        assert game._player_rect().bottom == game._elevator_solid[0].top
        assert game.grounded and not game._is_on_ladder()
    before = game.player_y
    game.step(3)
    assert game.player_y < before - 2 and not game.grounded
    entity = next(e for e in session.snapshot()["entities"] if e["id"] == "lift_0")
    assert entity["h"] == game._elevator_solid[0].height == 32


def test_riding_lift_reverses_without_pushing_player_through_a_ceiling():
    session = CaveSession()
    session.reset(2)
    game = session.game
    lift = game.elevators[0]
    lift.pos, lift.direction = 8.0, -1
    game.grid[6][lift.col] = "#"
    game._refresh_elevator_rects()
    game.player_x, game.player_y = lift.col * 32 + 5, 226
    game.vy = 0
    for _ in range(24):
        game.step(0)
        assert game._player_rect().top >= 224
        assert not game._rect_collides_solid(game._player_rect())
    assert lift.direction == 1


def test_freight_lift_can_be_boarded_from_spawn_with_real_actions():
    session = CaveSession()
    session.reset(2)
    game = session.game
    # Wait beside the shaft, then hop onto the returning pad above its floor thorn.
    for _ in range(470):
        game.step(0)
    for _ in range(30):
        game.step(2)
    for _ in range(14):
        game.step(5)
    boarded = False
    for _ in range(1150):
        game.step(0)
        lift = game._elevator_solid[0]
        if game._player_rect().bottom == lift.top and game.player_y < 600:
            boarded = True
        if boarded and game.player_y < 300:
            break
    assert boarded and game.player_y < 300 and game.health == 3 and not game.game_over


def test_descending_lift_cannot_embed_a_waiting_player_in_the_floor():
    session = CaveSession()
    session.reset(2)
    game = session.game
    game.thorns.clear()
    game.enemies.clear()
    lift = game.elevators[0]
    lift.pos, lift.direction = 21.0, 1
    game._refresh_elevator_rects()
    game.player_x, game.player_y, game.vy = 197, 706, 0
    for _ in range(12):
        game.step(0)
        assert not game._rect_collides_solid(game._player_rect())
    assert lift.direction == -1


@pytest.mark.parametrize("direction", [-1, 1])
def test_inverted_gravity_lift_ride_and_jump_use_the_underside(direction):
    session = CaveSession()
    session.reset(2)
    game = session.game
    lift = game.elevators[0]
    lift.pos, lift.direction = 8.0, direction
    game._refresh_elevator_rects()
    game.player_x, game.player_y, game.vy, game.gravity_dir = 197, 288, 0, -1
    for _ in range(24):
        game.step(0)
        assert game._player_rect().top == game._elevator_solid[0].bottom
        assert game.grounded
    before = game.player_y
    game.step(3)
    assert game.player_y > before + 2 and not game.grounded


@pytest.mark.parametrize(
    "level,index",
    [
        (level, index)
        for level, count in {2: 2, 5: 3, 11: 5, 12: 2, 14: 1}.items()
        for index in range(count)
    ],
)
def test_every_authored_lift_carries_without_overlap_or_phantom_climbing(level, index):
    session = CaveSession()
    session.reset(level)
    game = session.game
    game.enemies.clear()
    game.thorns.clear()
    lift = game.elevators[index]
    lift.pos = (lift.top + lift.bottom) / 2
    game._refresh_elevator_rects()
    game.player_x, game.player_y, game.vy = lift.col * 32 + 5, int(lift.pos * 32) - 30, 0
    for _ in range(120):
        game.step(0)
        assert game._player_rect().bottom == game._elevator_solid[index].top
        assert game.grounded and not game._is_on_ladder()
        assert not game._rect_collides_solid(game._player_rect())
