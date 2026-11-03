"""Player-facing cave edits must offer real, reachable tools without training drift."""

import pytest

from src.game.crystal_caves_entities import Bullet
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS
from src.unity_bridge.session import CaveSession


@pytest.mark.parametrize(
    "level,tile,kind", [(0, (6, 21), "p"), (2, (18, 17), "p"), (7, (8, 8), "z")]
)
def test_authored_tools_are_real_pickups_and_do_not_replace_terrain(level, tile, kind):
    session = CaveSession(level=level)
    game = session.game
    assert game.powerups.get(tile) == kind
    col, row = tile
    assert game.level.layout[row + 1][col] == "#" or (
        kind == "z" and game.level.layout[row][col + 1] == "H"
    )
    game.player_x, game.player_y = col * 32 + 4, row * 32 + 1
    result = session.handle({"op": "step", "actions": [0]})
    assert tile not in game.powerups
    assert (game.super_timer if kind == "p" else game.freeze_timer) > 0
    assert "pickup" in result["sounds"] if kind == "p" else "freeze" in result["sounds"]


def test_first_cave_powered_shot_can_be_collected_by_walking_from_spawn():
    session = CaveSession()
    for _ in range(60):
        result = session.handle({"op": "step", "actions": [2]})
    assert session.game.super_timer > 0
    assert result["health"] == 3 and not result["done"]


def test_player_content_does_not_add_powerups_to_training_maps():
    game = CaveSession(classic_controls=False).game
    assert (6, 21) not in game.powerups
    assert game.CAVES is HANDCRAFTED_LEVELS


@pytest.mark.parametrize("level,tile,kind", [(6, (7, 21), "p"), (14, (35, 18), "p")])
def test_later_powered_shots_are_supported_pickups(level, tile, kind):
    game = CaveSession(level=level).game
    assert game.powerups.get(tile) == kind
    assert game.level.layout[tile[1] + 1][tile[0]] == "#"


@pytest.mark.parametrize("level", [1, 5])
def test_early_ammo_is_available_before_the_first_corridor_threat(level):
    game = CaveSession(level=level).game
    assert (6, 21) in game.ammo_pickups
    assert game.level.layout[22][6] == "#"


def test_smelter_air_machine_is_accessible_from_its_corridor():
    game = CaveSession(level=6).game
    assert (18, 5) in game.air_tanks
    assert (20, 5) not in game.air_tanks
    game.bullets = [Bullet(18 * 32 + 10, 5 * 32 + 14, 0, 80)]
    game._update_bullets()
    assert (18, 5) not in game.air_tanks


def test_echoes_flyer_is_spawned_in_an_open_patrol_corridor():
    game = CaveSession(level=10).game
    flyer = next(e for e in game.enemies if e.kind == "flyer" and int(e.x) // 32 == 22)
    assert int(flyer.y) // 32 == 8
    start = flyer.x
    for _ in range(20):
        game._update_enemies()
    assert abs(flyer.x - start) > 8
    assert not game._rect_collides_solid(flyer.rect)


def test_powered_shots_and_freeze_remain_rare_across_the_campaign():
    caves = CaveSession().game.CAVES
    assert sum(row.count("p") for cave in caves for row in cave.layout) == 4
    assert sum(row.count("z") for cave in caves for row in cave.layout) == 1


def test_cascade_keep_bonus_shelf_is_reached_with_two_real_jumps():
    game = CaveSession(level=11).game
    # A geometry fixture isolates this optional route from enemy timing. The
    # normal engine still advances pickups, jumping, collision and elevators.
    game.enemies = []
    game.player_x, game.player_y = 8 * 32 + 4, 12 * 32 - game.PLAYER_HEIGHT
    game.vx = game.vy = 0.0

    def hop_to(col):
        game.step(5)
        for _ in range(90):
            game.step(2 if game.player_x < col * 32 + 4 else 0)
            if game.grounded:
                return
        pytest.fail("optional jump route did not reach a landing")

    hop_to(10)
    assert game._player_rect().bottom == 11 * 32
    hop_to(12)
    assert game._player_rect().bottom == 9 * 32
    assert (11, 10) not in game.treasures
    assert game.health == 3
