"""Behavioral regressions for the reference pass, including training isolation."""

import pytest

from src.game.crystal_caves_entities import Bullet, Enemy
from src.unity_bridge.classic_game import ClassicEnemy
from src.unity_bridge.mine import ENTRANCES
from src.unity_bridge.session import CaveSession


def shoot_enemy(game, enemy, powered=False, head=False):
    game.bullets.append(Bullet(enemy.x + 8, enemy.y + (5 if head else 15), 0, 80, powered))
    game._update_enemies()


def test_green_creature_has_a_real_head_collider_and_survives_four_ordinary_hits():
    classic = CaveSession().game
    enemy = next(e for e in classic.enemies if e.appearance == "dinosaur_enemy")
    assert enemy.height == 64 and enemy.rect.height == 64
    classic.freeze_timer = 100
    for remaining in (4, 3, 2, 1):
        shoot_enemy(classic, enemy, head=True)
        assert enemy.alive and enemy.health == remaining
        assert "enemy_hurt" in classic.audio.events
    shoot_enemy(classic, enemy, head=True)
    assert not enemy.alive
    # The unchanged training baseline used to kill every ground enemy in one hit.
    training = CaveSession(classic_controls=False).game
    old = training.enemies[-1]
    training.freeze_timer = 100
    shoot_enemy(training, old)
    assert not old.alive and Enemy(0, 0, 1).height == 24


def test_powered_shot_kills_green_creature_immediately():
    game = CaveSession().game
    enemy = next(e for e in game.enemies if e.appearance == "dinosaur_enemy")
    shoot_enemy(game, enemy, powered=True, head=True)
    assert not enemy.alive


def test_rock_wakes_then_walks_and_requires_a_powered_shot():
    game = CaveSession().game
    rock = next(e for e in game.enemies if e.appearance == "walking_rock")
    game.player_x, game.player_y = rock.x - 40, rock.y
    start = rock.x
    game._update_enemies()
    assert not rock.asleep and rock.wake_timer > 0 and rock.x == start
    game.freeze_timer = 100
    shoot_enemy(game, rock)
    assert rock.alive and rock.health == 1 and rock.x == start
    shoot_enemy(game, rock, powered=True)
    assert not rock.alive


def test_creature_awareness_is_blocked_by_a_wall():
    game = CaveSession().game
    creature = ClassicEnemy(128, 128, 1, appearance="dinosaur_enemy", health=5)
    game.player_x, game.player_y = 320, 150
    assert game._sees_player(creature)
    game.grid[5][7] = "#"
    game._refresh_static_tile_masks()
    assert not game._sees_player(creature)


def test_empty_gun_has_an_original_cue_without_repeating_every_frame():
    session = CaveSession()
    session.game.ammo = 0
    snapshot = session.handle({"op": "step", "actions": [6] * 8})
    assert snapshot["sounds"].count("empty") == 1
    assert session.game.ammo == 0 and not session.game.bullets


def test_main_mine_is_playable_and_enters_the_nearby_cave_without_a_fake_win():
    session = CaveSession()
    mine = session.handle({"op": "mine", "cleared": [2, 5]})
    assert mine["realm"] == "mine" and mine["level"] == -1 and len(mine["levels"]) == 16
    assert mine["cleared_caves"] == 2 and not mine["done"] and not mine["won"]
    assert not mine["human_only"] and not mine["recording"] and not mine["ai_available"]
    doors = [e for e in mine["entities"] if e["id"].startswith("entrance_")]
    assert len(doors) == len(ENTRANCES) == 16
    assert doors[2]["sprite"] == doors[5]["sprite"] == "mine_door_cleared"
    assert doors[0]["sprite"] == "mine_door"
    for _ in range(5):
        mine = session.handle({"op": "step", "actions": [2] * 8})
    assert mine["near_entrance"] == 0
    entered = session.handle({"op": "step", "actions": [9]})
    assert entered["portal_level"] == 0 and not entered["done"]
    cave = session.handle({"op": "reset", "level": entered["portal_level"]})
    assert cave["realm"] == "cave" and cave["level"] == 0 and cave["human_only"]
    back = session.handle({"op": "mine", "cleared": [2, 5]})
    assert back["near_entrance"] == 0


@pytest.mark.parametrize("cleared", [None, [True], [-1], [16], [1.5]])
def test_invalid_mine_request_leaves_the_current_cave_untouched(cleared):
    session = CaveSession()
    before = session.snapshot()
    with pytest.raises(ValueError):
        session.handle({"op": "mine", "cleared": cleared})
    assert session.snapshot() == before


def test_mine_does_not_feed_a_cave_checkpoint_incompatible_observations():
    session = CaveSession(policy=lambda state: state[:10])
    session.handle({"op": "mine"})
    with pytest.raises(ValueError, match="inside caves"):
        session.handle({"op": "mode", "mode": "ai"})
    assert session.mode == "human" and not session.recording_eligible


def test_mine_chains_hold_on_release_and_gun_controls_still_work():
    session = CaveSession()
    session.handle({"op": "mine"})
    game = session.game
    for _ in range(14):
        session.handle({"op": "step", "actions": [1]})
    assert game._is_on_ladder()
    y = game.player_y
    session.handle({"op": "step", "actions": [0] * 8})
    assert game.player_y == y
    session.handle({"op": "step", "actions": [9] * 8})
    assert game.player_y == pytest.approx(y + 16)
    session.handle({"op": "step", "actions": [3] * 8})
    assert game.player_y < y
    shot = session.handle({"op": "step", "actions": [6]})
    assert shot["ammo"] == 4 and "shoot" in shot["sounds"]
    assert any(entity["id"].startswith("bullet_") for entity in shot["entities"])
    assert not shot["done"] and not shot["won"]
