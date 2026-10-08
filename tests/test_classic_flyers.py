"""Distinct flying threats must obey collision, freeze and presentation rules."""

from src.game.crystal_caves_entities import Bullet, CaveSpec
from src.unity_bridge.classic_flyers import BatEgg
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visuals import art_sprites


def flyer_arena():
    session = CaveSession()
    rows = ["#" * 30] + ["#" + "." * 28 + "#" for _ in range(12)] + ["#" * 30]
    row = list(rows[4])
    row[10], row[20] = "F", "F"
    rows[4] = "".join(row)
    row = list(rows[12])
    row[3], row[25], row[26] = "P", "*", "E"
    rows[12] = "".join(row)
    session.game.CAVES = (CaveSpec("flyer arena", tuple(rows), (0, 0, 0), (255, 255, 255)),)
    session.game.use_eval_levels(1)
    session.reset(0)
    return session


def test_eye_roams_vertically_without_entering_walls():
    game = flyer_arena().game
    eye = next(e for e in game.enemies if e.appearance == "eye_flyer")
    start = eye.y
    for frame in range(1, 121):
        game.steps = frame
        game._update_enemies()
        assert not game._rect_collides_solid(eye.rect)
    assert abs(eye.y - start) > 16


def test_bat_drops_a_visible_bounded_egg():
    session = flyer_arena()
    game = session.game
    for frame in range(1, 241):
        game.steps = frame
        game._update_enemies()
    eggs = getattr(game, "bat_eggs", [])
    assert eggs and len(eggs) <= 8
    entities = session.snapshot()["entities"]
    assert any(e["sprite"] == "bat_egg" and e["w"] == 12 and e["h"] == 16 for e in entities)


def test_freeze_holds_both_flyers_and_existing_eggs():
    game = flyer_arena().game
    for frame in range(1, 241):
        game.steps = frame
        game._update_enemies()
    eggs = getattr(game, "bat_eggs", [])
    assert eggs
    before = [(e.x, e.y) for e in game.enemies] + [(e.x, e.y) for e in eggs]
    game.freeze_timer = 100
    for _ in range(20):
        game._update_enemies()
        game._check_player_danger()
    assert before == [(e.x, e.y) for e in game.enemies] + [(e.x, e.y) for e in eggs]


def test_falling_egg_hits_player_once_and_is_removed():
    game = flyer_arena().game
    player = game._player_rect()
    game.bat_eggs = [BatEgg(0, player.centerx - 6, player.top - 17, vy=4)]
    game._check_player_danger()
    assert game.health == 2
    assert not game.bat_eggs
    game._check_player_danger()
    assert game.health == 2


def test_egg_hits_a_floor_without_damaging_player_below_it():
    game = flyer_arena().game
    game.bat_eggs = [BatEgg(0, 12 * 32, 13 * 32 - 17, vy=4)]
    assert not game._update_bat_eggs()
    assert not game.bat_eggs
    assert game.health == 3


def test_a_bullet_can_break_a_frozen_egg():
    game = flyer_arena().game
    game.freeze_timer = 100
    game.bat_eggs = [BatEgg(0, 300, 100)]
    game.bullets = [Bullet(300, 104, 0, 80)]
    assert not game._update_bat_eggs()
    assert not game.bat_eggs and not game.bullets
    assert any(e.kind == "spark" for e in game.visual_events)


def test_reset_removes_eggs_and_observation_matches_the_visible_hazard():
    session = flyer_arena()
    game = session.game
    game.bat_eggs = [BatEgg(4, 10 * 32, 6 * 32)]
    assert game._tile_code(10, 6) == game.TILE_CODES[game.SPIKE]
    assert any(e["id"] == "egg_4" for e in session.snapshot()["entities"])
    assert art_sprites()["bat_egg"].get_size() == (12, 16)
    session.reset(0)
    assert not game.bat_eggs
    assert not any(e["id"].startswith("egg_") for e in session.snapshot()["entities"])
