"""Pink snake identity uses taller art above the unchanged patrol body."""

import numpy as np
import pygame
import pytest

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_creatures import creature_sprites, slug


@pytest.mark.parametrize("frame", range(4))
def test_snake_has_an_upright_pink_neck_yellow_eye_and_a_floor_anchored_tail(frame):
    image = slug(frame)
    assert image.get_size() == (24, 32)
    pixels = pygame.image.tobytes(image, "RGBA")
    assert set(pixels[3::4]) == {0, 255}
    pink = sum(
        pixels[n] > pixels[n + 1] * 1.6 and pixels[n + 2] > pixels[n + 1] * 1.2
        for n in range(0, len(pixels), 4)
        if pixels[n + 3]
    )
    assert pink > 120
    assert tuple(image.get_at((19, 2))) == (255, 238, 97, 255)
    assert image.get_at((2, 15)).a == 0  # No low crawling segments under an eye stalk.
    assert image.get_at((12, 2)).a == 0  # The head remains narrow and points forward.
    assert image.get_at((4, 30)).a == 255  # Tail curl follows the floor.
    assert any(image.get_at((x, 31)).a == 255 for x in range(24))
    assert image.get_bounding_rect().bottom == 32


def test_snake_poses_alias_and_bottom_anchor_are_deterministic_for_both_facings():
    sprites = creature_sprites()
    frames = [pygame.image.tobytes(sprites[f"slug_enemy_{frame}"], "RGBA") for frame in range(4)]
    assert len(set(frames)) == 4
    assert pygame.image.tobytes(sprites["slug_enemy"], "RGBA") == frames[0]
    for frame, expected in enumerate(frames):
        image = slug(frame)
        assert pygame.image.tobytes(image, "RGBA") == expected
        assert pygame.image.tobytes(slug(frame + 4), "RGBA") == expected
        assert pygame.transform.flip(image, True, False).get_bounding_rect().bottom == 32


def test_bottom_anchored_snake_art_clears_every_authored_spawn_and_horizontal_patrol_row():
    session = CaveSession()
    count = 0
    for level in range(len(session.cave_game.CAVES)):
        session.reset(level)
        game = session.game
        for enemy in game.enemies:
            if enemy.appearance != "slug_enemy":
                continue
            count += 1
            assert enemy.width == enemy.height == 24
            assert enemy.y % 32 == 8
            assert not game._rect_collides_solid(enemy.rect)
            # The eight added pixels extend upward from the same floor, through
            # all possible clear body positions at this authored patrol height.
            for x in range(len(game.level.layout[0]) * 32 - 23):
                body = pygame.Rect(x, int(enemy.y), 24, 24)
                if game._rect_collides_solid(body):
                    continue
                art = pygame.Rect(x, int(enemy.y) - 8, 24, 32)
                assert art.bottom == body.bottom
                assert not game._rect_collides_solid(art), (level, x, enemy.y)
    assert count > 0


@pytest.mark.parametrize("classic", [False, True])
def test_snake_authoring_does_not_change_collision_rectangles_or_observations(classic):
    session = CaveSession(classic_controls=classic)
    game = session.game
    before = game.get_state().copy()
    bodies = [(enemy.x, enemy.y, enemy.vx, tuple(enemy.rect)) for enemy in game.enemies]
    creature_sprites()
    session.snapshot()
    assert [(enemy.x, enemy.y, enemy.vx, tuple(enemy.rect)) for enemy in game.enemies] == bodies
    assert all(
        enemy.height == 24
        for enemy in game.enemies
        if getattr(enemy, "appearance", "") != "dinosaur_enemy"
    )
    np.testing.assert_array_equal(game.get_state(), before)
