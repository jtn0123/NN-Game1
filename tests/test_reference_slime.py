"""Flying slime identity follows the recording without changing game state."""

import numpy as np
import pygame
import pytest

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_creatures import creature_sprites, eye_flyer
from src.unity_bridge.visual_materials import INK, MAGENTA, WHITE


@pytest.mark.parametrize("frame", range(4))
def test_slime_has_a_tapered_green_body_and_two_stacked_eyes(frame):
    image = eye_flyer(frame)
    assert image.get_size() == (24, 24)
    pixels = pygame.image.tobytes(image, "RGBA")
    assert set(pixels[3::4]) == {0, 255}
    colors = {tuple(pixels[n : n + 3]) for n in range(0, len(pixels), 4) if pixels[n + 3]}
    assert MAGENTA not in colors
    green = sum(
        pixels[n + 1] > pixels[n] and pixels[n + 1] > pixels[n + 2] * 1.5
        for n in range(0, len(pixels), 4)
        if pixels[n + 3]
    )
    assert green > 150
    # The right-facing recording silhouette has two eyes stacked on its broad front.
    for y in (8, 15):
        assert tuple(image.get_at((18, y))) == (*WHITE, 255)
        assert tuple(image.get_at((20, y + 1))) == (*INK, 255)
    assert image.get_at((2, 12)).a == 255  # Narrow rear point remains visible.
    assert image.get_at((2, 7)).a == image.get_at((2, 18)).a == 0


def test_slime_alias_and_four_poses_are_deterministic_native_pixels():
    sprites = creature_sprites()
    frames = [pygame.image.tobytes(sprites[f"eye_flyer_{frame}"], "RGBA") for frame in range(4)]
    assert len(set(frames)) == 4
    assert pygame.image.tobytes(sprites["eye_flyer"], "RGBA") == frames[0]
    for frame, expected in enumerate(frames):
        assert pygame.image.tobytes(eye_flyer(frame), "RGBA") == expected
        assert pygame.image.tobytes(eye_flyer(frame + 4), "RGBA") == expected


@pytest.mark.parametrize("classic", [False, True])
def test_authoring_and_alias_lookup_leave_collision_and_observations_unchanged(classic):
    session = CaveSession(classic_controls=classic)
    game = session.game
    state = game.get_state().copy()
    bodies = [(enemy.x, enemy.y, enemy.vx, tuple(enemy.rect)) for enemy in game.enemies]
    for _ in range(3):
        creature_sprites()
    assert [(enemy.x, enemy.y, enemy.vx, tuple(enemy.rect)) for enemy in game.enemies] == bodies
    assert all(enemy.width == 24 for enemy in game.enemies)
    np.testing.assert_array_equal(game.get_state(), state)
