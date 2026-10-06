"""Blue cave materials keep exact collision opacity and world-space joins."""

from itertools import product

import numpy as np
import pygame

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_rooms import blue_cobble, blue_cobble_field
from src.unity_bridge.visuals import terrain, theme_index


def test_blue_cobble_crops_continue_across_native_tile_boundaries():
    # A shifted multi-tile patch crosses course and varied-width cluster origins.
    # Rendering tiles separately must match a single field draw at every pixel.
    origin = (96, 64)
    expected = blue_cobble_field((96, 96), origin)
    assembled = pygame.Surface((96, 96), pygame.SRCALPHA)
    for row in range(3):
        for col in range(3):
            assembled.blit(blue_cobble(col + 3, row + 2, (False,) * 4), (col * 32, row * 32))
    assert pygame.image.tobytes(assembled, "RGBA") == pygame.image.tobytes(expected, "RGBA")


def test_blue_cobble_edges_keep_whole_collision_cells_opaque():
    for edges in product((False, True), repeat=4):
        image = blue_cobble(7, 5, edges)
        assert image.get_size() == (32, 32)
        assert np.all(pygame.surfarray.array_alpha(image) == 255)


def test_authored_blue_rooms_preserve_solid_and_empty_terrain_cells():
    session = CaveSession()
    original = tuple(cave.layout for cave in session.game.CAVES)
    levels = [level for level in range(len(session.game.CAVES)) if theme_index(level) == 4]
    assert levels == [4, 13]
    for level in levels:
        session.reset(level)
        layout = session.terrain_layout()
        alpha = pygame.surfarray.array_alpha(terrain(layout, level))
        for row, cells in enumerate(layout):
            for col, tile in enumerate(cells):
                region = alpha[col * 32 : (col + 1) * 32, row * 32 : (row + 1) * 32]
                if tile == "#":
                    assert np.all(region == 255)
                elif tile == ".":
                    assert np.all(region == 0)
    assert tuple(cave.layout for cave in session.game.CAVES) == original
