"""Mine material wear preserves the published native silhouettes and actor bays."""

import hashlib

import numpy as np
import pygame

from src.unity_bridge.mine import ENTRANCES, MINE_SPEC
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_equipment import barrel, warning_sign
from src.unity_bridge.visual_fidelity import mine_surfaces
from src.unity_bridge.visual_mine_dressing import TORCH_CELLS, mine_fixture_placements

# Digests record the existing 1280x768 silhouette, not the changing material colors.
TERRAIN_ALPHA = "73b6e54d9b98332ebd16c323a8fd9d8fc760b184d6b9227d225545e9ba0059d5"
PROPS_ALPHA = "bcef4f85493f01be1a0b7adff8a8aafb24190418080c6706205111e5f7cd7789"
WALL_RGBA = "421eb5811e0f5051017e6f25d1d9064bcd74f4f81e4be33e168f7f588ca04249"
NON_TIMBER_TERRAIN = "646efed21b70e0a999bc1d465488c44d1b9b9d11d23deb1623fe3c74d848f105"
ACCEPTED_TIMBER_TERRAIN = "ee9749d5d8606810de7e89c11c7f2b44650ef3de1764bcce4f091e2f12e64a4c"
ACCEPTED_TIMBER_PROPS = "d4d1a3eae299da442ff9f901da90a43cf3fe11d32c47c4a6d186fb7155bc09bf"


def rgba(image):
    return np.frombuffer(pygame.image.tobytes(image, "RGBA"), np.uint8).reshape(
        image.get_height(), image.get_width(), 4
    )


def test_accepted_mine_materials_preserve_published_wall_and_native_silhouettes():
    wall, terrain, props = mine_surfaces(MINE_SPEC.layout)
    assert wall.get_size() == terrain.get_size() == props.get_size() == (1280, 768)
    assert hashlib.sha256(rgba(wall).tobytes()).hexdigest() == WALL_RGBA
    for image, digest in ((terrain, TERRAIN_ALPHA), (props, PROPS_ALPHA)):
        alpha = rgba(image)[:, :, 3]
        assert set(np.unique(alpha)) == {0, 255}
        assert hashlib.sha256(alpha.tobytes()).hexdigest() == digest


def test_mine_rock_wall_keeps_dark_palette_and_breaks_the_old_16px_grid():
    wall, _, _ = mine_surfaces(MINE_SPEC.layout)
    pixels = rgba(wall)
    assert wall.get_size() == (1280, 768)
    assert np.unique(pixels[:, :, 3]).tolist() == [255]
    palette = set(map(tuple, np.unique(pixels[:, :, :3].reshape(-1, 3), axis=0)))
    assert palette == {(5, 7, 14), (18, 23, 34), (31, 38, 51), (40, 47, 59)}
    luminance = (pixels[:, :, :3] * np.array((0.2126, 0.7152, 0.0722))).sum(2).mean()
    assert 16 < luminance <= 20.26, "Dark mortar must retain foreground contrast"
    repetition = np.all(pixels[:, 16:, :3] == pixels[:, :-16, :3], axis=2).mean()
    assert repetition < 0.6, "Large uneven clusters must break the old 16-pixel wallpaper"


def test_rock_revision_preserves_every_accepted_foreground_pixel():
    _, terrain, props = mine_surfaces(MINE_SPEC.layout)
    assert hashlib.sha256(rgba(terrain).tobytes()).hexdigest() == ACCEPTED_TIMBER_TERRAIN
    assert hashlib.sha256(rgba(props).tobytes()).hexdigest() == ACCEPTED_TIMBER_PROPS


def test_worn_beams_have_dark_undersides_and_varied_broken_grain():
    _, terrain, _ = mine_surfaces(MINE_SPEC.layout)
    pixels = rgba(terrain)
    for row in (7, 11, 15, 19):
        top = pixels[row * 32 + 3 : row * 32 + 10, 7 * 32 : 17 * 32, :3].mean()
        bottom = pixels[row * 32 + 22 : row * 32 + 28, 7 * 32 : 17 * 32, :3].mean()
        assert top - bottom > 40, "A beam's lower face should be darker than its worn upper face"
    # Several actual adjacent boards must differ; a few copies of one scratch are insufficient.
    boards = {pixels[7 * 32 : 8 * 32, col * 32 : (col + 1) * 32].tobytes() for col in range(7, 19)}
    assert len(boards) >= 8


def test_timber_wear_preserves_every_chain_and_stone_terrain_pixel():
    _, terrain, _ = mine_surfaces(MINE_SPEC.layout)
    pixels = rgba(terrain)
    untouched = np.ones(pixels.shape[:2], dtype=bool)
    for row, line in enumerate(MINE_SPEC.layout):
        for col, cell in enumerate(line):
            if row in (7, 11, 15, 19) and cell == "#":
                untouched[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32] = False
    assert hashlib.sha256(pixels[untouched].tobytes()).hexdigest() == NON_TIMBER_TERRAIN


def test_post_and_brace_wear_is_repeatable_and_confined_to_existing_faces():
    first = mine_surfaces(MINE_SPEC.layout)
    second = mine_surfaces(MINE_SPEC.layout)
    assert all(
        pygame.image.tobytes(a, "RGBA") == pygame.image.tobytes(b, "RGBA")
        for a, b in zip(first, second)
    )
    props = rgba(first[2])
    posts = {
        props[row * 32 : (row + 1) * 32, 18 * 32 + 6 : 18 * 32 + 26].tobytes()
        for row in (9, 10, 13, 14, 17, 18, 21, 22)
    }
    assert (
        len(posts) >= 6
    ), "Long supports should have distinct short grain rather than one repeated stripe"
    # Caps retain both diagonal arms and the existing central upright.
    for col in (6, 18, 33):
        x, y = col * 32, 8 * 32
        assert props[y + 3, x - 7, 3] == props[y + 3, x + 39, 3] == 255
        assert props[y + 31, x + 16, 3] == 255
        assert props[y + 29, x - 7, 3] == props[y + 29, x + 39, 3] == 0


def test_wear_preserves_fixtures_labels_torches_and_game_state():
    session = CaveSession()
    snapshot = session.handle({"op": "mine"})
    observation = session.game.get_state().copy()
    _, _, image = mine_surfaces(MINE_SPEC.layout)
    props = rgba(image)
    art = {"barrel": barrel(), "danger_sign": warning_sign()}
    assert len(mine_fixture_placements(MINE_SPEC.layout)) == 6
    for fixture in mine_fixture_placements(MINE_SPEC.layout):
        expected = rgba(art[fixture["sprite"]])
        x, y = fixture["col"] * 32, fixture["row"] * 32
        h, w = expected.shape[:2]
        np.testing.assert_array_equal(props[y : y + h, x : x + w], expected)
    protected = set(ENTRANCES) | {(col, row - 1) for col, row in ENTRANCES} | set(TORCH_CELLS)
    protected |= {
        (col, row)
        for row, line in enumerate(MINE_SPEC.layout)
        for col, cell in enumerate(line)
        if cell in "HP"
    }
    for col, row in protected:
        assert not np.any(props[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32, 3])
    np.testing.assert_array_equal(session.game.get_state(), observation)
    assert session.snapshot()["entities"] == snapshot["entities"]
