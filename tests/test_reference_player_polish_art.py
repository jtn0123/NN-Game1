"""Lifted boots and white hit flashes retain each action pose's native artwork."""

import hashlib

import numpy as np
import pygame
import pytest

from src.unity_bridge.visual_materials import INK, WHITE
from src.unity_bridge.visuals import art_sprites, explorer


@pytest.mark.parametrize(
    ("pose", "frame", "unchanged_rows_hash"),
    [
        ("jump", 0, "ba810231a8c5326512fc78d15dc54da15a24b196f8044cb114f4ed2eb7c2650d"),
        ("climb", 0, "3132188c5c74fedb548617da1661c5fae68b2b2a68ce58c7bd4feef4ad04a68b"),
        ("climb", 1, "2d03ac957b0912a9848e3d845be6e537f8d130201a6cc94110383d032d6fb37f"),
    ],
)
def test_lifted_boot_cleanup_removes_only_detached_bottom_row_pixels(
    pose, frame, unchanged_rows_hash
):
    image = explorer(pose, frame)
    assert image.get_size() == (24, 32)
    raw = pygame.image.tobytes(image, "RGBA")
    # The original last row held exactly two isolated red sole pixels. Every
    # byte above it must remain unchanged; the now-empty row has no RGB debris.
    assert hashlib.sha256(raw[: 24 * 31 * 4]).hexdigest() == unchanged_rows_hash
    assert raw[24 * 31 * 4 :] == bytes(24 * 4)


def pixels(image):
    return np.frombuffer(pygame.image.tobytes(image, "RGBA"), dtype=np.uint8).reshape(
        image.get_height(), image.get_width(), 4
    )


def test_white_player_hit_variants_keep_each_action_pose_mask_and_native_bounds():
    art = art_sprites()
    names = [name for name in art if name.startswith("mylo_") and not name.endswith("_hit")]
    # Each existing pose, including legacy aliases, must have the corresponding
    # flash so action selection cannot fall back to a missing or standing sprite.
    assert len(names) == 19
    for name in names:
        normal = art[name]
        flash = art[name + "_hit"]
        assert normal.get_size() == flash.get_size() == (24, 32)
        source, white = pixels(normal), pixels(flash)
        np.testing.assert_array_equal(white[:, :, 3], source[:, :, 3])
        assert set(np.unique(white[:, :, 3])) == {0, 255}
        ink = np.all(source[:, :, :3] == INK, axis=2) & (source[:, :, 3] == 255)
        body = (source[:, :, 3] == 255) & ~ink
        assert np.all(white[:, :, :3][ink] == INK)
        assert np.all(white[:, :, :3][body] == WHITE)
        assert np.count_nonzero(body) > 100
        np.testing.assert_array_equal(white[source[:, :, 3] == 0], source[source[:, :, 3] == 0])


def test_hit_art_generation_leaves_normal_player_poses_and_aliases_intact():
    art = art_sprites()
    poses = ("idle", "jump", "fall", "land", "hurt", "climb", "shoot", "shoot_air", "idle_look")
    for pose in poses:
        np.testing.assert_array_equal(pixels(art["mylo_" + pose]), pixels(explorer(pose)))
    for frame in range(4):
        expected = pixels(explorer("walk", frame))
        np.testing.assert_array_equal(pixels(art[f"mylo_walk_{frame + 1}"]), expected)
        np.testing.assert_array_equal(pixels(art[f"mylo_run_{frame}"]), expected)
    for frame in range(2):
        np.testing.assert_array_equal(
            pixels(art[f"mylo_climb_{frame}"]), pixels(explorer("climb", frame))
        )
