"""Airborne shooting preserves the muzzle; warmer hearts keep their HUD footprint."""

import hashlib
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

from src.unity_bridge.visuals import art_sprites, explorer

RESOURCES = Path(__file__).resolve().parents[1] / "unity/Assets/Resources"


def rgba(surface):
    return np.asarray(
        Image.frombytes("RGBA", surface.get_size(), pygame.image.tobytes(surface, "RGBA"))
    )


def test_airborne_shoot_art_preserves_head_torso_and_muzzle_with_bent_legs():
    pose = art_sprites()["mylo_shoot_air"]
    assert pose.get_size() == (24, 32)
    airborne = rgba(pose)
    standing = rgba(explorer("shoot"))
    jump = rgba(explorer("jump"))
    np.testing.assert_array_equal(airborne[:22], standing[:22])
    assert not np.array_equal(airborne[22:], standing[22:])
    # The ordinary jump carries its gun lower over the right hip. Compare the
    # leg regions that are clear of that weapon before checking the boot row.
    np.testing.assert_array_equal(airborne[24:31, :13], jump[24:31, :13])
    np.testing.assert_array_equal(airborne[27:31], jump[27:31])
    assert not np.any(airborne[31, :, 3])
    assert set(np.unique(airborne[:, :, 3])) == {0, 255}


def test_airborne_shoot_export_matches_author_and_keeps_the_native_pose_bounds():
    with Image.open(RESOURCES / "Sprites/mylo_shoot_air.png") as image:
        assert image.size == (24, 32)
        exported = np.asarray(image.convert("RGBA"))
    np.testing.assert_array_equal(exported, rgba(art_sprites()["mylo_shoot_air"]))


def test_warm_heart_export_preserves_original_alpha_and_live_slot_dimensions():
    with Image.open(RESOURCES / "Interface/heart.png") as image:
        assert image.size == (18, 16)
        image = image.convert("RGBA")
        pixels = np.asarray(image)
        mask_hash = hashlib.sha256(image.getchannel("A").tobytes()).hexdigest()
    assert mask_hash == "f975071293cb78177c00a1150e75cd1342a2c11834df3d53d318b3d31fadb80a"
    assert set(np.unique(pixels[:, :, 3])) == {0, 255}
    # A broad lit lobe and magenta shadow make the small icon legible in its slots.
    warm = np.all(pixels[:, :, :3] == (255, 211, 66), axis=2)
    shadow = np.all(pixels[:, :, :3] == (117, 25, 79), axis=2)
    assert np.count_nonzero(warm) >= 8
    assert np.count_nonzero(shadow) >= 20
    assert np.max(np.where(warm)[1]) < 9
    assert np.min(np.where(shadow)[1]) >= 9
