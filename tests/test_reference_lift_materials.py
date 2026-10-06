"""The recognizable split lift instrument preserves its rideable footprint."""

import hashlib

import numpy as np
import pygame
import pytest

from src.unity_bridge.visual_materials import GOLD, INK, RED
from src.unity_bridge.visual_mechanisms import hover_lift, mechanism_sprites

# Copied before the fourth reference series: landing surfaces and strut poses.
BASELINE_ALPHA = "2a2d875d7526c183b6388020dc4a0cf933250504720b8a9b73a89ffdccaecf54"
BASELINE_LIP = "a78d8be1a62d08712332bf5295372ea4a2531f1343af3979914d2be601c1ada0"
BASELINE_STRUTS = (
    "3e0a445b008b301fc3adcecbb45942118ec43343c3372900bbab034b676b0058",
    "d78b20de8d05c2e58710bf7617dc0b687b0eaaf8d579c3be5630793955bf2337",
)


def rgba(image):
    return np.frombuffer(pygame.image.tobytes(image, "RGBA"), np.uint8).reshape(32, 32, 4)


@pytest.mark.parametrize("frame", range(4))
def test_split_lift_face_preserves_landing_lip_strut_poses_and_native_alpha(frame):
    image = hover_lift(frame)
    assert image.get_size() == (32, 32)
    pixels = rgba(image)
    assert set(np.unique(pixels[:, :, 3])) == {0, 255}
    assert hashlib.sha256(pixels[:, :, 3].tobytes()).hexdigest() == BASELINE_ALPHA
    assert hashlib.sha256(pixels[3:6, 5:27].tobytes()).hexdigest() == BASELINE_LIP
    assert hashlib.sha256(pixels[23:].tobytes()).hexdigest() == BASELINE_STRUTS[frame % 2]


def test_lift_has_a_broad_dark_instrument_split_by_a_pale_vertical_divider():
    for frame in range(4):
        pixels = rgba(hover_lift(frame))[:, :, :3]
        panel = pixels[7:19, 5:27]
        assert (
            np.all(panel == INK, axis=2).mean() > 0.55
        ), "The instrument should fill the housing's front"
        divider = pixels[8:18, 17]
        pale = np.all(divider >= (169, 169, 194), axis=1)
        assert (
            np.count_nonzero(pale) >= 8
        ), "The left/right instrument sections need a readable divider"
        left, right = panel[:, :12], panel[:, 13:]
        assert np.count_nonzero(np.all(left == RED, axis=2)) >= 5
        assert np.count_nonzero(np.all(right == GOLD[2], axis=2)) >= 8


def test_all_four_lift_pose_exports_and_legacy_alias_remain_repeatable():
    images = mechanism_sprites()
    assert pygame.image.tobytes(images["elevator"], "RGBA") == pygame.image.tobytes(
        hover_lift(), "RGBA"
    )
    for frame in range(4):
        expected = pygame.image.tobytes(hover_lift(frame), "RGBA")
        assert pygame.image.tobytes(images[f"elevator_{frame}"], "RGBA") == expected
        assert pygame.image.tobytes(hover_lift(frame), "RGBA") == expected
