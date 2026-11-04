"""HD Blender creature art stays a drop-in replacement for the native pixel sprites."""

import importlib.util
import sys
from pathlib import Path

import pygame
import pytest

from src.unity_bridge.visual_creatures import HD_SCALE, creature_sprites, hd_creature_sprites

ROOT = Path(__file__).resolve().parents[1]
HD = hd_creature_sprites()
PIXEL = creature_sprites()


def bounds(image: pygame.Surface) -> pygame.Rect:
    return image.get_bounding_rect(min_alpha=128)


def test_every_moving_creature_pose_has_hd_art():
    expected = {
        name
        for name in PIXEL
        if not name.startswith(("defeat_", "dinosaur_enemy_hit_"))
        or name.startswith("dinosaur_enemy_hit_")
    }
    assert set(HD) == expected
    assert len(HD) == 30


@pytest.mark.parametrize("name", sorted(HD))
def test_hd_art_is_four_times_the_pixel_sprite_and_covers_the_same_footprint(name):
    hd, pixel = HD[name], PIXEL[name]
    assert hd.get_size() == (pixel.get_width() * HD_SCALE, pixel.get_height() * HD_SCALE)
    hd_box, pixel_box = bounds(hd), bounds(pixel)
    for hd_edge, pixel_edge in zip(
        (hd_box.left, hd_box.top, hd_box.right, hd_box.bottom),
        (pixel_box.left, pixel_box.top, pixel_box.right, pixel_box.bottom),
    ):
        assert abs(hd_edge / HD_SCALE - pixel_edge) <= 1.5, (name, hd_box, pixel_box)


@pytest.mark.parametrize("name", ["slug_enemy", "dinosaur_enemy"])
def test_tall_creatures_keep_their_feet_on_the_bottom_row_in_hd(name):
    # Unity anchors these sprites by their bottom edge on the patrol body.
    for frame in range(4):
        image = HD[f"{name}_{frame}"]
        assert image.get_height() - bounds(image).bottom <= HD_SCALE


def test_hd_hit_poses_flash_white_inside_the_same_silhouette():
    for frame in range(4):
        normal, hit = HD[f"dinosaur_enemy_{frame}"], HD[f"dinosaur_enemy_hit_{frame}"]
        assert pygame.image.tobytes(normal, "RGBA")[3::4] == pygame.image.tobytes(hit, "RGBA")[3::4]
        pixels = pygame.image.tobytes(hit, "RGBA")
        solid = [tuple(pixels[n : n + 3]) for n in range(0, len(pixels), 4) if pixels[n + 3] == 255]
        # Only white and ink (and their blends along contours) remain: no green.
        assert all(color[1] - min(color[0], color[2]) < 24 for color in solid)
        assert sum(color == (255, 255, 255) for color in solid) > len(solid) / 2


def test_hd_animation_frames_differ_and_reload_identically():
    for name in ("dinosaur_enemy", "bat_enemy", "slug_enemy", "eye_flyer", "walking_rock"):
        frames = [pygame.image.tobytes(HD[f"{name}_{frame}"], "RGBA") for frame in range(4)]
        assert len(set(frames)) == 4
        assert frames == [
            pygame.image.tobytes(hd_creature_sprites()[f"{name}_{frame}"], "RGBA")
            for frame in range(4)
        ]
    assert pygame.image.tobytes(HD["walking_rock_sleep"], "RGBA") != frames[0]


def test_blender_material_ids_stay_distinct_after_srgb_encoding():
    spec = importlib.util.spec_from_file_location(
        "render_blender_sprites", ROOT / "scripts/render_blender_sprites.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    colors = list(module.IDS)
    assert len(colors) == len(set(colors)) == 24
    nearest = min(
        sum((a - b) ** 2 for a, b in zip(first, second))
        for index, first in enumerate(colors)
        for second in colors[index + 1 :]
    )
    assert nearest > 300  # Far apart enough to survive any single-texel rounding.
