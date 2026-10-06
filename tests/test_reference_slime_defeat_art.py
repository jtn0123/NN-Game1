"""Slime defeat is a red pixel pulse; accepted creature artwork stays intact."""

import hashlib

import pygame
import pytest

from src.unity_bridge import visual_creatures
from src.unity_bridge.visual_creatures import creature_sprites


@pytest.mark.parametrize("phase", range(4))
def test_slime_defeat_stages_are_centered_warm_red_with_hard_native_alpha(phase):
    image = creature_sprites()[f"defeat_slime_{phase}"]
    assert image.get_size() == (32, 32)
    pixels = pygame.image.tobytes(image, "RGBA")
    assert set(pixels[3::4]) == {0, 255}
    colors = {tuple(pixels[n : n + 3]) for n in range(0, len(pixels), 4) if pixels[n + 3]}
    assert all(red > green * 2 and red > blue * 1.5 for red, green, blue in colors)
    bounds = image.get_bounding_rect()
    assert bounds.center == (16, 16)
    assert 0 < bounds.left < bounds.right < 32
    assert 0 < bounds.top < bounds.bottom < 32


def test_red_pulse_has_four_deterministic_expanding_then_disappearing_shapes():
    first = creature_sprites()
    second = creature_sprites()
    names = [f"defeat_slime_{phase}" for phase in range(4)]
    frames = [pygame.image.tobytes(first[name], "RGBA") for name in names]
    assert len(set(frames)) == 4
    assert frames == [pygame.image.tobytes(second[name], "RGBA") for name in names]
    assert frames == [
        pygame.image.tobytes(visual_creatures.slime_pulse(phase + 4), "RGBA") for phase in range(4)
    ]
    widths = [first[name].get_bounding_rect().width for name in names]
    assert widths[0] < widths[1] < widths[2]
    assert widths[3] < widths[2]


def test_all_preexisting_creature_poses_match_the_accepted_fourth_round_baseline():
    # This hash was recorded directly from the accepted fourth-round baseline
    # PNGs, then independently matched against the source author's RGBA bytes.
    # Only the four new defeat sprites are excluded; slime, snake, bat, rock,
    # dinosaur, hit and bone poses all remain protected.
    existing = {
        name: image
        for name, image in creature_sprites().items()
        if not name.startswith("defeat_slime_")
    }
    assert len(existing) == 34
    digest = hashlib.sha256()
    for name, image in sorted(existing.items()):
        digest.update(name.encode() + b"\0")
        digest.update(bytes(image.get_size()))
        digest.update(pygame.image.tobytes(image, "RGBA"))
    assert digest.hexdigest() == "eec3d1b82df400806afcc2afe88bd4f7defc0b5b68f4d53433c9153fa5dc9824"
