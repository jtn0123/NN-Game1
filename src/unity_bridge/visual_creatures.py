"""Creature poses: Blender-modelled enemies plus hand-pixelled hit and defeat effects."""

from __future__ import annotations

from pathlib import Path

import pygame

from .visual_materials import INK, WHITE, canvas

BLENDER_SPRITES = Path(__file__).resolve().parents[2] / "ArtSource/Blender/sprites"


def blender_sprite(name: str, pose: str) -> pygame.Surface:
    """A creature pose modelled in Blender (``ArtSource/Blender/creatures.py``).

    ``scripts/render_blender_sprites.py`` renders and converts the committed
    PNGs; loading them keeps the export free of any Blender dependency.
    """
    return pygame.image.load(str(BLENDER_SPRITES / f"{name}_{pose}.png")).copy()


def dinosaur(frame: int = 0) -> pygame.Surface:
    """Upright green dinosaur: open jaw, teeth, brow, gold claws, four-pose stride."""
    return blender_sprite("dinosaur_enemy", str(frame % 4))


def bat(frame: int = 0) -> pygame.Surface:
    """Violet-black bat with lit finger bones, yellow eyes and tiny fangs."""
    return blender_sprite("bat_enemy", str(frame % 4))


def slug(frame: int = 0) -> pygame.Surface:
    """Upright pink snake; swaying neck, pale belly, flicking tongue, floor-hugging tail."""
    return blender_sprite("slug_enemy", str(frame % 4))


def eye_flyer(frame: int = 0) -> pygame.Surface:
    """Glossy two-eyed flying slime; the legacy alias keeps scene and model data stable."""
    return blender_sprite("eye_flyer", str(frame % 4))


def walking_rock(frame: int = 0, asleep: bool = False) -> pygame.Surface:
    """Faceted mossy boulder on stubby purple legs; sleeping form is a plain rock."""
    return blender_sprite("walking_rock", "sleep" if asleep else str(frame % 4))


def dinosaur_hit(frame: int = 0) -> pygame.Surface:
    """Hard white hit response with the original pose's outline and alpha."""
    image = dinosaur(frame)
    for yy in range(image.get_height()):
        for xx in range(image.get_width()):
            color = image.get_at((xx, yy))
            if color.a and (color.r, color.g, color.b) != INK:
                image.set_at((xx, yy), WHITE)
    return image


def dinosaur_bones(frame: int = 0) -> pygame.Surface:
    """Four discrete angular breakup poses, with hard native-pixel edges."""
    image = canvas()
    poses = (
        [
            [(10, 10), (13, 13), (16, 10)],
            [(20, 9), (20, 14), (24, 17)],
            [(11, 18), (15, 21), (13, 24)],
            [(21, 20), (24, 24)],
        ],
        [
            [(5, 8), (9, 9), (11, 14)],
            [(21, 6), (26, 10), (23, 14)],
            [(7, 22), (12, 25), (15, 22)],
            [(22, 22), (27, 27)],
        ],
        [
            [(3, 12), (8, 9), (11, 10)],
            [(21, 4), (25, 7), (28, 4)],
            [(5, 25), (9, 28), (13, 25)],
            [(24, 23), (28, 20), (28, 27)],
        ],
        [[(3, 18), (6, 15), (8, 16)], [(25, 3), (28, 5)], [(6, 29), (9, 27)], [(26, 26), (28, 29)]],
    )
    phase = frame % 4
    for points in poses[phase]:
        pygame.draw.lines(image, INK, False, points, 3)
        pygame.draw.lines(image, WHITE, False, points)
        for xx, yy in (points[0], points[-1]):
            image.set_at((xx - 1, yy), WHITE)
            image.set_at((xx + 1, yy), WHITE)
    for xx, yy in ((15 - phase * 2, 7), (27, 16 + phase * 2)):
        image.set_at((xx, yy), (218, 77, 119))
    return image


def slime_pulse(frame: int = 0) -> pygame.Surface:
    """Four native red defeat stages; placement and lifetime follow the event."""
    image = canvas((32, 32))
    dark, mid, light = (119, 28, 54), (198, 53, 67), (246, 79, 86)
    phase = frame % 4
    radius = (2, 5, 4, 2)[phase]
    pygame.draw.polygon(
        image,
        dark,
        [(16, 16 - radius), (16 + radius, 16), (16, 16 + radius), (16 - radius, 16)],
    )
    if phase < 2:
        pygame.draw.polygon(
            image,
            mid,
            [
                (16, 16 - radius),
                (16 + radius - 1, 16),
                (16, 16 + radius - 1),
                (17 - radius, 16),
            ],
        )
    else:
        pygame.draw.rect(image, mid, (14, 13, 5, 7) if phase == 2 else (15, 15, 3, 3))
        reach = 10 if phase == 2 else 7
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            pygame.draw.line(
                image,
                mid,
                (16 + dx * (radius + 1), 16 + dy * (radius + 1)),
                (16 + dx * reach, 16 + dy * reach),
                2 if phase == 2 else 1,
            )
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            pygame.draw.line(image, mid, (16 + dx * 5, 16 + dy * 5), (16 + dx * 7, 16 + dy * 7))
    image.set_at((16, 15), light)
    return image


CREATURES = {
    "bat_enemy": bat,
    "slug_enemy": slug,
    "dinosaur_enemy": dinosaur,
    "eye_flyer": eye_flyer,
    "walking_rock": walking_rock,
}


def creature_sprites() -> dict[str, pygame.Surface]:
    images = {
        f"{name}_{frame}": draw(frame) for name, draw in CREATURES.items() for frame in range(4)
    }
    images.update({name: images[f"{name}_0"] for name in CREATURES})
    images.update({f"dinosaur_enemy_hit_{frame}": dinosaur_hit(frame) for frame in range(4)})
    images.update({f"defeat_bones_{frame}": dinosaur_bones(frame) for frame in range(4)})
    images.update({f"defeat_slime_{frame}": slime_pulse(frame) for frame in range(4)})
    images["walking_rock_sleep"] = walking_rock(asleep=True)
    return images
