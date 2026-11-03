"""Short pixel cycles with distinct, readable arcade creature silhouettes."""

from __future__ import annotations

from pathlib import Path

import pygame

from .visual_materials import GOLD, INK, MAGENTA, STEEL, WHITE, canvas

LEAF = (22, 151, 70)
MOSS = (11, 57, 38)
LIME = (148, 255, 138)
BELLY = (174, 211, 84)
PURPLE = (142, 42, 163)
VIOLET = (64, 24, 84)


BLENDER_SPRITES = Path(__file__).resolve().parents[2] / "ArtSource/Blender/sprites"


def dinosaur(frame: int = 0) -> pygame.Surface:
    """Hunched profile T-rex: open toothed jaw, forward eye, curled tail, clawed arms.

    Modelled and posed in Blender (``ArtSource/Blender/dinosaur_scene.py``) and
    rendered down to the creature palette by ``scripts/render_blender_sprites.py``.
    The 24px width matches the collision body, so the snout never enters a wall.
    """
    return pygame.image.load(str(BLENDER_SPRITES / f"dinosaur_{frame % 4}.png")).copy()


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


def eye_flyer(frame: int = 0) -> pygame.Surface:
    """Two-eyed flying slime; the legacy alias keeps scene and model data stable."""
    image = canvas((24, 24))
    shade, mid, light, rim = (43, 88, 40), (95, 155, 52), (132, 185, 68), (186, 211, 99)
    phase = frame % 4
    upper = (5, 4, 5, 6)[phase]
    lower = (18, 19, 18, 17)[phase]
    pygame.draw.polygon(
        image,
        INK,
        [
            (0, 11),
            (3, 9),
            (6, 8),
            (9, 6),
            (12, upper),
            (21, upper),
            (23, 7),
            (23, 18),
            (21, lower + 2),
            (12, lower + 2),
            (9, lower),
            (6, 17),
            (3, 15),
            (0, 14),
        ],
    )
    pygame.draw.polygon(
        image,
        mid,
        [
            (2, 11),
            (5, 10),
            (8, 9),
            (11, 7),
            (13, upper + 2),
            (20, upper + 2),
            (21, 7),
            (21, 18),
            (20, lower),
            (13, lower),
            (10, lower - 1),
            (7, 16),
            (4, 14),
            (2, 13),
        ],
    )
    pygame.draw.lines(
        image,
        light,
        False,
        [(2, 11), (5, 10), (8, 9), (11, 7), (13, upper + 1), (20, upper + 1)],
    )
    pygame.draw.line(image, rim, (13, upper + 1), (19, upper + 1))
    pygame.draw.lines(image, shade, False, [(7, 16), (11, lower), (18, lower + 1), (21, lower)])
    for yy in (8, 15):
        pygame.draw.rect(image, shade, (17, yy - 1, 5, 5))
        pygame.draw.rect(image, WHITE, (18, yy, 3, 3))
        pygame.draw.line(image, INK, (20, yy), (20, yy + 2))
    for xx in ((12, 13, 11, 12)[phase], 19):
        pygame.draw.rect(image, INK, (xx - 1, lower + 1, 4, 3))
        pygame.draw.line(image, (97, 112, 119), (xx, lower + 2), (xx + 1, lower + 2))
    return image


def walking_rock(frame: int = 0, asleep: bool = False) -> pygame.Surface:
    """A craggy gray boulder and purple legs; sleeping form is a plain rock."""
    image = canvas((24, 24))
    bob = 3 if asleep else (0, 1, 0, 1)[frame % 4]
    points = [(1, 12), (5, 6), (11, 3), (18, 4), (22, 10), (23, 17), (19, 21), (4, 21), (0, 17)]
    points = [(x, min(23, y + bob)) for x, y in points]
    pygame.draw.polygon(image, INK, points)
    pygame.draw.polygon(
        image,
        STEEL[2],
        [
            (3, 12 + bob),
            (7, 7 + bob),
            (12, 5 + bob),
            (17, 6 + bob),
            (20, 11 + bob),
            (21, 17 + bob),
            (17, 20 + bob),
            (5, 20 + bob),
        ],
    )
    pygame.draw.polygon(
        image, STEEL[3], [(5, 11 + bob), (9, 7 + bob), (13, 6 + bob), (16, 9 + bob), (12, 12 + bob)]
    )
    pygame.draw.polygon(
        image,
        GOLD[0],
        [(16, 8 + bob), (20, 12 + bob), (19, 17 + bob), (15, 16 + bob), (13, 13 + bob)],
    )
    pygame.draw.lines(image, STEEL[0], False, [(7, 12 + bob), (10, 15 + bob), (8, 19 + bob)])
    pygame.draw.line(image, STEEL[4], (8, 8 + bob), (12, 6 + bob))
    if not asleep:
        left, right = ((3, 16), (5, 17), (7, 15), (4, 13))[frame % 4]
        for x in (left, right):
            pygame.draw.rect(image, INK, (x, 19, 6, 5))
            pygame.draw.line(image, PURPLE, (x + 1, 19), (x + 2, 22), 2)
            pygame.draw.line(image, MAGENTA, (x + 1, 23), (x + 4, 23))
    return image


def bat(frame: int = 0) -> pygame.Surface:
    """Thin near-black wings and two tiny yellow eyes, without a green face."""
    image = canvas((24, 24))
    dark, membrane, rim, eye = (8, 10, 19), (17, 20, 32), (47, 48, 66), (255, 238, 97)
    outlines = (
        [
            (11, 13),
            (8, 9),
            (6, 7),
            (3, 4),
            (1, 1),
            (0, 2),
            (1, 8),
            (3, 7),
            (4, 12),
            (6, 11),
            (8, 16),
            (10, 17),
        ],
        [(11, 13), (8, 11), (5, 8), (2, 6), (0, 6), (1, 12), (3, 10), (5, 15), (7, 13), (9, 18)],
        [
            (11, 13),
            (8, 13),
            (5, 12),
            (1, 11),
            (0, 12),
            (2, 17),
            (4, 15),
            (6, 19),
            (8, 16),
            (10, 19),
        ],
        [
            (11, 13),
            (8, 15),
            (5, 17),
            (2, 18),
            (1, 16),
            (2, 21),
            (4, 20),
            (6, 22),
            (8, 18),
            (10, 19),
        ],
    )
    points = outlines[frame % 4]
    for flip in (False, True):
        wing = [(23 - xx, yy) if flip else (xx, yy) for xx, yy in points]
        pygame.draw.polygon(image, dark, wing)
        # A quiet one-pixel rim makes dark wings readable against black masonry.
        edge = [(23 - xx, yy) if flip else (xx, yy) for xx, yy in points[:5]]
        pygame.draw.lines(image, rim, False, edge)
        if frame % 4 < 3:
            image.set_at((20 if flip else 3, 9 + frame % 4), membrane)
    pygame.draw.polygon(
        image, dark, [(10, 10), (13, 10), (15, 13), (14, 17), (12, 20), (10, 18), (9, 14)]
    )
    pygame.draw.line(image, membrane, (11, 15), (12, 18))
    for xx in (10, 13):
        image.set_at((xx, 11), eye)
    return image


def slug(frame: int = 0) -> pygame.Surface:
    """Upright pink snake; keep the legacy alias and authoritative patrol body."""
    image = canvas((24, 32))
    phase = frame % 4
    dark, mid, light, rim = (119, 29, 78), (208, 56, 108), (244, 96, 139), (255, 151, 166)
    bend = (0, -1, 0, 1)[phase]
    pygame.draw.polygon(
        image,
        INK,
        [
            (13, 1),
            (20, 1),
            (23, 3),
            (23, 6),
            (19, 6),
            (18, 10),
            (17 + bend, 15),
            (18 + bend, 20),
            (21, 24),
            (21, 28),
            (19, 31),
            (3, 31),
            (1, 29),
            (0, 29),
            (0, 26),
            (2, 27),
            (3, 29),
            (9, 29),
            (10, 26),
            (13 + bend, 23),
            (11 + bend, 18),
            (10 + bend, 12),
            (11, 6),
            (13, 4),
        ],
    )
    pygame.draw.polygon(
        image,
        mid,
        [
            (14, 3),
            (21, 3),
            (21, 5),
            (18, 5),
            (16, 9),
            (15 + bend, 13),
            (16 + bend, 19),
            (19, 24),
            (19, 28),
            (17, 30),
            (4, 30),
            (4, 29),
            (10, 29),
            (11, 26),
            (15 + bend, 23),
            (13 + bend, 18),
            (12 + bend, 13),
            (12, 8),
            (14, 5),
        ],
    )
    pygame.draw.lines(image, light, False, [(14, 2), (19, 2), (21, 3)])
    pygame.draw.lines(
        image,
        light,
        False,
        [(12, 7), (11 + bend, 12), (12 + bend, 19), (14 + bend, 23), (11, 26), (10, 29), (4, 30)],
    )
    pygame.draw.lines(
        image, dark, False, [(16, 9), (16 + bend, 15), (17 + bend, 20), (19, 24), (19, 27)]
    )
    pygame.draw.line(image, light, (1, 28), (3, 30))
    image.set_at((13, 3), rim)
    image.set_at((19, 2), (255, 238, 97))
    image.set_at((22, 5), (242, 56, 68))
    if phase == 2:
        image.set_at((23, 5), (242, 56, 68))
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
