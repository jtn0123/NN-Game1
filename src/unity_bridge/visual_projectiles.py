"""Discrete silver capsule art centered on the existing 10x4 projectile body."""

from __future__ import annotations

import pygame

from .visual_materials import INK, STEEL, WHITE, canvas


def capsule(frame: int = 0) -> pygame.Surface:
    """The 16x8 artwork overhangs its unchanged collider by 3px/2px per side."""
    frame %= 4
    image = canvas((16, 8))
    if frame % 2:
        # Tail vanes turn through hard pixel poses inside fixed artwork bounds.
        pygame.draw.polygon(image, INK, [(2, 0), (6, 0), (8, 3), (6, 7), (2, 7), (3, 4)])
        pygame.draw.polygon(
            image, STEEL[3] if frame == 1 else STEEL[1], [(3, 1), (5, 1), (6, 3), (4, 3)]
        )
        pygame.draw.polygon(
            image, STEEL[1] if frame == 1 else STEEL[3], [(4, 5), (6, 5), (5, 6), (3, 6)]
        )
        image.set_at((4, 1 if frame == 1 else 6), WHITE)
    pygame.draw.polygon(
        image, INK, [(1, 2), (13, 2), (15, 3), (15, 5), (13, 6), (1, 6), (0, 5), (0, 3)]
    )
    pygame.draw.rect(image, (104, 107, 149), (2, 3, 11, 3))
    pygame.draw.line(image, (161, 166, 194), (2, 3), (12, 3))
    pygame.draw.line(image, (64, 68, 100), (2, 5), (12, 5))
    pygame.draw.line(image, STEEL[3], (13, 3), (13, 5))
    image.set_at((14, 4), WHITE)
    image.set_at((1, 4), STEEL[2])
    pygame.draw.line(image, WHITE, (8 if frame < 2 else 5, 4), (9 if frame < 2 else 6, 4))
    return image


def bat_egg() -> pygame.Surface:
    image = canvas((12, 16))
    pygame.draw.polygon(
        image, INK, [(5, 0), (8, 1), (11, 8), (10, 13), (7, 15), (3, 14), (0, 10), (1, 5)]
    )
    pygame.draw.polygon(
        image, STEEL[3], [(5, 2), (7, 2), (9, 8), (8, 12), (5, 13), (2, 10), (3, 5)]
    )
    pygame.draw.line(image, WHITE, (5, 3), (4, 7), 2)
    pygame.draw.line(image, STEEL[1], (7, 9), (6, 12), 2)
    return image


def projectile_sprites() -> dict[str, pygame.Surface]:
    images = {f"bullet_{frame}": capsule(frame) for frame in range(4)}
    images["bullet"] = images["bullet_0"]
    images["bat_egg"] = bat_egg()
    return images
