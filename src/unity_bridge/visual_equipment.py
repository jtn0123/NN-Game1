"""Pixel pressure vessels, two-tone barrels and reference-style warning plates."""

from __future__ import annotations

from typing import Sequence

import pygame

from .visual_materials import GOLD, INK, RED, RUST, STEEL, WHITE, YELLOW, canvas, lettering


def air_render_height(layout: Sequence[str], col: int, row: int) -> int:
    """Expand upward only into an authored empty cell; keep the original base."""
    return 64 if row > 0 and layout[row - 1][col] == "." else 32


def air_tank(frame: int = 0) -> pygame.Surface:
    image = canvas((32, 64))
    # Short valve neck above a long silver vessel; no rescaling of the small tank.
    pygame.draw.rect(image, INK, (9, 0, 11, 7))
    pygame.draw.rect(image, STEEL[3], (10, 2, 9, 4))
    pygame.draw.line(image, GOLD[3], (12, 1), (16, 1))
    pygame.draw.line(image, WHITE, (10, 2), (16, 2))
    pygame.draw.rect(image, INK, (2, 6, 27, 58))
    pygame.draw.rect(image, STEEL[3], (3, 7, 25, 55))
    pygame.draw.line(image, WHITE, (3, 7), (26, 7))
    pygame.draw.line(image, WHITE, (3, 8), (3, 60))
    pygame.draw.line(image, STEEL[1], (27, 8), (27, 61), 2)
    pygame.draw.line(image, STEEL[0], (4, 62), (27, 62))
    pygame.draw.rect(image, INK, (6, 11, 18, 17))
    pygame.draw.rect(image, STEEL[2], (7, 12, 16, 15))
    pygame.draw.rect(image, INK, (9, 14, 12, 10))
    for x, y, w, h in (
        ((15, 14, 5, 4), (10, 20, 5, 3)) if frame % 2 else ((10, 14, 5, 4), (15, 18, 5, 6))
    ):
        pygame.draw.rect(image, RED, (x, y, w, h))
        pygame.draw.line(image, (184, 35, 61), (x, y + h - 1), (x + w - 1, y + h - 1))
    pygame.draw.rect(image, STEEL[1], (5, 31, 22, 14))
    pygame.draw.rect(image, (215, 224, 225), (6, 32, 20, 12))
    lettering(image, "AIR", 7, 34, (178, 32, 49))
    pygame.draw.rect(image, INK, (6, 48, 18, 9))
    for x, color in ((7, RED), (12, YELLOW), (18, (40, 158, 79))):
        pygame.draw.rect(image, color, (x, 50, 5, 4))
    pygame.draw.line(image, STEEL[2], (6, 59), (23, 59))
    pygame.draw.line(image, RUST[0], (7, 61), (11, 61))
    image.set_at((9, 60), RUST[1])
    pygame.draw.line(image, STEEL[1], (20, 8), (23, 8))
    image.set_at((21, 9), STEEL[4])
    return image


def barrel() -> pygame.Surface:
    image = canvas()
    pygame.draw.polygon(
        image, INK, [(4, 1), (26, 1), (29, 4), (29, 29), (26, 31), (4, 31), (2, 28), (2, 4)]
    )
    pygame.draw.rect(image, (18, 60, 71), (4, 3, 24, 26))
    pygame.draw.rect(image, (41, 121, 143), (5, 4, 12, 24))
    pygame.draw.rect(image, (74, 180, 193), (6, 5, 4, 23))
    pygame.draw.line(image, (192, 238, 233), (6, 5), (6, 27))
    pygame.draw.rect(image, (113, 27, 71), (17, 4, 10, 24))
    pygame.draw.rect(image, (184, 42, 97), (18, 5, 4, 23))
    pygame.draw.line(image, (232, 91, 143), (18, 6), (18, 26))
    for y in (2, 14, 28):
        pygame.draw.rect(image, INK, (3, y, 26, 4))
        pygame.draw.line(image, STEEL[3], (4, y + 1), (27, y + 1))
        pygame.draw.line(image, STEEL[1], (4, y + 2), (27, y + 2))
    pygame.draw.line(image, (16, 59, 70), (10, 9), (12, 10))
    pygame.draw.line(image, (100, 181, 191), (10, 10), (12, 11))
    pygame.draw.line(image, (70, 23, 53), (24, 22), (26, 21))
    image.set_at((24, 23), (167, 71, 107))
    pygame.draw.line(image, RUST[0], (23, 30), (26, 30))
    image.set_at((25, 29), RUST[1])
    return image


def warning_sign(reverse_gravity: bool = False) -> pygame.Surface:
    image = canvas((64, 32))
    pygame.draw.rect(image, INK, (0, 2, 64, 29))
    pygame.draw.rect(image, (118, 29, 52), (1, 3, 62, 26))
    pygame.draw.rect(image, (173, 39, 61), (3, 5, 58, 22))
    pygame.draw.line(image, (225, 77, 79), (2, 3), (61, 3))
    pygame.draw.line(image, (74, 22, 43), (2, 28), (61, 28))
    for x in (3, 60):
        image.set_at((x, 5), STEEL[4])
        image.set_at((x, 26), STEEL[1])
    if reverse_gravity:
        lettering(image, "REVERSE", 11, 6, YELLOW)
        lettering(image, "GRAVITY", 11, 18, YELLOW)
    else:
        lettering(image, "DANGER", 14, 12, YELLOW)
        pygame.draw.line(image, GOLD[2], (7, 8), (10, 8))
        pygame.draw.line(image, GOLD[2], (53, 24), (56, 24))
    pygame.draw.line(image, STEEL[1], (46, 3), (49, 3))
    image.set_at((47, 4), STEEL[2])
    pygame.draw.line(image, RUST[0], (5, 28), (9, 28))
    return image


def equipment_sprites() -> dict[str, pygame.Surface]:
    return {
        "air_tank_tall": air_tank(),
        "air_tank_tall_0": air_tank(),
        "air_tank_tall_1": air_tank(1),
        "barrel": barrel(),
        "danger_sign": warning_sign(),
        "reverse_gravity_sign": warning_sign(True),
    }
