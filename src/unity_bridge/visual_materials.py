"""Shared pixel colors and deliberate material wear for the retro artwork."""

from __future__ import annotations

import pygame

from src.game.crystal_caves_art import CrystalCavesArt

Color = tuple[int, int, int]
TILE = 32
INK: Color = (7, 12, 23)
WHITE: Color = (255, 255, 255)
YELLOW: Color = (255, 255, 85)
GREEN: Color = (85, 255, 85)
CYAN: Color = (85, 255, 255)
RED: Color = (255, 85, 85)
BLUE: Color = (85, 85, 255)
MAGENTA: Color = (255, 85, 255)
STEEL = ((33, 44, 63), (67, 82, 101), (116, 139, 158), (174, 197, 212), (226, 240, 242))
RUST = ((78, 42, 33), (147, 73, 38), (204, 122, 57))
GOLD = ((99, 48, 26), (177, 102, 29), (238, 176, 49), YELLOW, (255, 246, 193))
PINK = ((75, 19, 71), (140, 28, 139), (210, 45, 206), MAGENTA, (255, 177, 239))
GEM_RAMPS = {
    BLUE: ((20, 23, 85), (39, 49, 176), BLUE, (142, 183, 255), (209, 237, 255)),
    GREEN: ((11, 57, 38), (22, 151, 70), GREEN, (148, 255, 138), (218, 255, 201)),
    YELLOW: (GOLD[0], GOLD[1], GOLD[2], YELLOW, GOLD[4]),
    RED: ((87, 17, 37), (184, 35, 61), RED, (255, 152, 144), (255, 220, 201)),
    MAGENTA: ((64, 24, 84), (142, 42, 163), MAGENTA, (230, 153, 255), (253, 224, 255)),
}


def canvas(size: tuple[int, int] = (32, 32)) -> pygame.Surface:
    return pygame.Surface(size, pygame.SRCALPHA)


def bolt(image: pygame.Surface, x: int, y: int) -> None:
    pygame.draw.rect(image, STEEL[0], (x, y, 3, 3))
    image.set_at((x, y), STEEL[4])
    image.set_at((x + 1, y + 1), STEEL[2])


def scratch(
    image: pygame.Surface,
    x: int,
    y: int,
    length: int,
    dark: Color = STEEL[0],
    light: Color = STEEL[3],
) -> None:
    """A short gouge with a one-pixel lit edge, not scattered noise."""
    pygame.draw.line(image, dark, (x, y), (x + length, y - 1))
    pygame.draw.line(image, light, (x, y + 1), (x + max(1, length - 1), y))


def metal(image: pygame.Surface, rect: tuple[int, int, int, int], worn: bool = True) -> None:
    x, y, w, h = rect
    pygame.draw.rect(image, INK, rect)
    pygame.draw.rect(image, STEEL[2], (x + 1, y + 1, w - 2, h - 2))
    pygame.draw.line(image, STEEL[4], (x + 1, y + 1), (x + w - 2, y + 1))
    pygame.draw.line(image, STEEL[3], (x + 1, y + 2), (x + 1, y + h - 3))
    pygame.draw.line(image, STEEL[1], (x + 2, y + h - 2), (x + w - 2, y + h - 2))
    pygame.draw.line(image, STEEL[0], (x + w - 2, y + 2), (x + w - 2, y + h - 2))
    if worn and w >= 16 and h >= 7:
        scratch(image, x + 3, y + 3, 3)
        pygame.draw.line(image, STEEL[1], (x + w - 8, y + 1), (x + w - 5, y + 1))
        pygame.draw.line(image, STEEL[3], (x + w - 8, y + 2), (x + w - 6, y + 2))
        pygame.draw.line(image, RUST[0], (x + 2, y + h - 3), (x + 5, y + h - 3))
        image.set_at((x + 3, y + h - 4), RUST[1])
        image.set_at((x + 4, y + h - 3), RUST[2])


def lettering(image: pygame.Surface, text: str, x: int, y: int, color: Color = WHITE) -> None:
    image.blit(CrystalCavesArt().text(text, color, scale=1), (x, y))


def raygun() -> pygame.Surface:
    image = canvas((24, 12))
    pygame.draw.polygon(
        image,
        INK,
        [(0, 3), (3, 1), (21, 1), (23, 3), (23, 7), (10, 7), (8, 11), (3, 11), (4, 7), (0, 7)],
    )
    pygame.draw.rect(image, STEEL[2], (2, 3, 20, 3))
    pygame.draw.line(image, STEEL[4], (4, 2), (20, 2))
    pygame.draw.line(image, STEEL[1], (2, 6), (19, 6))
    pygame.draw.rect(image, STEEL[0], (4, 7, 5, 4))
    for yy in (7, 9):
        pygame.draw.line(image, GOLD[1], (5, yy), (7, yy))
    for xx in (12, 15):
        pygame.draw.line(image, STEEL[0], (xx, 3), (xx, 5))
        image.set_at((xx + 1, 3), STEEL[3])
    pygame.draw.rect(image, STEEL[1], (20, 3, 3, 3))
    pygame.draw.rect(image, CYAN, (21, 4, 2, 1))
    scratch(image, 5, 4, 3)
    image.set_at((9, 2), STEEL[0])
    image.set_at((10, 3), RUST[1])
    return image
