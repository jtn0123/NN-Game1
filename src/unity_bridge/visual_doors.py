"""Native colored cave gates with clear-headroom presentation footprints."""

from __future__ import annotations

from typing import Sequence

import pygame

from .visual_materials import INK, RED, RUST, STEEL, WHITE, YELLOW, canvas


def door_render_height(layout: Sequence[str], col: int, row: int) -> int:
    """Keep the original base; expand only into an authored empty cell above."""
    return 64 if row > 0 and layout[row - 1][col] == "." else 32


def door_headroom_cells(layout: Sequence[str]) -> set[tuple[int, int]]:
    """Dressing must leave the upper half of every expanded gate/exit visible."""
    return {
        (col, row - 1)
        for row, cells in enumerate(layout)
        for col, symbol in enumerate(cells)
        if symbol in "DEd" and door_render_height(layout, col, row) == 64
    }


def cave_door(color: str = "green", height: int = 64, opened: bool = False) -> pygame.Surface:
    """A small upper window and broad colored lower panel inside yellow trim."""
    body, light, shade = {
        "green": ((64, 116, 75), (105, 156, 88), (32, 70, 49)),
        "red": ((153, 42, 57), (208, 74, 75), (83, 25, 43)),
        "blue": ((47, 68, 136), (83, 116, 196), (30, 38, 82)),
    }[color]
    image = canvas((32, height))
    pygame.draw.rect(image, INK, (2, 0, 28, height))
    pygame.draw.rect(image, STEEL[2], (3, 1, 26, height - 2))
    pygame.draw.line(image, WHITE, (3, 1), (28, 1))
    pygame.draw.line(image, STEEL[4], (3, 2), (3, height - 2))
    pygame.draw.line(image, STEEL[0], (28, 2), (28, height - 2))
    pygame.draw.rect(image, body, (5, 3, 22, height - 5))
    pygame.draw.rect(image, YELLOW, (5, 3, 22, height - 5), 1)
    pygame.draw.line(image, light, (7, 5), (24, 5))
    pygame.draw.line(image, shade, (25, 5), (25, height - 4))
    window_y = 8 if height == 64 else 6
    window_h = 12 if height == 64 else 8
    pygame.draw.rect(image, shade, (7, window_y - 1, 18, window_h + 3))
    pygame.draw.rect(image, INK, (8, window_y, 16, window_h))
    pygame.draw.line(image, YELLOW, (8, window_y - 1), (24, window_y - 1))
    pygame.draw.line(image, light, (8, window_y + window_h + 1), (24, window_y + window_h + 1))
    # The window's red lock lamp becomes green when the real exit is unlocked.
    status = (85, 255, 85) if opened else RED
    pygame.draw.rect(image, status, (19, window_y + 3, 3, 4))
    image.set_at((19, window_y + 3), WHITE if opened else (255, 162, 149))
    divider = 30 if height == 64 else 16
    pygame.draw.line(image, INK, (6, divider), (26, divider), 2)
    pygame.draw.line(image, YELLOW, (7, divider - 1), (24, divider - 1))
    pygame.draw.line(image, light, (7, divider + 1), (24, divider + 1))
    pygame.draw.rect(image, shade, (7, divider + 3, 18, height - divider - 7))
    if opened:
        pygame.draw.rect(image, INK, (9, divider + 4, 15, height - divider - 9))
        pygame.draw.line(image, light, (8, divider + 4), (8, height - 5))
        center_y = divider + (height - divider) // 2
        pygame.draw.lines(
            image, (85, 255, 85), False, [(15, center_y - 4), (19, center_y), (15, center_y + 4)], 2
        )
        pygame.draw.line(image, WHITE, (12, center_y), (18, center_y))
    else:
        pygame.draw.rect(image, body, (8, divider + 4, 16, height - divider - 9))
        pygame.draw.line(image, light, (8, divider + 4), (8, height - 5))
        pygame.draw.line(image, shade, (23, divider + 5), (23, height - 5))
    # Sparse edge wear leaves the window, trim and colored panel distinct.
    pygame.draw.line(image, STEEL[1], (23, 1), (25, 1))
    image.set_at((24, 2), STEEL[3])
    pygame.draw.line(image, RUST[0], (7, height - 2), (11, height - 2))
    image.set_at((9, height - 3), RUST[1])
    pygame.draw.line(image, shade, (19, height - 6), (21, height - 7))
    return image


def door_sprites() -> dict[str, pygame.Surface]:
    images = {}
    for color in ("red", "blue"):
        images[f"door_{color}"] = cave_door(color, 32)
        images[f"door_{color}_tall"] = cave_door(color)
    for opened in (False, True):
        name = "exit_open" if opened else "exit_locked"
        images[name] = cave_door(height=32, opened=opened)
        images[name + "_tall"] = cave_door(opened=opened)
    return images
