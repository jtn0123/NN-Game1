"""Dark, uneven native-pixel rock clusters for the main mine's rear wall."""

from __future__ import annotations

import pygame

from .visual_materials import Color, canvas

MORTAR: Color = (5, 7, 14)
SHADOW: Color = (18, 23, 34)
FACE: Color = (31, 38, 51)
EDGE: Color = (40, 47, 59)


def _rock_hash(col: int, row: int, salt: int = 0) -> int:
    """Stable coordinate variation, independent of the game and random state."""
    value = (col * 374761393 + row * 668265263 + salt * 982451653) & 0xFFFFFFFF
    value = ((value ^ (value >> 13)) * 1274126177) & 0xFFFFFFFF
    return value ^ (value >> 16)


def mine_rock_wall(size: tuple[int, int]) -> pygame.Surface:
    """Large jittered silhouettes, small uneven fillers and dark mortar gaps."""
    width, height = size
    wall = canvas(size)
    wall.fill(MORTAR)
    outlines = (
        ((2, 4), (5, 1), (11, 0), (15, 3), (16, 9), (13, 14), (7, 16), (2, 12), (0, 7)),
        ((0, 5), (2, 2), (8, 0), (14, 1), (16, 5), (15, 10), (12, 16), (6, 15), (2, 11)),
        ((1, 2), (6, 0), (12, 2), (13, 5), (16, 7), (14, 13), (9, 16), (3, 14), (0, 9)),
        ((0, 7), (4, 2), (9, 0), (14, 4), (16, 10), (13, 15), (5, 16), (2, 12)),
    )
    for row in range(-1, height // 28 + 2):
        for col in range(-1, width // 32 + 2):
            variation = _rock_hash(col, row)
            x = col * 32 + (14 if row % 2 else 0) + variation % 13 - 6
            y = row * 28 + (variation >> 5) % 15 - 7
            w = 18 + (variation >> 9) % 12
            h = 15 + (variation >> 14) % 14
            shape = outlines[(variation >> 19) % len(outlines)]
            points = [(x + xx * w // 16, y + yy * h // 16) for xx, yy in shape]
            pygame.draw.polygon(wall, SHADOW, points)
            inset = [(x + 2 + xx * (w - 5) // 16, y + 2 + yy * (h - 5) // 16) for xx, yy in shape]
            pygame.draw.polygon(wall, FACE, inset)
            # A broken top-left facet occupies only a few native pixels.
            if variation % 3 == 0:
                pygame.draw.line(wall, EDGE, (x + w // 3, y + 3), (x + w // 3 + 2, y + 2))
            if variation % 4 == 0:
                pygame.draw.lines(
                    wall,
                    SHADOW,
                    False,
                    [
                        (x + w - 6, y + h // 2),
                        (x + w - 8, y + h // 2 + 3),
                        (x + w - 5, y + h // 2 + 5),
                    ],
                )
            for chip in range(3):
                filler = _rock_hash(col, row, chip + 1)
                if chip == 0:
                    fx, fy = x + 25 + filler % 9, y + (filler >> 5) % 25
                elif chip == 1:
                    fx, fy = x + filler % 24, y + 23 + (filler >> 5) % 8
                else:
                    fx, fy = x + filler % 28 - 3, y + (filler >> 5) % 29 - 2
                fw, fh = 3 + (filler >> 9) % 5, 4 + (filler >> 13) % 5
                pygame.draw.polygon(
                    wall,
                    SHADOW,
                    [
                        (fx + 1, fy),
                        (fx + fw - 1, fy),
                        (fx + fw, fy + fh // 2),
                        (fx + fw - 2, fy + fh),
                        (fx, fy + fh - 1),
                        (fx, fy + 2),
                    ],
                )
                pygame.draw.rect(wall, FACE, (fx + 1, fy + 1, max(1, fw - 3), max(1, fh - 3)))
    return wall
