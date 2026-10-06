"""Recognizable mine props and materials observed in the gameplay reference."""

from __future__ import annotations

from typing import Sequence

import pygame

from .visual_materials import (
    GOLD,
    INK,
    MAGENTA,
    RED,
    STEEL,
    WHITE,
    YELLOW,
    canvas,
    lettering,
    raygun,
)
from .visual_mine_dressing import draw_mine_fixtures
from .visual_mine_rocks import mine_rock_wall


def pickup_sprites() -> dict[str, pygame.Surface]:
    images = {}
    gun = canvas()
    gun.blit(raygun(), (2, 11))
    images["ammo"] = gun
    chest = canvas()
    pygame.draw.polygon(chest, INK, [(3, 13), (6, 8), (23, 8), (28, 13), (28, 28), (3, 28)])
    pygame.draw.polygon(chest, GOLD[0], [(5, 13), (8, 10), (22, 10), (26, 13), (26, 26), (5, 26)])
    pygame.draw.rect(chest, GOLD[1], (6, 16, 19, 9))
    pygame.draw.line(chest, YELLOW, (6, 12), (23, 12))
    pygame.draw.line(chest, GOLD[3], (6, 17), (24, 17))
    for x in (8, 22):
        pygame.draw.rect(chest, STEEL[1], (x, 11, 3, 15))
        pygame.draw.line(chest, STEEL[3], (x, 12), (x, 24))
    pygame.draw.rect(chest, INK, (14, 16, 4, 6))
    pygame.draw.rect(chest, YELLOW, (15, 17, 2, 3))
    pygame.draw.line(chest, GOLD[0], (12, 23), (16, 24))
    images["treasure"] = chest
    for name, letter, color in (("power_shot", "P", MAGENTA), ("gravity", "G", (92, 212, 70))):
        pill = canvas()
        pygame.draw.ellipse(pill, INK, (3, 6, 26, 21))
        pygame.draw.ellipse(pill, color, (5, 8, 22, 17))
        pygame.draw.ellipse(pill, WHITE, (7, 9, 17, 13))
        pygame.draw.line(pill, YELLOW, (8, 8), (21, 8))
        pygame.draw.line(pill, INK, (8, 24), (22, 24))
        lettering(pill, letter, 13, 12, INK)
        if name == "gravity":
            pill = canvas()
            pygame.draw.rect(pill, INK, (3, 3, 26, 26))
            pygame.draw.rect(pill, STEEL[3], (4, 4, 24, 24))
            pygame.draw.line(pill, WHITE, (4, 4), (27, 4))
            pygame.draw.line(pill, STEEL[4], (4, 5), (4, 26))
            pygame.draw.line(pill, STEEL[0], (27, 5), (27, 27))
            pygame.draw.line(pill, STEEL[0], (5, 27), (26, 27))
            pygame.draw.rect(pill, (77, 78, 116), (6, 6, 20, 20))
            pygame.draw.line(pill, (109, 111, 146), (6, 6), (25, 6))
            pygame.draw.line(pill, (44, 47, 76), (25, 7), (25, 25))
            # This pickup starts inverted gravity: its arrow always points UP.
            pygame.draw.polygon(
                pill, INK, [(16, 7), (7, 16), (11, 16), (11, 26), (21, 26), (21, 16), (25, 16)]
            )
            pygame.draw.polygon(
                pill, YELLOW, [(16, 8), (8, 16), (12, 16), (12, 25), (20, 25), (20, 16), (24, 16)]
            )
            pygame.draw.polygon(
                pill, RED, [(16, 11), (12, 15), (14, 15), (14, 23), (18, 23), (18, 15), (20, 15)]
            )
            pygame.draw.line(pill, (183, 37, 55), (17, 15), (17, 23))
            pill.set_at((16, 8), WHITE)
            # A short worn frame edge leaves the direction indicator pristine.
            pygame.draw.line(pill, STEEL[1], (21, 4), (23, 4))
            pill.set_at((22, 5), STEEL[3])
        images[name] = pill
    stop = canvas()
    pygame.draw.polygon(
        stop, INK, [(10, 3), (21, 3), (29, 11), (29, 22), (21, 30), (10, 30), (2, 22), (2, 11)]
    )
    pygame.draw.polygon(
        stop, WHITE, [(11, 5), (20, 5), (27, 12), (27, 21), (20, 28), (11, 28), (4, 21), (4, 12)]
    )
    pygame.draw.polygon(
        stop, RED, [(11, 7), (20, 7), (25, 12), (25, 21), (20, 26), (11, 26), (6, 21), (6, 12)]
    )
    # Native tiny lettering still reads STOP at 2x; no modern icon substitution.
    lettering(stop, "STOP", 6, 13, WHITE)
    images["freeze"] = stop
    return images


def mine_door(cleared: bool = False) -> pygame.Surface:
    door = canvas()
    pygame.draw.rect(door, INK, (2, 0, 28, 32))
    pygame.draw.rect(door, STEEL[3], (4, 2, 24, 30))
    pygame.draw.rect(door, STEEL[1], (7, 4, 18, 28))
    pygame.draw.line(door, WHITE, (4, 2), (27, 2))
    pygame.draw.line(door, STEEL[4], (4, 3), (4, 30))
    pygame.draw.line(door, INK, (25, 4), (25, 31))
    pygame.draw.rect(door, INK, (10, 6, 12, 10))
    pygame.draw.rect(door, YELLOW, (11, 7, 10, 7))
    pygame.draw.line(door, GOLD[1], (12, 14), (20, 14))
    pygame.draw.rect(door, (85, 255, 85) if cleared else RED, (12, 23, 9, 4))
    if cleared:
        pygame.draw.lines(door, INK, False, [(13, 24), (15, 26), (19, 23)])
    pygame.draw.line(door, STEEL[0], (7, 29), (13, 29))
    door.set_at((8, 28), STEEL[2])
    return door


def torch(frame: int) -> pygame.Surface:
    image = canvas()
    pygame.draw.rect(image, INK, (13, 18, 7, 13))
    pygame.draw.rect(image, GOLD[0], (15, 18, 3, 12))
    pygame.draw.polygon(
        image, INK, [(10, 18), (12, 9), (16, 2 + frame % 3), (18, 9), (23, 15), (19, 23), (13, 23)]
    )
    pygame.draw.polygon(
        image, RED, [(12, 17), (15, 5 + frame % 3), (17, 12), (21, 16), (18, 21), (14, 21)]
    )
    pygame.draw.polygon(image, (255, 150, 35), [(14, 18), (16, 10 + frame % 2), (19, 17), (17, 21)])
    pygame.draw.line(image, YELLOW, (16, 16), (16, 20))
    # The reference's cool tapered wall socket occupies the existing hilt only.
    pygame.draw.rect(image, INK, (14, 23, 5, 8))
    pygame.draw.polygon(image, (105, 104, 143), [(14, 23), (18, 23), (17, 28), (16, 30), (15, 28)])
    pygame.draw.line(image, (222, 228, 232), (14, 23), (15, 27))
    pygame.draw.line(image, (49, 48, 80), (18, 23), (17, 27))
    image.set_at((15, 23), (169, 169, 194))
    return image


def mine_surfaces(layout: Sequence[str]) -> tuple[pygame.Surface, pygame.Surface, pygame.Surface]:
    cols, rows = len(layout[0]), len(layout)
    wall = mine_rock_wall((cols * 32, rows * 32))
    terrain, props = (canvas((cols * 32, rows * 32)) for _ in range(2))
    for row, line in enumerate(layout):
        for col, cell in enumerate(line):
            x, y = col * 32, row * 32
            if cell == "#":
                if row in (7, 11, 15, 19):
                    pygame.draw.rect(terrain, INK, (x, y, 32, 32))
                    pygame.draw.rect(terrain, (191, 114, 92), (x, y + 2, 32, 27))
                    pygame.draw.rect(terrain, (170, 93, 78), (x, y + 11, 32, 8))
                    pygame.draw.rect(terrain, (93, 46, 45), (x, y + 19, 32, 10))
                    pygame.draw.line(terrain, (249, 191, 124), (x, y + 2), (x + 31, y + 2))
                    # Broken grain follows the beam, with no full-width plank stripes.
                    seed = col * 17 + row * 29
                    for band, dy in enumerate((6, 10, 15, 21, 25)):
                        for stroke in range(3):
                            grain = seed + band * 37 + stroke * 11
                            start = 1 + grain % 25
                            length = min(2 + (grain // 7) % 6, 30 - start)
                            yy = y + dy + grain % 3 - 1
                            dark = (142, 77, 68) if dy < 19 else (87, 46, 44)
                            light = (217, 139, 102) if dy < 19 else (142, 77, 68)
                            pygame.draw.line(
                                terrain, dark, (x + start, yy), (x + start + length, yy)
                            )
                            if stroke == 1:
                                pygame.draw.line(
                                    terrain,
                                    light,
                                    (x + start, yy + 1),
                                    (x + start + length - 1, yy + 1),
                                )
                    # Small, shallow chips wear the top lip without changing its outline.
                    chip = 3 + seed % 23
                    pygame.draw.line(
                        terrain, (191, 114, 92), (x + chip, y + 2), (x + chip + 3, y + 2)
                    )
                    if seed % 4 == 0:
                        pygame.draw.lines(
                            terrain,
                            (87, 46, 44),
                            False,
                            [(x + 18, y + 11), (x + 15, y + 14), (x + 19, y + 16)],
                        )
                else:
                    pygame.draw.rect(terrain, (62, 66, 85), (x, y, 32, 32))
                    pygame.draw.rect(terrain, (88, 93, 107), (x + 2, y + 2, 28, 27))
                    pygame.draw.lines(
                        terrain,
                        (31, 34, 44),
                        False,
                        [(x + 3, y + 22), (x + 8, y + 19), (x + 18, y + 24)],
                    )
            elif cell == "H":
                for dy in range(0, 32, 8):
                    pygame.draw.rect(terrain, INK, (x + 12, y + dy, 8, 9))
                    pygame.draw.rect(terrain, STEEL[2], (x + 14, y + dy + 1, 4, 7), 1)
                    terrain.set_at((x + 14, y + dy + 2), WHITE)
            elif col in (6, 18, 33) and 7 <= row <= 22:
                pygame.draw.rect(props, INK, (x + 4, y, 24, 32))
                pygame.draw.rect(props, (142, 77, 68), (x + 6, y, 20, 32))
                pygame.draw.line(props, (219, 139, 104), (x + 8, y), (x + 8, y + 31), 2)
                pygame.draw.line(props, (93, 46, 45), (x + 22, y), (x + 22, y + 31), 2)
                # Each support segment has short vertical grain rather than a repeated stripe.
                seed = col * 19 + row * 31
                for stroke in range(7):
                    grain = seed + stroke * 43
                    xx, yy = x + 10 + grain % 11, y + grain // 3 % 25
                    length = 2 + grain // 7 % 6
                    color = (170, 93, 78) if stroke % 2 else (93, 46, 45)
                    pygame.draw.line(props, color, (xx, yy), (xx, yy + length))
                worn_y = y + seed % 27
                pygame.draw.line(props, (170, 93, 78), (x + 7, worn_y), (x + 7, worn_y + 3))
                if row in (8, 12, 16, 20):
                    for direction in (-1, 1):
                        points = [
                            (x + 16, y + 1),
                            (x + 16 + direction * 32, y + 1),
                            (x + 16, y + 27),
                        ]
                        pygame.draw.polygon(props, INK, points)
                        pygame.draw.polygon(
                            props,
                            (142, 77, 68),
                            [(x + 16, y + 3), (x + 16 + direction * 26, y + 3), (x + 16, y + 22)],
                        )
                        pygame.draw.polygon(
                            props,
                            (93, 46, 45),
                            [(x + 16, y + 5), (x + 16 + direction * 9, y + 5), (x + 16, y + 22)],
                        )
                        pygame.draw.line(
                            props,
                            (170, 93, 78),
                            (x + 16 + direction * 18, y + 6),
                            (x + 16 + direction * 7, y + 14),
                        )
                        pygame.draw.line(
                            props,
                            (93, 46, 45),
                            (x + 16 + direction * 12, y + 6),
                            (x + 16 + direction * 5, y + 11),
                        )
                        pygame.draw.line(
                            props,
                            (219, 139, 104),
                            (x + 16 + direction * 24, y + 3),
                            (x + 16, y + 24),
                        )
    draw_mine_fixtures(layout, props)
    return wall, terrain, props


def service_pipe_routes(layout: Sequence[str]) -> list[tuple[int, int, int, int]]:
    """Return the AIR connections so other scenery can leave their bays clear."""
    routes = []
    for row, line in enumerate(layout):
        for col, symbol in enumerate(line):
            if symbol != "O" or row < 2:
                continue
            # A real vertical run, then a horizontal connection along a clear bay.
            top = row - 1
            while top > max(1, row - 5) and layout[top - 1][col] == ".":
                top -= 1
            direction = -1 if col > len(line) // 2 else 1
            end = col
            for offset in range(1, 6):
                candidate = col + direction * offset
                if not 0 <= candidate < len(line) or layout[top][candidate] != ".":
                    break
                end = candidate
            routes.append((col, row, top, end))
    return routes


def service_pipe_cells(layout: Sequence[str]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for col, row, top, end in service_pipe_routes(layout):
        cells.update((col, y) for y in range(top, row))
        cells.update((x, top) for x in range(min(col, end), max(col, end) + 1))
    return cells


def connected_pipes(layout: Sequence[str], image: pygame.Surface) -> None:
    """Route non-colliding service runs into AIR machines, with ribbed elbows."""
    for col, row, top, end in service_pipe_routes(layout):
        x, y = col * 32 + 25, top * 32 + 9
        points = [(end * 32 + 16, y), (x, y), (x, row * 32 + 9)]
        for color, width in ((INK, 10), (STEEL[1], 8), (STEEL[3], 4)):
            pygame.draw.lines(image, color, False, points, width)
            pygame.draw.circle(image, color, (x, y), width // 2)
        for yy in range(y + 5, row * 32 + 9, 4):
            pygame.draw.line(image, STEEL[0], (x - 3, yy), (x + 3, yy))
            pygame.draw.line(image, STEEL[4], (x - 2, yy + 1), (x + 2, yy + 1))
        for xx in range(min(end * 32 + 16, x) + 4, max(end * 32 + 16, x) - 4, 4):
            pygame.draw.line(image, STEEL[0], (xx, y - 3), (xx, y + 3))
            pygame.draw.line(image, STEEL[4], (xx + 1, y - 2), (xx + 1, y + 2))
