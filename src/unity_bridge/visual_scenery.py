"""Non-colliding cave plants and silver supports seen in the reference rooms."""

from __future__ import annotations

from typing import Sequence, TypedDict

import pygame

from .visual_doors import door_headroom_cells
from .visual_fidelity import service_pipe_cells
from .visual_materials import INK, PINK, RUST, canvas


class DressingPlacement(TypedDict):
    sprite: str
    col: int
    row: int


def vine(rows: int) -> pygame.Surface:
    """Two crooked pink stems join into a rounded loop, rather than a ladder."""
    height = rows * 32
    image = canvas((32, height))
    left = [(7 + (0, 1, -1, 0)[y // 8 % 4], y) for y in range(0, height - 12, 8)]
    right = [(24 + (0, -1, 0, 1)[y // 8 % 4], y) for y in range(0, height - 12, 8)]
    loop = (
        left
        + [(8, height - 10), (12, height - 4), (19, height - 4), (23, height - 10)]
        + right[::-1]
    )
    pygame.draw.lines(image, INK, False, loop, 5)
    pygame.draw.lines(image, PINK[1], False, loop, 3)
    pygame.draw.lines(image, PINK[2], False, [(x - 1, y) for x, y in loop], 1)
    for y in range(10, height - 15, 16):
        pygame.draw.line(image, PINK[4], (6, y), (6, y + 3))
        pygame.draw.line(image, PINK[0], (25, y + 5), (25, y + 7))
    pygame.draw.lines(image, PINK[4], False, [(10, height - 8), (13, height - 5), (18, height - 5)])
    return image


def purple_mushroom() -> pygame.Surface:
    """Small angular magenta cap above green curled roots; decorative, no face."""
    image = canvas()
    for x, shift in ((8, -2), (15, 2), (22, -1)):
        root = [(x, 21), (x + shift, 24), (x - shift, 28), (x - 2, 31)]
        pygame.draw.lines(image, INK, False, root, 4)
        pygame.draw.lines(image, (102, 167, 37), False, root, 2)
        pygame.draw.line(image, (172, 219, 70), (x, 22), (x + shift, 24))
    pygame.draw.polygon(
        image, INK, [(3, 21), (4, 14), (10, 9), (13, 4), (18, 5), (22, 10), (27, 14), (29, 21)]
    )
    pygame.draw.polygon(
        image, PINK[1], [(5, 19), (6, 15), (12, 10), (14, 6), (17, 7), (21, 12), (25, 15), (27, 19)]
    )
    pygame.draw.polygon(image, PINK[2], [(7, 17), (12, 11), (14, 7), (17, 8), (17, 17)])
    pygame.draw.line(image, PINK[4], (12, 10), (14, 7))
    pygame.draw.line(image, PINK[0], (5, 20), (27, 20))
    pygame.draw.line(image, (183, 54, 125), (7, 18), (22, 18))
    image.set_at((23, 16), PINK[0])
    return image


def silver_column(rows: int, flared: bool = False) -> pygame.Surface:
    """A slim, fluted silver shaft with a wide cap and anchored foot."""
    height = rows * 32
    image = canvas((32, height))
    dark, shade, body = (49, 48, 80), (105, 104, 143), (169, 169, 194)
    light, gleam = (222, 228, 232), (250, 253, 239)
    pygame.draw.rect(image, INK, (8, 0, 17, height))
    pygame.draw.rect(image, body, (10, 0, 13, height))
    pygame.draw.line(image, gleam, (10, 2), (10, height - 3), 2)
    pygame.draw.line(image, light, (13, 2), (13, height - 3))
    pygame.draw.line(image, shade, (18, 1), (18, height - 2), 3)
    pygame.draw.line(image, dark, (22, 1), (22, height - 2))
    if flared:
        pygame.draw.polygon(image, INK, [(0, 3), (31, 3), (23, 14), (21, 22), (11, 22), (8, 14)])
        pygame.draw.polygon(image, body, [(3, 5), (28, 5), (21, 14), (20, 21), (12, 21), (10, 13)])
        pygame.draw.lines(image, gleam, False, [(5, 6), (12, 14), (13, 22)], 2)
        pygame.draw.lines(image, shade, False, [(25, 7), (19, 14), (19, 21)], 2)
    for y, bottom in ((0, False), (height - 6, True)):
        pygame.draw.rect(image, INK, (0, y, 32, 6))
        pygame.draw.line(image, gleam, (2, y + 1), (29, y + 1))
        pygame.draw.rect(image, body, (1, y + 2, 30, 2))
        pygame.draw.line(image, dark, (2, y + 4), (29, y + 4))
        if bottom:
            pygame.draw.line(image, shade, (6, y + 2), (9, y + 2))
            pygame.draw.line(image, RUST[0], (22, y + 3), (26, y + 3))
            image.set_at((24, y + 2), RUST[1])
    # One worn seam per column leaves broad reflective bands readable.
    pygame.draw.line(image, shade, (13, height - 19), (15, height - 20))
    image.set_at((13, height - 18), light)
    return image


def ventilation_grille() -> pygame.Surface:
    """A square silver wall grille with narrow dark vertical ventilation slots."""
    image = canvas((64, 64))
    pygame.draw.rect(image, INK, (0, 0, 64, 64))
    pygame.draw.rect(image, (174, 176, 198), (1, 1, 62, 62))
    pygame.draw.line(image, (239, 243, 239), (1, 1), (62, 1), 2)
    pygame.draw.line(image, (221, 226, 231), (1, 2), (1, 61), 2)
    pygame.draw.line(image, (101, 102, 134), (61, 3), (61, 61), 2)
    pygame.draw.line(image, (101, 102, 134), (3, 61), (60, 61), 2)
    pygame.draw.rect(image, (112, 113, 144), (5, 5, 54, 54))
    for x in range(7, 58, 4):
        pygame.draw.rect(image, (14, 19, 38), (x, 7, 2, 49))
        pygame.draw.line(image, (226, 231, 231), (x + 2, 7), (x + 2, 55))
        image.set_at((x + 1, 56), (55, 63, 84))
    pygame.draw.line(image, (215, 219, 228), (5, 5), (58, 5))
    pygame.draw.line(image, (55, 59, 81), (5, 58), (58, 58))
    # Short worn frame edges leave the slots crisp and the silver face readable.
    pygame.draw.line(image, (98, 108, 124), (47, 1), (50, 1))
    pygame.draw.line(image, (180, 190, 199), (48, 2), (51, 2))
    pygame.draw.line(image, (101, 102, 134), (2, 49), (2, 52))
    pygame.draw.line(image, RUST[0], (49, 60), (53, 60))
    image.set_at((51, 59), RUST[1])
    return image


def scenery_sprites() -> dict[str, pygame.Surface]:
    images = {
        "purple_mushroom": purple_mushroom(),
        "ventilation_grille": ventilation_grille(),
    }
    images.update({f"vine_{rows}": vine(rows) for rows in range(2, 5)})
    for rows in range(2, 7):
        images[f"silver_column_{rows}"] = silver_column(rows)
        images[f"flared_column_{rows}"] = silver_column(rows, True)
    return images


def placement_cells(placement: DressingPlacement) -> set[tuple[int, int]]:
    name = placement["sprite"]
    height = (
        int(name.rsplit("_", 1)[1])
        if name.startswith(("vine_", "silver_column_", "flared_column_"))
        else 2 if name == "ventilation_grille" else 1
    )
    width = 2 if name in ("danger_sign", "reverse_gravity_sign", "ventilation_grille") else 1
    return {
        (placement["col"] + dx, placement["row"] + dy)
        for dy in range(height)
        for dx in range(width)
    }


def scenery_placements(layout: Sequence[str], theme: int) -> list[DressingPlacement]:
    """Reserve complete clear footprints; supports attach to existing platforms."""
    placements: list[DressingPlacement] = []
    occupied = service_pipe_cells(layout) | door_headroom_cells(layout)

    def tile(col: int, row: int) -> str:
        return layout[row][col] if 0 <= row < len(layout) and 0 <= col < len(layout[row]) else ""

    def clear(col: int, row: int, height: int, width: int = 1) -> bool:
        return all(
            tile(col + dx, row + dy) == "." and (col + dx, row + dy) not in occupied
            for dy in range(height)
            for dx in range(-1, width + 1)
        )

    def place(name: str, col: int, row: int) -> None:
        placement: DressingPlacement = {"sprite": name, "col": col, "row": row}
        placements.append(placement)
        for c, r in placement_cells(placement):
            occupied.update((c + dx, r) for dx in (-1, 0, 1))

    if theme in (4, 5, 7):
        count = 0
        for row in range(len(layout) - 2, 0, -1):
            row_count = 0
            candidates = sorted(range(3, len(layout[row]) - 3), key=lambda c: (c + row) % 7)
            for col in candidates:
                if not all(tile(col + dx, row - 1) == "#" for dx in (-1, 0, 1)):
                    continue
                height = 0
                while height < 7 and tile(col, row + height) == ".":
                    height += 1
                if (
                    not 2 <= height <= 6
                    or tile(col, row + height) != "#"
                    or not clear(col, row, height)
                ):
                    continue
                if any(
                    abs(p["col"] - col) < 7
                    and p["row"] < row + height
                    and row <= max(r for _, r in placement_cells(p))
                    for p in placements
                ):
                    continue
                kind = "flared_column" if theme == 5 else "silver_column"
                place(f"{kind}_{height}", col, row)
                count += 1
                row_count += 1
                if row_count == 2 or count == 5:
                    break
            if count == 5:
                break
    if theme == 4:
        # Long loops belong in open ceiling bays, away from traps and ladders.
        for row in range(1, len(layout) - 2):
            for col in range(4, len(layout[row]) - 4):
                if (col + row * 3) % 11 or tile(col, row - 1) != "#":
                    continue
                height = next((h for h in (4, 3, 2) if clear(col, row, h)), 0)
                if height:
                    place(f"vine_{height}", col, row)
        mushrooms = 0
        for row in range(len(layout) - 2, 0, -1):
            candidates = sorted(range(3, len(layout[row]) - 3), key=lambda c: (c + row * 2) % 13)
            for col in candidates:
                if tile(col, row + 1) == "#" and clear(col, row, 1):
                    place("purple_mushroom", col, row)
                    mushrooms += 1
                    break
            if mushrooms == 4:
                break
    if theme == 0:
        # Grilles belong to clear tall wall bays. Leave a whole empty row below
        # each one so floor equipment and future door artwork keep their space.
        vents = 0
        for row in range(len(layout) - 4, 0, -1):
            candidates = sorted(range(3, len(layout[row]) - 4), key=lambda c: (c + row) % 11)
            for col in candidates:
                if not clear(col, row, 2, 2):
                    continue
                if not all(tile(col + dx, row + 2) == "." for dx in (0, 1)):
                    continue
                if any(
                    p["sprite"] == "ventilation_grille"
                    and abs(p["col"] - col) < 8
                    and abs(p["row"] - row) < 5
                    for p in placements
                ):
                    continue
                place("ventilation_grille", col, row)
                vents += 1
                if vents == 2:
                    break
            if vents == 2:
                break
    # Signs describe actual nearby hazards/effects, not arbitrary wall panels.
    # Place them after the room's plants/supports so they cannot cover that art.
    warnings = 0
    for row in range(len(layout) - 1, -1, -1):
        for col, symbol in enumerate(layout[row]):
            if symbol == "g":
                name = "reverse_gravity_sign"
            elif symbol in "^~tv" and warnings < 2:
                name = "danger_sign"
                if any(
                    p["sprite"] == name and abs(p["col"] - col) <= 5 and abs(p["row"] - row) <= 2
                    for p in placements
                ):
                    continue
            else:
                continue
            for dx, dy in ((-3, -1), (2, -1), (-3, 0), (2, 0), (-1, -2), (-1, 1)):
                if clear(col + dx, row + dy, 1, 2):
                    place(name, col + dx, row + dy)
                    warnings += name == "danger_sign"
                    break
    return placements
