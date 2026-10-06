"""Sparse reference inventory in the main mine's existing clear timber bays."""

from __future__ import annotations

from typing import Sequence, TypedDict

import pygame

from .mine import ENTRANCES
from .visual_equipment import barrel, warning_sign

TORCH_CELLS = tuple((col, row) for row in (6, 10, 14, 18) for col in (11, 27))
SUPPORT_COLUMNS = (6, 18, 33)


class MineFixture(TypedDict):
    sprite: str
    col: int
    row: int


def mine_fixture_cells(fixture: MineFixture) -> set[tuple[int, int]]:
    width = 2 if fixture["sprite"] == "danger_sign" else 1
    return {(fixture["col"] + dx, fixture["row"]) for dx in range(width)}


def mine_reserved_cells(layout: Sequence[str]) -> set[tuple[int, int]]:
    """Door labels, torches and support arms also occupy unmarked empty cells."""
    protected = {
        (col, row)
        for row, line in enumerate(layout)
        for col, symbol in enumerate(line)
        if symbol != "."
    }
    protected.update(ENTRANCES)
    protected.update((col, row - 1) for col, row in ENTRANCES)
    protected.update(TORCH_CELLS)
    for row, line in enumerate(layout):
        for col in SUPPORT_COLUMNS:
            if 7 <= row <= 22 and col < len(line) and line[col] == ".":
                protected.add((col, row))
                if row in (8, 12, 16, 20):
                    protected.update((col + dx, row) for dx in (-1, 1))
    return protected


def mine_fixture_placements(layout: Sequence[str]) -> list[MineFixture]:
    """Few single barrels and two wall plates, with complete side clearances."""
    placements: list[MineFixture] = []
    protected = mine_reserved_cells(layout)
    # The reference shows intact cyan/magenta barrels at quiet timber-bay edges
    # and occasional DANGER plates on the dark wall between mine door tiers.
    candidates = (
        ("barrel", 13, 6),
        ("barrel", 20, 10),
        ("barrel", 29, 14),
        ("barrel", 13, 18),
        ("danger_sign", 16, 4),
        ("danger_sign", 25, 12),
    )
    for sprite, col, row in candidates:
        width = 2 if sprite == "danger_sign" else 1
        if not all(
            0 <= row < len(layout)
            and 0 <= col + dx < len(layout[row])
            and layout[row][col + dx] == "."
            and (col + dx, row) not in protected
            for dx in range(-1, width + 1)
        ):
            continue
        if sprite == "barrel" and (row + 1 >= len(layout) or layout[row + 1][col] != "#"):
            continue
        fixture: MineFixture = {"sprite": sprite, "col": col, "row": row}
        placements.append(fixture)
        for c, r in mine_fixture_cells(fixture):
            protected.update((c + dx, r) for dx in (-1, 0, 1))
    return placements


def draw_mine_fixtures(layout: Sequence[str], image: pygame.Surface) -> None:
    art = {"barrel": barrel(), "danger_sign": warning_sign()}
    for fixture in mine_fixture_placements(layout):
        image.blit(art[fixture["sprite"]], (fixture["col"] * 32, fixture["row"] * 32))
