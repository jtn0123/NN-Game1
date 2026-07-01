"""Body-fit reachability audit for Crystal Caves generated levels.

The generator's historical oracle was a permissive tile flood. This module keeps
the check cheap enough for generation while using the live player dimensions,
door rules, and climbable ladder/rail tiles so generated levels cannot strand
objectives in pockets the real body cannot route to.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence, Set, Tuple

from .crystal_caves import CrystalCaves
from .crystal_caves_entities import CaveSpec

Cell = Tuple[int, int]


@dataclass(frozen=True)
class PhysicsReachabilityReport:
    """Summary of geometry reachability under player-size-aware routing."""

    winnable: bool
    reachable_switches: int
    total_switches: int
    reachable_objectives: int
    total_objectives: int
    unreachable_switches: Tuple[Cell, ...]
    unreachable_objectives: Tuple[Cell, ...]


def audit_physics_winnability(spec: CaveSpec) -> PhysicsReachabilityReport:
    """Audit staged level completion with player-size-aware reachability.

    Switches are checked with every door closed. Crystals and the exit are checked
    with every door colour open, matching the generator's staged solvability
    contract while replacing single-cell permissiveness with live body geometry.
    """

    rows = spec.layout
    switches = tuple(sorted(_find_any(rows, {CrystalCaves.SWITCH, CrystalCaves.SWITCH2})))
    doors = _find_any(rows, {CrystalCaves.DOOR, CrystalCaves.DOOR2})
    all_colors = {_door_color(rows[r][c]) for c, r in doors}

    closed_reach = physical_reachable_tiles(rows, open_colors=set())
    unreachable_switches = tuple(s for s in switches if s not in closed_reach)

    objectives = tuple(sorted(_find(rows, CrystalCaves.CRYSTAL) + _find(rows, CrystalCaves.EXIT)))
    open_reach = physical_reachable_tiles(rows, open_colors=all_colors)
    unreachable_objectives = tuple(obj for obj in objectives if obj not in open_reach)

    return PhysicsReachabilityReport(
        winnable=not unreachable_switches and not unreachable_objectives,
        reachable_switches=len(switches) - len(unreachable_switches),
        total_switches=len(switches),
        reachable_objectives=len(objectives) - len(unreachable_objectives),
        total_objectives=len(objectives),
        unreachable_switches=unreachable_switches,
        unreachable_objectives=unreachable_objectives,
    )


def physical_reachable_tiles(
    rows_or_spec: Sequence[str] | CaveSpec,
    *,
    open_colors: Iterable[str] | None = None,
) -> Set[Cell]:
    """Return cells reachable by walking, falling, climbing, and conservative jumps."""

    rows = rows_or_spec.layout if isinstance(rows_or_spec, CaveSpec) else rows_or_spec
    start = _find(rows, CrystalCaves.PLAYER)[0]
    queue: deque[Cell] = deque([start])
    seen: Set[Cell] = {start}
    open_set = set(open_colors or ())
    while queue:
        cell = queue.popleft()
        for nxt in _neighbors(rows, cell, open_set):
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def _neighbors(rows: Sequence[str], cell: Cell, open_colors: Set[str]) -> Set[Cell]:
    col, row = cell
    out: Set[Cell] = set()

    def add(c: int, r: int) -> None:
        if _body_fits(rows, c, r, open_colors):
            out.add((c, r))

    # Real movement allows horizontal steering both on the ground and in the air.
    add(col - 1, row)
    add(col + 1, row)

    # Gravity/fall.
    add(col, row + 1)

    if _climbable(rows, col, row):
        add(col, row - 1)
        add(col, row + 1)

    if _supported(rows, col, row, open_colors) or _climbable(rows, col, row):
        # Conservative jump envelope: enough for nearby ledges, but no freeform
        # multi-tile drift through sealed pockets.
        for dc in range(-2, 3):
            for dr in (-1, -2, -3):
                add(col + dc, row + dr)
    return out


def _body_fits(rows: Sequence[str], col: int, row: int, open_colors: Set[str]) -> bool:
    x = col * CrystalCaves.TILE_SIZE + 5
    y = row * CrystalCaves.TILE_SIZE + 1
    left = x // CrystalCaves.TILE_SIZE
    right = (x + CrystalCaves.PLAYER_WIDTH - 1) // CrystalCaves.TILE_SIZE
    top = y // CrystalCaves.TILE_SIZE
    bottom = (y + CrystalCaves.PLAYER_HEIGHT - 1) // CrystalCaves.TILE_SIZE
    return not any(
        _solid_at(rows, c, r, open_colors)
        for r in range(top, bottom + 1)
        for c in range(left, right + 1)
    )


def _supported(rows: Sequence[str], col: int, row: int, open_colors: Set[str]) -> bool:
    x = col * CrystalCaves.TILE_SIZE + 5
    y = row * CrystalCaves.TILE_SIZE + 2
    left = x // CrystalCaves.TILE_SIZE
    right = (x + CrystalCaves.PLAYER_WIDTH - 1) // CrystalCaves.TILE_SIZE
    top = y // CrystalCaves.TILE_SIZE
    bottom = (y + CrystalCaves.PLAYER_HEIGHT - 1) // CrystalCaves.TILE_SIZE
    return any(
        _solid_at(rows, c, r, open_colors)
        for r in range(top, bottom + 1)
        for c in range(left, right + 1)
    )


def _solid_at(rows: Sequence[str], col: int, row: int, open_colors: Set[str]) -> bool:
    if row < 0 or col < 0 or row >= len(rows) or col >= len(rows[0]):
        return True
    ch = rows[row][col]
    if ch == CrystalCaves.SOLID:
        return True
    if ch in (CrystalCaves.DOOR, CrystalCaves.DOOR2):
        return _door_color(ch) not in open_colors
    return False


def _climbable(rows: Sequence[str], col: int, row: int) -> bool:
    if row < 0 or col < 0 or row >= len(rows) or col >= len(rows[0]):
        return False
    return rows[row][col] in (CrystalCaves.LADDER, CrystalCaves.ELEVATOR)


def _door_color(ch: str) -> str:
    return "blue" if ch == CrystalCaves.DOOR2 else "red"


def _find(rows: Sequence[str], ch: str) -> Tuple[Cell, ...]:
    return tuple((c, r) for r, row in enumerate(rows) for c, x in enumerate(row) if x == ch)


def _find_any(rows: Sequence[str], chars: Set[str]) -> Tuple[Cell, ...]:
    return tuple((c, r) for r, row in enumerate(rows) for c, x in enumerate(row) if x in chars)
