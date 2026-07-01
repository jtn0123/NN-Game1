"""Procedural Crystal Caves level generator (platform-network model).

The level is mostly OPEN (the dense textured back-wall carries the "full" look)
and threaded with a varied web of thin platforms — so it reads less blocky and
more random than a carve-and-fill cave, matching the reference game. Every level

  * starts on the planet surface under space (sky_rows), Mylo descends a shaft,
  * guarantees full reachability under the engine's jump-aware physics (the exit
    sits behind a switch-gated door; every crystal + the switch are reachable),
  * carries one of three themed biomes.

Public API:
  generate_cave(seed, theme=None) -> CaveSpec   # verified, drop-in for CAVES
  grade_cave(spec) -> dict                        # 0-100 rubric score
  cave_reachable(rows, start, doors_open) -> set  # the shared solvability flood
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .crystal_caves_entities import CaveSpec

ROWS, COLS, SKY, JUMP = 18, 44, 3, 3

SOLID, EMPTY = "#", "."
PLAYER, EXIT, DOOR, SWITCH = "P", "E", "D", "s"
DOOR2, SWITCH2 = "d", "S"  # a second colour-keyed lever/door pair
CRYSTAL, AMMO, TREASURE = "*", "A", "$"
POWER, GRAV, FREEZE = "p", "g", "z"
SPIKE, ACID, CRAWLER, FLYER = "^", "~", "M", "F"
ELEVATOR = "="  # a vertical-shaft lift platform; rideable up/down within its run
LADDER = "H"  # climbable chain/ladder tile for narrow vertical connectors

# Colour-keyed lever/door pairs: a switch opens only the door of its colour.
DOOR_CHARS = {DOOR, DOOR2}
SWITCH_CHARS = {SWITCH, SWITCH2}
DOOR_COLOR = {DOOR: "red", DOOR2: "blue"}
SWITCH_COLOR = {SWITCH: "red", SWITCH2: "blue"}

Grid = List[List[str]]
Cell = Tuple[int, int]

THEMES: Dict[str, dict] = {
    "blue_rock": {
        "index": 0,
        "background": (9, 12, 22),
        "accent": (80, 190, 255),
        "hazards": [SPIKE],
        "powerups": [POWER, FREEZE],
        "enemies": [CRAWLER, CRAWLER, FLYER],
    },
    "rust": {
        "index": 1,
        "background": (12, 10, 18),
        "accent": (255, 188, 80),
        "hazards": [ACID, SPIKE],
        "powerups": [POWER, GRAV],
        "enemies": [CRAWLER, FLYER],
    },
    "gray_tech": {
        "index": 2,
        "background": (8, 15, 13),
        "accent": (120, 255, 155),
        "hazards": [SPIKE, ACID],
        "powerups": [GRAV, FREEZE],
        "enemies": [FLYER, CRAWLER],
    },
}
THEME_NAMES = tuple(THEMES.keys())


def cave_reachable(rows, start: Cell, doors_open, jump: int = JUMP) -> Set[Cell]:
    """Tiles the player can occupy from ``start`` under jump-aware physics: walk
    or air-drift sideways, fall through EMPTY, jump up to ``jump`` tiles while
    grounded. ``doors_open`` is either a bool (open/close every door) or a set of
    open colours (a door opens only if its colour is in the set). Accepts rows of
    strings or a list-of-lists grid (the single solvability oracle generators use)."""
    rows_n, cols_n = len(rows), len(rows[0])

    open_set = doors_open if isinstance(doors_open, (set, frozenset)) else None

    def door_open(ch: str) -> bool:
        if open_set is not None:
            return DOOR_COLOR.get(ch, "") in open_set
        return bool(doors_open)

    def is_open(c: int, r: int) -> bool:
        if not (0 <= r < rows_n and 0 <= c < cols_n):
            return False
        ch = rows[r][c]
        if ch == SOLID:
            return False
        if ch in DOOR_CHARS:
            return door_open(ch)
        return True

    def grounded(c: int, r: int) -> bool:
        # standing inside an elevator shaft => the platform supports you (it rides
        # the full run, so any shaft cell is a valid footing)
        if rows[r][c] in (ELEVATOR, LADDER):
            return True
        if r + 1 >= rows_n:
            return True
        below = rows[r + 1][c]
        return below == SOLID or (below in DOOR_CHARS and not door_open(below))

    def transport_run(c: int, r: int):
        """Yield every cell of a contiguous climb/ride shaft through (c, r)."""
        ch = rows[r][c]
        top = r
        while top - 1 >= 0 and rows[top - 1][c] == ch:
            top -= 1
        bot = r
        while bot + 1 < rows_n and rows[bot + 1][c] == ch:
            bot += 1
        for rr in range(top, bot + 1):
            yield c, rr

    sc, sr = start
    f0 = jump if grounded(sc, sr) else 0
    seen: Set[Tuple[int, int, int]] = set()
    tiles: Set[Cell] = set()
    queue: deque = deque([(sc, sr, f0)])
    while queue:
        c, r, f = queue.popleft()
        if (c, r, f) in seen:
            continue
        seen.add((c, r, f))
        tiles.add((c, r))
        if grounded(c, r):
            f = jump
        # riding a lift or climbing a ladder: free vertical travel to any cell in its shaft
        if rows[r][c] in (ELEVATOR, LADDER):
            for ec, er in transport_run(c, r):
                queue.append((ec, er, jump))
        if is_open(c, r + 1):
            queue.append((c, r + 1, 0))
        for dc in (-1, 1):
            if is_open(c + dc, r):
                nf = jump if grounded(c + dc, r) else f
                queue.append((c + dc, r, nf))
        if f > 0 and is_open(c, r - 1):
            queue.append((c, r - 1, f - 1))
    return tiles


def cave_reachable_keyed(rows, start: Cell, jump: int = JUMP) -> Set[Cell]:
    """Reachable tiles when each colour-keyed lever opens its own door the moment
    the player can reach it (a fixpoint). The multi-lock solvability oracle: a
    lever opens only its colour and may itself sit behind a different door."""
    open_colors: Set[str] = set()
    while True:
        reach = cave_reachable(rows, start, open_colors, jump)
        newly = {
            SWITCH_COLOR[rows[r][c]]
            for (c, r) in reach
            if rows[r][c] in SWITCH_CHARS and SWITCH_COLOR[rows[r][c]] not in open_colors
        }
        if not newly:
            return reach
        open_colors |= newly


def _band(r: int, surface: int) -> int:
    """Difficulty band 0..3 from the surface down to the floor."""
    span = max(1, ROWS - surface)
    return min(3, (r - surface) * 4 // span)


# ---------------------------------------------------------------------------
# Level FAMILIES — each carves a distinct terrain style learned (clean-room) from
# the VGMaps reference atlas; the shared pipeline then adds the surface entrance,
# exit chamber, objectives, hazards, and verifies solvability. Signature:
#   family(grid, rng, surface, shaft) -> None   (mutates grid below the surface)
# ---------------------------------------------------------------------------


def _open_interior(grid: Grid) -> None:
    for r in range(SKY + 1, ROWS - 1):
        for c in range(1, COLS - 1):
            grid[r][c] = EMPTY


def _family_platform_network(grid: Grid, rng: random.Random, surface: int, shaft: int) -> None:
    """Carved shelf-gallery rooms (open-platforms / room-grid archetype): thick
    rock with open 2-tall galleries every few rows whose floors are the shelves,
    so platforms read as ledges cut from the rock rather than islands floating in
    a void. Solid pillars break each gallery into rooms; adjacent galleries are
    linked by short jumpable vertical shafts so the cave stays fully traversable.

    Unlike the old open-interior version this leaves the surrounding rock SOLID
    (the base grid), matching the dense look of the snake/maze families."""
    gallery_rows = [surface + 3, surface + 7, surface + 11]
    rooms_by_row: Dict[int, List[Tuple[int, int]]] = {}
    for r in gallery_rows:
        c = rng.randint(1, 3)
        while c < COLS - 2:
            run = rng.randint(11, 17)  # wider rooms cut from the rock
            end = min(c + run, COLS - 1)
            rooms_by_row.setdefault(r, []).append((c, end))
            for cc in range(c, end):
                grid[r][cc] = SOLID  # the shelf / platform floor
                grid[r - 1][cc] = EMPTY  # stand
                grid[r - 2][cc] = EMPTY  # headroom
            # an occasional rock pillar rises through a room to vary it
            if rng.random() < 0.08 and end - c > 8:
                grid[r - 1][rng.randint(c + 1, end - 2)] = SOLID
            c = end + rng.randint(5, 8)  # rock wall between rooms
        segments = rooms_by_row[r]
        span_start, span_end = segments[0][0], segments[-1][1]
        for cc in range(span_start, span_end):
            grid[r - 2][cc] = EMPTY
            grid[r - 1][cc] = EMPTY

    # link adjacent galleries with vertical shafts punched straight through the
    # rock at arbitrary columns (same approach the maze uses to stay fully
    # connected); the carved span plus headroom is <= a jump so the player climbs
    # out as well as drops in. Unconstrained columns => no sealed-off pockets.
    def in_room(col: int, segments: List[Tuple[int, int]]) -> bool:
        return any(start + 1 <= col < end - 1 for start, end in segments)

    def carve_landing(col: int, row: int) -> None:
        for rr in (row - 2, row - 1):
            for cc in range(max(1, col - 1), min(COLS - 1, col + 2)):
                grid[rr][cc] = EMPTY

    used_connector_cols: Set[int] = set()
    for i in range(len(gallery_rows) - 1):
        r0, r1 = gallery_rows[i], gallery_rows[i + 1]
        upper = rooms_by_row[r0]
        lower = rooms_by_row[r1]
        candidates = [
            c
            for c in range(2, COLS - 2)
            if in_room(c, upper)
            and in_room(c, lower)
            and not any(abs(c - used) < 10 for used in used_connector_cols)
        ]
        if not candidates:
            candidates = [
                c
                for c in range(2, COLS - 2)
                if (in_room(c, upper) or in_room(c, lower))
                and not any(abs(c - used) < 10 for used in used_connector_cols)
            ]
        if not candidates:
            continue
        c = rng.choice(candidates)
        used_connector_cols.add(c)
        carve_landing(c, r0)
        carve_landing(c, r1)
        for r in range(r0 - 2, r1 + 1):
            grid[r][c] = LADDER
            if c + 1 < COLS - 1 and rng.random() < 0.35:
                grid[r][c + 1] = EMPTY
    # mine-shaft drop into the top gallery
    for r in range(surface + 1, gallery_rows[0]):
        grid[r][shaft] = EMPTY


def _family_snake_bands(grid: Grid, rng: random.Random, surface: int, shaft: int) -> None:
    """Staggered platform rooms with vertical breaks, not full-width lanes."""
    _open_interior(grid)
    rows = [surface + 3, surface + 6, surface + 9, surface + 12]
    for idx, r in enumerate(rows):
        anchors = [3, 23, 34] if idx % 2 == 0 else [8, 18, 30]
        rng.shuffle(anchors)
        for start in anchors:
            c0 = max(1, min(COLS - 3, start + rng.randint(-2, 2)))
            c1 = min(COLS - 1, c0 + rng.randint(5, 8))
            for c in range(c0, c1):
                grid[r][c] = SOLID
                if rng.random() < 0.40 and r + 1 < ROWS - 1:
                    grid[r + 1][c] = SOLID
        # Short wall ribs make small rooms and break the side-to-side visual read.
        for _ in range(rng.randint(2, 3)):
            c = rng.randint(4, COLS - 5)
            for rr in range(max(surface + 1, r - 2), min(ROWS - 1, r + 2)):
                grid[rr][c] = SOLID

    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]
        for _ in range(1):
            c = rng.randint(4, COLS - 5)
            for rr in range(r0 - 1, r1 + 1):
                grid[rr][c] = LADDER
                if c + 1 < COLS - 1 and rng.random() < 0.35:
                    grid[rr][c + 1] = EMPTY


def _family_terrain_climb(grid: Grid, rng: random.Random, surface: int, shaft: int) -> None:
    """Open upper area with sparse platforms over a stepped solid terrain mound
    rising from the floor (Ep1 L1 / L13 style)."""
    _open_interior(grid)
    shelf_rows = list(range(surface + 2, ROWS - 4, 2))
    for r in shelf_rows:
        anchors = [3, 11, 19, 27, 35]
        rng.shuffle(anchors)
        chosen = anchors[: rng.randint(3, 4)]
        if surface + 4 <= r <= ROWS - 6:
            chosen.append(rng.choice([17, 22, 27]))
        for start in chosen:
            c0 = max(1, min(COLS - 3, start + rng.randint(-2, 2)))
            length = rng.randint(4, 7)
            for cc in range(c0, min(c0 + length, COLS - 1)):
                grid[r][cc] = SOLID
        if rng.random() < 0.65 and r + 1 < ROWS - 1:
            c0 = rng.randint(5, COLS - 10)
            for cc in range(c0, min(c0 + rng.randint(3, 6), COLS - 1)):
                grid[r + 1][cc] = SOLID
    for c in (rng.randint(12, 16), rng.randint(21, 25), rng.randint(30, 34)):
        top = surface + rng.randint(3, 5)
        bottom = ROWS - rng.randint(4, 6)
        for r in range(top, max(top, bottom)):
            if grid[r][c] == EMPTY:
                grid[r][c] = LADDER
    peak = rng.randint(COLS // 3, 2 * COLS // 3)
    peak_h = rng.randint(5, 8)
    slope = rng.choice([2, 3])
    for c in range(1, COLS - 1):
        height = max(0, peak_h - abs(c - peak) // slope)
        for r in range(ROWS - 1 - height, ROWS - 1):
            grid[r][c] = SOLID


def _family_corridor_maze(grid: Grid, rng: random.Random, surface: int, shaft: int) -> None:
    """Dense solid mass with thin winding corridors carved through it — the most
    common reference archetype (Ep1 L4 spiral / corridor maze). Interior stays
    SOLID; thin 1-tall walkways every 3 rows are carved with solid pillars left
    between runs (winding), then linked by short jumpable vertical connectors."""
    rows = [surface + 3, surface + 7, surface + 11]
    for r in rows:
        for cc in range(1, COLS - 1):
            grid[r][cc] = EMPTY
            grid[r - 1][cc] = EMPTY
            grid[r - 2][cc] = EMPTY
        c = rng.randint(7, 10)
        while c < COLS - 6:
            width = rng.randint(2, 3)
            for cc in range(c, min(c + width, COLS - 1)):
                # Ceiling teeth and short buttresses add maze texture without
                # sealing each side into disconnected chunks.
                grid[r - 2][cc] = SOLID
                if rng.random() < 0.45:
                    grid[r - 1][cc] = SOLID
            c += rng.randint(8, 12)
    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]
        for _ in range(rng.randint(2, 3)):  # fewer, wider links across the width
            c = rng.randint(2, COLS - 3)
            width = rng.choice([1, 2, 2])
            for r in range(r0 - 2, r1 + 1):
                grid[r][c] = EMPTY
                if width == 2 and c + 1 < COLS - 1:
                    grid[r][c + 1] = EMPTY
    for r in range(surface + 1, rows[0] + 1):
        grid[r][shaft] = EMPTY  # drop the shaft into the top corridor


FAMILIES = {
    "platform_network": _family_platform_network,
    "snake_bands": _family_snake_bands,
    "terrain_climb": _family_terrain_climb,
    "corridor_maze": _family_corridor_maze,
}
FAMILY_NAMES = tuple(FAMILIES.keys())


def _seal_unreachable_open(grid: Grid, start: Cell, surface: int) -> None:
    """Fill every open cave tile the player can't reach with solid rock, so the
    remaining open space is exactly the connected, walkable level ("walk through
    everything"). This never erodes solid, so it works for open *and* dense-maze
    families; the exit chamber (reachable with the door open) is left intact."""
    reach = cave_reachable(grid, start, doors_open=True)
    for r in range(surface + 1, ROWS - 1):
        for c in range(1, COLS - 1):
            if grid[r][c] in (EMPTY, LADDER) and (c, r) not in reach:
                grid[r][c] = SOLID


def _standing_tiles(grid: Grid, reach: Set[Cell], surface: int) -> List[Cell]:
    return sorted(
        (
            (c, r)
            for (c, r) in reach
            if grid[r][c] == EMPTY
            and r > surface
            and r + 1 < ROWS
            and (grid[r + 1][c] == SOLID or grid[r + 1][c] in DOOR_CHARS)
        ),
        key=lambda t: (t[1], t[0]),
    )


def _place_gated_pocket(
    grid: Grid,
    start: Cell,
    standing: List[Cell],
    surface: int,
    door_char: str = DOOR,
    avoid: Optional[Set[Cell]] = None,
) -> Optional[Tuple[Cell, Cell]]:
    """Wall off a small drop-slot capped by a (colour-keyed) DOOR and leave its
    floor EMPTY for a crystal. In the real game the switch opens an *obstacle* (a
    colour-keyed door), not the exit; gating one crystal behind the door makes
    the switch mandatory (you need every crystal) while the exit opens simply by
    collecting them all. Returns (pocket_cell, door_cell), or None if nothing
    gates cleanly. Tries the deepest candidates first for a tucked-away pocket."""
    avoid = avoid or set()
    reach = cave_reachable(grid, start, doors_open=True)
    for ex, ey in sorted(standing, key=lambda t: (-t[1], t[0])):
        if ey < ROWS - 7:  # deepest-first -> once too shallow, give up
            break
        if ey - 2 <= surface or not (1 < ex < COLS - 2):
            continue
        if grid[ey - 1][ex] != EMPTY or grid[ey - 2][ex] != EMPTY:
            continue
        if (ex, ey) in avoid or (ex, ey - 1) in avoid:
            continue
        if (ex, ey - 2) not in reach:  # need a landing to drop in from
            continue
        walls = ((ex - 1, ey), (ex + 1, ey), (ex - 1, ey - 1), (ex + 1, ey - 1))
        snap = [(c, r, grid[r][c]) for c, r in (*walls, (ex, ey - 1), (ex, ey))]
        for c, r in walls:
            grid[r][c] = SOLID
        grid[ey - 1][ex] = door_char
        open_reach = cave_reachable(grid, start, doors_open=True)
        closed_reach = cave_reachable(grid, start, doors_open=False)
        # pocket floor reachable only with the door open => switch required
        if (ex, ey) in open_reach and (ex, ey) not in closed_reach and (ex, ey - 2) in open_reach:
            return (ex, ey), (ex, ey - 1)
        for c, r, value in snap:
            grid[r][c] = value
    return None


def _place_open_exit(
    grid: Grid, start: Cell, standing: List[Cell], surface: int, avoid: Set[Cell]
) -> Optional[Cell]:
    """Place the exit on an open, reachable standing tile deep in the cave. The
    exit is NOT walled — collecting every crystal is what unlocks it (the border
    flips green in the original), so it just needs to be reachable once the door
    is open. Prefers deep, central tiles for top-to-bottom route flow."""
    px, py = start
    reach = cave_reachable(grid, start, doors_open=True)
    cands = [
        (c, r)
        for (c, r) in sorted(standing, key=lambda t: (-t[1], abs(t[0] - COLS // 2)))
        if (c, r) in reach
        and (c, r) not in avoid
        and r >= ROWS - 8
        and grid[r][c] == EMPTY
        and abs(c - px) + abs(r - py) > 6
    ]
    if not cands:
        return None
    ex, ey = cands[0]
    grid[ey][ex] = EXIT
    return ex, ey


def _place_threats(
    grid: Grid,
    rng: random.Random,
    standing: List[Cell],
    start: Cell,
    surface: int,
    spec: Dict[str, Any],
    diff: Dict[str, Any],
    crystal_cells: List[Cell],
) -> None:
    """Place hazards and enemies. About half the hazard budget guards chokepoints
    — the floor tiles flanking a crystal, the approach you must cross to grab it —
    so threats are part of the route puzzle as in the original, not pure ambient
    scatter. The rest (and all enemies) spread through the deeper, harder bands.
    Reachability ignores hazards, so a guarded crystal stays winnable (you cross
    or jump the hazard); this only raises the stakes of the approach."""
    px, py = start
    hazards, enemies = spec["hazards"], spec["enemies"]
    hazard_budget = rng.randint(*diff["hazards"])
    enemy_budget = rng.randint(*diff["enemies"])
    floors = [
        (c, r)
        for (c, r) in standing
        if grid[r][c] == EMPTY and r >= surface + 4 and abs(c - px) + abs(r - py) > 4
    ]
    rng.shuffle(floors)
    floor_set = set(floors)

    if hazard_budget > 0:
        guard_spots = [
            (cc + dc, cr)
            for (cc, cr) in crystal_cells
            for dc in (-1, 1)
            if (cc + dc, cr) in floor_set
        ]
        rng.shuffle(guard_spots)
        for c, r in guard_spots[: hazard_budget // 2 + 1]:
            if hazard_budget <= 0:
                break
            if grid[r][c] != EMPTY:
                continue
            grid[r][c] = rng.choice(hazards)
            hazard_budget -= 1
            floor_set.discard((c, r))

    for c, r in floors:
        if grid[r][c] != EMPTY:
            continue
        band = _band(r, surface)
        if hazard_budget > 0 and rng.random() < 0.05 + 0.05 * band:
            grid[r][c] = rng.choice(hazards)
            hazard_budget -= 1
        elif enemy_budget > 0 and rng.random() < 0.05 + 0.03 * band:
            grid[r][c] = rng.choice(enemies)
            enemy_budget -= 1


def _place_elevators(grid: Grid, rng: random.Random, surface: int, shaft: int) -> None:
    """Convert one or two tall, wall-flanked vertical shafts into elevator lifts.
    The flood already treats an ELEVATOR run as rideable (a superset of jumpable),
    so this never breaks solvability — it adds authentic switch-free vertical
    transport and an easier ascent than a precise jump (good for the agent too)."""
    runs: List[Tuple[int, int, int]] = []
    for c in range(2, COLS - 2):
        if c == shaft:
            continue
        r = surface + 1
        while r < ROWS - 1:
            shaft_cell = (
                grid[r][c] in (EMPTY, LADDER)
                and grid[r][c - 1] == SOLID
                and grid[r][c + 1] == SOLID
            )
            if shaft_cell:
                top = r
                while (
                    r < ROWS - 1
                    and grid[r][c] in (EMPTY, LADDER)
                    and grid[r][c - 1] == SOLID
                    and grid[r][c + 1] == SOLID
                ):
                    r += 1
                if r - top >= 3:
                    runs.append((c, top, r - 1))
            else:
                r += 1
    rng.shuffle(runs)
    used_cols: Set[int] = set()
    placed = 0
    target = rng.randint(1, 2)
    for c, top, bot in runs:
        if placed >= target:
            break
        if any(abs(c - uc) < 3 for uc in used_cols):
            continue
        for rr in range(top, bot + 1):
            grid[rr][c] = ELEVATOR
        used_cols.add(c)
        placed += 1


def _place_ladders(
    grid: Grid,
    surface: int,
    shaft: int,
    *,
    max_runs: Optional[int] = None,
    min_len: int = 2,
    min_spacing: int = 5,
) -> None:
    """Turn decorative one-wide vertical shafts into real climbable ladders.

    The renderer already drew these connectors as ladders. Making the tile
    explicit keeps the generator oracle, rendering, and live physics aligned:
    the player can climb the same shafts the level art tells them to climb.
    """

    runs: List[Tuple[int, int, int]] = []
    for c in range(1, COLS - 1):
        r = surface + 1
        while r < ROWS - 1:
            shaft_cell = grid[r][c] == EMPTY and grid[r][c - 1] == SOLID and grid[r][c + 1] == SOLID
            if not shaft_cell:
                r += 1
                continue
            top = r
            while (
                r < ROWS - 1
                and grid[r][c] == EMPTY
                and grid[r][c - 1] == SOLID
                and grid[r][c + 1] == SOLID
            ):
                r += 1
            bottom = r - 1
            if bottom - top + 1 < min_len:
                continue
            runs.append((c, top, bottom))

    used_cols: Set[int] = set()
    placed = 0
    for c, top, bottom in sorted(runs, key=lambda run: (run[1], run[0])):
        if max_runs is not None and placed >= max_runs:
            break
        if any(abs(c - used) < min_spacing for used in used_cols):
            continue
        for rr in range(top, bottom + 1):
            grid[rr][c] = LADDER
        used_cols.add(c)
        placed += 1


def _place_platform_connector_ladders(grid: Grid, surface: int) -> None:
    """Mark existing platform-network connector cuts as short climbable ladders.

    This is deliberately late and local: the generator first builds normal open
    rooms/pockets, then only the already-carved vertical cuts through shelf floors
    become climbable. That keeps the visual closer to authored ladder segments
    instead of adding full-screen scaffold columns.
    """

    for c in range(2, COLS - 2):
        r = surface + 1
        while r < ROWS - 1:
            if grid[r][c] != EMPTY:
                r += 1
                continue
            top = r
            shelf_cuts = 0
            while r < ROWS - 1 and grid[r][c] == EMPTY:
                if grid[r][c - 1] == SOLID and grid[r][c + 1] == SOLID:
                    shelf_cuts += 1
                r += 1
            bottom = r - 1
            run_len = bottom - top + 1
            if 4 <= run_len <= 7 and shelf_cuts >= 1:
                for rr in range(top, bottom + 1):
                    grid[rr][c] = LADDER


# Difficulty presets scale the objective/threat budget so a curriculum can start
# on a learnable floor (few crystals, no threats) and ramp to the full game. The
# terrain family is unchanged; only what gets placed on it changes.
DIFFICULTY: Dict[str, Dict[str, Any]] = {
    # First-objective routing floor: same full cave footprint and normal spawn,
    # but the single crystal is placed near the entrance route. This is for
    # training the "leave spawn and collect one real cave crystal" skill before
    # asking the policy to route across the whole tutorial cave.
    "route_floor": {
        "crystals": (1, 1),
        "ammo": 1,
        "hazards": (0, 0),
        "enemies": (0, 0),
        "locks": 0,
        "route_floor": True,
    },
    # Stricter first-objective scaffold: the crystal sits on the catch ledge
    # directly under the entrance shaft. This removes the fall-past timing problem
    # from route_floor while still practicing "walk to shaft, descend, collect".
    "route_catch": {
        "crystals": (1, 1),
        "ammo": 1,
        "hazards": (0, 0),
        "enemies": (0, 0),
        "locks": 0,
        "route_catch": True,
    },
    # Middle first-objective scaffold: the crystal is still walk/fall reachable
    # from the entrance route, but it is laterally offset from the shaft catch.
    # This bridges catch-ledge mastery to normal tutorial crystals without adding
    # jump timing, hazards, locks, or multiple objectives.
    "route_offset": {
        "crystals": (1, 1),
        "ammo": 1,
        "hazards": (0, 0),
        "enemies": (0, 0),
        "locks": 0,
        "route_offset": True,
    },
    # The simplest winnable level: one crystal on the open route, no switch/lock,
    # no threats — "collect them all" is trivially satisfiable, so the agent only
    # has to grab a crystal and reach the exit. The curriculum's first rung.
    "tutorial": {"crystals": (1, 1), "ammo": 1, "hazards": (0, 0), "enemies": (0, 0), "locks": 0},
    # Bridges the tutorial->easy cliff: multiple crystals to teach collect-and-route,
    # but NO lock (so no gated 1-wide drop-pocket) — every crystal stays on the open
    # walkable route. Mirrors the real game's opening rooms (several crystals along the
    # walking floor, colour-keyed doors arrive later).
    "easy_open": {"crystals": (2, 3), "ammo": 1, "hazards": (0, 0), "enemies": (0, 0), "locks": 0},
    "easy": {"crystals": (2, 3), "ammo": 2, "hazards": (0, 0), "enemies": (0, 0), "locks": 1},
    "normal": {"crystals": (8, 10), "ammo": 3, "hazards": (3, 5), "enemies": (2, 3), "locks": 1},
}
DIFFICULTY_NAMES = tuple(DIFFICULTY.keys())


def _attempt(seed: int, theme: str, family: str, difficulty: str = "normal") -> Optional[CaveSpec]:
    rng = random.Random(seed)
    spec = THEMES[theme]
    diff = DIFFICULTY.get(difficulty, DIFFICULTY["normal"])
    grid: Grid = [[SOLID] * COLS for _ in range(ROWS)]

    # sky band + planet surface
    for r in range(SKY):
        for c in range(COLS):
            grid[r][c] = EMPTY
    surface = SKY
    for c in range(COLS):
        grid[surface][c] = SOLID

    px, py = 3, SKY - 1
    shaft = px + rng.randint(4, 7)

    # the chosen family carves the cave terrain below the surface
    FAMILIES[family](grid, rng, surface, shaft)

    # the mine shaft entrance + a catch ledge, punched after the family so the
    # drop from the surface is always open regardless of style
    grid[py][px] = EMPTY
    grid[surface][shaft] = EMPTY
    grid[surface][shaft + 1] = EMPTY
    for r in range(surface, surface + 3):
        grid[r][shaft] = EMPTY
    for c in range(shaft - 1, shaft + 3):
        if 1 <= c <= COLS - 2:
            grid[surface + 3][c] = SOLID

    ladder_caps = {
        "platform_network": 0,
        "snake_bands": 0,
        "terrain_climb": 1,
        "corridor_maze": 2,
    }
    _place_ladders(
        grid,
        surface,
        shaft,
        max_runs=ladder_caps.get(family),
        min_len=3,
        min_spacing=7,
    )

    _seal_unreachable_open(grid, (px, py), surface)

    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    standing = _standing_tiles(grid, reach_open, surface)
    if len(standing) < 16 or not any(t[1] >= ROWS - 6 for t in standing):
        return None

    grid[py][px] = PLAYER

    # Gate one crystal behind a switch-controlled door per colour (the switch
    # opens an obstacle, as in the original, not the exit). A gated crystal makes
    # its switch mandatory because every crystal is needed to unlock the exit.
    # "normal" caves sometimes add a second, differently-coloured lock (a blue
    # lever opens only the blue door) for an authentic multi-key puzzle. The
    # "tutorial" tier uses no lock at all (locks == 0): one open crystal + exit.
    n_locks = diff.get("locks", 1)
    locks: List[Tuple[Cell, Cell, str, str]] = []
    if n_locks >= 1:
        gated = _place_gated_pocket(grid, (px, py), standing, surface, door_char=DOOR)
        if gated is None:
            return None
        locks.append((*gated, DOOR, SWITCH))
        if difficulty == "normal" and rng.random() < 0.5:
            taken = {gated[0], gated[1]}
            gated2 = _place_gated_pocket(
                grid, (px, py), standing, surface, door_char=DOOR2, avoid=taken
            )
            if gated2 is not None:
                locks.append((*gated2, DOOR2, SWITCH2))

    # the exit sits openly in the cave; collecting all crystals unlocks it
    _seal_unreachable_open(grid, (px, py), surface)
    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    standing = _standing_tiles(grid, reach_open, surface)
    pocket_cells = {lk[0] for lk in locks}
    exit_avoid = set(pocket_cells)
    if diff.get("route_catch"):
        exit_avoid.add((shaft, surface + 2))
    exit_cell = _place_open_exit(grid, (px, py), standing, surface, avoid=exit_avoid)
    if exit_cell is None:
        return None
    ex, ey = exit_cell

    _seal_unreachable_open(grid, (px, py), surface)
    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    standing = _standing_tiles(grid, reach_open, surface)

    # Coverage gate: reject layouts whose walkable space collapsed to one corner
    # (carved families can seal off poorly-linked regions, leaving dead rock with
    # buried items). Require the standing tiles to span most of the width and
    # depth so the cave reads as a full level; the pipeline retries otherwise.
    if standing:
        col_span = max(c for c, _ in standing) - min(c for c, _ in standing)
        row_span = max(r for _, r in standing) - min(r for _, r in standing)
        if len(standing) < 24 or col_span < int(COLS * 0.6) or row_span < (ROWS - SKY) // 2:
            return None

    # each switch must be reachable with every door CLOSED (so no lock depends on
    # another); place it near the door it controls for a short, readable conduit.
    reach_closed = cave_reachable(grid, (px, py), doors_open=False)
    used_switch_cells: Set[Cell] = set()
    for pocket_cell, door_cell, _door_char, switch_char in locks:
        dx, dy = door_cell
        switch_cands = [
            t
            for t in standing
            if t in reach_closed
            and SKY + 2 < t[1]
            and t not in pocket_cells
            and t != (ex, ey)
            and t not in used_switch_cells
            and abs(t[0] - px) + abs(t[1] - py) > 5
        ]
        if not switch_cands:
            return None
        switch_cands.sort(key=lambda t: (t[0] - dx) ** 2 + (t[1] - dy) ** 2)
        near_door = switch_cands[: max(1, len(switch_cands) // 4)]
        sx, sy = rng.choice(near_door)
        grid[sy][sx] = switch_char
        used_switch_cells.add((sx, sy))

    # one gated crystal occupies each pocket; the rest are placed below
    for pc in pocket_cells:
        grid[pc[1]][pc[0]] = CRYSTAL

    free = [
        t for t in standing if grid[t[1]][t[0]] == EMPTY and abs(t[0] - px) + abs(t[1] - py) > 3
    ]
    rng.shuffle(free)

    # L3: on lock-free tiers (tutorial/easy_open) guarantee at least one crystal sits
    # on the WALK-ONLY route (no jump needed), so a policy that has only learned to
    # walk/fall can still collect a crystal and produce the dense crystal signal that
    # bootstraps learning. take() pops from the end, so append a walk-reachable cell
    # last to ensure it becomes the first crystal placed.
    if n_locks == 0:
        reach_walk = cave_reachable(grid, (px, py), doors_open=True, jump=0)
        walk_free = [t for t in free if t in reach_walk]
        if not walk_free:
            return None
        if diff.get("route_floor"):
            guaranteed = min(
                walk_free,
                key=lambda t: (abs(t[0] - px) + abs(t[1] - py), t[1], t[0]),
            )
        elif diff.get("route_catch"):
            catch_tile = (shaft, surface + 2)
            guaranteed = (
                catch_tile
                if catch_tile in walk_free
                else min(
                    walk_free,
                    key=lambda t: (
                        abs(t[0] - catch_tile[0]) + abs(t[1] - catch_tile[1]),
                        t[1],
                        t[0],
                    ),
                )
            )
        elif diff.get("route_offset"):
            catch_tile = (shaft, surface + 2)
            shaft_cols = {shaft, shaft + 1}
            offset_free = [
                t
                for t in walk_free
                if t[0] not in shaft_cols
                and t[1] >= surface + 2
                and 2 <= abs(t[0] - catch_tile[0]) + abs(t[1] - catch_tile[1]) <= 10
            ]
            if not offset_free:
                offset_free = [t for t in walk_free if t[0] not in shaft_cols]
            if not offset_free:
                offset_free = walk_free
            guaranteed = min(
                offset_free,
                key=lambda t: (
                    abs(t[0] - catch_tile[0]) + abs(t[1] - catch_tile[1]),
                    t[1],
                    t[0],
                ),
            )
        else:
            guaranteed = walk_free[0]
        free.remove(guaranteed)
        free.append(guaranteed)

    placed_items: Set[Cell] = set(pocket_cells) | {exit_cell} | used_switch_cells

    def take(n: int, *, min_distance: int = 0) -> List[Cell]:
        chosen: List[Cell] = []
        if min_distance:
            for cell in reversed(free[:]):
                if len(chosen) >= n:
                    break
                if any(
                    abs(cell[0] - other[0]) + abs(cell[1] - other[1]) < min_distance
                    for other in placed_items
                ):
                    continue
                free.remove(cell)
                chosen.append(cell)
                placed_items.add(cell)
        while len(chosen) < n and free:
            cell = free.pop()
            chosen.append(cell)
            placed_items.add(cell)
        return chosen

    crystal_cells = list(pocket_cells)
    for c, r in take(max(1, rng.randint(*diff["crystals"]) - len(pocket_cells)), min_distance=5):
        grid[r][c] = CRYSTAL
        crystal_cells.append((c, r))
    for c, r in take(diff["ammo"], min_distance=4):
        grid[r][c] = AMMO
    for c, r in take(1, min_distance=4):
        grid[r][c] = rng.choice(spec["powerups"])
    for c, r in take(1, min_distance=4):
        grid[r][c] = TREASURE

    # Hazards + enemies. A share of the hazard budget guards chokepoints — the
    # tiles right beside a crystal — so threats are part of the route puzzle (as
    # in the original) rather than pure ambient scatter; the rest spreads deep.
    _place_threats(grid, rng, standing, (px, py), surface, spec, diff, crystal_cells)

    # authentic vertical transport: convert a tall shaft or two into elevators
    _place_elevators(grid, rng, surface, shaft)

    rows = ["".join(row) for row in grid]
    if not _solvable(rows):
        return None
    # L5: easier tiers must be solvable with the locomotion that tier is meant to
    # require — enforce a per-tier walk/ride-reachable coverage floor, not just the
    # jump-aware solvability above (a tutorial level that needs a jump is unwinnable
    # for a policy that hasn't learned to jump).
    if not _walk_coverage_ok(rows, difficulty):
        return None
    return CaveSpec(
        name=f"Generated {family} {theme}",
        layout=tuple(rows),
        background=spec["background"],
        accent=spec["accent"],
        sky_rows=SKY,
    )


def _find(rows, ch: str) -> List[Cell]:
    return [(c, r) for r, row in enumerate(rows) for c, x in enumerate(row) if x == ch]


def _find_any(rows, chars: Set[str]) -> List[Cell]:
    return [(c, r) for r, row in enumerate(rows) for c, x in enumerate(row) if x in chars]


def _solvable(rows) -> bool:
    player = _find(rows, PLAYER)[0]
    exit_ = _find(rows, EXIT)[0]
    crystals = _find(rows, CRYSTAL)
    switches = _find_any(rows, SWITCH_CHARS)
    # with every door shut, all levers must be reachable (no lock needs another)
    reach_closed = cave_reachable(rows, player, doors_open=False)
    if not all(s in reach_closed for s in switches):
        return False
    # with each lever opening its own colour (fixpoint), all crystals + the exit
    # must be reachable
    reach = cave_reachable_keyed(rows, player)
    return all(c in reach for c in crystals) and exit_ in reach


# Minimum fraction of crystals reachable WITHOUT jumping (walk + fall + ride elevators)
# per difficulty tier. None = jump-aware solvability only (the historical behaviour).
_WALK_COVERAGE_FLOOR: Dict[str, float] = {
    "route_floor": 1.0,
    "route_catch": 1.0,
    "route_offset": 1.0,
    "tutorial": 1.0,
    "easy_open": 0.6,
    "easy": 0.4,
}


def _walk_coverage_ok(rows, difficulty: str) -> bool:
    """Whether the level is solvable with the locomotion the tier requires.

    cave_reachable(jump=0) floods walking, falling, and elevator-riding (the elevator
    shaft is treated as free vertical travel), but NOT jumping — so this certifies the
    objectives a non-jumping policy can actually reach. ``normal`` keeps the prior
    jump-aware-only guarantee.
    """
    floor = _WALK_COVERAGE_FLOOR.get(difficulty)
    if floor is None:
        return True
    player = _find(rows, PLAYER)[0]
    crystals = _find(rows, CRYSTAL)
    if not crystals:
        return True
    reach_walk = cave_reachable(rows, player, doors_open=True, jump=0)
    walk_crystals = sum(1 for c in crystals if c in reach_walk)
    if difficulty in {"route_floor", "route_catch", "route_offset", "tutorial"}:
        # The single crystal AND the exit must be walk/ride reachable.
        exit_ = _find(rows, EXIT)[0]
        return walk_crystals == len(crystals) and exit_ in reach_walk
    return walk_crystals / len(crystals) >= floor


def generate_cave(
    seed: int,
    theme: Optional[str] = None,
    family: Optional[str] = None,
    difficulty: str = "normal",
) -> CaveSpec:
    """Generate a verified, themed, drop-in CaveSpec in the given level family
    (default: chosen by seed). Retries with fresh seeds until solvable.

    difficulty scales the objective/threat budget ("easy" = a learnable floor
    with few crystals and no threats; "normal" = the full game). The quality
    rubric (>=80) only applies to "normal"; easier presets intentionally fail
    it (too few crystals) so they accept the first solvable layout."""
    if theme is None:
        theme = THEME_NAMES[seed % len(THEME_NAMES)]
    if family is None:
        family = FAMILY_NAMES[seed % len(FAMILY_NAMES)]
    if difficulty == "normal":
        for attempt in range(60):
            spec = _attempt(seed * 101 + attempt, theme, family, difficulty)
            if spec is not None and grade_cave(spec)["score"] >= 80 and _body_fit_winnable(spec):
                return spec
    # Fall back to (or, for easy presets, target) the first solvable layout.
    for attempt in range(60):
        spec = _attempt(seed * 101 + attempt, theme, family, difficulty)
        if spec is not None and _body_fit_winnable(spec):
            return spec
    raise RuntimeError(f"could not generate a solvable {family} cave for seed {seed}")


def _body_fit_winnable(spec: CaveSpec) -> bool:
    """Cheap final gate that keeps generated objectives physically reachable."""

    from .crystal_caves_physics_reach import audit_physics_winnability

    return audit_physics_winnability(spec).winnable


def grade_cave(spec: CaveSpec) -> dict:
    """Score a level 0-100 on the level-gen rubric (platform-aware)."""
    rows = spec.layout
    total = ROWS * COLS
    solid = sum(row.count(SOLID) for row in rows)
    density = solid / total

    player = _find(rows, PLAYER)[0]
    crystals = _find(rows, CRYSTAL)
    exit_ = _find(rows, EXIT)[0]
    switches = _find_any(rows, SWITCH_CHARS)
    hazards = _find(rows, SPIKE) + _find(rows, ACID)

    reach_open = cave_reachable(rows, player, doors_open=True)
    reach_closed = cave_reachable(rows, player, doors_open=False)
    reach_keyed = cave_reachable_keyed(rows, player)
    grid = [list(r) for r in rows]
    standing = _standing_tiles(grid, reach_open, SKY)
    standing_conn = sum(1 for t in standing if t in reach_open) / len(standing) if standing else 0.0

    solvable = (
        all(s in reach_closed for s in switches)
        and all(c in reach_keyed for c in crystals)
        and exit_ in reach_keyed
    )
    crystal_bands = {_band(r, SKY) for (_, r) in crystals}
    # The switch must matter: at least one crystal sits behind a door (reachable
    # only once it's open), so collecting every crystal genuinely requires the
    # switch — the exit itself opens by collecting them all, as in the original.
    gated_crystals = [c for c in crystals if c in reach_open and c not in reach_closed]
    switch_gates_crystal = len(switches) >= 1 and len(gated_crystals) >= 1

    checks = {
        "solvable": solvable,
        "density": round(density, 3),
        "density_ok": 0.22 <= density <= 0.82,
        "top_entrance": player[1] <= SKY,
        "exit_near_bottom": exit_[1] >= ROWS - 5,
        "standing_connectivity": round(standing_conn, 3),
        "fully_connected": standing_conn >= 0.999,
        "crystals": len(crystals),
        "crystal_bands": len(crystal_bands),
        "switches": len(switches),
        "hazards": len(hazards),
        "switch_gates_crystal": switch_gates_crystal,
    }

    score = 0
    score += 25 if checks["solvable"] else 0
    score += 12 if checks["density_ok"] else 0
    score += 12 if checks["top_entrance"] else 0
    score += 8 if (checks["crystal_bands"] >= 3 and checks["exit_near_bottom"]) else 0
    score += int(12 * standing_conn)
    score += 10 if (8 <= len(crystals) <= 16 and len(switches) >= 1) else 0
    score += 8 if (len(hazards) >= 3) else 0
    score += 6 if checks["switch_gates_crystal"] else 0
    score += 7 if (len(switches) >= 1 and exit_[1] >= ROWS - 5 and len(crystals) >= 8) else 0
    checks["score"] = score
    return checks
