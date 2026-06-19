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
from typing import Dict, List, Optional, Set, Tuple

from .crystal_caves_entities import CaveSpec

ROWS, COLS, SKY, JUMP = 18, 44, 3, 3

SOLID, EMPTY = "#", "."
PLAYER, EXIT, DOOR, SWITCH = "P", "E", "D", "s"
CRYSTAL, AMMO, TREASURE = "*", "A", "$"
POWER, GRAV, FREEZE = "p", "g", "z"
SPIKE, ACID, CRAWLER, FLYER = "^", "~", "M", "F"

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


def cave_reachable(rows, start: Cell, doors_open: bool, jump: int = JUMP) -> Set[Cell]:
    """Tiles the player can occupy from ``start`` under jump-aware physics: walk
    or air-drift sideways, fall through EMPTY, jump up to ``jump`` tiles while
    grounded. A DOOR blocks unless ``doors_open``. Accepts rows of strings or a
    list-of-lists grid (the single solvability oracle the generator trusts)."""
    rows_n, cols_n = len(rows), len(rows[0])

    def is_open(c: int, r: int) -> bool:
        if not (0 <= r < rows_n and 0 <= c < cols_n):
            return False
        ch = rows[r][c]
        if ch == SOLID:
            return False
        if ch == DOOR:
            return doors_open
        return True

    def grounded(c: int, r: int) -> bool:
        if r + 1 >= rows_n:
            return True
        below = rows[r + 1][c]
        return below == SOLID or (below == DOOR and not doors_open)

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
        if is_open(c, r + 1):
            queue.append((c, r + 1, 0))
        for dc in (-1, 1):
            if is_open(c + dc, r):
                nf = jump if grounded(c + dc, r) else f
                queue.append((c + dc, r, nf))
        if f > 0 and is_open(c, r - 1):
            queue.append((c, r - 1, f - 1))
    return tiles


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
    """Open level threaded with a varied web of thin platforms (open-platform /
    small-platform maps): randomized length, spacing, jitter, thickness, pillars."""
    _open_interior(grid)
    for r in range(surface + 2, ROWS - 1):
        if r % 2 == 0:
            continue
        depth = (r - surface) / max(1, ROWS - surface)
        place_prob = 0.64 - 0.16 * depth
        gap_min = 2 + int(3 * depth)
        c = rng.randint(1, 4)
        while c < COLS - 2:
            if rng.random() < place_prob:
                length = rng.randint(2, 8)
                jitter = rng.choice([0, 0, 0, -1, 1])
                row = max(surface + 2, min(ROWS - 2, r + jitter))
                thick = 2 if rng.random() < 0.16 else 1
                end = min(c + length, COLS - 1)
                for cc in range(c, end):
                    grid[row][cc] = SOLID
                    if thick == 2 and row + 1 < ROWS - 1:
                        grid[row + 1][cc] = SOLID
                if rng.random() < 0.16:
                    for pr in range(row + 1, min(row + 1 + rng.randint(2, 3), ROWS - 1)):
                        grid[pr][c] = SOLID
                c = end + rng.randint(gap_min, gap_min + 3)
            else:
                c += rng.randint(2, 4)


def _family_snake_bands(grid: Grid, rng: random.Random, surface: int, shaft: int) -> None:
    """Thick horizontal bands stacked with staggered end-gaps, forcing a
    left-right-left snake descent (Ep2 L7 style)."""
    _open_interior(grid)
    side = rng.randint(0, 1)
    for r in range(surface + 3, ROWS - 2, 3):
        gap = rng.randint(6, 10)
        thickness = rng.choice([1, 2, 2])
        cols = range(1, COLS - 1 - gap) if side == 0 else range(1 + gap, COLS - 1)
        for c in cols:
            for t in range(thickness):
                if r + t < ROWS - 1:
                    grid[r + t][c] = SOLID
        side ^= 1


def _family_terrain_climb(grid: Grid, rng: random.Random, surface: int, shaft: int) -> None:
    """Open upper area with sparse platforms over a stepped solid terrain mound
    rising from the floor (Ep1 L1 / L13 style)."""
    _open_interior(grid)
    for r in range(surface + 2, ROWS - 7, 2):
        c = rng.randint(2, 6)
        while c < COLS - 3:
            if rng.random() < 0.5:
                length = rng.randint(3, 7)
                end = min(c + length, COLS - 1)
                for cc in range(c, end):
                    grid[r][cc] = SOLID
                c = end + rng.randint(3, 6)
            else:
                c += rng.randint(3, 5)
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
    rows = list(range(surface + 2, ROWS - 1, 3))
    for r in rows:
        c = rng.randint(1, 3)
        while c < COLS - 1:
            run = rng.randint(7, 14)  # long corridors -> rows overlap and connect
            for cc in range(c, min(c + run, COLS - 1)):
                grid[r][cc] = EMPTY
            c += run + rng.randint(2, 3)  # solid pillar -> winding maze
    for i in range(len(rows) - 1):
        r0, r1 = rows[i], rows[i + 1]
        for _ in range(rng.randint(5, 8)):  # plenty of links across the width
            c = rng.randint(2, COLS - 3)
            for r in range(r0, r1 + 1):  # connector spans <= jump so it climbs
                grid[r][c] = EMPTY
                grid[r - 1][c] = EMPTY  # headroom at the connector for the jump
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
            if grid[r][c] == EMPTY and (c, r) not in reach:
                grid[r][c] = SOLID


def _standing_tiles(grid: Grid, reach: Set[Cell], surface: int) -> List[Cell]:
    return sorted(
        (
            (c, r)
            for (c, r) in reach
            if grid[r][c] == EMPTY
            and r > surface
            and r + 1 < ROWS
            and grid[r + 1][c] in (SOLID, DOOR)
        ),
        key=lambda t: (t[1], t[0]),
    )


def _place_gated_exit(grid: Grid, start: Cell, standing: List[Cell], surface: int) -> Optional[Cell]:
    """Place the exit in a walled drop-slot capped by a DOOR, so the switch is
    ALWAYS required (the exit's only access is dropping through the door). Tries
    the deepest standing tiles that have reachable open space above; mutates the
    grid and returns the exit cell, or None if no candidate gates cleanly."""
    reach = cave_reachable(grid, start, doors_open=True)
    for ex, ey in sorted(standing, key=lambda t: (-t[1], t[0])):
        if ey < ROWS - 7:  # deepest-first -> once too shallow, give up
            break
        if ey - 2 <= surface or not (1 < ex < COLS - 2):
            continue
        if grid[ey - 1][ex] != EMPTY or grid[ey - 2][ex] != EMPTY:
            continue
        if (ex, ey - 2) not in reach:  # need a landing to drop in from
            continue
        walls = ((ex - 1, ey), (ex + 1, ey), (ex - 1, ey - 1), (ex + 1, ey - 1))
        snap = [(c, r, grid[r][c]) for c, r in (*walls, (ex, ey - 1), (ex, ey))]
        for c, r in walls:
            grid[r][c] = SOLID
        grid[ey - 1][ex] = DOOR
        grid[ey][ex] = EXIT
        open_reach = cave_reachable(grid, start, doors_open=True)
        closed_reach = cave_reachable(grid, start, doors_open=False)
        if (
            (ex, ey - 2) in open_reach
            and (ex, ey) in open_reach
            and (ex, ey) not in closed_reach
        ):
            return ex, ey
        for c, r, value in snap:
            grid[r][c] = value
    return None


def _attempt(seed: int, theme: str, family: str) -> Optional[CaveSpec]:
    rng = random.Random(seed)
    spec = THEMES[theme]
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

    _seal_unreachable_open(grid, (px, py), surface)

    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    standing = _standing_tiles(grid, reach_open, surface)
    if len(standing) < 16 or not any(t[1] >= ROWS - 6 for t in standing):
        return None

    grid[py][px] = PLAYER

    # exit in a switch-gated drop-slot (the door always genuinely gates it)
    exit_tile = _place_gated_exit(grid, (px, py), standing, surface)
    if exit_tile is None:
        return None
    ex, ey = exit_tile

    # the slot walls may have orphaned tiny pockets -> re-seal and recompute
    _seal_unreachable_open(grid, (px, py), surface)
    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    standing = _standing_tiles(grid, reach_open, surface)

    # switch must be reachable with the door CLOSED (it opens the door)
    reach_closed = cave_reachable(grid, (px, py), doors_open=False)
    switch_cands = [
        t
        for t in standing
        if t in reach_closed
        and SKY + 2 < t[1]
        and t != (ex, ey)
        and abs(t[0] - px) + abs(t[1] - py) > 5
    ]
    if not switch_cands:
        return None
    sx, sy = rng.choice(switch_cands)
    grid[sy][sx] = SWITCH

    free = [
        t
        for t in standing
        if grid[t[1]][t[0]] == EMPTY and abs(t[0] - px) + abs(t[1] - py) > 3
    ]
    rng.shuffle(free)

    def take(n: int) -> List[Cell]:
        return [free.pop() for _ in range(min(n, len(free)))]

    for c, r in take(rng.randint(10, 14)):
        grid[r][c] = CRYSTAL
    for c, r in take(3):
        grid[r][c] = AMMO
    for c, r in take(1):
        grid[r][c] = rng.choice(spec["powerups"])
    for c, r in take(1):
        grid[r][c] = TREASURE

    # hazards + enemies, weighted toward the deeper (harder) bands
    floors = [
        t
        for t in standing
        if grid[t[1]][t[0]] == EMPTY and t[1] >= surface + 4 and abs(t[0] - px) + abs(t[1] - py) > 4
    ]
    rng.shuffle(floors)
    hazard_budget = rng.randint(6, 9)
    enemy_budget = rng.randint(3, 6)
    for c, r in floors:
        if grid[r][c] != EMPTY:
            continue
        band = _band(r, surface)
        if hazard_budget > 0 and rng.random() < 0.05 + 0.05 * band:
            grid[r][c] = rng.choice(spec["hazards"])
            hazard_budget -= 1
        elif enemy_budget > 0 and rng.random() < 0.05 + 0.03 * band:
            grid[r][c] = rng.choice(spec["enemies"])
            enemy_budget -= 1

    rows = ["".join(row) for row in grid]
    if not _solvable(rows):
        return None
    return CaveSpec(
        name=f"Generated {theme}",
        layout=tuple(rows),
        background=spec["background"],
        accent=spec["accent"],
        sky_rows=SKY,
    )


def _find(rows, ch: str) -> List[Cell]:
    return [(c, r) for r, row in enumerate(rows) for c, x in enumerate(row) if x == ch]


def _solvable(rows) -> bool:
    player = _find(rows, PLAYER)[0]
    exit_ = _find(rows, EXIT)[0]
    crystals = _find(rows, CRYSTAL)
    switches = _find(rows, SWITCH)
    reach_closed = cave_reachable(rows, player, doors_open=False)
    if not all(s in reach_closed for s in switches):
        return False
    reach_open = cave_reachable(rows, player, doors_open=True)
    return all(c in reach_open for c in crystals) and exit_ in reach_open


def generate_cave(
    seed: int, theme: Optional[str] = None, family: Optional[str] = None
) -> CaveSpec:
    """Generate a verified, themed, drop-in CaveSpec in the given level family
    (default: chosen by seed). Retries with fresh seeds until solvable."""
    if theme is None:
        theme = THEME_NAMES[seed % len(THEME_NAMES)]
    if family is None:
        family = FAMILY_NAMES[seed % len(FAMILY_NAMES)]
    for attempt in range(60):
        spec = _attempt(seed * 101 + attempt, theme, family)
        if spec is not None and grade_cave(spec)["score"] >= 80:
            return spec
    # Fall back to the first solvable layout regardless of score.
    for attempt in range(60):
        spec = _attempt(seed * 101 + attempt, theme, family)
        if spec is not None:
            return spec
    raise RuntimeError(f"could not generate a solvable {family} cave for seed {seed}")


def grade_cave(spec: CaveSpec) -> dict:
    """Score a level 0-100 on the level-gen rubric (platform-aware)."""
    rows = spec.layout
    total = ROWS * COLS
    solid = sum(row.count(SOLID) for row in rows)
    density = solid / total

    player = _find(rows, PLAYER)[0]
    crystals = _find(rows, CRYSTAL)
    exit_ = _find(rows, EXIT)[0]
    switches = _find(rows, SWITCH)
    hazards = _find(rows, SPIKE) + _find(rows, ACID)

    reach_open = cave_reachable(rows, player, doors_open=True)
    reach_closed = cave_reachable(rows, player, doors_open=False)
    grid = [list(r) for r in rows]
    standing = _standing_tiles(grid, reach_open, SKY)
    standing_conn = (
        sum(1 for t in standing if t in reach_open) / len(standing) if standing else 0.0
    )

    solvable = (
        all(s in reach_closed for s in switches)
        and all(c in reach_open for c in crystals)
        and exit_ in reach_open
    )
    crystal_bands = {_band(r, SKY) for (_, r) in crystals}

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
        "door_gates_exit": exit_ not in reach_closed,
    }

    score = 0
    score += 25 if checks["solvable"] else 0
    score += 12 if checks["density_ok"] else 0
    score += 12 if checks["top_entrance"] else 0
    score += 8 if (checks["crystal_bands"] >= 3 and checks["exit_near_bottom"]) else 0
    score += int(12 * standing_conn)
    score += 10 if (8 <= len(crystals) <= 16 and len(switches) >= 1) else 0
    score += 8 if (len(hazards) >= 3) else 0
    score += 6 if checks["door_gates_exit"] else 0
    score += 7 if (len(switches) >= 1 and exit_[1] >= ROWS - 5 and len(crystals) >= 8) else 0
    checks["score"] = score
    return checks
