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


def _carve_platforms(grid: Grid, rng: random.Random, surface: int) -> None:
    """Thread a varied web of thin platforms across the open interior. Platform
    length, spacing, vertical jitter, thickness, and support pillars are all
    randomized, and gaps widen toward the bottom (rising difficulty)."""
    for r in range(surface + 2, ROWS - 1):
        if r % 2 == 0:
            continue  # leave open lanes between platform rows
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
                # occasional support pillar dropping from a platform end
                if rng.random() < 0.16:
                    for pr in range(row + 1, min(row + 1 + rng.randint(2, 3), ROWS - 1)):
                        grid[pr][c] = SOLID
                c = end + rng.randint(gap_min, gap_min + 3)
            else:
                c += rng.randint(2, 4)


Rect = Tuple[int, int, int, int]  # (col0, row0, col1, row1), half-open


def _build_exit_chamber(grid: Grid, rng: random.Random) -> Tuple[Cell, Cell, Rect]:
    """Seal a small exit chamber at the bottom whose only entry is a DOOR, so the
    exit genuinely requires the switch. Returns (door_pos, exit_pos, rect)."""
    width = 5
    x0 = rng.randint(max(2, COLS - 4 - width), COLS - 2 - width)
    roof = ROWS - 4
    for r in range(roof, ROWS):
        for c in range(x0 - 1, x0 + width + 1):
            grid[r][c] = SOLID
    for r in range(roof + 1, ROWS - 1):
        for c in range(x0, x0 + width):
            grid[r][c] = EMPTY
    door_c = x0 + width // 2
    grid[roof][door_c] = DOOR
    # a split landing platform above the door so the player can drop through it
    for c in range(door_c - 2, door_c + 3):
        if c != door_c and 1 <= c <= COLS - 2:
            grid[roof - 2][c] = SOLID
    exit_pos = (door_c, ROWS - 2)
    rect = (x0 - 1, roof - 2, x0 + width + 1, ROWS)
    return (door_c, roof), exit_pos, rect


def _prune_unreachable(grid: Grid, start: Cell, surface: int, protect: Rect) -> None:
    """Turn floating platform bits the player can never stand on back into open
    background, so connectivity == 'every platform is usable'. The exit chamber
    ``protect`` rect is left intact so its sealed walls keep gating the exit."""
    pc0, pr0, pc1, pr1 = protect
    for _ in range(6):
        reach = cave_reachable(grid, start, doors_open=True)
        changed = False
        for r in range(surface + 1, ROWS - 1):
            for c in range(1, COLS - 1):
                if pc0 <= c < pc1 and pr0 <= r < pr1:
                    continue
                if grid[r][c] == SOLID and (c, r - 1) not in reach and (c, r) not in reach:
                    grid[r][c] = EMPTY
                    changed = True
        if not changed:
            break


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


def _attempt(seed: int, theme: str) -> Optional[CaveSpec]:
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

    # open interior below the surface
    for r in range(surface + 1, ROWS - 1):
        for c in range(1, COLS - 1):
            grid[r][c] = EMPTY

    px, py = 3, SKY - 1
    grid[py][px] = EMPTY
    shaft = px + rng.randint(4, 7)
    grid[surface][shaft] = EMPTY
    grid[surface][shaft + 1] = EMPTY
    for c in range(shaft - 1, shaft + 3):
        if 1 <= c <= COLS - 2:
            grid[surface + 3][c] = SOLID

    _carve_platforms(grid, rng, surface)
    (door_pos, exit_pos, chamber) = _build_exit_chamber(grid, rng)
    _prune_unreachable(grid, (px, py), surface, chamber)

    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    if exit_pos not in reach_open or door_pos not in reach_open:
        return None
    standing = _standing_tiles(grid, reach_open, surface)
    if len(standing) < 18:
        return None

    grid[py][px] = PLAYER
    grid[exit_pos[1]][exit_pos[0]] = EXIT

    # switch must be reachable with the door CLOSED (it opens the door)
    reach_closed = cave_reachable(grid, (px, py), doors_open=False)
    switch_cands = [
        t
        for t in standing
        if t in reach_closed and SKY + 2 < t[1] < ROWS - 4 and abs(t[0] - px) + abs(t[1] - py) > 5
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


def generate_cave(seed: int, theme: Optional[str] = None) -> CaveSpec:
    """Generate a verified, themed, drop-in CaveSpec. Retries with fresh seeds
    until a fully solvable level is produced."""
    if theme is None:
        theme = THEME_NAMES[seed % len(THEME_NAMES)]
    for attempt in range(40):
        spec = _attempt(seed * 101 + attempt, theme)
        if spec is not None and grade_cave(spec)["score"] >= 80:
            return spec
    # Fall back to the first solvable layout regardless of score.
    for attempt in range(40):
        spec = _attempt(seed * 101 + attempt, theme)
        if spec is not None:
            return spec
    raise RuntimeError(f"could not generate a solvable cave for seed {seed}")


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
        "density_ok": 0.22 <= density <= 0.50,
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
