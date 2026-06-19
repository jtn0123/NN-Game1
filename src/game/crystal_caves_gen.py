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
CRYSTAL, AMMO, TREASURE = "*", "A", "$"
POWER, GRAV, FREEZE = "p", "g", "z"
SPIKE, ACID, CRAWLER, FLYER = "^", "~", "M", "F"
ELEVATOR = "="  # a vertical-shaft lift platform; rideable up/down within its run

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
        # standing inside an elevator shaft => the platform supports you (it rides
        # the full run, so any shaft cell is a valid footing)
        if rows[r][c] == ELEVATOR:
            return True
        if r + 1 >= rows_n:
            return True
        below = rows[r + 1][c]
        return below == SOLID or (below == DOOR and not doors_open)

    def elevator_run(c: int, r: int):
        """Yield every cell of the contiguous elevator shaft through (c, r)."""
        top = r
        while top - 1 >= 0 and rows[top - 1][c] == ELEVATOR:
            top -= 1
        bot = r
        while bot + 1 < rows_n and rows[bot + 1][c] == ELEVATOR:
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
        # riding an elevator: free vertical travel to any cell in its shaft
        if rows[r][c] == ELEVATOR:
            for ec, er in elevator_run(c, r):
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
    gallery_rows = list(range(surface + 3, ROWS - 1, 3))
    for r in gallery_rows:
        c = rng.randint(1, 3)
        while c < COLS - 2:
            run = rng.randint(6, 12)  # wide rooms cut from the rock
            end = min(c + run, COLS - 1)
            for cc in range(c, end):
                grid[r][cc] = SOLID  # the shelf / platform floor
                grid[r - 1][cc] = EMPTY  # stand
                grid[r - 2][cc] = EMPTY  # headroom
            # an occasional rock pillar rises through a room to vary it
            if rng.random() < 0.16 and end - c > 4:
                grid[r - 1][rng.randint(c + 1, end - 2)] = SOLID
            c = end + rng.randint(2, 4)  # rock wall between rooms
    # link adjacent galleries with vertical shafts punched straight through the
    # rock at arbitrary columns (same approach the maze uses to stay fully
    # connected); the carved span plus headroom is <= a jump so the player climbs
    # out as well as drops in. Unconstrained columns => no sealed-off pockets.
    for i in range(len(gallery_rows) - 1):
        r0, r1 = gallery_rows[i], gallery_rows[i + 1]
        for _ in range(rng.randint(5, 8)):
            c = rng.randint(2, COLS - 3)
            for r in range(r0 - 2, r1 + 1):
                grid[r][c] = EMPTY
    # mine-shaft drop into the top gallery
    for r in range(surface + 1, gallery_rows[0]):
        grid[r][shaft] = EMPTY


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


def _place_gated_pocket(
    grid: Grid, start: Cell, standing: List[Cell], surface: int
) -> Optional[Tuple[Cell, Cell]]:
    """Wall off a small drop-slot capped by a DOOR and leave its floor EMPTY for
    a crystal. In the real game the switch opens an *obstacle* (a colour-keyed
    door), not the exit; gating one crystal behind the door makes the switch
    mandatory (you need every crystal) while the exit opens simply by collecting
    them all. Returns (pocket_cell, door_cell), or None if nothing gates cleanly.
    Tries the deepest candidates first for a satisfying tucked-away pocket."""
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
        open_reach = cave_reachable(grid, start, doors_open=True)
        closed_reach = cave_reachable(grid, start, doors_open=False)
        # pocket floor reachable only with the door open => switch required
        if (ex, ey) in open_reach and (ex, ey) not in closed_reach and (ex, ey - 2) in open_reach:
            return (ex, ey), (ex, ey - 1)
        for c, r, value in snap:
            grid[r][c] = value
    return None


def _place_open_exit(
    grid: Grid, start: Cell, standing: List[Cell], surface: int, avoid: Cell
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
        and (c, r) != avoid
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
                grid[r][c] == EMPTY and grid[r][c - 1] == SOLID and grid[r][c + 1] == SOLID
            )
            if shaft_cell:
                top = r
                while r < ROWS - 1 and grid[r][c] == EMPTY and grid[r][c - 1] == SOLID and grid[r][c + 1] == SOLID:
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


# Difficulty presets scale the objective/threat budget so a curriculum can start
# on a learnable floor (few crystals, no threats) and ramp to the full game. The
# terrain family is unchanged; only what gets placed on it changes.
DIFFICULTY: Dict[str, Dict[str, Any]] = {
    "easy": {"crystals": (2, 3), "ammo": 2, "hazards": (0, 0), "enemies": (0, 0)},
    "normal": {"crystals": (10, 14), "ammo": 3, "hazards": (6, 9), "enemies": (3, 6)},
}
DIFFICULTY_NAMES = tuple(DIFFICULTY.keys())


def _attempt(
    seed: int, theme: str, family: str, difficulty: str = "normal"
) -> Optional[CaveSpec]:
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

    _seal_unreachable_open(grid, (px, py), surface)

    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    standing = _standing_tiles(grid, reach_open, surface)
    if len(standing) < 16 or not any(t[1] >= ROWS - 6 for t in standing):
        return None

    grid[py][px] = PLAYER

    # gate one crystal behind a switch-controlled door (the switch opens an
    # obstacle, as in the original, not the exit). One gated crystal makes the
    # switch mandatory because every crystal is needed to unlock the exit.
    gated = _place_gated_pocket(grid, (px, py), standing, surface)
    if gated is None:
        return None
    pocket_cell, door_cell = gated

    # the exit sits openly in the cave; collecting all crystals unlocks it
    _seal_unreachable_open(grid, (px, py), surface)
    reach_open = cave_reachable(grid, (px, py), doors_open=True)
    standing = _standing_tiles(grid, reach_open, surface)
    exit_cell = _place_open_exit(grid, (px, py), standing, surface, avoid=pocket_cell)
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

    # switch must be reachable with the door CLOSED (it opens the door); place it
    # near the door it controls so the cable is a short, readable conduit.
    reach_closed = cave_reachable(grid, (px, py), doors_open=False)
    dx, dy = door_cell
    switch_cands = [
        t
        for t in standing
        if t in reach_closed
        and SKY + 2 < t[1]
        and t not in (pocket_cell, (ex, ey))
        and abs(t[0] - px) + abs(t[1] - py) > 5
    ]
    if not switch_cands:
        return None
    switch_cands.sort(key=lambda t: (t[0] - dx) ** 2 + (t[1] - dy) ** 2)
    near_door = switch_cands[: max(1, len(switch_cands) // 4)]
    sx, sy = rng.choice(near_door)
    grid[sy][sx] = SWITCH

    # the gated crystal occupies the pocket; the rest are placed below
    grid[pocket_cell[1]][pocket_cell[0]] = CRYSTAL

    free = [
        t
        for t in standing
        if grid[t[1]][t[0]] == EMPTY and abs(t[0] - px) + abs(t[1] - py) > 3
    ]
    rng.shuffle(free)

    def take(n: int) -> List[Cell]:
        return [free.pop() for _ in range(min(n, len(free)))]

    crystal_cells = [pocket_cell]
    for c, r in take(max(1, rng.randint(*diff["crystals"]) - 1)):
        grid[r][c] = CRYSTAL
        crystal_cells.append((c, r))
    for c, r in take(diff["ammo"]):
        grid[r][c] = AMMO
    for c, r in take(1):
        grid[r][c] = rng.choice(spec["powerups"])
    for c, r in take(1):
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
    return CaveSpec(
        name=f"Generated {family} {theme}",
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
            if spec is not None and grade_cave(spec)["score"] >= 80:
                return spec
    # Fall back to (or, for easy presets, target) the first solvable layout.
    for attempt in range(60):
        spec = _attempt(seed * 101 + attempt, theme, family, difficulty)
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
    # The switch must matter: at least one crystal sits behind the door (reachable
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
