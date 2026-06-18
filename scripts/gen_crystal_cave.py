"""Crystal Caves auto level generator — PROTOTYPE v0 (practice script).

Proves the core pipeline against the two hard constraints:
  1. The player entrance is at the very TOP of the level; the level descends.
  2. Every area is reachable ("walk through everything") — guaranteed by the
     same jump-aware reachability flood the solvability test uses: any open tile
     the player cannot reach is filled back to solid rock.

It generates a layout, self-grades it against a rubric, and (optionally) renders
a PNG via the real CrystalCaves renderer. This is a starting point to refine
once the level-design spec lands.

    python scripts/gen_crystal_cave.py --seed 1 --render /tmp/gen.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROWS, COLS = 18, 44
JUMP = 3
SKY = 3  # open rows at the top rendered as space + planet surface

SOLID, EMPTY = "#", "."
PLAYER, EXIT, DOOR, SWITCH = "P", "E", "D", "s"
CRYSTAL, AMMO, TREASURE = "*", "A", "$"
POWER, GRAV, FREEZE, AIRT = "p", "g", "z", "O"
SPIKE, ACID = "^", "~"
CRAWLER, FLYER = "M", "F"

Grid = List[List[str]]


def _open(grid: Grid, c: int, r: int, doors_open: bool) -> bool:
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return False
    ch = grid[r][c]
    if ch == SOLID:
        return False
    if ch == DOOR:
        return doors_open
    return True


def _grounded(grid: Grid, c: int, r: int, doors_open: bool) -> bool:
    if r + 1 >= ROWS:
        return True
    below = grid[r + 1][c]
    return below == SOLID or (below == DOOR and not doors_open)


def reachable(grid: Grid, start: Tuple[int, int], doors_open: bool) -> Set[Tuple[int, int]]:
    """Tiles the player can occupy from ``start`` (walk, fall, jump up to 3)."""
    sc, sr = start
    f0 = JUMP if _grounded(grid, sc, sr, doors_open) else 0
    seen: Set[Tuple[int, int, int]] = set()
    tiles: Set[Tuple[int, int]] = set()
    queue: deque = deque([(sc, sr, f0)])
    while queue:
        c, r, f = queue.popleft()
        if (c, r, f) in seen:
            continue
        seen.add((c, r, f))
        tiles.add((c, r))
        if _grounded(grid, c, r, doors_open):
            f = JUMP
        if _open(grid, c, r + 1, doors_open):
            queue.append((c, r + 1, 0))
        for dc in (-1, 1):
            if _open(grid, c + dc, r, doors_open):
                nf = JUMP if _grounded(grid, c + dc, r, doors_open) else f
                queue.append((c + dc, r, nf))
        if f > 0 and _open(grid, c, r - 1, doors_open):
            queue.append((c, r - 1, f - 1))
    return tiles


def generate(seed: int) -> List[str]:
    rng = random.Random(seed)
    grid: Grid = [[SOLID] * COLS for _ in range(ROWS)]

    def carve(c: int, r: int) -> None:
        if 1 <= c <= COLS - 2 and SKY <= r <= ROWS - 2:
            grid[r][c] = EMPTY

    # --- outer-space sky band + planet surface (the top entrance) ---
    for r in range(SKY):
        for c in range(COLS):
            grid[r][c] = EMPTY  # open sky, edge to edge
    surface = SKY
    for c in range(COLS):
        grid[surface][c] = SOLID  # solid planet ground

    # Mylo stands on the surface (left); the mine shaft drops down to the right.
    px, py = 3, SKY - 1
    grid[py][px] = EMPTY
    shaft = px + rng.randint(5, 9)
    for r in range(surface, surface + 3):
        grid[r][shaft] = EMPTY
        grid[r][shaft + 1] = EMPTY

    # --- descending spine from the shaft bottom ---
    cx, cy = shaft, surface + 2
    carve(cx, cy)
    spine: List[Tuple[int, int]] = [(cx, cy)]
    while cy < ROWS - 3:
        direction = rng.choice([-1, 1])
        for _ in range(rng.randint(4, 10)):
            if 2 <= cx + direction <= COLS - 3:
                cx += direction
                carve(cx, cy)
                carve(cx, cy - 1)  # headroom so the corridor is jumpable
                spine.append((cx, cy))
        for _ in range(rng.randint(1, 3)):
            if cy < ROWS - 3:
                cy += 1
                carve(cx, cy)
                carve(cx, cy - 1)
                spine.append((cx, cy))

    # --- branches off the spine for exploration + density control ---
    for _ in range(rng.randint(7, 10)):
        bx, by = rng.choice(spine)
        direction = rng.choice([-1, 1])
        for _ in range(rng.randint(4, 9)):
            if 2 <= bx + direction <= COLS - 3:
                bx += direction
                carve(bx, by)
                carve(bx, by - 1)

    # --- connectivity guarantee: fill every CAVE tile the player can't reach ---
    reach = reachable(grid, (px, py), doors_open=True)
    for r in range(surface + 1, ROWS - 1):
        for c in range(1, COLS - 1):
            if grid[r][c] == EMPTY and (c, r) not in reach:
                grid[r][c] = SOLID
    reach = reachable(grid, (px, py), doors_open=True)

    # standing spots: reachable open tiles with solid floor below
    standing = sorted(
        (
            (c, r)
            for (c, r) in reach
            if grid[r][c] == EMPTY and r + 1 < ROWS and grid[r + 1][c] == SOLID
        ),
        key=lambda t: (t[1], t[0]),
    )

    # --- objectives ---
    grid[py][px] = PLAYER

    # exit at the lowest, then left-most, reachable standing spot
    ex, ey = max(standing, key=lambda t: (t[1], -t[0]))
    grid[ey][ex] = EXIT

    free = [
        (c, r)
        for (c, r) in standing
        if grid[r][c] == EMPTY and abs(c - px) + abs(r - py) > 3
    ]
    rng.shuffle(free)

    def take(n: int) -> List[Tuple[int, int]]:
        out = [free.pop() for _ in range(min(n, len(free)))]
        return out

    for c, r in take(rng.randint(9, 12)):
        grid[r][c] = CRYSTAL
    for c, r in take(3):
        grid[r][c] = AMMO
    for c, r in take(1):
        grid[r][c] = rng.choice([POWER, GRAV, FREEZE])
    for c, r in take(1):
        grid[r][c] = TREASURE

    # short spike runs as hazards (passable but damaging) — only down in the
    # cave, never on the clean surface entrance.
    floors = [
        (c, r)
        for (c, r) in standing
        if grid[r][c] == EMPTY and r >= surface + 3 and (c, r) not in (free or [(ex, ey)])
    ]
    rng.shuffle(floors)
    for c, r in floors[:6]:
        if grid[r][c] == EMPTY:
            grid[r][c] = SPIKE

    return ["".join(row) for row in grid]


def grade(rows: List[str]) -> Dict[str, object]:
    grid = [list(r) for r in rows]
    total = ROWS * COLS
    solid = sum(row.count(SOLID) for row in grid)

    def find(ch: str) -> List[Tuple[int, int]]:
        return [(c, r) for r, row in enumerate(grid) for c, x in enumerate(row) if x == ch]

    player = find(PLAYER)[0]
    crystals = find(CRYSTAL)
    exit_ = find(EXIT)[0]
    reach = reachable(grid, player, doors_open=True)

    # Connectivity is measured over the CAVE (skip the decorative sky band).
    open_tiles = [
        (c, r)
        for r in range(SKY, ROWS)
        for c in range(COLS)
        if grid[r][c] != SOLID
    ]
    connected = sum(1 for t in open_tiles if t in reach) / max(1, len(open_tiles))

    checks = {
        "density": round(solid / total, 3),
        "density_ok": 0.45 <= solid / total <= 0.85,
        "top_entrance": player[1] <= 3,
        "exit_near_bottom": exit_[1] >= ROWS - 6,
        "exit_reachable": exit_ in reach,
        "crystals": len(crystals),
        "crystals_reachable": sum(1 for c in crystals if c in reach),
        "all_crystals_reachable": all(c in reach for c in crystals),
        "connectivity": round(connected, 3),
        "fully_connected": connected >= 0.999,
    }
    score = 100
    if not checks["density_ok"]:
        score -= 15
    if not checks["top_entrance"]:
        score -= 20
    if not checks["exit_near_bottom"]:
        score -= 10
    if not checks["exit_reachable"]:
        score -= 25
    if not checks["all_crystals_reachable"]:
        score -= 25
    if not checks["fully_connected"]:
        score -= 15
    if checks["crystals"] < 8:
        score -= 10
    checks["score"] = max(0, score)
    return checks


def render_png(rows: List[str], out: str) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from config import Config
    from src.game.crystal_caves import CrystalCaves
    from src.game.crystal_caves_entities import CaveSpec

    pygame.init()
    cfg = Config()
    game = CrystalCaves(cfg, headless=False)
    spec = CaveSpec(
        name="GENERATED",
        layout=tuple(rows),
        background=(9, 12, 22),
        accent=(80, 190, 255),
        sky_rows=SKY,
    )
    game.level = spec
    game._load_level(spec)
    # Generated caves have no authored CAVE_DRESSING yet — suppress the episode-0
    # dressing that would otherwise bleed in at hard-coded coordinates.
    game._draw_authored_dressing = lambda *a, **k: None  # type: ignore[method-assign]
    surface = pygame.Surface((cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT))
    game.render(surface)
    pygame.image.save(surface, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", type=str, default="")
    args = parser.parse_args()

    rows = generate(args.seed)
    report = grade(rows)
    print(f"seed={args.seed}  grade={report['score']}/100")
    for key, value in report.items():
        if key != "score":
            print(f"  {key}: {value}")
    if args.render:
        render_png(rows, args.render)
        print(f"rendered -> {args.render}")


if __name__ == "__main__":
    main()
