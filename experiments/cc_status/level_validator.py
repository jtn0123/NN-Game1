"""Level validator: automatic design-quality gate for Crystal Caves levels.

Every authored level must pass a battery of INDEPENDENT checks, each one a
separate simulated "run" over the level, so a bad edit is caught the moment
it is made rather than after a training run wastes hours on junk:

  1. WIN-REQUIREMENTS run — plays by the real rules: doors start locked, a
     lever opens only its own colour once physically reached, the exit needs
     every crystal. All crystals, all levers and the exit must be reachable
     under that lock ordering (``cave_reachable_keyed``).
  2. GEMS run — with gates open, every crystal is reachable. Hazards can never
     wall a gem off in the real game (spikes/acid are passable at the cost of
     1 HP + knockback + invulnerability), so gem reachability is a pure
     geometry check; the heart COST is measured separately (check 5).
  3. ENEMY run — every enemy spawns in open space and its patrol range
     intersects the player-reachable area: a threat the player can never meet
     is decoration, not gameplay.
  4. LADDER / ELEVATOR run — every ladder run and elevator shaft touches the
     player-reachable area; unreachable climbing gear is junk geometry.
  5. HAZARD-BUDGET run — a cheapest-hits search (Dijkstra where entering a
     hazard tile costs 1) from spawn to every objective: no single objective
     may require more hits than a full-health player can survive (2), and the
     per-level count of hazard-taxed objectives is reported for design review.

All checks run on the jump-aware TILE oracle (milliseconds per level), so the
whole 16-level battery stays test-suite friendly. The slower physics-faithful
macro walkthrough (frame-exact movement) remains the deep certification and
already gates the suite in tests/test_crystal_caves_handcrafted.py.

Run:  python -m experiments.cc_status.level_validator
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.game.crystal_caves_gen import (  # noqa: E402
    DOOR_CHARS,
    JUMP,
    LADDER,
    cave_reachable,
    cave_reachable_keyed,
)

Cell = Tuple[int, int]

SOLID, ELEVATOR = "#", "="
HAZARDS = {"^", "~"}
CRYSTAL, EXIT_CH, PLAYER = "*", "E", "P"
SWITCH_CHARS = {"s", "S"}
ENEMY_CHARS = {"M", "F"}
MAX_AFFORDABLE_HITS = 2  # 3 HP: a full-health player survives two hazard hits


def _find(rows: Tuple[str, ...], chars: Set[str]) -> List[Cell]:
    return [(c, r) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in chars]


def _spawn(rows: Tuple[str, ...]) -> Cell:
    return _find(rows, {PLAYER})[0]


def _runs(rows: Tuple[str, ...], ch: str) -> List[List[Cell]]:
    """Contiguous vertical runs of ``ch`` (ladder rails / elevator shafts)."""
    runs: List[List[Cell]] = []
    seen: Set[Cell] = set()
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            if tile != ch or (c, r) in seen:
                continue
            run = []
            rr = r
            while rr < len(rows) and rows[rr][c] == ch:
                run.append((c, rr))
                seen.add((c, rr))
                rr += 1
            runs.append(run)
    return runs


def _solid(rows: Tuple[str, ...], c: int, r: int) -> bool:
    if not (0 <= r < len(rows) and 0 <= c < len(rows[0])):
        return True
    return rows[r][c] == SOLID


def _enemy_patrol(rows: Tuple[str, ...], spawn: Cell, kind: str) -> Set[Cell]:
    """The horizontal band an enemy sweeps, mirroring the engine's patrol rules:
    advance until a wall (both kinds) or a ledge (crawlers only), then flip."""
    c0, r = spawn
    cells = {spawn}
    for step in (-1, 1):
        c = c0
        while True:
            nxt = c + step
            if _solid(rows, nxt, r):
                break
            if kind == "crawler" and not _solid(rows, nxt, r + 1):
                break  # crawlers turn at ledges
            c = nxt
            cells.add((c, r))
    return cells


def _min_hits_to(rows: Tuple[str, ...], start: Cell) -> Dict[Cell, int]:
    """Cheapest-hazard-hits distance to every tile: Dijkstra over the same
    (cell, remaining-jump) motion states as ``cave_reachable``, where stepping
    ONTO a spike/acid tile costs 1 hit and every other move costs 0. Doors open
    (the endgame state in which gem collection happens)."""
    rows_n, cols_n = len(rows), len(rows[0])

    def is_open(c: int, r: int) -> bool:
        return 0 <= r < rows_n and 0 <= c < cols_n and rows[r][c] != SOLID

    def grounded(c: int, r: int) -> bool:
        if rows[r][c] in (ELEVATOR, LADDER):
            return True
        if r + 1 >= rows_n:
            return True
        return rows[r + 1][c] == SOLID

    def hit_cost(c: int, r: int) -> int:
        return 1 if rows[r][c] in HAZARDS else 0

    def shaft(c: int, r: int) -> List[Cell]:
        ch = rows[r][c]
        top = r
        while top - 1 >= 0 and rows[top - 1][c] == ch:
            top -= 1
        bot = r
        while bot + 1 < rows_n and rows[bot + 1][c] == ch:
            bot += 1
        return [(c, rr) for rr in range(top, bot + 1)]

    sc, sr = start
    best: Dict[Tuple[int, int, int], int] = {}
    out: Dict[Cell, int] = {}
    f0 = JUMP if grounded(sc, sr) else 0
    heap: List[Tuple[int, int, int, int]] = [(0, sc, sr, f0)]
    while heap:
        hits, c, r, f = heapq.heappop(heap)
        if best.get((c, r, f), 1 << 30) < hits:
            continue
        best[(c, r, f)] = hits
        out[(c, r)] = min(out.get((c, r), 1 << 30), hits)
        if grounded(c, r):
            f = JUMP
        moves: List[Tuple[Cell, int]] = []
        if rows[r][c] in (ELEVATOR, LADDER):
            moves.extend(((cell, JUMP) for cell in shaft(c, r)))
        if is_open(c, r + 1):
            moves.append(((c, r + 1), 0))
        for dc in (-1, 1):
            if is_open(c + dc, r):
                moves.append(((c + dc, r), JUMP if grounded(c + dc, r) else f))
        if f > 0 and is_open(c, r - 1):
            moves.append(((c, r - 1), f - 1))
        for (nc, nr), nf in moves:
            nh = hits + hit_cost(nc, nr)
            if nh < best.get((nc, nr, nf), 1 << 30):
                best[(nc, nr, nf)] = nh
                heapq.heappush(heap, (nh, nc, nr, nf))
    return out


def validate_level(layout: Tuple[str, ...]) -> Dict:
    """All validator runs for one level; see the module docstring for the list."""
    rows = tuple(layout)
    spawn = _spawn(rows)
    crystals = set(_find(rows, {CRYSTAL}))
    switches = set(_find(rows, SWITCH_CHARS))
    exits = _find(rows, {EXIT_CH})
    exit_pos = exits[0] if exits else None

    # 1. WIN-REQUIREMENTS run: real lock ordering (levers open their own colour).
    keyed_reach = cave_reachable_keyed(rows, spawn)
    win_missing = sorted((crystals | switches | ({exit_pos} if exit_pos else set())) - keyed_reach)

    # 2. GEMS run: gates open, every crystal reachable.
    open_reach = cave_reachable(rows, spawn, True)
    gems_missing = sorted(crystals - open_reach)

    # 3. ENEMY run: spawns in open space, patrol overlaps player space.
    enemy_report = []
    for c, r in _find(rows, ENEMY_CHARS):
        kind = "crawler" if rows[r][c] == "M" else "flyer"
        in_wall = _solid(rows, c, r)
        patrol = _enemy_patrol(rows, (c, r), kind)
        encounterable = bool(patrol & open_reach)
        enemy_report.append(
            {
                "spawn": (c, r),
                "kind": kind,
                "in_wall": in_wall,
                "patrol_cells": len(patrol),
                "encounterable": encounterable,
            }
        )

    # 4. LADDER / ELEVATOR run: every run/shaft touches reachable space.
    ladder_unreachable = [run[0] for run in _runs(rows, LADDER) if not (set(run) & open_reach)]
    elevator_unreachable = [run[0] for run in _runs(rows, ELEVATOR) if not (set(run) & open_reach)]

    # 5. HAZARD-BUDGET run: cheapest hits from spawn to each objective.
    min_hits = _min_hits_to(rows, spawn)
    taxed = {
        obj: min_hits.get(obj, 1 << 30)
        for obj in crystals | switches | ({exit_pos} if exit_pos else set())
        if min_hits.get(obj, 1 << 30) > 0
    }
    unaffordable = sorted(t for t, hits in taxed.items() if hits > MAX_AFFORDABLE_HITS)

    return {
        "win_requirements_missing": win_missing,
        "gems_missing": gems_missing,
        "enemies": enemy_report,
        "enemies_in_walls": [e["spawn"] for e in enemy_report if e["in_wall"]],
        "enemies_unencounterable": [e["spawn"] for e in enemy_report if not e["encounterable"]],
        "ladders_unreachable": ladder_unreachable,
        "elevators_unreachable": elevator_unreachable,
        "hazard_taxed_objectives": {k: v for k, v in sorted(taxed.items())},
        "hazard_unaffordable": unaffordable,
        "ok": not (
            win_missing
            or gems_missing
            or ladder_unreachable
            or elevator_unreachable
            or unaffordable
            or any(e["in_wall"] or not e["encounterable"] for e in enemy_report)
        ),
    }


def main() -> int:
    from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS

    bad = 0
    print(
        f"{'level':<22} {'win':>4} {'gems':>5} {'enemy':>6} {'ladder':>7} "
        f"{'elev':>5} {'hz-taxed':>9} {'verdict':>8}"
    )
    for lv in HANDCRAFTED_LEVELS:
        res = validate_level(lv.layout)
        verdict = "OK" if res["ok"] else "JUNK"
        if not res["ok"]:
            bad += 1
        print(
            f"{lv.name:<22} "
            f"{'ok' if not res['win_requirements_missing'] else 'MISS':>4} "
            f"{'ok' if not res['gems_missing'] else 'MISS':>5} "
            f"{'ok' if not (res['enemies_in_walls'] or res['enemies_unencounterable']) else 'BAD':>6} "
            f"{'ok' if not res['ladders_unreachable'] else 'MISS':>7} "
            f"{'ok' if not res['elevators_unreachable'] else 'MISS':>5} "
            f"{len(res['hazard_taxed_objectives']):>9} "
            f"{verdict:>8}"
        )
        for key in (
            "win_requirements_missing",
            "gems_missing",
            "enemies_in_walls",
            "enemies_unencounterable",
            "ladders_unreachable",
            "elevators_unreachable",
            "hazard_unaffordable",
        ):
            if res[key]:
                print(f"    {key}: {res[key][:8]}")
    print(f"\n{len(HANDCRAFTED_LEVELS) - bad}/{len(HANDCRAFTED_LEVELS)} levels pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
