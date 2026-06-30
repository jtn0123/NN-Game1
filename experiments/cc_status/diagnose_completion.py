"""RUN-15 completion diagnostic (agent-free, structural).

RUN-14 showed that on `normal` the compass collects ~83% of crystals but the exit
NEVER unlocks (it never gets the last ones) and even TRAIN wins are 0. Before building
a fix, pin down WHETHER normal levels are even winnable within the engine's limits — a
perfect-agent upper bound, no policy involved.

Two questions per held-out level:
  1. REACHABLE? — doors-open geodesic reachability of every crystal + the exit from spawn
     (procedural levels are generator-guaranteed solvable, so this is a sanity check; <100%
     would mean a level is structurally unwinnable and 0 wins is partly mechanical).
  2. FITS THE STEP BUDGET? — a greedy nearest-crystal GEODESIC tour (collect all, then
     reach exit), doors-open (an OPTIMISTIC lower bound on path length), converted to game
     steps via TILE_SIZE/MOVE_SPEED. If even this optimistic tour exceeds MAX_STEPS=3000,
     the step limit caps wins for ANY agent → the fix is budget/curriculum, not policy.
     Also flags tours whose longest single inter-crystal leg exceeds the no-progress
     window (MAX_STEPS_WITHOUT_PROGRESS=720 steps), which would trip the early-timeout.

Run (fast, CPU, no training):
    python -m experiments.cc_status.diagnose_completion --difficulties easy,normal --games 40
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cc_status.lever_ab import make_config  # noqa: E402
from src.game.crystal_caves import CrystalCaves  # noqa: E402


def _passable_doors_open(game: CrystalCaves) -> np.ndarray:
    """Boolean traversability with every door treated OPEN (optimistic tour): only solid
    walls block. Doors/crystals/exit/switches are not in `game.grid` (they live in separate
    sets and never block movement once open), so `grid != '#'` is exactly the open map."""
    return np.array(
        [
            [game.grid[r][c] != game.SOLID for c in range(game.level_cols)]
            for r in range(game.level_rows)
        ],
        dtype=bool,
    )


def _bfs_dist(passable: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    """4-connected tile BFS distance field from start over passable tiles."""
    field: dict[tuple[int, int], int] = {start: 0}
    q: deque[tuple[int, int]] = deque([start])
    h, w = passable.shape
    while q:
        c, r = q.popleft()
        d = field[(c, r)] + 1
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c + dc, r + dr
            if 0 <= nc < w and 0 <= nr < h and passable[nr, nc] and (nc, nr) not in field:
                field[(nc, nr)] = d
                q.append((nc, nr))
    return field


def _greedy_tour_tiles(
    passable: np.ndarray, start: tuple[int, int], crystals: list, exit_pos: tuple[int, int]
) -> tuple[float, float, bool]:
    """Greedy nearest-objective geodesic tour: collect all crystals then reach exit.
    Returns (total_tiles, longest_single_leg_tiles, all_reachable)."""
    cur = start
    remaining = list(crystals)
    total = 0.0
    longest = 0.0
    while remaining:
        field = _bfs_dist(passable, cur)
        reach = [(field[c], c) for c in remaining if c in field]
        if not reach:
            return total, longest, False  # a crystal is unreachable doors-open
        d, nxt = min(reach, key=lambda x: x[0])
        total += d
        longest = max(longest, float(d))
        cur = nxt
        remaining.remove(nxt)
    field = _bfs_dist(passable, cur)
    if exit_pos not in field:
        return total, longest, False
    total += field[exit_pos]
    longest = max(longest, float(field[exit_pos]))
    return total, longest, True


def diagnose(difficulty: str, games: int) -> dict[str, Any]:
    # Same config path as the RUN-14 A/B (procedural, platform_network, difficulty-aware
    # generation) so the held-out eval levels match what was actually trained/evaluated.
    cfg = make_config({}, difficulty=difficulty)
    game = CrystalCaves(cfg, headless=True)
    game.use_eval_levels(games)
    game.reset_eval_cursor()

    steps_per_tile = game.TILE_SIZE / game.MOVE_SPEED
    rows_out: list[dict[str, Any]] = []
    for _ in range(games):
        game.reset()
        total_crystals = len(game.crystals)
        start = game._player_tile()
        passable = _passable_doors_open(game)
        tour_tiles, longest_leg, reachable = _greedy_tour_tiles(
            passable, start, list(game.crystals), game.exit_pos
        )
        tour_steps = tour_tiles * steps_per_tile
        longest_leg_steps = longest_leg * steps_per_tile
        rows_out.append(
            {
                "crystals": total_crystals,
                "reachable_doors_open": reachable,
                "tour_steps": tour_steps,
                "longest_leg_steps": longest_leg_steps,
                "fits_max_steps": tour_steps <= game.MAX_STEPS,
                "leg_under_noprogress": longest_leg_steps <= game.MAX_STEPS_WITHOUT_PROGRESS,
            }
        )

    n = len(rows_out)
    ts = np.array([r["tour_steps"] for r in rows_out])
    return {
        "difficulty": difficulty,
        "games": n,
        "max_steps": game.MAX_STEPS,
        "no_progress_window": game.MAX_STEPS_WITHOUT_PROGRESS,
        "mean_crystals": float(np.mean([r["crystals"] for r in rows_out])),
        "pct_reachable_doors_open": float(np.mean([r["reachable_doors_open"] for r in rows_out])),
        "tour_steps_mean": float(ts.mean()),
        "tour_steps_median": float(np.median(ts)),
        "tour_steps_p90": float(np.percentile(ts, 90)),
        "tour_steps_max": float(ts.max()),
        "pct_tour_fits_3000": float(np.mean([r["fits_max_steps"] for r in rows_out])),
        "pct_legs_under_noprogress": float(np.mean([r["leg_under_noprogress"] for r in rows_out])),
        "budget_used_mean_frac": float(ts.mean() / game.MAX_STEPS),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difficulties", default="easy,normal")
    p.add_argument("--games", type=int, default=40)
    args = p.parse_args()
    for diff in [d.strip() for d in args.difficulties.split(",") if d.strip()]:
        r = diagnose(diff, args.games)
        print(
            f"\n==== {diff.upper()}  (n={r['games']}, MAX_STEPS={r['max_steps']}, "
            f"no-progress window={r['no_progress_window']}) ===="
        )
        print(f"  mean crystals/level        {r['mean_crystals']:.1f}")
        print(
            f"  all crystals reachable     {r['pct_reachable_doors_open']*100:5.1f}%  (doors-open)"
        )
        print(
            f"  optimistic tour steps      mean {r['tour_steps_mean']:.0f}  median "
            f"{r['tour_steps_median']:.0f}  p90 {r['tour_steps_p90']:.0f}  max {r['tour_steps_max']:.0f}"
        )
        print(
            f"  tour FITS in {r['max_steps']} steps   {r['pct_tour_fits_3000']*100:5.1f}%  "
            f"(perfect-agent upper bound on win rate)"
        )
        print(f"  longest leg < no-progress  {r['pct_legs_under_noprogress']*100:5.1f}%")
        print(
            f"  budget used by movement    {r['budget_used_mean_frac']*100:5.1f}%  "
            f"(before jumps/backtracking/hazard-dodging)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
