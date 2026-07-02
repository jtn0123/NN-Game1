"""Compass-honesty and trap audit for the hand-crafted Crystal Caves levels.

Tests the red-team hypothesis behind the RUN-24 failure taxonomy: the shaping
compass descends a 4-connected symmetric BFS field (up == down), while the
player's real motion is gravity-bound (max jump ~3 tiles, one-way drops). If
the two disagree, the compass can point at objectives the player can never
reach again — manufacturing "far, oscillating" stalls and trapped episodes.

For every level we build a *physics-faithful motion graph* using the oracle's
macro simulator (level_reach): nodes are resting cells (grounded / on-ladder),
edges are macro moves the engine can actually perform. Doors are treated as
OPEN (the post-switch endgame where traps bite). Then, per resting cell X:

  trapped(X)       — some objective (crystal/switch/exit) can NEVER be touched
                     from X: reaching X with those objectives outstanding makes
                     the episode unwinnable.
  dead(X)          — NO remaining objective is touchable from X at all.
  compass_lie(X)   — the compass field is finite at X (it says "a route
                     exists") but the specific nearest objective it descends
                     toward cannot be touched from X.
  false_hope(X)    — compass finite at X while zero objectives are touchable:
                     the agent is hard-trapped and the compass still beckons.

Run:  python -m experiments.cc_status.compass_audit [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.cc_status.level_reach import (  # noqa: E402
    _MACROS,
    _Body,
    _player_tile,
    _run_macro,
    _step,
    CRYSTAL,
    EXIT,
    LevelSim,
    MAX_FRAMES,
    MAX_RESTING,
    SWITCHES,
    TILE,
)

Cell = Tuple[int, int]


def _settle(sim: LevelSim, cell: Cell) -> _Body:
    """A resting body at ``cell``, dropped to the ground exactly like the oracle BFS."""
    b = _Body(cell[0] * TILE + 5, cell[1] * TILE + 1)
    if not sim.on_ladder(b.x, b.y):
        for _ in range(MAX_FRAMES):
            _step(sim, b, 0, False)
            if b.grounded or sim.on_ladder(b.x, b.y):
                break
    return b


def build_motion_graph(
    layout: Tuple[str, ...],
) -> Tuple[LevelSim, Set[Cell], Dict[Cell, Set[Cell]], Dict[Cell, Set[Cell]]]:
    """Physics-faithful directed motion graph over resting cells, doors open.

    Returns (sim, nodes, edges, touched_from) where ``edges[a]`` are the resting
    cells one macro away from ``a`` and ``touched_from[a]`` are the OBJECTIVE
    tiles brushed while performing macros that start at ``a``.
    """
    sim = LevelSim(layout)  # no closed doors: endgame connectivity
    objectives = sim.crystals | sim.switches | {sim.exit}

    sc, sr = sim.start
    b = _Body(sc * TILE + 5, sr * TILE + 1)
    for _ in range(MAX_FRAMES):
        _step(sim, b, 0, False)
        if b.grounded or sim.on_ladder(b.x, b.y):
            break
    start_cell = _player_tile(b.x, b.y)

    nodes: Set[Cell] = {start_cell}
    edges: Dict[Cell, Set[Cell]] = {}
    touched_from: Dict[Cell, Set[Cell]] = {}
    queue: deque = deque([start_cell])
    while queue and len(nodes) < MAX_RESTING:
        cell = queue.popleft()
        body = _settle(sim, cell)
        out: Set[Cell] = set()
        touch: Set[Cell] = set()
        for program in _MACROS:
            out.update(_run_macro(sim, body, program, touch))
        edges[cell] = out
        touched_from[cell] = touch & objectives
        for nxt in out:
            if nxt not in nodes:
                nodes.add(nxt)
                queue.append(nxt)
    return sim, nodes, edges, touched_from


def compass_field(sim: LevelSim, sources: Set[Cell]) -> Dict[Cell, Tuple[int, Cell]]:
    """The live engine's shaping field, replicated: multi-source 4-connected BFS
    over non-solid tiles with SYMMETRIC vertical edges (the up==down assumption
    under audit), doors open. Maps cell -> (distance, nearest source label)."""
    field: Dict[Cell, Tuple[int, Cell]] = {}
    queue: deque = deque()
    for src in sources:
        if not sim.solid_at(*src):
            field[src] = (0, src)
            queue.append(src)
    while queue:
        col, row = queue.popleft()
        dist, label = field[(col, row)]
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (col + dc, row + dr)
            if nxt in field or sim.solid_at(*nxt):
                continue
            field[nxt] = (dist + 1, label)
            queue.append(nxt)
    return field


def audit_level(layout: Tuple[str, ...]) -> Dict:
    sim, nodes, edges, touched_from = build_motion_graph(layout)
    objectives = sim.crystals | sim.switches | {sim.exit}
    field = compass_field(sim, sim.crystals or {sim.exit})

    # Touchable objective set from each node = union of touched_from over the
    # node's descendants (computed by BFS per node; graphs are ~a few hundred nodes).
    touchable: Dict[Cell, Set[Cell]] = {}
    for node in nodes:
        seen = {node}
        queue = deque([node])
        acc: Set[Cell] = set()
        while queue:
            cur = queue.popleft()
            acc |= touched_from.get(cur, set())
            for nxt in edges.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        touchable[node] = acc

    trapped: List[Cell] = []
    dead: List[Cell] = []
    lies: List[Cell] = []
    false_hope: List[Cell] = []
    for node in nodes:
        reach = touchable[node]
        if objectives - reach:
            trapped.append(node)
        crystals_reach = reach & sim.crystals
        if not reach:
            dead.append(node)
        entry = field.get(node)
        if entry is not None and sim.crystals:
            _, nearest = entry
            if nearest not in crystals_reach:
                lies.append(node)
            if not crystals_reach:
                false_hope.append(node)

    n = len(nodes)
    return {
        "nodes": n,
        "trapped": sorted(trapped),
        "dead": sorted(dead),
        "compass_lies": sorted(lies),
        "false_hope": sorted(false_hope),
        "trapped_frac": len(trapped) / n if n else 0.0,
        "dead_frac": len(dead) / n if n else 0.0,
        "compass_lie_frac": len(lies) / n if n else 0.0,
        "false_hope_frac": len(false_hope) / n if n else 0.0,
    }


def cross_check_oracle(layout: Tuple[str, ...]) -> Dict[str, float]:
    """Compare the tile oracle (cave_reachable — the live trapped detector's engine)
    against the physics motion graph on every reachable resting cell: any
    (cell, crystal) pair the oracle calls unreachable while the physics graph
    touches it is a FALSE TRAPPED verdict waiting to happen. This is the check
    that exposed the ladder-blind oracle (35% false pairs before the fix)."""
    from src.game.crystal_caves_gen import cave_reachable

    sim, nodes, _edges, _touched = build_motion_graph(layout)
    keep = {"#", "=", "H"}
    rows = ["".join(ch if ch in keep else "." for ch in row) for row in layout]
    miss = sum(
        1 for node in nodes for c in sim.crystals if c not in cave_reachable(rows, node, True)
    )
    pairs = len(nodes) * max(1, len(sim.crystals))
    return {"pairs": pairs, "oracle_false_unreachable": miss, "miss_frac": miss / pairs}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write per-level JSON here")
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="also compare the tile oracle (trapped detector) against physics truth",
    )
    args = parser.parse_args(argv)

    from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS

    rows = []
    print(
        f"{'level':<28} {'cells':>5} {'trapped':>8} {'dead':>6} {'compass-lie':>12} {'false-hope':>11}"
    )
    for lv in HANDCRAFTED_LEVELS:
        res = audit_level(lv.layout)
        rows.append(
            {
                "name": lv.name,
                **{k: v for k, v in res.items() if "frac" in k or k == "nodes"},
                "trapped_cells": [list(c) for c in res["trapped"]],
                "false_hope_cells": [list(c) for c in res["false_hope"]],
            }
        )
        print(
            f"{lv.name:<28} {res['nodes']:>5} {res['trapped_frac']:>8.3f} {res['dead_frac']:>6.3f}"
            f" {res['compass_lie_frac']:>12.3f} {res['false_hope_frac']:>11.3f}"
        )
    mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
    print(
        f"\n{'MEAN':<28} {'':>5} {mean('trapped_frac'):>8.3f} {mean('dead_frac'):>6.3f}"
        f" {mean('compass_lie_frac'):>12.3f} {mean('false_hope_frac'):>11.3f}"
    )
    if args.cross_check:
        print("\ntile-oracle vs physics cross-check (false-unreachable (cell,crystal) pairs):")
        tot_p = tot_m = 0
        for lv in HANDCRAFTED_LEVELS:
            cc = cross_check_oracle(lv.layout)
            tot_p += cc["pairs"]
            tot_m += cc["oracle_false_unreachable"]
            print(f"  {lv.name:<28} {cc['miss_frac']:.3f}")
        print(f"  TOTAL {tot_m}/{tot_p} = {tot_m / max(1, tot_p):.4f} (must be ~0)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
