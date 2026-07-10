# ruff: noqa: E402
"""Go-Explore demo harvester for the fixed Crystal Caves level set.

Machine-generates replay-verified winning demos WITHOUT a human player and
without the HP-budget planner (which collapses on elevator/door levels).

Method (Go-Explore, Ecoffet et al. 2019, first-return phase):
- Maintain an archive of visited "cells" (coarse game states). Each cell
  stores a deep-copied engine snapshot plus the full action trace from reset.
- Repeatedly pick a promising frontier cell (most gems collected, least
  explored), restore its snapshot, and roll a short burst of biased random
  actions. New cells reached are added with their extended traces.
- Because episode termination (death / timeout / stall clock) happens inside
  the live engine during exploration, every archived trace is by construction
  a legal episode prefix; a trace whose final step wins IS a demo.
- Every candidate win is re-verified open-loop from a fresh reset via
  demo_extract.verify_stored — the ground truth for determinism drift.

Output JSONs use the same schema as demo_extract, so diagnose_gap's
--demo-dir / --demo-pretrain / --demo-reset-p consume them unchanged.

Run:  python -m experiments.cc_status.go_explore [--levels 0,1,...]
          [--budget 200000] [--rollout 60] [--seed 0] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import Config
from experiments.cc_status.demo_extract import verify_stored
from experiments.cc_status.demo_planners import route_floor_scripted_action
from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS

Cell = Tuple[int, int, int, int, bool, int, int]


@dataclass
class Entry:
    snap: Any
    trace: List[int]
    steps: int
    chosen: int = 0


@dataclass
class LevelResult:
    level: int
    name: str
    won: bool = False
    trace: List[int] = field(default_factory=list)
    env_steps: int = 0
    archive_size: int = 0
    best_remaining: int = 10**9
    exit_unlocked_seen: bool = False
    seconds: float = 0.0
    verify_ok: bool = False


_MAX_STEPS_OVERRIDE = 0


def _fresh_game(level_index: int) -> CrystalCaves:
    spec = HANDCRAFTED_LEVELS[level_index]
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    if _MAX_STEPS_OVERRIDE:
        cfg.CRYSTAL_CAVES_MAX_STEPS_OVERRIDE = _MAX_STEPS_OVERRIDE
    game = CrystalCaves(cfg, headless=True)
    game.CAVES = (spec,)
    game._eval_caves = (spec,)
    game._randomize_levels = False
    game.use_eval_levels(1)
    game.reset_eval_cursor()
    game.reset()
    return game


def _cell(game: CrystalCaves) -> Cell:
    tx, ty = game._player_tile()
    # Key on the exact remaining-crystal SET, not its size: two different
    # partially-collected subsets must not alias, or a shorter trace with the
    # "wrong" subset overwrites the frontier snapshot of the one that can
    # actually continue.
    return (
        int(tx),
        int(ty),
        len(game.crystals),
        hash(frozenset(game.crystals)) & 0xFFFFFFFF,
        bool(game.exit_unlocked),
        hash(frozenset(getattr(game, "open_colors", ()) or ())) & 0xFFFF,
        int(getattr(game, "health", 0)),
    )


def _action_sampler(game: CrystalCaves, rng: random.Random):
    labels = [str(x) for x in getattr(game, "ACTION_LABELS", [])]
    weights = {
        "IDLE": 0.2,
        "LEFT": 3.0,
        "RIGHT": 3.0,
        "JUMP": 1.0,
        "LEFT_JUMP": 2.0,
        "RIGHT_JUMP": 2.0,
        "SHOOT": 0.4,
        "LEFT_SHOOT": 0.4,
        "RIGHT_SHOOT": 0.4,
        "INTERACT": 0.6,
    }
    ids = list(range(len(labels)))
    ws = [weights.get(labels[i], 1.0) for i in ids]

    def sample() -> int:
        return rng.choices(ids, weights=ws, k=1)[0]

    return sample


# --- oracle route planner -----------------------------------------------------
# BFS over level_reach's physics-macro graph, recording exact frame actions.
# Executed legs steer the REAL game; enemies/hazards perturb execution, and
# Go-Explore's archive absorbs those perturbations.

from experiments.cc_status import level_reach as _lr


def _act_id(game: CrystalCaves, move_dir: int, jump: bool) -> int:
    if jump:
        return {-1: game.LEFT_JUMP, 0: game.JUMP, 1: game.RIGHT_JUMP}[move_dir]
    return {-1: game.LEFT, 0: game.IDLE, 1: game.RIGHT}[move_dir]


def _door_chars_closed(game: CrystalCaves) -> frozenset:
    open_colors = set(getattr(game, "open_colors", ()) or ())
    # engine colour pairing: door char opens when its colour is thrown; a colour
    # is "open" once in open_colors — map door chars whose colour is still shut.
    colour_of = {"D": "red", "d": "blue"}
    return frozenset(ch for ch in ("D", "d") if colour_of[ch] not in open_colors)


_ROUTE_CACHE: Dict[
    Tuple[str, Tuple[int, int], frozenset], Dict[Tuple[int, int], List[Tuple[int, bool]]]
] = {}


def _route_tree(
    game: CrystalCaves, max_nodes: int = 1200
) -> Dict[Tuple[int, int], List[Tuple[int, bool]]]:
    """One BFS over the macro graph from the player's cell; returns earliest
    frame-paths to EVERY tile touched. Cached per (level, start-cell, doors) —
    the exploit arm relaunches from the same frontier constantly, so cache hits
    dominate and the sweep cost amortizes to nothing."""
    doors = _door_chars_closed(game)
    start_cell = game._player_tile()
    key = (
        str(getattr(game.level, "name", "")),
        start_cell,
        doors,
        int(game.player_x) // 2,
        int(game.player_y) // 2,
    )
    hit = _ROUTE_CACHE.get(key)
    if hit is not None:
        return hit
    layout = tuple(game.level.layout)
    hazards = _hazard_cells(layout)
    sim = _lr.LevelSim(layout, closed_doors=doors)
    b = _lr._Body(float(game.player_x), float(game.player_y))
    b.grounded = sim.is_on_surface(b.x, b.y)
    from collections import deque as _deque

    tree: Dict[Tuple[int, int], List[Tuple[int, bool]]] = {}
    seen = {start_cell}
    queue = _deque([(b, [])])
    nodes = 0
    while queue and nodes < max_nodes:
        body, frames = queue.popleft()
        nodes += 1
        for program in _lr._MACROS:
            nb = _lr._Body(body.x, body.y, 0.0, 0.0, body.grounded, body.coyote)
            leg: List[Tuple[int, bool]] = []
            prev_cell = _lr._player_tile(nb.x, nb.y)
            for seg_frames, move_dir, jump in program:
                stop = False
                for _ in range(seg_frames):
                    _lr._step(sim, nb, move_dir, jump)
                    leg.append((move_dir, jump))
                    if _hazard_near(hazards, nb.x, nb.y):
                        # trajectory comes within the safety margin of a hazard
                        # — friction slides during enemy waits drift a few px,
                        # so zero-clearance routes take damage in the engine
                        stop = True
                        break
                    if len(frames) + len(leg) > 2600:
                        stop = True
                        break
                    for tc in sim._tiles_for_rect(nb.x, nb.y):
                        if tc not in tree:
                            tree[tc] = frames + list(leg)
                    cell = _lr._player_tile(nb.x, nb.y)
                    if (nb.grounded or sim.on_ladder(nb.x, nb.y)) and cell != prev_cell:
                        prev_cell = cell
                        if cell not in seen:
                            seen.add(cell)
                            queue.append(
                                (
                                    _lr._Body(nb.x, nb.y, 0.0, 0.0, nb.grounded, nb.coyote),
                                    frames + list(leg),
                                )
                            )
                    if nb.y > (sim.rows + 2) * _lr.TILE:
                        stop = True
                        break
                if stop:
                    break
    if len(_ROUTE_CACHE) > 4000:
        _ROUTE_CACHE.clear()
    _ROUTE_CACHE[key] = tree
    return tree


_HAZARD_MARGIN = 6


def _hazard_near(hazards, x: float, y: float) -> bool:
    # margin is HORIZONTAL only: grounded friction slides drift sideways a few
    # px, but vertical position doesn't drift — and tight-corridor spike hops
    # legitimately clear hazards by ~2px vertically.
    left = int(x - _HAZARD_MARGIN) // _lr.TILE
    right = (int(x) + _lr.PW + _HAZARD_MARGIN - 1) // _lr.TILE
    top = int(y) // _lr.TILE
    bottom = (int(y) + _lr.PH - 1) // _lr.TILE
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (c, r) in hazards:
                return True
    return False


def _hazard_cells(layout):
    return {(c, r) for r, row in enumerate(layout) for c, ch in enumerate(row) if ch in ("^", "~")}


def local_grab_route(
    game: CrystalCaves, target: Tuple[int, int], max_nodes: int = 90
) -> List[Tuple[int, bool]]:
    """Small, uncached macro search for short post-knockback grabs (~0.1s)."""
    doors = _door_chars_closed(game)
    layout = tuple(game.level.layout)
    hazards = _hazard_cells(layout)
    sim = _lr.LevelSim(layout, closed_doors=doors)
    b = _lr._Body(float(game.player_x), float(game.player_y))
    b.grounded = sim.is_on_surface(b.x, b.y)
    from collections import deque as _deque

    seen = {_lr._player_tile(b.x, b.y)}
    queue = _deque([(b, [])])
    nodes = 0
    while queue and nodes < max_nodes:
        body, frames = queue.popleft()
        nodes += 1
        for program in _lr._MACROS:
            nb = _lr._Body(body.x, body.y, 0.0, 0.0, body.grounded, body.coyote)
            leg: List[Tuple[int, bool]] = []
            prev_cell = _lr._player_tile(nb.x, nb.y)
            for seg_frames, move_dir, jump in program:
                stop = False
                for _ in range(seg_frames):
                    _lr._step(sim, nb, move_dir, jump)
                    leg.append((move_dir, jump))
                    if _hazard_near(hazards, nb.x, nb.y):
                        # trajectory comes within the safety margin of a hazard
                        # — friction slides during enemy waits drift a few px,
                        # so zero-clearance routes take damage in the engine
                        stop = True
                        break
                    if len(frames) + len(leg) > 400:
                        stop = True
                        break
                    for tc in sim._tiles_for_rect(nb.x, nb.y):
                        if tc == target:
                            return frames + leg
                    cell = _lr._player_tile(nb.x, nb.y)
                    if (nb.grounded or sim.on_ladder(nb.x, nb.y)) and cell != prev_cell:
                        prev_cell = cell
                        if cell not in seen:
                            seen.add(cell)
                            queue.append(
                                (
                                    _lr._Body(nb.x, nb.y, 0.0, 0.0, nb.grounded, nb.coyote),
                                    frames + list(leg),
                                )
                            )
                    if nb.y > (sim.rows + 2) * _lr.TILE:
                        stop = True
                        break
                if stop:
                    break
    return []


def oracle_route_actions(
    game: CrystalCaves, target: Tuple[int, int], max_nodes: int = 1200
) -> List[Tuple[int, bool]]:
    """(move_dir, jump) frames from the player's position to touching target."""
    return _route_tree(game, max_nodes=max_nodes).get(target, [])


def _enemy_threat(game: CrystalCaves, next_move_dir: int):
    """Nearest live enemy within striking range ahead/overhead.

    Returns (kind, dx) or None. Callers hop over same-row crawlers (waiting is
    fatal in narrow corridors — the crawler walks INTO an idle player) and wait
    out everything else."""
    px = float(game.player_x)
    py = float(game.player_y)
    best = None
    for e in getattr(game, "enemies", ()) or ():
        if not getattr(e, "alive", True):
            continue
        dx = float(e.x) - px
        dy = float(e.y) - py
        if abs(dy) > 40:
            continue
        if abs(dx) < 26 or (next_move_dir and 0 < dx * next_move_dir < 56):
            if best is None or abs(dx) < abs(best[1]):
                best = (str(getattr(e, "kind", "")), dx, dy)
    return best


_ACTION_MJ = None


def _hazard_reflex(game: CrystalCaves, sim, hazards, action: int, rng) -> int:
    """One-frame lookahead: veto random actions whose next frame overlaps a
    hazard rect; try up to 3 alternates, else idle."""
    global _ACTION_MJ
    if _ACTION_MJ is None:
        _ACTION_MJ = {
            game.IDLE: (0, False),
            game.LEFT: (-1, False),
            game.RIGHT: (1, False),
            game.JUMP: (0, True),
            game.LEFT_JUMP: (-1, True),
            game.RIGHT_JUMP: (1, True),
        }
    for _ in range(4):
        mj = _ACTION_MJ.get(action)
        if mj is None:
            return action  # shoot/interact don't move
        b = _lr._Body(float(game.player_x), float(game.player_y), 0.0, 0.0, True, 6)
        _lr._step(sim, b, mj[0], mj[1])
        if not _hazard_near(hazards, b.x, b.y):
            return action
        action = rng.choice((game.IDLE, game.LEFT, game.RIGHT, game.JUMP))
    return game.IDLE


# --- tour-order optimization ---------------------------------------------------
# The sweep's universal failure mode: greedy nearest-target collection wastes the
# 3000-step episode clock, pinning frontiers at 2800-2900 steps with 1-6 gems
# left. Order objectives like a delivery route instead: switches first, then
# gems by a nearest-neighbour + 2-opt tour over true tile route distances.

_TOUR_CACHE: Dict[Tuple[str, frozenset, frozenset], List[Tuple[int, int]]] = {}


def _tile_walkable(layout, c: int, r: int) -> bool:
    if r < 0 or c < 0 or r >= len(layout) or c >= len(layout[0]):
        return False
    return layout[r][c] != "#"  # doors treated as open (post-switch world)


def _tile_field(layout, start):
    from collections import deque as _dq

    dist = {start: 0}
    q = _dq([start])
    while q:
        c, r = q.popleft()
        d = dist[(c, r)]
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c + dc, r + dr)
            if n not in dist and _tile_walkable(layout, n[0], n[1]):
                # vertical movement is ~2x slower than walking in real frames
                # (ladder climb ~8 frames/tile vs walk ~4), so weight it or the
                # "optimal" tour is physically expensive
                dist[n] = d + (2 if dr else 1)
                q.append(n)
    return dist


def _objective_tour(game: CrystalCaves) -> List[Tuple[int, int]]:
    """Ordered objective list: unused switches (nearest-first), then gems by a
    2-opt-improved tour over tile route distances. Cached per state."""
    layout = tuple(game.level.layout)
    gems = frozenset(game.crystals)
    switches = frozenset(game.switches - game.used_switches)
    key = (str(getattr(game.level, "name", "")), gems, switches)
    hit = _TOUR_CACHE.get(key)
    if hit is not None:
        return hit

    pos = game._player_tile()
    exit_tile = tuple(getattr(game, "exit_pos", ()) or ()) or None
    nodes = list(switches) + list(gems)
    fields = {n: _tile_field(layout, n) for n in nodes}
    pos_field = _tile_field(layout, pos)

    def dist_from(node, target):
        f = pos_field if node == pos else fields[node]
        return f.get(target, 10_000)

    order = []
    cur = pos
    todo = set(switches)
    while todo:
        nxt = min(todo, key=lambda n: dist_from(cur, n))
        order.append(nxt)
        todo.discard(nxt)
        cur = nxt
    gem_order = []
    todo = set(gems)
    while todo:
        nxt = min(todo, key=lambda n: dist_from(cur, n))
        gem_order.append(nxt)
        todo.discard(nxt)
        cur = nxt

    start_anchor = order[-1] if order else pos

    def tour_len(seq):
        # anchor the tour at the EXIT: the last gem should leave the player
        # near the door, or the final leg wastes the little clock that's left
        total = 0
        prev = start_anchor
        for n in seq:
            total += dist_from(prev, n)
            prev = n
        if exit_tile is not None and seq:
            total += fields[seq[-1]].get(exit_tile, 10_000)
        return total

    improved = True
    passes = 0
    best_len = tour_len(gem_order)
    while improved and passes < 30:
        improved = False
        passes += 1
        for i in range(len(gem_order) - 1):
            for j in range(i + 2, len(gem_order) + 1):
                cand = gem_order[:i] + gem_order[i:j][::-1] + gem_order[j:]
                cand_len = tour_len(cand)
                if cand_len < best_len:
                    gem_order, best_len = cand, cand_len
                    improved = True
    result = order + gem_order
    if len(_TOUR_CACHE) > 500:
        _TOUR_CACHE.clear()
    _TOUR_CACHE[key] = result
    return result


def _select(archive: Dict[Cell, Entry], rng: random.Random) -> Cell:
    cells = list(archive.keys())
    roll = rng.random()
    if roll < 0.5:
        # split the exploit arm: mostly push the deepest lineage, but keep
        # feeding the deepest FULL-HEALTH lineage so a damage-free run can
        # overtake the risky one (first-to-depth lineages tend to arrive hurt,
        # and pure depth-first exploitation ratchets onto them).
        healthy = [c for c in cells if c[6] >= 3]
        if healthy and rng.random() < 0.6:
            cells = healthy
        # hard exploit arm: half of all rollouts launch from the single best
        # frontier (fewest gems remaining, then shortest trace, then most HP) —
        # this concentration is what actually pushes the frontier forward; the
        # weighted/uniform arms below keep mid-tree branches alive for one-way
        # drops and HP-variant recovery.
        return min(
            cells,
            key=lambda c: (
                c[2],
                -len(archive[c].snap.used_switches),
                -c[6],
                archive[c].steps,
            ),
        )
    if roll < 0.65:
        return rng.choice(cells)
    best_remaining = min(c[2] for c in cells)
    ws = []
    for c in cells:
        e = archive[c]
        w = 1.0 / (1.0 + e.chosen) ** 0.5
        # frontier bias: strongly prefer cells closest to full collection,
        # and anything with the exit already unlocked.
        w *= 4.0 ** (-(c[2] - best_remaining) / 3.0)
        # at equal collection progress, prefer shorter prefixes — trace budget
        # (episode cap 3000) is the binding constraint on deep levels.
        w *= 1.5 ** (-e.steps / 600.0)
        # HP is grab-budget for threat-guarded gems (tank a hit, collect during
        # invulnerability); prefer frontiers that can still afford hits.
        w *= 1.05 ** c[6]
        if c[4]:
            w *= 8.0
        ws.append(w)
    return rng.choices(cells, weights=ws, k=1)[0]


def _replay_cells(level_index: int, trace: List[int]) -> Tuple[Any, List[Cell], bool]:
    """Open-loop replay from reset; returns (game, per-step cells, done_flag)."""
    g = _fresh_game(level_index)
    cells: List[Cell] = [_cell(g)]
    done = False
    for a in trace:
        _s, _r, done, _info = g.step(int(a))
        if done:
            break
        cells.append(_cell(g))
    return g, cells, done


def compact_trace(level_index: int, trace: List[int]) -> List[int]:
    """Greedy loop-cutting: repeatedly remove the largest trace segment that
    starts and ends in the same cell, keeping the cut only when an open-loop
    replay still reaches an equal-or-better state (enemy phases shift after a
    splice, so every cut is re-verified in the live engine)."""
    _g, cells, done = _replay_cells(level_index, trace)
    if done:
        return trace
    target_remaining = cells[-1][2]
    improved = True
    while improved:
        improved = False
        seen: Dict[Cell, int] = {}
        loops: List[Tuple[int, int]] = []
        for idx, c in enumerate(cells):
            if c in seen:
                loops.append((seen[c], idx))
            else:
                seen[c] = idx
        for i, j in sorted(loops, key=lambda p: p[0] - p[1]):
            if j - i < 20:
                continue
            candidate = trace[:i] + trace[j:]
            g2, cells2, done2 = _replay_cells(level_index, candidate)
            if not done2 and cells2[-1][2] <= target_remaining:
                trace, cells = candidate, cells2
                target_remaining = cells2[-1][2]
                improved = True
                break
    return trace


def explore_level(
    level_index: int,
    *,
    budget: int = 200_000,
    rollout: int = 60,
    seed: int = 0,
    max_trace: int = 0,  # 0 = derive from the game's episode cap
    log_every: int = 25_000,
    scripted_p: float = 0.7,
) -> LevelResult:
    rng = random.Random(seed * 1000 + level_index)
    spec = HANDCRAFTED_LEVELS[level_index]
    res = LevelResult(level=level_index, name=spec.name)
    started = time.time()

    game = _fresh_game(level_index)
    if not max_trace:
        max_trace = int(game.MAX_STEPS) - 100
    sample = _action_sampler(game, rng)
    root = _cell(game)
    archive: Dict[Cell, Entry] = {root: Entry(snap=copy.deepcopy(game), trace=[], steps=0)}
    res.best_remaining = root[2]
    next_log = log_every
    last_improve_steps = 0
    last_best = res.best_remaining

    while res.env_steps < budget and not res.won:
        cell = _select(archive, rng)
        entry = archive[cell]
        entry.chosen += 1
        if entry.steps >= max_trace:
            continue
        g = copy.deepcopy(entry.snap)
        trace = list(entry.trace)
        done = False

        def burst_len() -> int:
            # ladder climbs need long HELD jump inputs (~8 frames/tile), so mix
            # short steering bursts with long holds.
            return rng.randint(10, 45) if rng.random() < 0.35 else rng.randint(2, 6)

        burst_action = sample()
        burst_left = burst_len()
        # Guided rollout: plan a physics-exact route to the current objective on
        # the level_reach macro graph (enemy-free), execute it frame-by-frame in
        # the real game; enemies perturb it and the archive absorbs the result.
        plan: List[Tuple[int, bool]] = []
        # Expensive-first ordering: while at full health, target the remaining
        # gem closest to a hazard (the ones that may cost HP), leaving safe
        # gems for the low-HP endgame — the depth race otherwise crowns
        # HP-spending lineages that arrive broke at the guarded cluster.
        override_target: Tuple[int, int] | None = None
        ammo_left = int(getattr(g, "ammo", 0))
        restock = getattr(g, "ammo_pickups", None)
        if ammo_left <= 1 and restock:
            # rockets are the crawler answer; restock before anything else
            px, py = g._player_tile()
            override_target = min(restock, key=lambda a: abs(a[0] - px) + abs(a[1] - py))
        elif g.crystals or (g.switches - g.used_switches):
            # follow the optimized tour instead of nearest-Euclidean targeting
            tour = _objective_tour(g)
            for node in tour:
                if node in g.crystals or node in (g.switches - g.used_switches):
                    override_target = node
                    break
        # Plan only from cache-warm cells or the exploit frontier (one cold
        # sweep, then warm) — cold-sweeping from every random diversity cell
        # collapses throughput ~100x.
        cache_key = (
            str(getattr(g.level, "name", "")),
            g._player_tile(),
            _door_chars_closed(g),
            int(g.player_x) // 2,
            int(g.player_y) // 2,
        )
        exploit_cell = min(archive, key=lambda cc: (cc[2], -cc[6], archive[cc].steps))
        if rng.random() < scripted_p and (cache_key in _ROUTE_CACHE or cell == exploit_cell):
            if override_target is not None:
                plan = oracle_route_actions(g, override_target)
            else:
                target, _d = g._current_target()
                if target is not None:
                    plan = oracle_route_actions(g, (target[1], target[2]))
        plan_i = 0
        guided = bool(plan)
        _layout = tuple(g.level.layout)
        _sim = _lr.LevelSim(_layout, closed_doors=_door_chars_closed(g))
        _hz = _hazard_cells(_layout)
        scripted_burst = False
        stale = 0
        prev_health = int(getattr(g, "health", 0))
        damage_replans = 0
        for _ in range(rollout):
            if plan_i < len(plan):
                move_dir, jump = plan[plan_i]
                # Wait-for-the-crawler: if an enemy is in striking range in our
                # movement direction, idle this frame (do not consume the plan)
                # and let the deterministic patrol pass. Cap the wait so a
                # hovering flyer cannot freeze the rollout forever.
                threat = (
                    _enemy_threat(g, move_dir)
                    if g.health > 0 and getattr(g, "grounded", False)
                    else None
                )
                if (
                    threat is not None
                    and (threat[0] == "crawler" or abs(threat[2]) <= 16)
                    and int(getattr(g, "ammo", 0)) > 0
                ):
                    # same-row crawler ahead: shoot it — a dead crawler clears
                    # the lane permanently, and shooting doesn't move the body,
                    # so the plan stays position-true (just paused).
                    burst_action = g.RIGHT_SHOOT if threat[1] > 0 else g.LEFT_SHOOT
                elif threat is not None and threat[0] == "crawler" and 28 <= abs(threat[1]) <= 58:
                    # out of ammo: hop over it (waiting is fatal — it walks
                    # into an idle player); hop diverges, so drop the plan
                    hop_dir = 1 if threat[1] > 0 else -1
                    burst_action = _act_id(g, hop_dir, True)
                    plan = []
                    plan_i = 0
                elif threat is not None and threat[0] != "crawler" and stale % 100 < 78:
                    # flyers etc: pause and let them drift off; grounded
                    # friction stops the body so the plan resumes position-true
                    burst_action = g.IDLE
                else:
                    plan_i += 1
                    burst_action = _act_id(g, move_dir, jump)
            else:
                if guided:
                    # guided rollouts end with their plan — random tails next
                    # to hazard-guarded gems are where the HP bleed happens
                    break
                if burst_left <= 0:
                    scripted_burst = rng.random() < 0.4
                    burst_action = sample()
                    burst_left = burst_len()
                burst_left -= 1
                if scripted_burst:
                    variant = "recovery" if rng.random() < 0.3 else "direct"
                    burst_action = route_floor_scripted_action(
                        g, variant=variant, stale_steps=stale
                    )
                else:
                    burst_action = _hazard_reflex(g, _sim, _hz, burst_action, rng)
            # The movement controller never presses INTERACT; when standing on
            # the targeted lever, throw it deterministically.
            target, _dist = g._current_target()
            if (
                target is not None
                and target[0] == "switch"
                and g._player_tile() == (target[1], target[2])
            ):
                burst_action = g.INTERACT
                plan = []
                plan_i = 0
            _s, _r, done, _info = g.step(burst_action)
            stale += 1
            trace.append(int(burst_action))
            hp = int(getattr(g, "health", 0))
            if hp < prev_health and rng.random() < 0.03:
                near = [
                    (e.kind, int(e.x) // 32, int(e.y) // 32)
                    for e in (getattr(g, "enemies", ()) or ())
                    if getattr(e, "alive", True)
                    and abs(float(e.x) - g.player_x) < 64
                    and abs(float(e.y) - g.player_y) < 64
                ]
                print(
                    f"  [dmg] tile={g._player_tile()} hp={hp} near={near}",
                    flush=True,
                )
            if hp < prev_health and hp > 0 and not done:
                # Knocked back by a guard hazard/enemy: ~70 invulnerability
                # frames make the hazard temporarily safe — replan straight
                # through it to the target and grab while invulnerable.
                if damage_replans < 2:
                    damage_replans += 1
                    target, _d2 = g._current_target()
                    if target is not None:
                        plan = local_grab_route(g, (target[1], target[2]))
                        plan_i = 0
            prev_health = hp
            res.env_steps += 1
            if done:
                if getattr(g, "won", False):
                    res.won = True
                    res.trace = trace
                break
            c = _cell(g)
            res.best_remaining = min(res.best_remaining, c[2])
            res.exit_unlocked_seen = res.exit_unlocked_seen or c[4]
            known = archive.get(c)
            if known is None or len(trace) < known.steps:
                if len(trace) <= max_trace:
                    archive[c] = Entry(snap=copy.deepcopy(g), trace=list(trace), steps=len(trace))
        if res.best_remaining < last_best:
            last_best = res.best_remaining
            last_improve_steps = res.env_steps
            # continuous endgame compaction: every deepening whose trace is
            # already fat gets trimmed immediately — the sweep's finish-line
            # failures all sit within ~125 steps of the 3000 clock
            if res.best_remaining <= 8:
                fcell = min(archive, key=lambda c: (c[2], archive[c].steps))
                fent = archive[fcell]
                if fent.steps > 2400:
                    short = compact_trace(level_index, list(fent.trace))
                    if len(short) < fent.steps:
                        g4, _c4, d4 = _replay_cells(level_index, short)
                        if not d4:
                            archive[_cell(g4)] = Entry(
                                snap=copy.deepcopy(g4),
                                trace=short,
                                steps=len(short),
                            )
        elif res.env_steps - last_improve_steps > (40_000 if res.best_remaining <= 6 else 120_000):
            # stalled: compact the best frontier prefix and re-root exploration
            frontier_cell = min(archive, key=lambda c: (c[2], archive[c].steps))
            fe = archive[frontier_cell]
            print(
                f"  [{spec.name}] stalled; frontier remaining gems: "
                f"{sorted(fe.snap.crystals)} hp={fe.snap.health} "
                f"switches_used={sorted(fe.snap.used_switches)} "
                f"open={sorted(fe.snap.open_colors)} trace={fe.steps}",
                flush=True,
            )
            short = compact_trace(level_index, list(fe.trace))
            g3, cells3, done3 = _replay_cells(level_index, short)
            if not done3 and len(short) < fe.steps:
                print(
                    f"  [{spec.name}] compacted frontier {fe.steps} -> {len(short)} steps; re-rooting",
                    flush=True,
                )
                archive[_cell(g3)] = Entry(snap=copy.deepcopy(g3), trace=short, steps=len(short))
            last_improve_steps = res.env_steps
        if res.env_steps >= next_log:
            next_log += log_every
            frontier_steps = min(
                (e.steps for c, e in archive.items() if c[2] == res.best_remaining),
                default=0,
            )
            print(
                f"  [{spec.name}] steps={res.env_steps} archive={len(archive)} "
                f"best_remaining={res.best_remaining} frontier_trace={frontier_steps} "
                f"exit_unlocked={res.exit_unlocked_seen}",
                flush=True,
            )

    res.archive_size = len(archive)
    res.seconds = time.time() - started
    if res.won:
        res.verify_ok = verify_stored(level_index, res.trace)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="", help="comma-separated level indices; empty = all")
    parser.add_argument("--budget", type=int, default=200_000, help="env steps per level")
    parser.add_argument("--rollout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scripted-p", type=float, default=0.7)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Episode step-cap override (0 = game default 3000). Demos harvested "
        "above the default cap are only valid for training runs using the same cap.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_REPO_ROOT, "experiments", "cc_status", "data", "demos_goexplore"),
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    global _MAX_STEPS_OVERRIDE
    _MAX_STEPS_OVERRIDE = int(args.max_steps or 0)
    indices = (
        [int(x) for x in args.levels.split(",") if x.strip() != ""]
        if args.levels
        else list(range(len(HANDCRAFTED_LEVELS)))
    )
    manifest: List[Dict[str, Any]] = []
    n_won = 0
    for index in indices:
        spec = HANDCRAFTED_LEVELS[index]
        res = explore_level(
            index,
            budget=args.budget,
            rollout=args.rollout,
            seed=args.seed,
            scripted_p=args.scripted_p,
        )
        status = "WON " if (res.won and res.verify_ok) else "FAIL"
        print(
            f"{status} {spec.name}: steps={res.env_steps} archive={res.archive_size} "
            f"best_remaining={res.best_remaining} exit_unlocked={res.exit_unlocked_seen} "
            f"trace={len(res.trace)} verify={res.verify_ok} {res.seconds:.0f}s",
            flush=True,
        )
        if res.won and res.verify_ok:
            n_won += 1
            payload = {
                "level": index,
                "name": spec.name,
                "actions": res.trace,
                "replay": {
                    "won": True,
                    "steps": len(res.trace),
                    "source": "go_explore",
                    "env_steps": res.env_steps,
                    "seed": args.seed,
                },
                "config": {"CRYSTAL_CAVES_IMPORTED": True},
            }
            path = out_dir / f"level{index:02d}_{spec.name.replace(' ', '_')}.json"
            path.write_text(json.dumps(payload))
        manifest.append(
            {
                "level": index,
                "name": spec.name,
                "status": "won" if (res.won and res.verify_ok) else "explored",
                "env_steps": res.env_steps,
                "archive": res.archive_size,
                "best_remaining": res.best_remaining,
                "exit_unlocked_seen": res.exit_unlocked_seen,
                "trace_len": len(res.trace),
            }
        )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{n_won}/{len(indices)} Go-Explore demos verified WINNING in the live engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
