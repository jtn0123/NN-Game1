"""Extract a WINNING demonstration (action sequence) for each hand-crafted level.

Architecture: PLAN -> EXECUTE (closed-loop) -> REPLAN FROM REALITY.

Planning runs on the same physics simulation as the winnability oracle
(`level_reach`), leg by leg (levers nearest-first, then crystals nearest-first,
then the exit), with Dijkstra over short movement macros (cost = frames, hazard
contact heavily priced). Execution replays the plan's per-frame actions in the
LIVE engine with three reactive layers the plan is blind to: shoot/wait for
patrolling enemies (never toward an air generator), re-align at every planned
rest checkpoint so drift cannot compound, and — when knockback still derails the
run — RE-PLAN the remaining objectives from the live game state and continue.

Only sequences whose live replay actually WINS are stored, as JSON action lists
(the engine is deterministic given a pinned level and actions, so demo consumers
regenerate exact transitions by replay).

Run:  python -m experiments.cc_status.demo_extract [--out DIR]
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import Config  # noqa: E402
from experiments.cc_status.level_reach import (  # noqa: E402
    MAX_FRAMES,
    PH,
    PW,
    TILE,
    LevelSim,
    _Body,
    _player_tile,
    _step,
)
from src.game.crystal_caves import CrystalCaves  # noqa: E402
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402

# engine action ids (pinned by the action-space tests)
IDLE, LEFT, RIGHT, JUMP, LEFT_JUMP, RIGHT_JUMP = 0, 1, 2, 3, 4, 5
SHOOT, LEFT_SHOOT, RIGHT_SHOOT, INTERACT = 6, 7, 8, 9

_ACTION_TO_INPUT = {
    IDLE: (0, False),
    LEFT: (-1, False),
    RIGHT: (1, False),
    JUMP: (0, True),
    LEFT_JUMP: (-1, True),
    RIGHT_JUMP: (1, True),
    INTERACT: (0, False),
}

# Short movement macros so Dijkstra composes tight paths (the oracle's larger
# exploration strides made tours that blew the 3000-step episode horizon).
_D = (-1, 1)
_PLAN_MACROS: List[List[Tuple[int, int, bool]]] = (
    [[(n, d, False)] for d in _D for n in (6, 14, 30, 60)]
    + [[(1, d, True), (n, d, False)] for d in _D for n in (16, 34)]
    + [[(1, 0, True), (12, 0, False), (n, d, False)] for d in _D for n in (10, 24)]
    + [[(1, d, True), (11, d, False), (24, -d, False)] for d in _D]
    + [[(n, 0, True)] for n in (8, 20, 44)]
    + [[(n, 0, False)] for n in (8, 20)]
    + [[(n, d, True)] for d in _D for n in (20, 44)]
)

_MAX_LEG_EXPANSIONS = 150_000
_HAZARD_EDGE_COST = 600  # frames-equivalent price for clipping spikes/acid
_ENEMY_HIT_COST = 900  # frames-equivalent price for tanking one enemy hit (-1 HP)
_INVULN_FRAMES = 70  # engine post-hit invulnerability (crystal_caves INVULN_FRAMES)
_STEP_BUDGET = 2900  # must win inside the engine's 3000-step horizon
_MAX_REPLANS = 24

_SWITCH_CHARS = {"s": "D", "S": "d"}  # lever char -> door char it opens


_HORIZON = 3000  # engine episode cap; enemy trajectories precomputed to here
_EW = _EH = 24  # engine Enemy rect size


def enemy_spawns(layout: Tuple[str, ...]) -> List[Tuple[float, float, float, str]]:
    """(x, y, vx, kind) at reset, mirroring the engine's _load_level spawns."""
    out: List[Tuple[float, float, float, str]] = []
    for row, line in enumerate(layout):
        for col, ch in enumerate(line):
            if ch == "M":
                out.append((col * TILE + 4.0, row * TILE + 8.0, 1.1, "crawler"))
            elif ch == "F":
                out.append((col * TILE + 4.0, row * TILE + 4.0, 1.6, "flyer"))
    return out


def enemy_trajectories(
    sim: LevelSim,
    enemies: List[Tuple[float, float, float, str]],
    horizon: int = _HORIZON,
) -> List[List[Tuple[float, float]]]:
    """Per-frame (x, y) for each enemy, simulated with the engine's patrol rules
    (advance; flip on solid collision, crawlers also flip at ledges). Enemies
    ignore the player entirely, so their paths are a pure function of time —
    which is what lets the planner time routes through patrol gaps."""

    def collides(x: float, y: float) -> bool:
        lo_c, hi_c = int(x) // TILE, (int(x) + _EW - 1) // TILE
        lo_r, hi_r = int(y) // TILE, (int(y) + _EH - 1) // TILE
        return any(
            sim.solid_at(col, row) for row in range(lo_r, hi_r + 1) for col in range(lo_c, hi_c + 1)
        )

    tracks: List[List[Tuple[float, float, float]]] = []
    for x, y, vx, kind in enemies:
        track: List[Tuple[float, float, float]] = []
        ex, ey, evx = x, y, vx
        for _ in range(horizon):
            track.append((ex, ey, evx))
            ex += evx
            if kind == "flyer":
                if collides(ex, ey):
                    ex -= evx
                    evx = -evx
            else:
                ahead_x = ex + (_EW + 2 if evx > 0 else -2)
                foot_row = int((ey + _EH + 2) // TILE)
                ahead_col = int(ahead_x // TILE)
                if collides(ex, ey) or not sim.solid_at(ahead_col, foot_row):
                    ex -= evx
                    evx = -evx
        tracks.append(track)
    return tracks


def _contact(x: float, y: float, t: int, tracks: List[List[Tuple[float, float]]]) -> bool:
    """Player rect at (x, y) overlaps any enemy rect at frame t (lethal in plans)."""
    if t >= _HORIZON:
        t = _HORIZON - 1
    px2 = x + PW
    py2 = y + PH
    for track in tracks:
        ex, ey = track[t][0], track[t][1]
        if x < ex + _EW and px2 > ex and y < ey + _EH and py2 > ey:
            return True
    return False


def _frame_action(move_dir: int, jump: bool) -> int:
    if jump:
        return {(-1): LEFT_JUMP, 0: JUMP, 1: RIGHT_JUMP}[move_dir]
    return {(-1): LEFT, 0: IDLE, 1: RIGHT}[move_dir]


def _run_macro(
    sim: LevelSim,
    start: _Body,
    program: List[Tuple[int, int, bool]],
    stop_on: Optional[Tuple[int, int]] = None,
    t0: int = 0,
    tracks: Optional[List[List[Tuple[float, float]]]] = None,
    contact_lethal: bool = True,
    invuln_0: int = -1,
) -> Tuple[_Body, List[int], bool, int, int, bool, int]:
    """Run one macro; return (end body, actions, hit stop_on, hazard clips,
    enemy clips, died, invuln_until).

    ``tracks`` are precomputed enemy trajectories: overlapping an enemy at the
    absolute frame (t0 + local index) is LETHAL during planning when
    ``contact_lethal`` — otherwise it is COUNTED (enemy clips) and followed by
    the engine's invulnerability window, so the planner can price hits as
    spendable HP instead of demanding a 0-damage speedrun (the red-team finding:
    perfectionism, not level difficulty, was why 0/16 routes planned).
    ``invuln_0`` carries a still-running invulnerability window in from the
    previous leg — a hit spent at the instant a crystal was touched must let
    the NEXT leg walk out of the enemy for free, exactly like the engine does.
    ``stop_on`` cuts the macro at the exact frame the player rect overlaps that
    tile, so legs end at the pickup instead of overshooting a full stride."""
    b = _Body(start.x, start.y, 0.0, 0.0, start.grounded, start.coyote)
    actions: List[int] = []
    hazard_clips = 0
    enemy_clips = 0
    invuln_until = max(invuln_0, t0 - 1)
    t = t0
    for frames, move_dir, jump in program:
        for _ in range(frames):
            _step(sim, b, move_dir, jump)
            actions.append(_frame_action(move_dir, jump))
            t += 1
            if tracks is not None and t > invuln_until and _contact(b.x, b.y, t, tracks):
                if contact_lethal:
                    return b, actions, False, hazard_clips, enemy_clips, True, invuln_until
                enemy_clips += 1
                invuln_until = t + _INVULN_FRAMES
            lo_c, hi_c = int(b.x) // TILE, (int(b.x) + PW - 1) // TILE
            lo_r, hi_r = int(b.y) // TILE, (int(b.y) + PH - 1) // TILE
            for row in range(lo_r, hi_r + 1):
                for col in range(lo_c, hi_c + 1):
                    if 0 <= row < sim.rows and 0 <= col < sim.cols:
                        if sim.grid[row][col] in "^~":
                            hazard_clips += 1
                        if stop_on == (col, row):
                            return b, actions, True, hazard_clips, enemy_clips, False, invuln_until
            if b.y > (sim.rows + 2) * TILE:
                return b, actions, False, hazard_clips, enemy_clips, False, invuln_until
    for _ in range(MAX_FRAMES):  # settle so legs chain from stable footing
        if b.grounded or sim.on_ladder(b.x, b.y):
            break
        _step(sim, b, 0, False)
        actions.append(IDLE)
        t += 1
        if tracks is not None and t > invuln_until and _contact(b.x, b.y, t, tracks):
            if contact_lethal:
                return b, actions, False, hazard_clips, enemy_clips, True, invuln_until
            enemy_clips += 1
            invuln_until = t + _INVULN_FRAMES
    return b, actions, False, hazard_clips, enemy_clips, False, invuln_until


def _plan_leg(
    sim: LevelSim,
    start: _Body,
    *,
    target: Tuple[int, int],
    reach: str,
    t0: int = 0,
    tracks: Optional[List[List[Tuple[float, float]]]] = None,
    hp_budget: int = 0,
    start_invuln: int = -1,
) -> Optional[Tuple[_Body, List[int], int, int]]:
    """Frame-cost Dijkstra over rest states until the target is satisfied.

    reach="touch": player rect overlaps the target tile (crystals, exit).
    reach="adjacent": end RESTING within Chebyshev 1 (a lever's INTERACT range).
    ``hp_budget`` is how many enemy hits this leg may tank (priced at
    _ENEMY_HIT_COST each, so contact-free routes still win when they exist);
    ``start_invuln`` is a still-running invulnerability frame carried over from a
    hit spent late in the previous leg.
    Returns (end body, actions, enemy hits spent, invuln_until at leg end).
    """
    if reach == "adjacent" and (
        tracks is None or t0 <= start_invuln or not _contact(start.x, start.y, t0, tracks)
    ):
        c0, r0 = _player_tile(start.x, start.y)
        if max(abs(c0 - target[0]), abs(r0 - target[1])) <= 1:
            return start, [], 0, start_invuln
    # A*: priority = elapsed frames + penalties + admissible walk-time heuristic;
    # state keyed by (cell, time bucket, hits) because the same cell can be lethal
    # at one frame and safe 30 frames later (patrol phase).
    walk_dist = _bfs_distances(sim, target)
    frames_per_tile = TILE / 4.2

    def h(cell: Tuple[int, int]) -> float:
        return walk_dist.get(cell, 200) * frames_per_tile * 0.9

    start_cell = _player_tile(start.x, start.y)
    best: Dict[Tuple[int, int, int, int], int] = {(start_cell[0], start_cell[1], t0 // 24, 0): 0}
    counter = 0
    heap: List[Tuple[float, int, int, int, int, int, _Body, List[int]]] = [
        (h(start_cell), 0, counter, t0, 0, start_invuln, start, [])
    ]
    expansions = 0
    while heap and expansions < _MAX_LEG_EXPANSIONS:
        _f, cost, _, t, hits, invuln, body, actions = heapq.heappop(heap)
        cell = _player_tile(body.x, body.y)
        if cost > best.get((cell[0], cell[1], t // 24, hits), 1 << 30):
            continue
        for program in _PLAN_MACROS:
            expansions += 1
            end, macro_actions, hit, clips, eclips, died, end_invuln = _run_macro(
                sim,
                body,
                program,
                stop_on=target if reach == "touch" else None,
                t0=t,
                tracks=tracks,
                contact_lethal=hits >= hp_budget,
                invuln_0=invuln,
            )
            if died:
                continue
            new_hits = hits + eclips
            if new_hits > hp_budget:
                continue
            if hit:
                return end, actions + macro_actions, new_hits, end_invuln
            if not (end.grounded or sim.on_ladder(end.x, end.y)):
                continue
            end_t = t + len(macro_actions)
            if end_t > _HORIZON - 100:
                continue
            end_cell = _player_tile(end.x, end.y)
            if (
                reach == "adjacent"
                and max(abs(end_cell[0] - target[0]), abs(end_cell[1] - target[1])) <= 1
            ):
                return end, actions + macro_actions, new_hits, end_invuln
            new_cost = (
                cost
                + len(macro_actions)
                + (_HAZARD_EDGE_COST if clips else 0)
                + _ENEMY_HIT_COST * eclips
            )
            key = (end_cell[0], end_cell[1], end_t // 24, new_hits)
            if new_cost < best.get(key, 1 << 30):
                best[key] = new_cost
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        new_cost + h(end_cell),
                        new_cost,
                        counter,
                        end_t,
                        new_hits,
                        end_invuln,
                        end,
                        actions + macro_actions,
                    ),
                )
    return None


def _bfs_distances(sim: LevelSim, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """Maze-aware nearness for tour ordering (manhattan misorders mazes)."""
    dist = {start: 0}
    queue: deque = deque([start])
    while queue:
        col, row = queue.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (col + dc, row + dr)
            if nxt not in dist and not sim.solid_at(*nxt):
                dist[nxt] = dist[(col, row)] + 1
                queue.append(nxt)
    return dist


def _nearest(sim: LevelSim, here: Tuple[int, int], tiles: Set[Tuple[int, int]]) -> Tuple[int, int]:
    dist = _bfs_distances(sim, here)
    return min(tiles, key=lambda t: dist.get(t, 1 << 30))


def _resim_touches(sim: LevelSim, start: _Body, actions: List[int]) -> Set[Tuple[int, int]]:
    """Replay a leg in the planning sim, returning every tile the rect touches."""
    b = _Body(start.x, start.y, 0.0, 0.0, start.grounded, start.coyote)
    touched: Set[Tuple[int, int]] = set()
    for action in actions:
        move_dir, jump = _ACTION_TO_INPUT[action]
        _step(sim, b, move_dir, jump)
        lo_c, hi_c = int(b.x) // TILE, (int(b.x) + PW - 1) // TILE
        lo_r, hi_r = int(b.y) // TILE, (int(b.y) + PH - 1) // TILE
        for row in range(lo_r, hi_r + 1):
            for col in range(lo_c, hi_c + 1):
                touched.add((col, row))
    return touched


def plan_from(
    layout: Tuple[str, ...],
    *,
    start_xy: Tuple[float, float],
    switches_left: Set[Tuple[int, int]],
    crystals_left: Set[Tuple[int, int]],
    exit_pos: Tuple[int, int],
    open_door_chars: Set[str],
    enemies: Optional[List[Tuple[float, float, float, str]]] = None,
    hp_budget: int = 2,
) -> Optional[List[int]]:
    """Plan a winning action tail from an arbitrary (live) game state.

    ``enemies`` anchors the patrol simulation to the LIVE enemy states, so a
    replan mid-episode is timed against where the enemies actually are.
    ``hp_budget`` is the total enemy hits the whole route may tank (HP is a
    spendable resource: 3 hearts = 2 affordable hits from full health)."""
    doors_present = {dc for dc in ("D", "d") if any(dc in row for row in layout)}
    opened = set(open_door_chars)
    if enemies is None:
        enemies = enemy_spawns(layout)
    kinds = [kind for _x, _y, _vx, kind in enemies]

    def make_sim() -> LevelSim:
        return LevelSim(layout, closed_doors=frozenset(doors_present - opened))

    sim = make_sim()
    tracks = enemy_trajectories(sim, enemies)

    def reanchor(t: int) -> None:
        # Door state changed: enemy collision world changed. Re-simulate the
        # remaining horizon from each enemy's position/heading at plan-time t.
        nonlocal tracks
        anchored = [
            (
                track[min(t, _HORIZON - 1)][0],
                track[min(t, _HORIZON - 1)][1],
                track[min(t, _HORIZON - 1)][2],
                kind,
            )
            for track, kind in zip(tracks, kinds)
        ]
        fresh = enemy_trajectories(sim, anchored)
        tracks = [
            track[:t] + fresh_track[: _HORIZON - t] for track, fresh_track in zip(tracks, fresh)
        ]

    body = _Body(start_xy[0], start_xy[1])
    actions: List[int] = []
    for _ in range(MAX_FRAMES):
        if body.grounded or sim.on_ladder(body.x, body.y):
            break
        _step(sim, body, 0, False)
        actions.append(IDLE)

    hp_left = max(0, int(hp_budget))
    invuln = -1  # a hit spent late in one leg still protects the start of the next
    switches = set(switches_left)
    while switches:
        tile = _nearest(sim, _player_tile(body.x, body.y), switches)
        result = _plan_leg(
            sim,
            body,
            target=tile,
            reach="adjacent",
            t0=len(actions),
            tracks=tracks,
            hp_budget=hp_left,
            start_invuln=invuln,
        )
        if result is None:
            return None
        body, leg, spent, invuln = result
        hp_left -= spent
        actions.extend(leg)
        actions.append(INTERACT)
        switches.discard(tile)
        door_char = _SWITCH_CHARS.get(layout[tile[1]][tile[0]])
        if door_char:
            opened.add(door_char)
        sim = make_sim()
        reanchor(len(actions))

    remaining = set(crystals_left)
    while remaining:
        tile = _nearest(sim, _player_tile(body.x, body.y), remaining)
        result = _plan_leg(
            sim,
            body,
            target=tile,
            reach="touch",
            t0=len(actions),
            tracks=tracks,
            hp_budget=hp_left,
            start_invuln=invuln,
        )
        if result is None:
            return None
        prev = body
        body, leg, spent, invuln = result
        hp_left -= spent
        actions.extend(leg)
        remaining.discard(tile)
        remaining -= _resim_touches(sim, prev, leg)

    result = _plan_leg(
        sim,
        body,
        target=exit_pos,
        reach="touch",
        t0=len(actions),
        tracks=tracks,
        hp_budget=hp_left,
        start_invuln=invuln,
    )
    if result is None:
        return None
    _body, leg, _spent, _invuln = result
    actions.extend(leg)
    return actions


def _expected_and_checkpoints(
    layout: Tuple[str, ...],
    start_xy: Tuple[float, float],
    actions: List[int],
    open_door_chars: Set[str],
) -> Tuple[List[Tuple[float, float]], Set[int]]:
    """Planner trajectory (x,y before each index) + rest-checkpoint indices."""
    doors_present = {dc for dc in ("D", "d") if any(dc in row for row in layout)}
    sim = LevelSim(layout, closed_doors=frozenset(doors_present - set(open_door_chars)))
    b = _Body(start_xy[0], start_xy[1])
    expected: List[Tuple[float, float]] = []
    checkpoints: Set[int] = set()
    for index, action in enumerate(actions):
        expected.append((b.x, b.y))
        move_dir, jump = _ACTION_TO_INPUT[action]
        _step(sim, b, move_dir, jump)
        if b.grounded and abs(b.vx) < 0.3 and abs(b.vy) < 1.2:
            checkpoints.add(index + 1)
    return expected, checkpoints


class _Executor:
    """Closed-loop executor: combat layer + checkpoint re-sync + step accounting."""

    def __init__(self, game: CrystalCaves):
        self.game = game
        self.executed: List[int] = []
        self.steps = 0

    def _do(self, action: int) -> bool:
        self.executed.append(int(action))
        _s, _r, done, _info = self.game.step(int(action))
        self.steps += 1
        return done

    def _threat(self):
        g = self.game
        px = g.player_x + g.PLAYER_WIDTH / 2
        py = g.player_y + g.PLAYER_HEIGHT / 2
        for enemy in g.enemies:
            if not enemy.alive:
                continue
            ex = enemy.x + enemy.width / 2
            ey = enemy.y + enemy.height / 2
            if abs(ey - py) > 40:
                continue
            dx = ex - px
            if abs(dx) < 52 or (abs(dx) < 110 and dx * enemy.vx < 0):
                return enemy, dx
        return None, 0.0

    def _airtank_in_line(self, dx: float) -> bool:
        g = self.game
        pcol = int((g.player_x + g.PLAYER_WIDTH / 2) // g.TILE_SIZE)
        prow = int((g.player_y + g.PLAYER_HEIGHT / 2) // g.TILE_SIZE)
        step = 1 if dx > 0 else -1
        for col in range(pcol, pcol + step * 11, step):
            for row in (prow - 1, prow, prow + 1):
                if (col, row) in g.air_tanks:
                    return True
        return False

    def hunt(self, step_budget: int) -> str:
        """No contact-free route exists: the corridor is camped. Walk toward the
        nearest live enemy and shoot it (never toward an air generator), then let
        the caller replan. Returns 'killed_one' | 'done' | 'failed'."""
        g = self.game
        for _ in range(400):
            if g.game_over:
                return "done"
            if self.steps >= step_budget:
                return "failed"
            alive = [e for e in g.enemies if e.alive]
            if not alive or g.ammo <= 0:
                return "failed"
            px = g.player_x + g.PLAYER_WIDTH / 2
            py = g.player_y + g.PLAYER_HEIGHT / 2
            target = min(alive, key=lambda e: abs(e.x - px) + 2 * abs(e.y - py))
            n_alive = len(alive)
            dx = (target.x + target.width / 2) - px
            dy = (target.y + target.height / 2) - py
            if abs(dy) < 40 and g.shoot_cooldown == 0 and not self._airtank_in_line(dx):
                if self._do(RIGHT_SHOOT if dx > 0 else LEFT_SHOOT):
                    return "done"
            elif g.grounded:
                if self._do(RIGHT if dx > 0 else LEFT):
                    return "done"
            else:
                if self._do(IDLE):
                    return "done"
            if len([e for e in g.enemies if e.alive]) < n_alive:
                return "killed_one"
        return "failed"

    def run(
        self,
        actions: List[int],
        expected: List[Tuple[float, float]],
        checkpoints: Set[int],
        *,
        step_budget: int,
    ) -> str:
        """Execute a timed plan; returns 'done' | 'derailed' | 'budget'.

        The plan may PRICE IN enemy hits (HP is a spendable resource), but a
        LIVE hit — planned or accidental — always derails: knockback throws
        the player off the planned trajectory and every open-loop frame after
        it is mistimed against the patrols (running on cost the first two hits
        of every attempt to compounding desync). The caller replans from the
        post-knockback reality with the remaining HP budget, so a planned hit
        still buys passage through a camped corridor — it just hands control
        straight back to the planner afterwards."""
        g = self.game
        i = 0
        injected = 0
        start_health = g.health
        while i < len(actions):
            if g.game_over:
                return "done"
            if self.steps >= step_budget:
                return "budget"
            if g.health < start_health:
                return "derailed"  # hit taken: replan from post-knockback reality
            ex, ey = expected[i]
            dx_p = ex - g.player_x
            dy_p = ey - g.player_y
            at_cp = i in checkpoints or i == 0
            tol_x, tol_y = (6, 10) if at_cp else (26, 44)
            if (abs(dx_p) > tol_x or abs(dy_p) > tol_y) and g.grounded:
                if injected > 18:
                    return "derailed"  # drift already desynced the patrol phase
                injected += 1
                move = RIGHT if dx_p > 3 else LEFT if dx_p < -3 else IDLE
                if dy_p < -12 and abs(dx_p) < 48:
                    move = {RIGHT: RIGHT_JUMP, LEFT: LEFT_JUMP, IDLE: JUMP}[move]
                if self._do(move):
                    return "done"
                continue
            if at_cp and (abs(g.vx) > 0.4 or not g.grounded):
                if injected > 18:
                    return "derailed"
                injected += 1
                if self._do(IDLE):
                    return "done"
                continue
            if self._do(actions[i]):
                return "done"
            i += 1
        return "done" if g.game_over else "derailed"


def extract_level(level_index: int) -> Dict[str, Any]:
    """Plan/execute/replan until the live engine records a WIN (or give up)."""
    spec = HANDCRAFTED_LEVELS[level_index]
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    game = CrystalCaves(cfg, headless=True)
    game.CAVES = (spec,)
    game._eval_caves = (spec,)
    game._randomize_levels = False
    game.use_eval_levels(1)
    game.reset_eval_cursor()
    game.reset()

    executor = _Executor(game)
    replans = 0
    while replans <= _MAX_REPLANS and not game.game_over and executor.steps < _STEP_BUDGET:
        open_chars = {
            {"red": "D", "blue": "d"}[color]
            for color in game.open_colors
            if color in ("red", "blue")
        }
        plan = plan_from(
            spec.layout,
            start_xy=(game.player_x, game.player_y),
            switches_left=set(game.switches - game.used_switches),
            crystals_left=set(game.crystals),
            exit_pos=game.exit_pos,
            open_door_chars=open_chars,
            enemies=[(e.x, e.y, e.vx, e.kind) for e in game.enemies if e.alive],
            # The executor derails at the FIRST live hit, so planning more than one
            # hit ahead is wasted search space: budget exactly one plannable hit
            # while above the last heart (health-2 starved replans at 2 HP into
            # plan_failed; health-1 let plans queue multiple hits they never used).
            hp_budget=1 if game.health >= 2 else 0,
        )
        if plan is None:
            # No contact-free route: a patrol camps the corridor. Clear one enemy
            # with the gun, then replan against the new (deterministic) world.
            outcome = executor.hunt(_STEP_BUDGET)
            replans += 1
            if outcome == "killed_one" and not game.game_over:
                continue
            if not game.game_over:
                return {
                    "won": False,
                    "end_reason": "plan_failed",
                    "steps": executor.steps,
                    "replans": replans,
                    "crystals_remaining": len(game.crystals),
                    "health": game.health,
                    "actions": executor.executed,
                }
            break
        expected, checkpoints = _expected_and_checkpoints(
            spec.layout, (game.player_x, game.player_y), plan, open_chars
        )
        outcome = executor.run(plan, expected, checkpoints, step_budget=_STEP_BUDGET)
        replans += 1
        if outcome in ("done", "budget"):
            break
    return {
        "won": bool(game.won),
        "end_reason": "actions_exhausted" if not game.game_over else game._end_reason,
        "steps": executor.steps,
        "replans": replans,
        "crystals_remaining": len(game.crystals),
        "health": game.health,
        "actions": executor.executed,
    }


def verify_stored(level_index: int, actions: List[int]) -> bool:
    """Pure open-loop replay of a stored action list must reproduce the win."""
    spec = HANDCRAFTED_LEVELS[level_index]
    cfg = Config()
    cfg.CRYSTAL_CAVES_IMPORTED = True
    game = CrystalCaves(cfg, headless=True)
    game.CAVES = (spec,)
    game._eval_caves = (spec,)
    game._randomize_levels = False
    game.use_eval_levels(1)
    game.reset_eval_cursor()
    game.reset()
    for action in actions:
        _s, _r, done, _info = game.step(int(action))
        if done:
            return bool(game.won)
    return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract winning demos for each level.")
    parser.add_argument(
        "--out", default=str(_REPO_ROOT / "experiments" / "cc_status" / "data" / "demos")
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    n_won = 0
    for index, spec in enumerate(HANDCRAFTED_LEVELS):
        result = extract_level(index)
        stored = False
        if result["won"] and verify_stored(index, result["actions"]):
            stored = True
            n_won += 1
            payload = {
                "level": index,
                "name": spec.name,
                "actions": result["actions"],
                "replay": {k: v for k, v in result.items() if k != "actions"},
                "config": {"CRYSTAL_CAVES_IMPORTED": True},
            }
            path = out_dir / f"level{index:02d}_{spec.name.replace(' ', '_')}.json"
            path.write_text(json.dumps(payload))
        print(
            f"{'WON ' if result['won'] else 'FAIL'} {spec.name}: {result['steps']} steps, "
            f"replans={result.get('replans')}, end={result['end_reason']}, "
            f"crystals_left={result.get('crystals_remaining')}, hp={result.get('health')}"
            + ("" if stored or not result["won"] else "  [verify-replay FAILED]"),
            flush=True,
        )
        manifest.append(
            {
                "level": index,
                "name": spec.name,
                "status": "won" if stored else result["end_reason"],
                "steps": result["steps"],
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{n_won}/{len(HANDCRAFTED_LEVELS)} demos verified WINNING in the live engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
