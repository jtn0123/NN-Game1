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
    LevelSim,
    MAX_FRAMES,
    PH,
    PW,
    TILE,
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
_STEP_BUDGET = 2900  # must win inside the engine's 3000-step horizon
_MAX_REPLANS = 8

_SWITCH_CHARS = {"s": "D", "S": "d"}  # lever char -> door char it opens


def _frame_action(move_dir: int, jump: bool) -> int:
    if jump:
        return {(-1): LEFT_JUMP, 0: JUMP, 1: RIGHT_JUMP}[move_dir]
    return {(-1): LEFT, 0: IDLE, 1: RIGHT}[move_dir]


def _run_macro(
    sim: LevelSim,
    start: _Body,
    program: List[Tuple[int, int, bool]],
    stop_on: Optional[Tuple[int, int]] = None,
) -> Tuple[_Body, List[int], bool, int]:
    """Run one macro; return (end body, actions, hit stop_on, hazard clips).

    ``stop_on`` cuts the macro at the exact frame the player rect overlaps that
    tile, so legs end at the pickup instead of overshooting a full stride."""
    b = _Body(start.x, start.y, 0.0, 0.0, start.grounded, start.coyote)
    actions: List[int] = []
    hazard_clips = 0
    for frames, move_dir, jump in program:
        for _ in range(frames):
            _step(sim, b, move_dir, jump)
            actions.append(_frame_action(move_dir, jump))
            lo_c, hi_c = int(b.x) // TILE, (int(b.x) + PW - 1) // TILE
            lo_r, hi_r = int(b.y) // TILE, (int(b.y) + PH - 1) // TILE
            for row in range(lo_r, hi_r + 1):
                for col in range(lo_c, hi_c + 1):
                    if 0 <= row < sim.rows and 0 <= col < sim.cols:
                        if sim.grid[row][col] in "^~":
                            hazard_clips += 1
                        if stop_on == (col, row):
                            return b, actions, True, hazard_clips
            if b.y > (sim.rows + 2) * TILE:
                return b, actions, False, hazard_clips
    for _ in range(MAX_FRAMES):  # settle so legs chain from stable footing
        if b.grounded or sim.on_ladder(b.x, b.y):
            break
        _step(sim, b, 0, False)
        actions.append(IDLE)
    return b, actions, False, hazard_clips


def _plan_leg(
    sim: LevelSim, start: _Body, *, target: Tuple[int, int], reach: str
) -> Optional[Tuple[_Body, List[int]]]:
    """Frame-cost Dijkstra over rest states until the target is satisfied.

    reach="touch": player rect overlaps the target tile (crystals, exit).
    reach="adjacent": end RESTING within Chebyshev 1 (a lever's INTERACT range).
    """
    if reach == "adjacent":
        c0, r0 = _player_tile(start.x, start.y)
        if max(abs(c0 - target[0]), abs(r0 - target[1])) <= 1:
            return start, []
    best: Dict[Tuple[int, int], int] = {_player_tile(start.x, start.y): 0}
    counter = 0
    heap: List[Tuple[int, int, _Body, List[int]]] = [(0, counter, start, [])]
    expansions = 0
    while heap and expansions < _MAX_LEG_EXPANSIONS:
        cost, _, body, actions = heapq.heappop(heap)
        cell = _player_tile(body.x, body.y)
        if cost > best.get(cell, 1 << 30):
            continue
        for program in _PLAN_MACROS:
            expansions += 1
            end, macro_actions, hit, clips = _run_macro(
                sim, body, program, stop_on=target if reach == "touch" else None
            )
            if hit:
                return end, actions + macro_actions
            if not (end.grounded or sim.on_ladder(end.x, end.y)):
                continue
            end_cell = _player_tile(end.x, end.y)
            if (
                reach == "adjacent"
                and max(abs(end_cell[0] - target[0]), abs(end_cell[1] - target[1])) <= 1
            ):
                return end, actions + macro_actions
            new_cost = cost + len(macro_actions) + (_HAZARD_EDGE_COST if clips else 0)
            if new_cost < best.get(end_cell, 1 << 30):
                best[end_cell] = new_cost
                counter += 1
                heapq.heappush(heap, (new_cost, counter, end, actions + macro_actions))
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


def _nearest(sim: LevelSim, here: Tuple[int, int], tiles: Set[Tuple[int, int]]):
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
) -> Optional[List[int]]:
    """Plan a winning action tail from an arbitrary (live) game state."""
    doors_present = {dc for dc in ("D", "d") if any(dc in row for row in layout)}
    opened = set(open_door_chars)

    def make_sim() -> LevelSim:
        return LevelSim(layout, closed_doors=frozenset(doors_present - opened))

    sim = make_sim()
    body = _Body(start_xy[0], start_xy[1])
    actions: List[int] = []
    for _ in range(MAX_FRAMES):
        if body.grounded or sim.on_ladder(body.x, body.y):
            break
        _step(sim, body, 0, False)
        actions.append(IDLE)

    switches = set(switches_left)
    while switches:
        tile = _nearest(sim, _player_tile(body.x, body.y), switches)
        result = _plan_leg(sim, body, target=tile, reach="adjacent")
        if result is None:
            return None
        body, leg = result
        actions.extend(leg)
        actions.append(INTERACT)
        switches.discard(tile)
        door_char = _SWITCH_CHARS.get(layout[tile[1]][tile[0]])
        if door_char:
            opened.add(door_char)
        sim = make_sim()

    remaining = set(crystals_left)
    while remaining:
        tile = _nearest(sim, _player_tile(body.x, body.y), remaining)
        result = _plan_leg(sim, body, target=tile, reach="touch")
        if result is None:
            return None
        prev = body
        body, leg = result
        actions.extend(leg)
        remaining.discard(tile)
        remaining -= _resim_touches(sim, prev, leg)

    result = _plan_leg(sim, body, target=exit_pos, reach="touch")
    if result is None:
        return None
    _body, leg = result
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

    def run(
        self,
        actions: List[int],
        expected: List[Tuple[float, float]],
        checkpoints: Set[int],
        *,
        step_budget: int,
    ) -> str:
        """Execute a plan; returns 'done' | 'derailed' | 'budget'."""
        g = self.game
        i = 0
        repair_left = 700
        while i < len(actions):
            if g.game_over:
                return "done"
            if self.steps >= step_budget:
                return "budget"
            # combat layer: shoot the corridor threat, else wait for the patrol
            enemy, dx = self._threat()
            if enemy is not None and g.grounded and not g._is_on_ladder():
                if g.ammo > 0 and g.shoot_cooldown == 0 and not self._airtank_in_line(dx):
                    if self._do(RIGHT_SHOOT if dx > 0 else LEFT_SHOOT):
                        return "done"
                    continue
                if self._do(IDLE):
                    return "done"
                continue
            # re-sync layer: strict at rest checkpoints, loose mid-segment
            ex, ey = expected[i]
            dx_p = ex - g.player_x
            dy_p = ey - g.player_y
            at_cp = i in checkpoints or i == 0
            tol_x, tol_y = (6, 10) if at_cp else (26, 44)
            if (abs(dx_p) > tol_x or abs(dy_p) > tol_y) and g.grounded:
                if repair_left <= 0:
                    return "derailed"
                repair_left -= 1
                move = RIGHT if dx_p > 3 else LEFT if dx_p < -3 else IDLE
                if dy_p < -12 and abs(dx_p) < 48:
                    move = {RIGHT: RIGHT_JUMP, LEFT: LEFT_JUMP, IDLE: JUMP}[move]
                if self._do(move):
                    return "done"
                continue
            if at_cp and (abs(g.vx) > 0.4 or not g.grounded) and repair_left > 0:
                repair_left -= 1
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
        )
        if plan is None:
            return {"won": False, "end_reason": "plan_failed", "steps": executor.steps}
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
