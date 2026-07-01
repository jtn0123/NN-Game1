"""Physics-faithful winnability check for Crystal Caves levels.

This replicates the *exact* movement + collision the live CrystalCaves engine
uses (`_apply_player_input` + `_move_player` + `_move_axis` + `_solid_at`) and
runs a BFS over resting states (grounded / on-ladder) to answer one honest
question per level:

    Starting from P, using only walk / jump / climb (the mechanics this engine
    actually has), can the player physically touch every crystal, every switch,
    and the exit tile?

Doors are treated as PASSABLE (the win sequence throws every switch, which opens
every colour-keyed door), so this is a connectivity check under "all switches
thrown". Enemies and hazards are ignored — this measures geometry/reachability,
not combat survival.

Run:  python -m experiments.cc_status.level_reach
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Engine physics constants (mirrored from src/game/crystal_caves.py so this check
# is standalone and cannot be silently broken by an unrelated engine edit).
TILE = 32
PW, PH = 22, 30
MOVE_SPEED = 4.2
AIR_SPEED = 3.3
JUMP_SPEED = 10.5
GRAVITY = 0.52
MAX_FALL = 10.0
FRICTION = 0.82
CLIMB = 3.1
DESCEND = 2.0
COYOTE = 6

SOLID = "#"
LADDER = "H"
ELEVATOR = "="
DOORS = {"D", "d"}
CRYSTAL = "*"
SWITCHES = {"s", "S"}
EXIT = "E"
PLAYER = "P"

MAX_FRAMES = 70  # per macro trajectory
MAX_RESTING = 6000  # BFS safety cap


class LevelSim:
    def __init__(self, layout: Tuple[str, ...], closed_doors: frozenset = frozenset()):
        self.rows = len(layout)
        self.cols = max(len(r) for r in layout)
        self.grid = [row.ljust(self.cols, SOLID) for row in layout]
        self.closed_doors = frozenset(closed_doors)  # door chars treated as solid
        self.start = self._find(PLAYER)
        self.crystals = self._find_all(lambda ch: ch == CRYSTAL)
        self.switches = self._find_all(lambda ch: ch in SWITCHES)
        self.exit = self._find(EXIT)

    def _find(self, target: str) -> Tuple[int, int]:
        for r, row in enumerate(self.grid):
            c = row.find(target)
            if c != -1:
                return (c, r)
        raise ValueError(f"tile {target!r} not found")

    def _find_all(self, pred) -> Set[Tuple[int, int]]:
        return {(c, r) for r, row in enumerate(self.grid) for c, ch in enumerate(row) if pred(ch)}

    # --- collision (mirrors _solid_at / _rect_collides_solid). Doors listed in
    # ``closed_doors`` act as walls, exactly like the engine's locked doors. ---
    def solid_at(self, col: int, row: int) -> bool:
        if col < 0 or row < 0 or col >= self.cols or row >= self.rows:
            return True
        ch = self.grid[row][col]
        return ch == SOLID or ch in self.closed_doors

    def _tiles_for_rect(self, x: float, y: float):
        left = int(x) // TILE
        right = (int(x) + PW - 1) // TILE
        top = int(y) // TILE
        bottom = (int(y) + PH - 1) // TILE
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                yield col, row

    def rect_collides(self, x: float, y: float) -> bool:
        return any(self.solid_at(c, r) for c, r in self._tiles_for_rect(x, y))

    def on_ladder(self, x: float, y: float) -> bool:
        for c, r in self._tiles_for_rect(x, y):
            if 0 <= c < self.cols and 0 <= r < self.rows:
                if self.grid[r][c] in (LADDER, ELEVATOR):
                    return True
        return False

    def is_on_surface(self, x: float, y: float) -> bool:
        return self.rect_collides(x, y + 1)


class _Body:
    __slots__ = ("x", "y", "vx", "vy", "grounded", "coyote")

    def __init__(self, x, y, vx=0.0, vy=0.0, grounded=False, coyote=0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.grounded, self.coyote = grounded, coyote


def _move_axis(sim: LevelSim, b: _Body, dx: float, dy: float) -> None:
    remaining = dx if dx != 0 else dy
    if remaining == 0:
        return
    sign = 1.0 if remaining > 0 else -1.0
    axis_x = dx != 0
    while abs(remaining) > 0.001:
        step = sign * min(1.0, abs(remaining))
        nx = b.x + step if axis_x else b.x
        ny = b.y if axis_x else b.y + step
        if sim.rect_collides(nx, ny):
            if axis_x:
                b.vx = 0.0
            else:
                b.vy = 0.0
            return
        b.x, b.y = nx, ny
        remaining -= step


def _step(sim: LevelSim, b: _Body, move_dir: int, wants_jump: bool) -> None:
    """One engine frame: _apply_player_input + _move_player."""
    if sim.on_ladder(b.x, b.y):
        b.grounded = False
        b.coyote = 0
        if move_dir:
            b.vx = move_dir * AIR_SPEED
        else:
            b.vx *= FRICTION
            if abs(b.vx) < 0.05:
                b.vx = 0.0
        b.vy = -CLIMB if wants_jump else DESCEND
    else:
        b.grounded = sim.is_on_surface(b.x, b.y)
        b.coyote = COYOTE if b.grounded else max(0, b.coyote - 1)
        speed = MOVE_SPEED if b.grounded else AIR_SPEED
        if move_dir:
            b.vx = move_dir * speed
        else:
            b.vx *= FRICTION
            if abs(b.vx) < 0.05:
                b.vx = 0.0
        if wants_jump and b.coyote > 0:
            b.vy = -JUMP_SPEED
            b.grounded = False
            b.coyote = 0
        b.vy += GRAVITY
        b.vy = max(-MAX_FALL, min(MAX_FALL, b.vy))
    _move_axis(sim, b, b.vx, 0.0)
    _move_axis(sim, b, 0.0, b.vy)
    b.grounded = sim.is_on_surface(b.x, b.y)


def _player_tile(x: float, y: float) -> Tuple[int, int]:
    return (int((x + PW / 2) // TILE), int((y + PH / 2) // TILE))


def _touched(sim: LevelSim, x: float, y: float, touch: Set[Tuple[int, int]]) -> None:
    for c, r in sim._tiles_for_rect(x, y):
        touch.add((c, r))


# Macro control PROGRAMS: each is a list of (frames, move_dir, jump-held) segments,
# mirroring the input sequences a player/agent can actually feed the engine.
# Held-jump re-fires only from the ground (coyote) or climbs on a ladder; the
# engine grants full mid-air direction control (AIR_SPEED per frame), so the
# jump-then-reverse programs model up-and-over arcs.
_D = (-1, 1)
_MACROS: List[List[Tuple[int, int, bool]]] = (
    [[(MAX_FRAMES, d, False)] for d in _D]  # walk / fall off edges
    + [[(MAX_FRAMES, d, True)] for d in _D]  # repeated hops (+ climb on ladders)
    + [[(1, d, True), (MAX_FRAMES - 1, d, False)] for d in _D]  # jump, hold direction
    + [
        [(1, 0, True), (12, 0, False), (MAX_FRAMES - 13, d, False)] for d in _D
    ]  # vertical jump then drift after apex
    + [
        [(1, d, True), (11, d, False), (MAX_FRAMES - 12, -d, False)] for d in _D
    ]  # jump one way, reverse mid-air (up-and-over)
    + [[(MAX_FRAMES, 0, True)]]  # climb straight up a ladder
    + [[(MAX_FRAMES, 0, False)]]  # descend a ladder / wait
)


def _run_macro(
    sim: LevelSim,
    start: _Body,
    program: List[Tuple[int, int, bool]],
    touch: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Simulate one control program to completion; return resting cells reached."""
    b = _Body(start.x, start.y, 0.0, 0.0, start.grounded, start.coyote)
    rests: List[Tuple[int, int]] = []
    prev_cell: Tuple[int, int] | None = None

    def record() -> bool:
        nonlocal prev_cell
        _touched(sim, b.x, b.y, touch)
        cell = _player_tile(b.x, b.y)
        if (b.grounded or sim.on_ladder(b.x, b.y)) and cell != prev_cell:
            rests.append(cell)
            prev_cell = cell
        return b.y > (sim.rows + 2) * TILE

    for frames, move_dir, jump in program:
        for _ in range(frames):
            _step(sim, b, move_dir, jump)
            if record():
                return rests
    # settle: let it fall to a resting cell if the program ended airborne
    if not (b.grounded or sim.on_ladder(b.x, b.y)):
        for _ in range(MAX_FRAMES):
            _step(sim, b, 0, False)
            if record() or b.grounded or sim.on_ladder(b.x, b.y):
                break
    return rests


def analyze(layout: Tuple[str, ...], closed_doors: frozenset = frozenset()) -> Dict:
    sim = LevelSim(layout, closed_doors=closed_doors)
    # place player at start cell and settle to the ground
    sc, sr = sim.start
    b = _Body(sc * TILE + 5, sr * TILE + 1)
    for _ in range(MAX_FRAMES):
        _step(sim, b, 0, False)
        if b.grounded or sim.on_ladder(b.x, b.y):
            break
    start_cell = _player_tile(b.x, b.y)

    touch: Set[Tuple[int, int]] = set()
    seen: Set[Tuple[int, int]] = {start_cell}
    queue: deque = deque([_Body(b.x, b.y, 0.0, 0.0, b.grounded, b.coyote)])
    resting_cells = {start_cell}
    # map resting cell -> a representative body so we can re-expand
    while queue and len(seen) < MAX_RESTING:
        body = queue.popleft()
        for program in _MACROS:
            for cell in _run_macro(sim, body, program, touch):
                resting_cells.add(cell)
                if cell not in seen:
                    seen.add(cell)
                    # rebuild a resting body at this cell for expansion. On a ladder the
                    # player rests in place; otherwise settle it onto the ground below.
                    nx = cell[0] * TILE + 5
                    ny = cell[1] * TILE + 1
                    nb = _Body(nx, ny)
                    if not sim.on_ladder(nb.x, nb.y):
                        for _ in range(MAX_FRAMES):
                            _step(sim, nb, 0, False)
                            if nb.grounded or sim.on_ladder(nb.x, nb.y):
                                break
                    queue.append(nb)

    def reached(tiles: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        return {t for t in tiles if t in touch}

    cr = reached(sim.crystals)
    sw = reached(sim.switches)
    exit_ok = sim.exit in touch
    total_obj = len(sim.crystals) + len(sim.switches) + 1
    got_obj = len(cr) + len(sw) + (1 if exit_ok else 0)
    return {
        "crystals": (len(cr), len(sim.crystals)),
        "switches": (len(sw), len(sim.switches)),
        "exit": exit_ok,
        "winnable": got_obj == total_obj,
        "obj_frac": got_obj / total_obj,
        "missing_crystals": sorted(sim.crystals - cr),
        "missing_switches": sorted(sim.switches - sw),
        "resting_cells": resting_cells,
        "sim": sim,
    }


# door char -> the switch char that opens it (engine colour pairing)
SWITCH_FOR_DOOR = {"D": "s", "d": "S"}


def analyze_gated(layout: Tuple[str, ...]) -> Dict:
    """In-game lock-ordering solvability. Unlike ``analyze`` (which treats doors as
    open), doors start CLOSED (solid) and a colour opens only once one of its
    switches has been physically REACHED with the doors opened so far. Iterates to
    a fixpoint, so switch-behind-other-colour chains resolve, and a switch locked
    behind its OWN door is correctly reported as a deadlock."""
    present = {ch for row in layout for ch in row}
    doors_present = {dc for dc in DOORS if dc in present}
    opened: Set[str] = set()
    order: list = []
    while True:
        res = analyze(layout, closed_doors=frozenset(doors_present - opened))
        sim = res["sim"]
        reached_sw = sim.switches - set(res["missing_switches"])
        reach_chars = {sim.grid[r][c] for (c, r) in reached_sw}
        newly = {dc for dc in doors_present - opened if SWITCH_FOR_DOOR[dc] in reach_chars}
        if not newly:
            break
        opened |= newly
        order.append(sorted(newly))
    res["gated_winnable"] = res["winnable"]
    res["door_open_order"] = order
    res["doors_never_opened"] = sorted(doors_present - opened)
    return res


def door_value(layout: Tuple[str, ...]) -> Dict[str, bool]:
    """Per door colour: does keeping it closed actually block any objective?
    False = decorative (the player can route around it, so the switch puzzle
    gates nothing)."""
    present = {ch for row in layout for ch in row}
    out: Dict[str, bool] = {}
    for dc in DOORS:
        if dc in present:
            res = analyze(layout, closed_doors=frozenset({dc}))
            out[dc] = not res["winnable"]
    return out


def main() -> int:
    from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS as CC1_LEVELS

    win = 0
    for lvl in CC1_LEVELS:
        res = analyze(lvl.layout)
        cc, ct = res["crystals"]
        sc, st = res["switches"]
        tag = "WIN " if res["winnable"] else "FAIL"
        if res["winnable"]:
            win += 1
        print(
            f"{tag} {lvl.name}: crystals {cc}/{ct}  switches {sc}/{st}  "
            f"exit={'ok' if res['exit'] else 'NO'}  obj={res['obj_frac']:.2f}"
        )
        if not res["winnable"]:
            if res["missing_crystals"]:
                print(f"       missing crystals: {res['missing_crystals'][:12]}")
            if res["missing_switches"]:
                print(f"       missing switches: {res['missing_switches']}")
            if not res["exit"]:
                print(f"       exit {res['sim'].exit} unreachable")
    print(f"\n{win}/{len(CC1_LEVELS)} levels winnable (all objectives reachable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
