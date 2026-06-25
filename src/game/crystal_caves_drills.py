"""Hand-authored single-skill "drill" levels for Crystal Caves.

Each drill is a tiny, deliberately-shaped room that isolates ONE motor skill the
agent must master (walk-and-collect, jump up a ledge, jump a gap, drop-and-climb,
climb a staircase, collect-then-jump-to-exit). They are used two ways:

  1. Diagnostic — run a trained policy on each and read its per-skill win rate, so we
     know exactly which skills are missing instead of inferring it.
  2. Teaching — pre-train / interleave so the agent enters the full levels already
     knowing these motor skills.

True to the real 1991 game, whose opening levels each introduce one mechanic. Levels
are built on a solid 18x44 grid (matching the authored caves); the win rule is
unchanged — collect every crystal, then reach the exit.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .crystal_caves_entities import CaveSpec

_ROWS, _COLS = 18, 44
_SOLID, _EMPTY = "#", "."
_Cell = Tuple[int, int]  # (col, row)
_Rect = Tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive


def _drill(
    name: str,
    *,
    skill: str,
    opens: List[_Rect],
    player: _Cell,
    exit_cell: _Cell,
    crystals: Tuple[_Cell, ...] = (),
    extra: Tuple[Tuple[int, int, str], ...] = (),  # (col, row, char)
    accent: Tuple[int, int, int] = (120, 220, 255),
) -> CaveSpec:
    """Carve ``opens`` to empty on an all-solid grid, then place objects."""
    grid = [[_SOLID for _ in range(_COLS)] for _ in range(_ROWS)]
    for r0, c0, r1, c1 in opens:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                grid[r][c] = _EMPTY
    for col, row in crystals:
        grid[row][col] = "*"
    grid[exit_cell[1]][exit_cell[0]] = "E"
    grid[player[1]][player[0]] = "P"
    for col, row, char in extra:
        grid[row][col] = char
    layout = tuple("".join(row) for row in grid)
    return CaveSpec(
        name=name,
        layout=layout,
        background=(10, 12, 20),
        accent=accent,
        sky_rows=0,
    )


# A "+N" in the comments is how many tiles of vertical jump the skill needs (jump = 3).
DRILL_CAVES: Tuple[CaveSpec, ...] = (
    # D1 walk — collect on flat ground, then walk to the exit (sanity floor).
    _drill(
        "drill: walk + collect",
        skill="walk",
        opens=[(13, 1, 16, 42)],
        player=(3, 16),
        crystals=((22, 16),),
        exit_cell=(40, 16),
    ),
    # D2 jump-up (+2) — a raised shelf the agent must hop onto for the crystal/exit.
    _drill(
        "drill: jump up",
        skill="jump_up",
        opens=[(12, 1, 15, 22), (10, 23, 13, 42)],
        player=(3, 15),
        crystals=((30, 13),),
        exit_cell=(41, 13),
    ),
    # D3 jump-gap (+0, horizontal) — a 3-wide pit with acid below; jump across.
    _drill(
        "drill: jump a gap",
        skill="jump_gap",
        opens=[(11, 1, 14, 42), (15, 20, 16, 22)],
        player=(3, 14),
        crystals=((30, 14),),
        exit_cell=(41, 14),
        extra=((20, 16, "~"), (21, 16, "~"), (22, 16, "~")),
    ),
    # D4 drop-and-climb (+2 out) — crystal in a shallow pocket; fall in, climb back out.
    _drill(
        "drill: drop and climb",
        skill="drop_climb",
        opens=[(9, 1, 11, 42), (12, 18, 13, 22)],
        player=(3, 11),
        crystals=((20, 13),),
        exit_cell=(41, 11),
    ),
    # D5 staircase (+2 per step) — climb four ascending steps to the crystal/exit.
    _drill(
        "drill: staircase",
        skill="staircase",
        opens=[(1, 1, 15, 8), (1, 9, 13, 14), (1, 15, 11, 20), (1, 21, 9, 26), (1, 27, 7, 42)],
        player=(3, 15),
        crystals=((34, 7),),
        exit_cell=(41, 7),
    ),
    # D6 reach-the-exit (+2) — the wall we're stuck on: grab the floor crystal (unlocks
    # the exit), then jump up onto a shelf to reach the exit.
    _drill(
        "drill: collect then jump to exit",
        skill="reach_exit",
        opens=[(11, 1, 16, 29), (11, 30, 14, 33), (11, 34, 16, 42)],
        player=(3, 16),
        crystals=((15, 16),),
        exit_cell=(31, 14),
    ),
)


DRILL_BY_SKILL: Dict[str, CaveSpec] = {
    "walk": DRILL_CAVES[0],
    "jump_up": DRILL_CAVES[1],
    "jump_gap": DRILL_CAVES[2],
    "drop_climb": DRILL_CAVES[3],
    "staircase": DRILL_CAVES[4],
    "reach_exit": DRILL_CAVES[5],
}


BRIDGE_CAVES: Tuple[CaveSpec, ...] = (
    # B1 two-step climb — easier than the four-step staircase but no longer a
    # single isolated hop. Teaches chaining two upward transitions before the exit.
    _drill(
        "bridge: two-step climb",
        skill="bridge_two_step",
        opens=[(12, 1, 16, 16), (10, 17, 14, 29), (8, 30, 12, 42)],
        player=(3, 16),
        crystals=((22, 14), (34, 12)),
        exit_cell=(41, 12),
        accent=(160, 220, 120),
    ),
    # B2 safe gap — same horizontal-jump decision as the acid drill, but the pit
    # is non-lethal so exploration does not immediately teach jump avoidance.
    _drill(
        "bridge: safe gap collect",
        skill="bridge_safe_gap",
        opens=[(11, 1, 14, 42), (15, 19, 16, 23)],
        player=(3, 14),
        crystals=((15, 14), (31, 14)),
        exit_cell=(41, 14),
        accent=(160, 220, 120),
    ),
    # B3 flat switch route — puts the causal switch/door relationship into an
    # otherwise trivial route so full-level door logic gets clean practice.
    _drill(
        "bridge: switch opens route",
        skill="bridge_switch",
        opens=[(12, 1, 16, 42)],
        player=(3, 16),
        crystals=((13, 16), (34, 16)),
        exit_cell=(41, 16),
        extra=((22, 16, "s"), (28, 16, "D")),
        accent=(160, 220, 120),
    ),
    # B4 collect then hop to exit — a gentler version of the known full-game wall:
    # collect on the floor, then make a small vertical transition to the exit.
    _drill(
        "bridge: collect then low exit",
        skill="bridge_exit_hop",
        opens=[(12, 1, 16, 25), (11, 26, 15, 34), (10, 35, 14, 42)],
        player=(3, 16),
        crystals=((14, 16), (29, 15)),
        exit_cell=(40, 14),
        accent=(160, 220, 120),
    ),
    # B5 mini route — combines switch, two crystals, and a small exit climb in a
    # compact cave. This is intentionally still simpler than procedural tutorial.
    _drill(
        "bridge: mini full route",
        skill="bridge_mini_route",
        opens=[(12, 1, 16, 18), (11, 19, 15, 28), (10, 29, 14, 42)],
        player=(3, 16),
        crystals=((12, 16), (33, 14)),
        exit_cell=(41, 14),
        extra=((18, 16, "s"), (25, 15, "D")),
        accent=(160, 220, 120),
    ),
)


BRIDGE_BY_SKILL: Dict[str, CaveSpec] = {
    "two_step": BRIDGE_CAVES[0],
    "safe_gap": BRIDGE_CAVES[1],
    "switch": BRIDGE_CAVES[2],
    "exit_hop": BRIDGE_CAVES[3],
    "mini_route": BRIDGE_CAVES[4],
}
