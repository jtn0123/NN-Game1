"""Hand-authored single-skill "drill" levels for Crystal Caves.

Each drill is a tiny, deliberately-shaped room that isolates ONE motor skill the
agent must master (walk-and-collect, jump up a ledge, jump a gap, drop-and-climb,
climb a staircase, collect-then-jump-to-exit). Contact caves are even smaller
training-only variations focused on the final few tiles around an objective.
They are used two ways:

  1. Diagnostic — run a trained policy on each and read its per-skill win rate, so we
     know exactly which skills are missing instead of inferring it.
  2. Teaching — pre-train / interleave so the agent enters the full levels already
     knowing these motor skills.

True to the real 1991 game, whose opening levels each introduce one mechanic. Levels
are built on a solid 18x44 grid (matching the authored caves); the win rule is
unchanged — collect every crystal, then reach the exit.
"""

from __future__ import annotations

import random
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


CONTACT_CAVES: Tuple[CaveSpec, ...] = (
    # C1 flat contact — the shortest possible "walk into crystal, then exit" loop.
    _drill(
        "contact: floor",
        skill="contact_floor",
        opens=[(13, 1, 16, 42)],
        player=(18, 16),
        crystals=((21, 16),),
        exit_cell=(25, 16),
        accent=(255, 210, 120),
    ),
    # C2 jump-up contact — start close enough that the only missing behavior is the hop.
    _drill(
        "contact: jump up",
        skill="contact_jump_up",
        opens=[(13, 1, 16, 21), (11, 22, 14, 42)],
        player=(19, 16),
        crystals=((20, 16),),
        exit_cell=(24, 14),
        accent=(255, 210, 120),
    ),
    # C3 drop-return contact — crystal is in a shallow pocket, exit is back on the rim.
    _drill(
        "contact: drop return",
        skill="contact_drop_return",
        opens=[(10, 1, 12, 42), (13, 19, 14, 23)],
        player=(17, 12),
        crystals=((21, 14),),
        exit_cell=(25, 12),
        accent=(255, 210, 120),
    ),
    # C4 step pair contact — two small upward transitions, not the old four-step chain.
    _drill(
        "contact: step pair",
        skill="contact_step_pair",
        opens=[(11, 1, 15, 20), (10, 21, 14, 29), (9, 30, 13, 42)],
        player=(18, 15),
        crystals=((24, 14),),
        exit_cell=(32, 13),
        accent=(255, 210, 120),
    ),
    # C5 exit-after-crystal — the specific full-level failure mode in a compact room.
    _drill(
        "contact: exit after crystal",
        skill="contact_exit_after_crystal",
        opens=[(12, 1, 16, 25), (11, 26, 15, 34), (10, 35, 14, 42)],
        player=(18, 16),
        crystals=((22, 16),),
        exit_cell=(30, 15),
        accent=(255, 210, 120),
    ),
)


CONTACT_BY_SKILL: Dict[str, CaveSpec] = {
    "floor": CONTACT_CAVES[0],
    "jump_up": CONTACT_CAVES[1],
    "drop_return": CONTACT_CAVES[2],
    "step_pair": CONTACT_CAVES[3],
    "exit_after_crystal": CONTACT_CAVES[4],
}


def contact_pool_caves(pool_size: int, seed: int = 0) -> Tuple[CaveSpec, ...]:
    """Build a deterministic varied contact-training pool.

    These are still training-only rooms, but unlike ``CONTACT_CAVES`` they vary
    positions, heights, and route fragments so interleaved lanes cannot memorize
    five exact maps.
    """

    if pool_size <= 0:
        return ()
    rng = random.Random(seed)
    kinds = (
        "floor",
        "jump_up",
        "drop_return",
        "step_pair",
        "exit_after_crystal",
    )
    caves: list[CaveSpec] = []
    seen_layouts: set[Tuple[str, ...]] = set()
    attempts = 0
    max_attempts = max(100, pool_size * 80)
    while len(caves) < pool_size and attempts < max_attempts:
        kind = kinds[attempts % len(kinds)]
        spec = _contact_pool_variant(kind, index=len(caves), attempt=attempts, rng=rng)
        attempts += 1
        if spec.layout in seen_layouts:
            continue
        seen_layouts.add(spec.layout)
        caves.append(spec)
    if len(caves) < pool_size:
        raise RuntimeError(f"could only build {len(caves)} unique contact caves")
    return tuple(caves)


def _contact_pool_variant(
    kind: str,
    *,
    index: int,
    attempt: int,
    rng: random.Random,
) -> CaveSpec:
    accent = (255, 180 + (attempt * 17) % 60, 90 + (attempt * 23) % 80)
    if kind == "floor":
        row = rng.randint(14, 16)
        player_col = rng.randint(5, 20)
        crystal_col = player_col + rng.randint(2, 6)
        exit_col = min(41, crystal_col + rng.randint(2, 8))
        return _drill(
            f"contact pool: floor {index:03d}",
            skill="contact_pool_floor",
            opens=[(row - 3, 1, row, 42)],
            player=(player_col, row),
            crystals=((crystal_col, row),),
            exit_cell=(exit_col, row),
            accent=accent,
        )
    if kind == "jump_up":
        base_row = rng.randint(14, 16)
        shelf_delta = rng.randint(1, 3)
        shelf_row = base_row - shelf_delta
        split = rng.randint(18, 28)
        player_col = split - rng.randint(4, 9)
        crystal_col = max(player_col + 1, split - rng.randint(1, 4))
        exit_col = min(41, split + rng.randint(3, 11))
        return _drill(
            f"contact pool: jump up {index:03d}",
            skill="contact_pool_jump_up",
            opens=[(base_row - 3, 1, base_row, split), (shelf_row - 3, split + 1, shelf_row, 42)],
            player=(player_col, base_row),
            crystals=((crystal_col, base_row),),
            exit_cell=(exit_col, shelf_row),
            accent=accent,
        )
    if kind == "drop_return":
        top_row = rng.randint(10, 13)
        pocket_left = rng.randint(16, 23)
        pocket_width = rng.randint(3, 6)
        pocket_right = min(38, pocket_left + pocket_width)
        player_col = max(3, pocket_left - rng.randint(3, 8))
        exit_col = min(41, pocket_right + rng.randint(3, 8))
        crystal_col = rng.randint(pocket_left + 1, pocket_right - 1)
        return _drill(
            f"contact pool: drop return {index:03d}",
            skill="contact_pool_drop_return",
            opens=[
                (top_row - 2, 1, top_row, 42),
                (top_row + 1, pocket_left, top_row + 2, pocket_right),
            ],
            player=(player_col, top_row),
            crystals=((crystal_col, top_row + 2),),
            exit_cell=(exit_col, top_row),
            accent=accent,
        )
    if kind == "step_pair":
        base_row = rng.randint(15, 16)
        step_one = base_row - rng.randint(1, 2)
        step_two = max(8, step_one - rng.randint(1, 2))
        split_one = rng.randint(16, 23)
        split_two = rng.randint(split_one + 6, 34)
        player_col = split_one - rng.randint(4, 8)
        crystal_col = rng.randint(split_one + 2, split_two - 1)
        exit_col = min(41, split_two + rng.randint(3, 7))
        return _drill(
            f"contact pool: step pair {index:03d}",
            skill="contact_pool_step_pair",
            opens=[
                (base_row - 4, 1, base_row, split_one),
                (step_one - 4, split_one + 1, step_one, split_two),
                (step_two - 4, split_two + 1, step_two, 42),
            ],
            player=(player_col, base_row),
            crystals=((crystal_col, step_one),),
            exit_cell=(exit_col, step_two),
            accent=accent,
        )
    base_row = rng.randint(14, 16)
    shelf_row = base_row - rng.randint(1, 2)
    split = rng.randint(22, 31)
    player_col = split - rng.randint(6, 11)
    crystal_col = split - rng.randint(1, 4)
    exit_col = min(41, split + rng.randint(2, 8))
    return _drill(
        f"contact pool: exit after crystal {index:03d}",
        skill="contact_pool_exit_after_crystal",
        opens=[(base_row - 4, 1, base_row, split), (shelf_row - 3, split + 1, shelf_row, 42)],
        player=(player_col, base_row),
        crystals=((crystal_col, base_row),),
        exit_cell=(exit_col, shelf_row),
        accent=accent,
    )
