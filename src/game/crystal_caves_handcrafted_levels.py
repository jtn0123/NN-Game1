"""Hand-crafted, physics-verified Crystal-Caves-style levels.

Why these exist: the levels reverse-engineered from the original CC1 game data
were provably unplayable (0/16 winnable under real physics) because a tile's
solidity lives in the game's collision code, not in the level data or graphics we
can read — so the byte->tile mapping grossly over-solidified and mislabelled
objects. Rather than ship broken levels, these are authored by hand in the spirit
of Crystal Caves (collect every crystal, throw switches to open doors, dodge
spikes, reach the exit) and EVERY one is certified winnable by the physics-faithful
reachability oracle in experiments/cc_status/level_reach.py: from the player start,
using only the mechanics this engine actually has (walk / jump / climb), every
crystal, switch, and the exit is physically reachable.

Legend (engine tiles): '#' solid  '.' empty  'P' player  'E' exit  '*' crystal
'H' ladder  's' switch  'D' door  '^' spike.

Regenerate/verify with:  python -m experiments.cc_status.level_reach
"""

from __future__ import annotations

from .crystal_caves_entities import CaveSpec

HANDCRAFTED_LEVELS = (
    CaveSpec(
        name="First Steps",
        layout=(
            "########################################",
            "#......................................#",
            "#......................................#",
            "#...*.......*..........*.........*.....#",
            "#..###.....###........###.......###....#",
            "#......................................#",
            "#.P...*.........*..........*.......E...#",
            "########################################",
        ),
        background=(9, 12, 22),
        accent=(80, 190, 255),
    ),
    CaveSpec(
        name="Locked Door",
        layout=(
            "########################################",
            "#.........*..............*.............#",
            "#........###............###............#",
            "#..*.................#.......*.....*...#",
            "#.###....*......s....#....######...###.#",
            "#.......###.....#....D.................#",
            "#.P.............#....#............E....#",
            "########################################",
        ),
        background=(12, 10, 18),
        accent=(255, 188, 80),
    ),
    CaveSpec(
        name="The Shaft",
        layout=(
            "########################################",
            "#....*..E..*...H...*...................#",
            "#...#####.###..H..###..................#",
            "#.............#H#......................#",
            "#....*.....*..H..*....*................#",
            "#...###...###.H.###..###...............#",
            "#............#H#.......................#",
            "#....*.....*..H..*....*................#",
            "#...###...###.H.###..###...............#",
            "#............#H#.......................#",
            "#.P..........H.........*...............#",
            "########################################",
        ),
        background=(8, 15, 13),
        accent=(120, 255, 155),
    ),
    CaveSpec(
        name="Spike Run",
        layout=(
            "########################################",
            "#......................................#",
            "#...*.....*.......*........*.....*.....#",
            "#..###...###.....###......###...###....#",
            "#......................................#",
            "#.P..^^....^^^.....^^....^^^....^^...E.#",
            "########################################",
        ),
        background=(20, 8, 10),
        accent=(255, 120, 90),
    ),
    CaveSpec(
        name="Deep Dig",
        layout=(
            "########################################",
            "#..............*..........*......*.....#",
            "#.....H.......###........###....###....#",
            "#..*..H................................#",
            "#.###.H..........s....#........*.......#",
            "#.....H..........#....D.......###......#",
            "#..*..H..........#....#................#",
            "#.P.^^H..........#....#....^^.....E....#",
            "########################################",
        ),
        background=(10, 14, 24),
        accent=(150, 200, 255),
    ),
)
