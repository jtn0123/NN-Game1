"""Small, addressable human-cave edits layered over the accepted terrain baseline."""

from dataclasses import replace

from src.game.crystal_caves_entities import CaveSpec

# Zero-based cave index and (column, row, marker). Additions claim an empty cell;
# no pickup, trap, actor or supporting platform is replaced.
CONTENT_EDITS: dict[int, tuple[tuple[int, int, str], ...]] = {
    0: ((6, 21, "p"),),  # First raygun upgrade after the opening ammo pickup.
    # Two jumps offer a rewarded alternative to the long upper chain ascent.
    1: ((6, 21, "A"), (21, 5, "#"), (22, 5, "#"), (21, 4, "$")),
    2: ((18, 17, "p"),),  # Upgrade before the red-gate/rock pocket.
    5: ((6, 21, "A"),),  # Avoid a long unarmed trip to the upper ammo shelf.
    6: ((7, 21, "p"),),  # Rare upgrade before the smelter's main floor patrol.
    7: ((8, 8, "z"),),  # Choose freeze by climbing before the flyer crossing.
    # Dinosaur patrol floor, covered bat-room rest and central-chain return.
    8: ((1, 6, "#"), (4, 12, "#"), (15, 17, "#"), (16, 17, "#"), (15, 16, "$")),
    # Two short jumps link existing shelves. The chest is an optional reward,
    # so the familiar lift and ladder route still reaches every crystal.
    11: ((10, 11, "#"), (11, 11, "#"), (11, 10, "$")),
    # A visible falling trap guards a rewarded three-jump alternative. The
    # existing chain remains available while this branch reduces retracing.
    13: ((15, 17, "#"), (16, 17, "#"), (17, 16, "#"), (18, 16, "#"), (17, 15, "$")),
    14: ((35, 18, "p"),),  # A choice before the walking rock beside the lift.
}

# Move sealed actors and an off-route ammo pickup without changing their count.
RELOCATIONS = {
    6: ((20, 5, 18, 5, "O"),),
    8: ((31, 13, 21, 13, "A"),),
    10: ((24, 8, 22, 8, "F"),),
}

# Trim only the lower layer of a double-thick ceiling. The row above remains
# solid, supporting every object on the upper floor while opening jump space.
CEILING_CLEARANCE = {0: tuple((col, 19) for col in range(6, 11))}

# A shallow trench gives the opening/return jump room above both hazards without
# removing the upper floor or changing the classic jump arc and damage rules.
HAZARD_TRENCHES = {0: ((8, 21, "t"), (9, 21, "^"))}


def polish_caves(caves: tuple[CaveSpec, ...]) -> tuple[CaveSpec, ...]:
    result = list(caves)
    levels = (
        CONTENT_EDITS.keys()
        | RELOCATIONS.keys()
        | CEILING_CLEARANCE.keys()
        | HAZARD_TRENCHES.keys()
    )
    for level in sorted(levels):
        rows = [list(row) for row in result[level].layout]
        for col, row in CEILING_CLEARANCE.get(level, ()):
            if rows[row][col] != "#" or rows[row - 1][col] != "#":
                raise ValueError(f"cave {level + 1} ceiling {(col, row)} is not double stone")
            rows[row][col] = "."
        for col, row, marker in HAZARD_TRENCHES.get(level, ()):
            if rows[row][col] != marker or rows[row + 1][col] != "#" or rows[row + 2][col] != "#":
                raise ValueError(f"cave {level + 1} trench {(col, row)} lacks solid support")
            rows[row][col], rows[row + 1][col] = ".", marker
        for old_col, old_row, col, row, marker in RELOCATIONS.get(level, ()):
            if rows[old_row][old_col] != marker or rows[row][col] != ".":
                raise ValueError(f"cave {level + 1} relocation {(col, row)} is invalid")
            rows[old_row][old_col], rows[row][col] = ".", marker
        edits = CONTENT_EDITS.get(level, ())
        for col, row, marker in edits:
            if rows[row][col] != ".":
                raise ValueError(f"cave {level + 1} polish cell {(col, row)} is occupied")
            rows[row][col] = marker
        result[level] = replace(result[level], layout=tuple("".join(row) for row in rows))
    return tuple(result)
