"""Small traversal repairs for the Unity player's lower classic jump."""

from dataclasses import replace

from src.game.crystal_caves import CrystalCaves
from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS

from .classic_layouts import CLASSIC_LEVELS
from .classic_polish import polish_caves
from .classic_secrets import ClassicSecrets


def apply_classic_caves(game: CrystalCaves) -> None:
    if game.CAVES is not HANDCRAFTED_LEVELS:
        return
    caves = list(CLASSIC_LEVELS)
    # Connect existing platforms, preserving every collectible, gate and spawn.
    # The training maps remain unchanged; these repairs accompany human tuning.
    ladders = {3: ((13, 3, 6), (23, 3, 6)), 9: ((8, 5, 16),)}
    for level, connections in ladders.items():
        layout = [list(row) for row in caves[level].layout]
        for col, start, end in connections:
            for row in range(start, end + 1):
                if layout[row][col] not in ".#":
                    raise ValueError("classic ladder repair would replace a level object")
                layout[row][col] = "H"
        caves[level] = replace(caves[level], layout=tuple("".join(row) for row in layout))
    # Keep the silver spikes too. These existing trap sites gain retracting thorns.
    for level, tile in {
        0: (8, 21),
        2: (12, 21),
        5: (10, 21),
        7: (20, 21),
        11: (12, 20),
        14: (13, 21),
    }.items():
        col, row = tile
        layout = [list(line) for line in caves[level].layout]
        if layout[row][col] != "^" or layout[row + 1][col] != "#":
            raise ValueError("classic thorn site must replace a floor spike")
        layout[row][col] = "t"
        caves[level] = replace(caves[level], layout=tuple("".join(line) for line in layout))
    # Hanging orange traps occupy clear air beneath an existing ceiling. Every
    # platform, ladder and pickup remains in its authored location.
    for level, sites in {
        4: ((12, 17), (22, 17), (16, 5)),
        13: ((13, 15), (17, 7), (31, 7)),
    }.items():
        layout = [list(line) for line in caves[level].layout]
        for col, row in sites:
            if layout[row][col] != "." or layout[row - 1][col] != "#":
                raise ValueError("classic stalactite site must be clear beneath stone")
            layout[row][col] = "v"
        caves[level] = replace(caves[level], layout=tuple("".join(line) for line in layout))
    game.CAVES = polish_caves(tuple(caves))
    if isinstance(game, ClassicSecrets):
        game.secrets_enabled = True
