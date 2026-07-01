"""Render the hand-crafted Crystal Caves levels to PNGs for eyeballing.

Schematic top-down renders drawn straight from the tile grids (not the in-game
pixel art) so the layout/structure of each level can be reviewed. Writes one PNG
per level plus a 4x4 contact sheet.

Run:  python -m experiments.cc_status.render_levels [output_dir]
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pygame  # noqa: E402

from src.game.crystal_caves_handcrafted_levels import HANDCRAFTED_LEVELS  # noqa: E402

TS = 22  # tile size in px

# color per tile ('.' is the level background)
COL = {
    "#": (86, 74, 64),
    ".": None,
    "H": (150, 110, 60),
    "=": (70, 120, 200),
    "*": (90, 230, 255),
    "E": (60, 220, 90),
    "P": (255, 240, 120),
    "s": (230, 70, 70),
    "S": (80, 130, 255),
    "D": (150, 40, 40),
    "d": (40, 60, 150),
    "^": (200, 70, 70),
    "~": (120, 220, 70),
    "M": (240, 150, 60),
    "F": (200, 100, 240),
    "A": (230, 230, 230),
    "$": (255, 210, 70),
    "O": (140, 210, 255),
    "g": (255, 120, 200),
    "z": (120, 255, 220),
    "p": (255, 160, 90),
}
GLYPH = set("EPsSMFA$Ogzp")  # draw the letter on top of the marker


def render(spec, font, small) -> "pygame.Surface":
    layout = spec.layout
    h = len(layout) * TS
    w = len(layout[0]) * TS
    surf = pygame.Surface((w, h + 24))
    surf.fill(spec.background)
    surf.fill((20, 20, 24), pygame.Rect(0, h, w, 24))  # caption strip
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            x, y = c * TS, r * TS
            if ch == "#":
                pygame.draw.rect(surf, COL["#"], (x, y, TS, TS))
                pygame.draw.rect(surf, (110, 96, 84), (x, y, TS, TS), 1)
            elif ch == "H":
                pygame.draw.line(surf, COL["H"], (x + 4, y), (x + 4, y + TS), 3)
                pygame.draw.line(surf, COL["H"], (x + TS - 4, y), (x + TS - 4, y + TS), 3)
                pygame.draw.line(surf, COL["H"], (x + 4, y + TS // 2), (x + TS - 4, y + TS // 2), 2)
            elif ch == "=":
                pygame.draw.rect(surf, COL["="], (x, y + TS // 2 - 2, TS, 5))
            elif ch == "*":
                pygame.draw.polygon(
                    surf,
                    COL["*"],
                    [
                        (x + TS // 2, y + 3),
                        (x + TS - 3, y + TS // 2),
                        (x + TS // 2, y + TS - 3),
                        (x + 3, y + TS // 2),
                    ],
                )
            elif ch == "^":
                pygame.draw.polygon(
                    surf,
                    COL["^"],
                    [(x + 2, y + TS - 2), (x + TS // 2, y + 4), (x + TS - 2, y + TS - 2)],
                )
            elif ch == "~":
                pygame.draw.rect(surf, COL["~"], (x, y + TS // 2, TS, TS // 2))
            elif ch in ("D", "d"):
                pygame.draw.rect(surf, COL[ch], (x + 2, y, TS - 4, TS))
            elif COL.get(ch) is not None:
                pygame.draw.circle(surf, COL[ch], (x + TS // 2, y + TS // 2), TS // 2 - 2)
            if ch in GLYPH:
                g = small.render(ch, True, (10, 10, 10))
                surf.blit(g, (x + TS // 2 - g.get_width() // 2, y + TS // 2 - g.get_height() // 2))
    cap = font.render(spec.name, True, (235, 235, 235))
    surf.blit(cap, (6, h + 4))
    return surf


def main(argv) -> int:
    out_dir = argv[1] if len(argv) > 1 else os.path.join(_REPO_ROOT, "scratchpad", "cc_renders")
    os.makedirs(out_dir, exist_ok=True)
    pygame.init()
    font = pygame.font.SysFont("monospace", 14, bold=True)
    small = pygame.font.SysFont("monospace", 11, bold=True)

    surfs = []
    for spec in HANDCRAFTED_LEVELS:
        s = render(spec, font, small)
        surfs.append(s)
        pygame.image.save(s, os.path.join(out_dir, f"{spec.name.replace(' ', '_')}.png"))

    cols, rows, pad = 4, 4, 8
    cw = max(s.get_width() for s in surfs)
    chh = max(s.get_height() for s in surfs)
    sheet = pygame.Surface((cols * cw + pad * (cols + 1), rows * chh + pad * (rows + 1)))
    sheet.fill((12, 12, 16))
    for i, s in enumerate(surfs):
        sheet.blit(s, (pad + (i % cols) * (cw + pad), pad + (i // cols) * (chh + pad)))
    pygame.image.save(sheet, os.path.join(out_dir, "_contact_sheet.png"))
    print(f"rendered {len(surfs)} levels + contact sheet -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
