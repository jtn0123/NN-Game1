"""Semantic decoder for the original Crystal Caves Episode 1 level bytes.

Decodes ``experiments/cc_status/data/cc1_levels_raw.json`` (40-byte rows per
level, length byte already stripped) into per-tile semantic categories using
the authoritative byte->object mapping documented on ModdingWiki
("Crystal Caves Map Format") and implemented in Camoto libgamemaps
(fmt-map-ccaves-mapping.hpp). This replaces the earlier best-effort decode
(commit 2d4134b) which misread the I-beam/continuation codes as ladders/air
and did not know the concrete terrain family, garbling 6 of 16 levels.

Key format facts (verified against the raw bytes):
- 0x20 is empty air; 0x6E is NOT an object but a continuation cell ("NEXT")
  claimed by a multi-cell object (I-beam right ends, 2x2 exit, 4x4 signs,
  vine columns, door bottoms, ...).
- 0x5B starts a two-byte sign code (next byte selects the sign).
- 0x57 followed by 0x4C/0x52 is an exhaust sucker; other bytes after 0x57 are
  not consumed.
- Vine/chain codes (0x85-0x88) sit at the BOTTOM of a climbable column whose
  body cells above are NEXT bytes — the original's only climbables are these
  vines/chains; there are no ladders in the 1991 format.

Output: per-level ground-truth stats (gems incl. hidden, enemies, hazards,
switches/doors, platforms, vines, pickups, terrain density) plus optional
ASCII previews. No original layout is written into the repo docs — stats only.

Run:  python -m experiments.cc_status.cc1_decode [--ascii L6] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "cc1_levels_raw.json")

EMPTY = 0x20
NEXT = 0x6E
SIGN = 0x5B
SUCKER = 0x57

# byte -> (category, name). Categories drive the stats rollup.
CODES: dict[int, tuple[str, str]] = {
    0x59: ("player", "player start"),
    0x58: ("exit", "level exit (2x2)"),
    0x78: ("entrance", "level entrance door"),
    # gems (win condition: collect every gem)
    0x52: ("gem", "red gem"),
    0x2B: ("gem", "yellow gem"),
    0x62: ("gem", "green gem"),
    0x63: ("gem", "blue gem"),
    0x98: ("hidden_gem", "hidden gem in I-beam left"),
    0x99: ("hidden_gem", "hidden gem in I-beam mid"),
    0x9A: ("hidden_gem", "hidden gem in I-beam right"),
    # enemies
    0x23: ("enemy", "spider"),
    0x26: ("enemy", "robot"),
    0x2A: ("enemy", "brown walking ball"),
    0x2F: ("enemy", "flying bone"),
    0x3D: ("enemy", "purple wall enemy (left)"),
    0x45: ("enemy", "purple wall enemy (right)"),
    0x3F: ("enemy", "green stripy"),
    0x41: ("enemy", "green fish"),
    0x4D: ("enemy", "emu"),
    0x53: ("enemy", "purple snake"),
    0x5E: ("enemy", "bird"),
    0x6F: ("enemy", "dormant walking ball"),
    0x7E: ("enemy", "bat"),
    0xF2: ("enemy", "dinosaur"),
    0xF3: ("enemy", "blue ball"),
    0x39: ("enemy", "mine cart"),
    0x40: ("enemy", "tornado"),
    # hazards
    0x28: ("hazard", "stalactites 1"),
    0x29: ("hazard", "stalactites 2"),
    0x7C: ("hazard", "dropping stalactite"),
    0xF6: ("hazard", "stalagmite 1"),
    0xF7: ("hazard", "stalagmite 2"),
    0x49: ("hazard", "popup floor spike"),
    0x46: ("hazard", "flame"),
    0x4A: ("hazard", "flame tower"),
    0xD5: ("hazard", "flame (2)"),
    0x61: ("hazard", "laser left, vert-moving"),
    0x71: ("hazard", "laser left, static"),
    0x73: ("hazard", "laser right, vert-moving"),
    0x77: ("hazard", "laser right, static"),
    0x81: ("hazard", "laser left, vert-moving, switched"),
    0x82: ("hazard", "laser left, static, switched"),
    0x83: ("hazard", "laser right, vert-moving, switched"),
    0x84: ("hazard", "laser right, static, switched"),
    0x54: ("hazard", "hammer guide"),
    0x55: ("hazard", "crusher hammer"),
    0xFA: ("hazard", "leaking barrel"),
    0x75: ("hazard", "volcano eruption"),
    0x8E: ("hazard", "volcano top"),
    0x8F: ("hazard", "volcano bottom (4x1)"),
    # mushrooms: the touch-deadly pickups family in CC1
    0xAA: ("mushroom", "blue mushroom"),
    0xAB: ("mushroom", "red mushroom"),
    0xAC: ("mushroom", "green mushroom"),
    0xBB: ("mushroom", "purple mushroom"),
    # switches / doors / air
    0xA0: ("switch", "red switch"),
    0xA1: ("switch", "green switch"),
    0xA2: ("switch", "blue switch"),
    0x76: ("switch", "horizontal switch, off"),
    0xD8: ("switch", "horizontal switch, on"),
    0xA6: ("light_switch", "light switch (dark level)"),
    0xA3: ("door", "red door"),
    0xA4: ("door", "green door"),
    0xA5: ("door", "blue door"),
    0x24: ("air", "air compressor"),
    # pickups / treasure
    0x47: ("pickup", "gun/ammo"),
    0x5D: ("pickup", "P powerup"),
    0x8C: ("pickup", "G powerup (gravity)"),
    0x69: ("pickup", "stop sign (freeze)"),
    0xA7: ("pickup", "treasure chest"),
    0xA8: ("pickup", "chest key"),
    0xA9: ("pickup", "egg"),
    0xF4: ("pickup", "pick"),
    0xF5: ("pickup", "shovel"),
    0x8B: ("pickup", "candle"),
    0x91: ("pickup", "slime w/ chunks"),
    0x92: ("pickup", "slime w/ bones"),
    0x93: ("pickup", "slime w/ helmet"),
    0xFC: ("pickup", "slime w/ 3 chunks"),
    0xFD: ("pickup", "slime w/ 2 chunks"),
    0xFE: ("pickup", "slime w/ 2 bones"),
    # moving platforms
    0x48: ("moving_platform", "horiz platform, always on"),
    0x56: ("moving_platform", "vert platform"),
    0xCD: ("moving_platform", "horiz platform, switched"),
    0xD6: ("moving_platform", "vert platform, stationary"),
    0xD7: ("moving_platform", "vert platform, switched"),
    0x4E: ("moving_platform", "moon"),
    0x6D: ("moving_platform", "earth"),
    # climbables (the only ones in the 1991 format — no ladders)
    0x85: ("vine", "hanging chain w/ hook"),
    0x86: ("vine", "hanging double-chain"),
    0x87: ("vine", "purple vine"),
    0x88: ("vine", "green vine"),
    0xC3: ("decor", "red vine top (decor)"),
    0xC4: ("decor", "red vine mid+bottom (decor)"),
    # solid terrain (3x3 block-piece family, colour varies per level)
    0x72: ("solid", "block TL"),
    0x74: ("solid", "block TM"),
    0x79: ("solid", "block TR"),
    0x66: ("solid", "block BL"),
    0x67: ("solid", "block BM"),
    0x68: ("solid", "block BR"),
    0x34: ("solid", "block ML"),
    0x35: ("solid", "block MM"),
    0x36: ("solid", "block MR"),
    0x43: ("solid", "random concrete"),
    0x4B: ("solid", "concrete 0"),
    0x4C: ("solid", "concrete 1"),
    0x6B: ("solid", "concrete 2"),
    0x6C: ("solid", "concrete 3"),
    0x42: ("solid", "ice block"),
    0xB0: ("solid", "hidden block (head-butt)"),
    0xBD: ("solid", "\\ ledge"),
    0xBE: ("solid", "/ ledge"),
    # thin walkable platforms
    0x44: ("platform", "I-beam left"),
    0x64: ("platform", "I-beam mid"),
    0x5F: ("platform", "underscore platform"),
    0x94: ("platform", "golden handrail left"),
    0x95: ("platform", "golden handrail mid+right"),
    0x96: ("platform", "wooden handrail left"),
    0x97: ("platform", "wooden handrail mid+right"),
    # misc codes confirmed from the raw bytes
    0x32: ("decor", "first-row background filler"),
    0x6A: ("solid", "inverted rubble pile, mid"),
    0x70: ("solid", "inverted rubble pile, left+end"),
    0x80: ("decor", "sector alpha sign (2x2)"),
    # decor / background
    0x21: ("decor", "blue dripping pipe"),
    0x22: ("decor", "green hanging stuff 2"),
    0x3A: ("decor", "green hanging stuff 1"),
    0x25: ("platform", "green pipe vert"),
    0x2C: ("platform", "green pipe join"),
    0x2D: ("platform", "green pipe horiz"),
    0x2E: ("platform", "green pipe join"),
    0xBF: ("platform", "green pipe join"),
    0xC0: ("platform", "green pipe join"),
    0xC1: ("platform", "green pipe join"),
    0xC2: ("platform", "green pipe join"),
    0xC5: ("platform", "green pipe cross"),
    0xCF: ("platform", "green pipe exit"),
    0xD1: ("platform", "green pipe exit"),
    0xD9: ("platform", "green pipe join"),
    0xDA: ("platform", "green pipe join"),
    0x30: ("decor", "large chain end"),
    0x38: ("decor", "large chain"),
    0x5A: ("decor", "horizon/hill/light"),
    0x7A: ("solid", "invisible blocking tile"),
    0x89: ("decor", "tear revealing horiz bar"),
    0x8A: ("decor", "tear revealing vert bar"),
    0x90: ("decor", "funnel tube stem"),
    0x9F: ("decor", "large fan blades"),
    0xB1: ("decor", "thick horiz wooden post"),
    0xB2: ("decor", "thick vert wooden post"),
    0xB3: ("decor", "thin wooden post"),
    0xBA: ("decor", "metal support mid"),
    0xCA: ("decor", "metal support bottom"),
    0xCB: ("decor", "metal support top"),
    0xBC: ("decor", "tuft of grass"),
    0xC6: ("decor", "down arrow"),
    0xC7: ("decor", "up arrow"),
    0xC8: ("decor", "fish barrier"),
    0xCC: ("decor", "ceiling lump"),
    0xCE: ("decor", "low grass"),
    0xD0: ("decor", "control panel"),
    0xDB: ("decor", "earth (intro)"),
    0xDC: ("decor", "moon (intro)"),
    0xE0: ("decor", "funnel machine"),
    0xE7: ("decor", "thick purple post"),
    0xE8: ("platform", "corrugated pipe"),
    0xE9: ("platform", "corrugated pipe"),
    0xEA: ("platform", "corrugated pipe"),
    0xEB: ("platform", "corrugated pipe"),
    0xEC: ("platform", "corrugated pipe"),
    0xED: ("platform", "corrugated pipe"),
    0xF0: ("decor", "wooden Y beam"),
    0xF8: ("decor", "round glass thing"),
    0xF9: ("decor", "clean barrel"),
    0xFB: ("decor", "exploded barrel"),
}

# Multi-cell footprints (relative cells beyond the code cell that the object
# claims when those cells hold NEXT). 2x2 arrays in Camoto order.
FOOTPRINT_2X2 = {0x58, 0x55, 0x80, 0x9F, 0xE0, 0xF8}
FOOTPRINT_RIGHT1 = {0x44, 0x64, 0x98, 0x99, 0x6A, 0x70, 0x95, 0x97, 0x54, 0x75}
FOOTPRINT_BELOW1 = {0xA3, 0xA4, 0xA5, 0x24, 0x69, 0xC4}

ASCII_GLYPH = {
    "empty": " ",
    "solid": "#",
    "platform": "=",
    "gem": "*",
    "hidden_gem": "+",
    "enemy": "E",
    "hazard": "!",
    "mushroom": "m",
    "switch": "s",
    "light_switch": "L",
    "door": "D",
    "air": "A",
    "pickup": "$",
    "moving_platform": "~",
    "vine": "|",
    "player": "P",
    "exit": "X",
    "entrance": "e",
    "decor": ".",
    "sign": ".",
    "unknown": "?",
}


def _trim_trailing_junk(rows: list[list[int]]) -> tuple[list[list[int]], int]:
    """Drop trailing rows that are mostly bytes outside the documented format.

    The raw extraction for L16 overran the level end by one row of non-map
    bytes (0x00-0x0D etc.); a real map row never contains those.
    """
    known = set(CODES) | {EMPTY, NEXT, SIGN, SUCKER}
    trimmed = 0
    while rows and sum(1 for b in rows[-1] if b not in known) >= 5:
        rows = rows[:-1]
        trimmed += 1
    return rows, trimmed


def decode_level(rows: list[list[int]]) -> dict[str, Any]:
    """Decode one level's raw byte rows into a category grid + object counts."""
    rows, trimmed_rows = _trim_trailing_junk(rows)
    height = len(rows)
    width = len(rows[0])
    grid = [["empty"] * width for _ in range(height)]
    objects: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    unknown: Counter[int] = Counter()

    # Pass 1: assign categories for direct codes; remember NEXT cells.
    next_cells: set[tuple[int, int]] = set()
    sign_cells: set[tuple[int, int]] = set()
    y = 0
    for y in range(height):
        x = 0
        while x < width:
            b = rows[y][x]
            if b == EMPTY:
                x += 1
                continue
            if b == NEXT:
                next_cells.add((y, x))
                x += 1
                continue
            if b == SIGN and x + 1 < width:
                grid[y][x] = "sign"
                sign_cells.add((y, x + 1))
                grid[y][x + 1] = "sign"
                objects["sign"] += 1
                by_category["sign"] += 1
                x += 2
                continue
            if b == SUCKER and x + 1 < width and rows[y][x + 1] in (0x4C, 0x52):
                grid[y][x] = "hazard"
                grid[y][x + 1] = "hazard"
                objects["exhaust sucker"] += 1
                by_category["hazard"] += 1
                x += 2
                continue
            cat_name = CODES.get(b)
            if cat_name is None:
                unknown[b] += 1
                grid[y][x] = "unknown"
                x += 1
                continue
            cat, name = cat_name
            grid[y][x] = cat
            objects[name] += 1
            by_category[cat] += 1
            x += 1

    # Pass 2: resolve NEXT cells to the object that claims them.
    def claim(y: int, x: int, cat: str) -> None:
        if (y, x) in next_cells:
            grid[y][x] = cat
            next_cells.discard((y, x))

    for y in range(height):
        for x in range(width):
            b = rows[y][x]
            cat = grid[y][x]
            if b in FOOTPRINT_2X2:
                for dy, dx in ((0, 1), (1, 0), (1, 1)):
                    if y + dy < height and x + dx < width:
                        claim(y + dy, x + dx, cat)
            elif b in FOOTPRINT_RIGHT1 and x + 1 < width:
                claim(y, x + 1, cat)
            elif b in FOOTPRINT_BELOW1 and y + 1 < height:
                claim(y + 1, x, cat)
            elif b in (0x85, 0x86, 0x87, 0x88):
                # vine body extends UP through NEXT cells
                y2 = y - 1
                while y2 >= 0 and (y2, x) in next_cells:
                    claim(y2, x, "vine")
                    y2 -= 1

    # Remaining NEXT cells: continuation of an adjacent-left object (I-beam
    # chains longer than 2, sign bodies) — inherit left neighbour's category.
    for y, x in sorted(next_cells):
        grid[y][x] = grid[y][x - 1] if x > 0 else "decor"

    solid = sum(row.count("solid") for row in grid)
    platform = sum(row.count("platform") for row in grid)
    non_empty = sum(width - row.count("empty") for row in grid)
    return {
        "height": height,
        "width": width,
        "trimmed_junk_rows": trimmed_rows,
        "grid": grid,
        "objects": dict(objects),
        "by_category": dict(by_category),
        "unknown_bytes": {f"0x{b:02X}": n for b, n in sorted(unknown.items())},
        "stats": {
            "gems_visible": by_category.get("gem", 0),
            "gems_hidden": by_category.get("hidden_gem", 0),
            "gems_total": by_category.get("gem", 0) + by_category.get("hidden_gem", 0),
            "enemies": by_category.get("enemy", 0),
            "hazards": by_category.get("hazard", 0),
            "mushrooms": by_category.get("mushroom", 0),
            "switches": by_category.get("switch", 0),
            "doors": by_category.get("door", 0),
            "moving_platforms": by_category.get("moving_platform", 0),
            "vines": by_category.get("vine", 0),
            "pickups": by_category.get("pickup", 0),
            "dark_level": by_category.get("light_switch", 0) > 0,
            "solid_tiles": solid,
            "platform_tiles": platform,
            "non_empty_tiles": non_empty,
            "fill_frac": round(non_empty / (width * height), 3),
            "solid_frac": round((solid + platform) / (width * height), 3),
        },
    }


def ascii_render(decoded: dict[str, Any]) -> str:
    return "\n".join("".join(ASCII_GLYPH.get(c, "?") for c in row) for row in decoded["grid"])


def load_raw() -> dict[str, list[list[int]]]:
    with open(RAW_PATH, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascii", metavar="LEVEL", help="print ASCII preview of one level")
    parser.add_argument("--json", metavar="PATH", help="write full decoded stats JSON")
    args = parser.parse_args(argv)

    raw = load_raw()
    levels = {k: v for k, v in raw.items() if k.startswith("L")}
    decoded = {name: decode_level(rows) for name, rows in levels.items()}

    if args.ascii:
        if args.ascii not in decoded:
            print(f"unknown level {args.ascii}; have {sorted(decoded)}")
            return 1
        print(ascii_render(decoded[args.ascii]))
        return 0

    header = (
        f"{'lvl':4} {'size':7} {'gems':>4} {'hid':>3} {'enem':>4} {'haz':>3} "
        f"{'mush':>4} {'sw':>2} {'door':>4} {'mvpl':>4} {'vine':>4} {'pick':>4} "
        f"{'dark':>4} {'solid%':>6} {'fill%':>5} {'unk':>3}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(decoded, key=lambda n: int(n[1:])):
        s = decoded[name]["stats"]
        unk = sum(decoded[name]["unknown_bytes"].values())
        print(
            f"{name:4} {s['solid_tiles'] and ''or''}{decoded[name]['width']}x{decoded[name]['height']:<4} "
            f"{s['gems_visible']:>4} {s['gems_hidden']:>3} {s['enemies']:>4} {s['hazards']:>3} "
            f"{s['mushrooms']:>4} {s['switches']:>2} {s['doors']:>4} {s['moving_platforms']:>4} "
            f"{s['vines']:>4} {s['pickups']:>4} "
            f"{str(s['dark_level'])[:1]:>4} {s['solid_frac']:>6} {s['fill_frac']:>5} {unk:>3}"
        )
        if decoded[name]["unknown_bytes"]:
            print(f"     unknown: {decoded[name]['unknown_bytes']}")

    if args.json:
        slim = {name: {k: v for k, v in d.items() if k != "grid"} for name, d in decoded.items()}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
