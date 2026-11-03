"""Render the Blender-modelled creatures down to Crystal Caves pixel art.

    python scripts/render_blender_sprites.py                 # every creature
    python scripts/render_blender_sprites.py bat_enemy       # just one
    python scripts/render_blender_sprites.py --passes DIR    # convert existing passes

Blender (``ArtSource/Blender/creatures.py``) renders a flat slot/layer ID pass
and a studio-lit shading pass at 4x the final size. Each final pixel takes the
majority part of its 4x4 block; its brightness picks a tone from that slot's
palette ramp (parts behind the body sit one tone darker); an ink contour
separates nearer layers from farther ones; and a one-pixel ink outline wraps
the silhouette. The PNGs committed in ``ArtSource/Blender/sprites`` are what
the asset export uses, so building the game never needs Blender.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / "ArtSource/Blender"
SCENE = BLENDER / "creatures.py"
OUTPUT = BLENDER / "sprites"
SCALE = 4

sys.path.insert(0, str(ROOT))
from ArtSource.Blender.sprite_ids import FAR, all_ids  # noqa: E402
from src.unity_bridge.visual_materials import GOLD, INK, STEEL, WHITE  # noqa: E402

Color = tuple[int, int, int]
Part = tuple[str, int]  # (slot, layer)
Cell = tuple[Part, float] | None


@dataclass(frozen=True)
class Creature:
    size: tuple[int, int]
    frames: dict[str, int]  # output suffix -> Blender frame
    ramps: dict[str, tuple[Color, ...]]  # slot -> dark-to-light tones


EYE_YELLOW: Color = (255, 238, 97)
GREENS = ((24, 78, 26), (57, 144, 30), (85, 170, 0), (124, 203, 59), (174, 230, 108))
WALK = {str(frame): frame for frame in range(4)}

CREATURES: dict[str, Creature] = {
    "dinosaur_enemy": Creature(
        (24, 64),
        WALK,
        {
            "body": GREENS,
            "dark": ((11, 57, 38), (24, 78, 26)),
            "belly": ((110, 140, 40), (140, 178, 53), (174, 211, 84), (214, 236, 128)),
            "white": (WHITE,),
            "ink": (INK,),
            "accent": (GOLD[1], GOLD[2], GOLD[3]),
        },
    ),
    "bat_enemy": Creature(
        (24, 24),
        WALK,
        {
            # Readable against black masonry: violet membranes with lit finger bones.
            "body": ((22, 16, 40), (44, 30, 72), (74, 50, 112), (108, 76, 150)),
            "dark": ((14, 12, 28), (32, 26, 56), (60, 48, 96)),
            "belly": ((96, 78, 140), (150, 126, 196)),
            "eye": (EYE_YELLOW,),
            "white": (WHITE,),
        },
    ),
    "slug_enemy": Creature(
        (24, 32),
        WALK,
        {
            "body": ((86, 20, 60), (119, 29, 78), (208, 56, 108), (244, 96, 139), (255, 151, 166)),
            "belly": ((214, 92, 140), (240, 138, 172), (255, 182, 204)),
            "eye": (EYE_YELLOW,),
            "ink": (INK,),
            "tongue": ((242, 56, 68),),
        },
    ),
    "eye_flyer": Creature(
        (24, 24),
        WALK,
        {
            "body": ((24, 56, 30), (43, 88, 40), (95, 155, 52), (132, 185, 68), (186, 220, 99)),
            "belly": ((132, 185, 68), (186, 220, 99), (222, 245, 150)),
            "dark": ((24, 56, 30), (43, 88, 40)),
            "white": (WHITE,),
            "ink": (INK,),
        },
    ),
    "walking_rock": Creature(
        (24, 24),
        {**WALK, "sleep": 4},
        {
            "body": STEEL,
            "accent": ((34, 84, 48), (60, 130, 62), (110, 172, 82)),
            "belly": ((64, 24, 84), (112, 40, 140), (160, 72, 190)),
            "white": (WHITE,),
            "ink": (INK,),
        },
    ),
}

# Brightness quantiles that split a slot's pixels into its ramp's tones.
CUTS = {2: (0.5,), 3: (0.3, 0.75), 4: (0.15, 0.5, 0.85), 5: (0.10, 0.42, 0.80, 0.97)}


def srgb(linear: tuple[float, float, float]) -> Color:
    """The 8-bit value Blender writes for a linear scene color (Standard view)."""

    def encode(value: float) -> int:
        value = value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
        return round(max(0.0, min(1.0, value)) * 255)

    return (encode(linear[0]), encode(linear[1]), encode(linear[2]))


IDS: dict[Color, Part] = {srgb(color): (slot, layer) for slot, layer, color in all_ids()}


def pixel(image: Image.Image, point: tuple[int, int]) -> tuple[int, int, int, int]:
    """An RGBA pixel (the images are always converted to RGBA first)."""
    return cast(tuple[int, int, int, int], image.getpixel(point))


def part_of(rgb: Color) -> Part:
    key = min(IDS, key=lambda known: sum((a - b) ** 2 for a, b in zip(known, rgb)))
    return IDS[key]


def luminance(rgb: tuple[int, ...]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def sample(passes: Path, name: str, frame: int, creature: Creature) -> list[list[Cell]]:
    width, height = creature.size
    ids = Image.open(passes / f"{name}_{frame}_id.png").convert("RGBA")
    shade = Image.open(passes / f"{name}_{frame}_shade.png").convert("RGBA")
    if ids.size != (width * SCALE, height * SCALE) or shade.size != ids.size:
        raise ValueError(f"{name} frame {frame}: unexpected render size {ids.size}")
    grid: list[list[Cell]] = []
    for row in range(height):
        line: list[Cell] = []
        for col in range(width):
            votes: dict[Part, list[float]] = {}
            for dy in range(SCALE):
                for dx in range(SCALE):
                    point = (col * SCALE + dx, row * SCALE + dy)
                    r, g, b, a = pixel(ids, point)
                    if a >= 128:
                        votes.setdefault(part_of((r, g, b)), []).append(
                            luminance(pixel(shade, point))
                        )
            if sum(len(values) for values in votes.values()) * 2 < SCALE * SCALE:
                line.append(None)
                continue

            # Small features (eyes, teeth, claws) win ties so they survive downsampling.
            def weight(part: Part) -> int:
                return len(votes[part]) * (2 if len(creature.ramps[part[0]]) <= 2 else 1)

            part = max(votes, key=weight)
            values = votes[part]
            line.append((part, sum(values) / len(values)))
        grid.append(line)
    return grid


def smoothed_tone(grid: list[list[Cell]], tones: list[list[int]], col: int, row: int) -> int:
    """3x3 majority among same-slot neighbours: clean shading bands, no speckle."""
    cell = grid[row][col]
    assert cell is not None
    slot = cell[0][0]
    counts: dict[int, int] = {}
    for y in range(max(0, row - 1), min(len(grid), row + 2)):
        for x in range(max(0, col - 1), min(len(grid[0]), col + 2)):
            other = grid[y][x]
            if other is not None and other[0][0] == slot:
                counts[tones[y][x]] = counts.get(tones[y][x], 0) + (
                    2 if (x, y) == (col, row) else 1
                )
    return max(counts, key=lambda tone: (counts[tone], tone == tones[row][col]))


def convert_creature(passes: Path, output: Path, name: str, creature: Creature) -> list[Path]:
    width, height = creature.size
    grids = {
        suffix: sample(passes, name, frame, creature) for suffix, frame in creature.frames.items()
    }
    # Shared brightness cut points keep every pose's shading consistent.
    levels: dict[str, list[float]] = {}
    for grid in grids.values():
        for line in grid:
            for cell in line:
                if cell is not None:
                    levels.setdefault(cell[0][0], []).append(cell[1])
    cuts: dict[str, list[float]] = {}
    for slot, values in levels.items():
        ordered = sorted(values)
        fractions = CUTS.get(len(creature.ramps[slot]), ())
        cuts[slot] = [ordered[min(len(ordered) - 1, int(f * len(ordered)))] for f in fractions]
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, grid in grids.items():
        tones = [
            [0 if cell is None else sum(cell[1] > cut for cut in cuts[cell[0][0]]) for cell in line]
            for line in grid
        ]
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for row, line in enumerate(grid):
            for col, cell in enumerate(line):
                if cell is None:
                    continue
                (slot, layer), ramp = cell[0], creature.ramps[cell[0][0]]
                tone = smoothed_tone(grid, tones, col, row)
                if layer == FAR and len(ramp) > 2:
                    tone -= 1  # Parts behind the body recede.
                color = ramp[max(0, min(len(ramp) - 1, tone))]
                if any(
                    0 <= x < width
                    and 0 <= y < height
                    and (other := grid[y][x]) is not None
                    and other[0][1] > layer
                    and len(ramp) > 2
                    and len(creature.ramps[other[0][0]]) > 2
                    for x, y in ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1))
                ):
                    color = INK  # Contour where a nearer part overlaps this one.
                image.putpixel((col, row), (*color, 255))
        outlined = image.copy()
        for row in range(height):
            for col in range(width):
                if pixel(image, (col, row))[3]:
                    continue
                neighbours = ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1))
                if any(
                    0 <= x < width and 0 <= y < height and pixel(image, (x, y))[3]
                    for x, y in neighbours
                ):
                    outlined.putpixel((col, row), (*INK, 255))
        path = output / f"{name}_{suffix}.png"
        outlined.save(path)
        written.append(path)
    return written


def render(directory: Path, names: list[str]) -> None:
    blender = shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"
    command = [blender, "--background", "--factory-startup", "--python", str(SCENE)]
    subprocess.run([*command, "--", str(directory), *names], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help=f"creatures to render: {', '.join(CREATURES)}")
    parser.add_argument("--passes", type=Path, help="convert existing Blender passes")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    names = args.names or list(CREATURES)
    unknown = sorted(set(names) - set(CREATURES))
    if unknown:
        parser.error(f"unknown creature(s): {', '.join(unknown)}")

    def convert(passes: Path) -> list[Path]:
        return [
            path
            for name in names
            for path in convert_creature(passes, args.output, name, CREATURES[name])
        ]

    if args.passes:
        paths = convert(args.passes)
    else:
        with tempfile.TemporaryDirectory(prefix="crystal-caves-blender-") as directory:
            render(Path(directory), names)
            paths = convert(Path(directory))
    for path in paths:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
