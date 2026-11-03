"""Render Blender-modelled creatures down to Crystal Caves pixel art.

    python scripts/render_blender_sprites.py            # re-render with Blender, then convert
    python scripts/render_blender_sprites.py --passes D # convert existing passes in D

Blender renders a flat material-ID pass and a studio-lit shading pass at 4x the
final size (``ArtSource/Blender/dinosaur_scene.py``). Each final pixel takes the
majority material of its 4x4 block, its brightness picks a tone from that
material's palette ramp, and a one-pixel ink outline is added around the
silhouette. The committed PNGs in ``ArtSource/Blender/sprites`` are what the
asset export uses, so building the game never needs Blender.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "ArtSource/Blender/dinosaur_scene.py"
OUTPUT = ROOT / "ArtSource/Blender/sprites"
SCALE, WIDTH, HEIGHT, FRAMES = 4, 24, 64, 4

sys.path.insert(0, str(ROOT))
from src.unity_bridge.visual_materials import GOLD, INK, WHITE  # noqa: E402

Color = tuple[int, int, int]


def srgb(linear: tuple[float, float, float]) -> Color:
    """The 8-bit value Blender writes for a linear scene color (Standard view)."""

    def encode(value: float) -> int:
        value = value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055
        return round(max(0.0, min(1.0, value)) * 255)

    return (encode(linear[0]), encode(linear[1]), encode(linear[2]))


# Material ID colors from the Blender scene, each mapped to a palette ramp
# (dark to light). Greens match the hand-drawn creature palette.
RAMPS: dict[Color, tuple[Color, ...]] = {
    srgb((0.10, 0.80, 0.10)): (
        (24, 78, 26),
        (57, 144, 30),
        (85, 170, 0),
        (124, 203, 59),
        (174, 230, 108),
    ),
    srgb((0.10, 0.40, 0.10)): ((11, 57, 38), (24, 78, 26), (57, 144, 30)),
    srgb((0.90, 0.90, 0.10)): ((110, 140, 40), (140, 178, 53), (174, 211, 84)),
    srgb((1.0, 1.0, 1.0)): (WHITE,),
    srgb((0.0, 0.0, 1.0)): (INK,),
    srgb((1.0, 0.5, 0.0)): (GOLD[4],),
}


def nearest_material(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return min(RAMPS, key=lambda key: sum((a - b) ** 2 for a, b in zip(key, rgb)))


def pixel(image: Image.Image, point: tuple[int, int]) -> tuple[int, int, int, int]:
    """An RGBA pixel (the images are always converted to RGBA first)."""
    return cast(tuple[int, int, int, int], image.getpixel(point))


def luminance(rgb: tuple[int, ...]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def sample(passes: Path, frame: int) -> list[list[tuple[tuple[int, int, int], float] | None]]:
    ids = Image.open(passes / f"dinosaur_{frame}_id.png").convert("RGBA")
    shade = Image.open(passes / f"dinosaur_{frame}_shade.png").convert("RGBA")
    if ids.size != (WIDTH * SCALE, HEIGHT * SCALE) or shade.size != ids.size:
        raise ValueError(f"frame {frame}: unexpected render size {ids.size}")
    grid: list[list[tuple[tuple[int, int, int], float] | None]] = []
    for row in range(HEIGHT):
        line: list[tuple[tuple[int, int, int], float] | None] = []
        for col in range(WIDTH):
            votes: dict[tuple[int, int, int], list[float]] = {}
            for dy in range(SCALE):
                for dx in range(SCALE):
                    point = (col * SCALE + dx, row * SCALE + dy)
                    r, g, b, a = pixel(ids, point)
                    if a >= 128:
                        votes.setdefault(nearest_material((r, g, b)), []).append(
                            luminance(pixel(shade, point))
                        )
            opaque = sum(len(values) for values in votes.values())
            if opaque * 2 < SCALE * SCALE:
                line.append(None)
                continue
            # Small features (eyes, teeth, claws) win ties so they survive downsampling.
            material = max(
                votes, key=lambda key: len(votes[key]) * (2 if len(RAMPS[key]) == 1 else 1)
            )
            values = votes[material]
            line.append((material, sum(values) / len(values)))
        grid.append(line)
    return grid


def convert(passes: Path, output: Path) -> list[Path]:
    grids = [sample(passes, frame) for frame in range(FRAMES)]
    # Shared brightness cut points keep the four poses' shading consistent.
    levels: dict[tuple[int, int, int], list[float]] = {}
    for grid in grids:
        for line in grid:
            for cell in line:
                if cell is not None:
                    levels.setdefault(cell[0], []).append(cell[1])
    cuts: dict[tuple[int, int, int], list[float]] = {}
    for material, values in levels.items():
        ordered = sorted(values)
        count = len(RAMPS[material])
        fractions = {5: (0.10, 0.42, 0.80, 0.97), 3: (0.30, 0.75)}.get(count, ())
        cuts[material] = [ordered[min(len(ordered) - 1, int(f * len(ordered)))] for f in fractions]
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for frame, grid in enumerate(grids):
        tones = [
            [None if cell is None else sum(cell[1] > cut for cut in cuts[cell[0]]) for cell in line]
            for line in grid
        ]
        image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        for row, line in enumerate(grid):
            for col, cell in enumerate(line):
                if cell is None:
                    continue
                material = cell[0]
                tone = smoothed_tone(grid, tones, col, row)
                image.putpixel((col, row), (*RAMPS[material][tone], 255))
        outlined = image.copy()
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if pixel(image, (col, row))[3]:
                    continue
                neighbours = ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1))
                if any(
                    0 <= x < WIDTH and 0 <= y < HEIGHT and pixel(image, (x, y))[3]
                    for x, y in neighbours
                ):
                    outlined.putpixel((col, row), (*INK, 255))
        path = output / f"dinosaur_{frame}.png"
        outlined.save(path)
        written.append(path)
    return written


def smoothed_tone(grid: list, tones: list, col: int, row: int) -> int:
    """3x3 majority among same-material neighbours: clean shading bands, no speckle."""
    material = grid[row][col][0]
    counts: dict[int, int] = {}
    for y in range(max(0, row - 1), min(HEIGHT, row + 2)):
        for x in range(max(0, col - 1), min(WIDTH, col + 2)):
            cell = grid[y][x]
            if cell is not None and cell[0] == material:
                tone = tones[y][x]
                counts[tone] = counts.get(tone, 0) + (2 if (x, y) == (col, row) else 1)
    return max(counts, key=lambda tone: (counts[tone], tone == tones[row][col]))


def render(directory: Path) -> None:
    blender = shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"
    subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python",
            str(SCENE),
            "--",
            str(directory),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=Path, help="convert existing Blender passes")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.passes:
        paths = convert(args.passes, args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="crystal-caves-blender-") as directory:
            render(Path(directory))
            paths = convert(Path(directory), args.output)
    for path in paths:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
