"""Render the Blender-modelled creatures down to Crystal Caves pixel art.

    python scripts/render_blender_sprites.py                 # every creature
    python scripts/render_blender_sprites.py bat_enemy       # just one
    python scripts/render_blender_sprites.py --passes DIR    # convert existing passes

Blender (``ArtSource/Blender/creatures.py``) renders a flat slot/layer ID pass
and a studio-lit shading pass at 8x the final size, for two outputs:

- ``sprites/``: native pixel art. Each pixel takes the majority part of its
  8x8 block; its brightness picks a tone from that slot's palette ramp (parts
  behind the body sit one tone darker); an ink contour separates nearer layers
  from farther ones; a one-pixel ink outline wraps the silhouette.
- ``sprites_hd/``: 4x HD art with HD-only details, the same palette ramps
  blended smoothly, half-pixel contours and antialiased edges.

The committed PNGs are what the asset export uses, so building the game never
needs Blender.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BLENDER = ROOT / "ArtSource/Blender"
SCENE = BLENDER / "creatures.py"
OUTPUT = BLENDER / "sprites"
HD_OUTPUT = BLENDER / "sprites_hd"
SCALE = 8  # render pixels per sprite pixel (HD sprites are 4x)

sys.path.insert(0, str(ROOT))
from ArtSource.Blender.sprite_ids import BODY, FAR, SLOTS, all_ids  # noqa: E402
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
            "tongue": ((150, 70, 110), (210, 120, 160)),
        },
    ),
    "slug_enemy": Creature(
        (24, 32),
        WALK,
        {
            "body": ((86, 20, 60), (119, 29, 78), (208, 56, 108), (244, 96, 139), (255, 151, 166)),
            "dark": ((120, 26, 76), (160, 40, 100)),
            "white": (WHITE,),
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


# Every (slot, layer) pair as an index, in a fixed order.
PARTS: list[Part] = list(IDS.values())
ID_COLORS = np.array(list(IDS), dtype=np.int32)


def load(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    if image.size != size:
        raise ValueError(f"{path.name}: expected {size}, rendered {image.size}")
    return np.asarray(image, dtype=np.int32)


def decode(passes: Path, stem: str, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per render pixel: part index (-1 where empty) and studio-lit brightness."""
    ids = load(passes / f"{stem}_id.png", size)
    shade = load(passes / f"{stem}_shade.png", size)
    distance = ((ids[..., None, :3] - ID_COLORS[None, None]) ** 2).sum(axis=-1)
    parts = np.where(ids[..., 3] >= 128, distance.argmin(axis=-1), -1)
    brightness = shade[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    return parts, brightness


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by a disc of the given radius."""
    grown = mask.copy()
    padded = np.pad(mask, radius)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                grown |= padded[
                    radius + dy : radius + dy + height, radius + dx : radius + dx + width
                ]
    return grown


def sample(passes: Path, name: str, frame: int, creature: Creature) -> list[list[Cell]]:
    width, height = creature.size
    parts, brightness = decode(passes, f"{name}_{frame}", (width * SCALE, height * SCALE))
    grid: list[list[Cell]] = []
    for row in range(height):
        line: list[Cell] = []
        for col in range(width):
            block = np.s_[row * SCALE : (row + 1) * SCALE, col * SCALE : (col + 1) * SCALE]
            ids, values = parts[block].ravel(), brightness[block].ravel()
            if (ids >= 0).sum() * 2 < SCALE * SCALE:
                line.append(None)
                continue

            # Small features (eyes, teeth, claws) win ties so they survive downsampling.
            def weight(index: int) -> int:
                count = int((ids == index).sum())
                return count * (2 if len(creature.ramps[PARTS[index][0]]) <= 2 else 1)

            index = max((int(i) for i in np.unique(ids[ids >= 0])), key=weight)
            line.append((PARTS[index], float(values[ids == index].mean())))
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
    """The native-resolution palette sprite: one Blender part per game pixel."""
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
        image = np.zeros((height, width, 4), dtype=np.uint8)
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
                image[row, col] = (*color, 255)
        opaque = image[..., 3] > 0
        ring = dilate(opaque, 1) & ~opaque
        image[ring] = (*INK, 255)
        path = output / f"{name}_{suffix}.png"
        Image.fromarray(image).save(path)
        written.append(path)
    return written


def convert_hd(passes: Path, output: Path, name: str, creature: Creature) -> list[Path]:
    """The HD sprite: 4x detail with smooth palette-mapped shading.

    Shading keeps the same palette ramps as the pixel sprite but blends between
    tones continuously. Contours and the outline are about half a game pixel,
    and the 8x render is box-filtered to 4x for clean antialiased edges.
    """
    width, height = creature.size
    size = (width * SCALE, height * SCALE)
    decoded = {
        suffix: decode(passes, f"{name}_{frame}_hd", size)
        for suffix, frame in creature.frames.items()
    }
    slot_of = np.array([SLOTS.index(slot) for slot, _ in PARTS])
    layer_of = np.array([layer for _, layer in PARTS])
    # Brightness rank within each slot, shared by every pose.
    ranked: dict[int, np.ndarray] = {}
    for slot_index in range(len(SLOTS)):
        values = [b[(p >= 0) & (slot_of[p] == slot_index)] for p, b in decoded.values()]
        ranked[slot_index] = np.sort(np.concatenate(values))
    output.mkdir(parents=True, exist_ok=True)
    written = []
    variants = [("", False)] + ([("_hit", True)] if name == "dinosaur_enemy" else [])
    for suffix, (parts, brightness) in decoded.items():
        filled = parts >= 0
        slots = np.where(filled, slot_of[np.maximum(parts, 0)], -1)
        layers = np.where(filled, layer_of[np.maximum(parts, 0)], -1)
        multi = np.isin(slots, [SLOTS.index(s) for s, r in creature.ramps.items() if len(r) > 2])
        missing = {SLOTS[i] for i in np.unique(slots[filled])} - set(creature.ramps)
        if missing:
            raise ValueError(f"{name}: no palette ramp for slot(s) {sorted(missing)}")
        rgb = np.zeros((*parts.shape, 3))
        for slot, ramp in creature.ramps.items():
            mask = slots == SLOTS.index(slot)
            if not mask.any():
                continue
            tones = np.array(ramp, dtype=float)
            order = ranked[SLOTS.index(slot)]
            position = np.searchsorted(order, brightness[mask]) / max(1, len(order))
            # Parts behind the body recede by most of one tone.
            position = position - np.where(layers[mask] == FAR, 0.8 / max(1, len(ramp) - 1), 0)
            scaled = np.clip(position, 0, 1) * (len(ramp) - 1)
            low = np.floor(scaled).astype(int)
            high = np.minimum(low + 1, len(ramp) - 1)
            blend = (scaled - low)[:, None]
            rgb[mask] = tones[low] * (1 - blend) + tones[high] * blend
        ink = np.zeros(parts.shape, dtype=bool)
        for layer in (FAR, BODY):
            nearer = dilate(multi & (layers > layer), 2)
            ink |= multi & (layers == layer) & nearer
        ring = dilate(filled, 4) & ~filled
        for variant, hit in variants:
            color = rgb.copy()
            if hit:
                color[filled & (slots != SLOTS.index("ink"))] = WHITE
            color[ink | ring] = INK
            alpha = (filled | ring).astype(float)
            # 2x2 box filter with premultiplied alpha: 8x render -> 4x sprite.
            premultiplied = np.concatenate([color * alpha[..., None], alpha[..., None]], axis=-1)
            h, w = parts.shape
            reduced = premultiplied.reshape(h // 2, 2, w // 2, 2, 4).mean(axis=(1, 3))
            coverage = reduced[..., 3:]
            out = np.where(coverage > 0, reduced[..., :3] / np.maximum(coverage, 1e-6), 0)
            image = np.concatenate([out, coverage * 255], axis=-1).round().astype(np.uint8)
            stem = f"{name}{variant}_{suffix}"
            path = output / f"{stem}.png"
            Image.fromarray(image).save(path)
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
    parser.add_argument("--hd-output", type=Path, default=HD_OUTPUT)
    args = parser.parse_args()
    names = args.names or list(CREATURES)
    unknown = sorted(set(names) - set(CREATURES))
    if unknown:
        parser.error(f"unknown creature(s): {', '.join(unknown)}")

    def convert(passes: Path) -> list[Path]:
        return [
            path
            for name in names
            for path in [
                *convert_creature(passes, args.output, name, CREATURES[name]),
                *convert_hd(passes, args.hd_output, name, CREATURES[name]),
            ]
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
