"""Material-ID colors shared by the Blender scene and the pixel converter.

Plain Python with no Blender import, so both ``creatures.py`` (inside Blender)
and ``scripts/render_blender_sprites.py`` (outside it) read the same table.

Every modelled part has a *slot* (which palette ramp colors it) and a *layer*
(0 behind the body, 1 the body, 2 in front of it). The flat ID pass encodes
both; the converter draws an ink contour wherever a nearer layer overlaps a
farther one, the way a pixel artist separates a near leg from the torso.
"""

from __future__ import annotations

SLOTS = ("body", "dark", "belly", "white", "ink", "accent", "eye", "tongue")
LAYERS = (0, 1, 2)
FAR, BODY, NEAR = LAYERS


def id_color(slot: str, layer: int = BODY) -> tuple[float, float, float]:
    """A linear-light color unique to one (slot, layer) pair."""
    index = SLOTS.index(slot)
    return ((index % 4 + 1) / 5, (index // 4) * 0.5 + 0.2, layer * 0.35 + 0.1)


def all_ids() -> list[tuple[str, int, tuple[float, float, float]]]:
    return [(slot, layer, id_color(slot, layer)) for slot in SLOTS for layer in LAYERS]
