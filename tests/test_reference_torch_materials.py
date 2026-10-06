"""A cool wall socket keeps the published torch flame poses and footprint."""

import hashlib

import numpy as np
import pygame
import pytest

from src.unity_bridge.visual_fidelity import torch

# Region digests from the sprites copied before the fourth reference series.
BASELINE_ALPHA = (
    "3b084ddad368d69a20587353b1417448db1657ea31f07a34a92afbb394156960",
    "d3ecead0bd9028f7946e9750e937a4a00a29cc906011423513908b194096745f",
    "f3bfe763c1edd22115e7894aa6a268a899babaa39ce168d0423ae64b8175f65d",
    "3b084ddad368d69a20587353b1417448db1657ea31f07a34a92afbb394156960",
)
BASELINE_FLAMES = (
    "b1520b210311136588ed712844980517cb4998df0fb685bdaac731f1743b4e50",
    "a3c48c93ba7d3b570080464ba888a85a86e4ef0cfcc5732ca9ab8067f07d1159",
    "262342b2a4ad226e56443429fa71da4bf3f8395111baa746e8a6df52bb938bb6",
    "232d2552aee91b9fe9358830deb6c7cf7cb996ce6f68320fe62643e7ff179e64",
)


def rgba(image):
    return np.frombuffer(pygame.image.tobytes(image, "RGBA"), np.uint8).reshape(32, 32, 4)


@pytest.mark.parametrize("frame", range(4))
def test_cool_torch_socket_preserves_every_native_flame_pixel_and_alpha(frame):
    image = torch(frame)
    assert image.get_size() == (32, 32)
    pixels = rgba(image)
    assert set(np.unique(pixels[:, :, 3])) == {0, 255}
    assert hashlib.sha256(pixels[:, :, 3].tobytes()).hexdigest() == BASELINE_ALPHA[frame]
    assert hashlib.sha256(pixels[:23].tobytes()).hexdigest() == BASELINE_FLAMES[frame]


def test_exposed_torch_mount_reads_as_cool_metal_in_all_four_poses():
    cool_colors = {(49, 48, 80), (105, 104, 143), (169, 169, 194), (222, 228, 232)}
    mounts = []
    for frame in range(4):
        pixels = rgba(torch(frame))
        exposed = pixels[23:31, 13:20]
        opaque_colors = [tuple(color) for color in exposed[exposed[:, :, 3] == 255, :3]]
        assert sum(color in cool_colors for color in opaque_colors) >= 16
        assert (
            99,
            48,
            26,
        ) not in opaque_colors, "The exposed wall socket must not read as a brown stick"
        mounts.append(exposed.tobytes())
    assert (
        len(set(mounts)) == 1
    ), "The metal socket stays fixed while the existing flame changes pose"
