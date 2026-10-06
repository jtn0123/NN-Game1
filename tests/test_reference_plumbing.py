"""Decorative plumbing has a real AIR destination instead of modulo wallpaper."""

from pathlib import Path

import numpy as np
import pygame
from PIL import Image

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_doors import door_headroom_cells
from src.unity_bridge.visual_fidelity import connected_pipes, service_pipe_cells
from src.unity_bridge.visual_scenery import placement_cells
from src.unity_bridge.visuals import dressing, dressing_placements


def test_authored_cave_plumbing_keeps_only_connected_service_runs(tmp_path: Path):
    session = CaveSession()
    layouts = tuple(cave.layout for cave in session.game.CAVES)
    connected_rooms = 0
    for level, cave in enumerate(session.game.CAVES):
        session.reset(level)
        observation = session.game.get_state().copy()
        gameplay_sites = (session.game.air_tanks.copy(), session.game.doors.copy())
        placements = dressing_placements(cave.layout, level)
        assert not any(p["sprite"] == "pipe" for p in placements), level
        protected = service_pipe_cells(cave.layout) | door_headroom_cells(cave.layout)
        assert all(placement_cells(p).isdisjoint(protected) for p in placements), level
        actual, _ = dressing(cave.layout, level)
        expected = pygame.Surface(actual.get_size(), pygame.SRCALPHA)
        connected_pipes(cave.layout, expected)
        for name, image in (("actual", actual), ("connected", expected)):
            pygame.image.save(image, tmp_path / f"{name}_{level}.png")
        with Image.open(tmp_path / f"actual_{level}.png") as opened:
            actual_pixels = np.asarray(opened.convert("RGBA"))
        with Image.open(tmp_path / f"connected_{level}.png") as opened:
            expected_pixels = np.asarray(opened.convert("RGBA"))
        service_mask = expected_pixels[:, :, 3] > 0
        if np.any(service_mask):
            np.testing.assert_array_equal(
                actual_pixels[service_mask], expected_pixels[service_mask]
            )
            connected_rooms += 1
        np.testing.assert_array_equal(session.game.get_state(), observation)
        assert (session.game.air_tanks, session.game.doors) == gameplay_sites
    assert connected_rooms == 11
    assert tuple(cave.layout for cave in session.game.CAVES) == layouts


def test_former_unconnected_pipe_bays_export_as_quiet_wall_space(tmp_path: Path):
    session = CaveSession()
    # These empty bays formerly contained nothing except an arbitrary vertical
    # pipe stamp. They are separate from AIR routes and other authored fixtures.
    for level, col, row in ((0, 7, 4), (4, 19, 3), (8, 9, 2)):
        layout = session.game.CAVES[level].layout
        assert layout[row][col] == "."
        assert (col, row) not in service_pipe_cells(layout)
        decoration, _ = dressing(layout, level)
        path = tmp_path / f"quiet_bay_{level}.png"
        pygame.image.save(decoration, path)
        with Image.open(path) as opened:
            pixels = np.asarray(opened.convert("RGBA"))
        cell = pixels[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
        assert not np.any(cell[:, :, 3]), (level, col, row)
