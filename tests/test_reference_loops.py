"""Reference art must render in clear bays without changing authored gameplay."""

from pathlib import Path

import numpy as np
import pygame
from PIL import Image

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_fidelity import service_pipe_cells
from src.unity_bridge.visual_scenery import placement_cells, ventilation_grille
from src.unity_bridge.visuals import dressing, dressing_placements, theme_index


def test_ventilation_grille_png_keeps_the_native_square_slot_silhouette(tmp_path: Path):
    path = tmp_path / "ventilation_grille.png"
    pygame.image.save(ventilation_grille(), path)
    with Image.open(path) as opened:
        grille = np.asarray(opened.convert("RGBA"))
    assert grille.shape == (64, 64, 4)
    assert np.all(grille[:, :, 3] == 255)
    # These tall alternating slit faces distinguish a grille from a closed panel.
    slot_row = grille[20, 5:59, :3].mean(axis=1)
    dark_columns = np.flatnonzero(slot_row < 40)
    assert len(dark_columns) >= 24
    assert np.array_equal(grille[20, dark_columns + 5, :3], grille[45, dark_columns + 5, :3])


def test_grilles_stay_in_clear_red_wall_bays_and_export_without_overlaps(tmp_path: Path):
    session = CaveSession()
    before = tuple(cave.layout for cave in session.game.CAVES)
    path = tmp_path / "grille.png"
    pygame.image.save(ventilation_grille(), path)
    with Image.open(path) as opened:
        expected = np.asarray(opened.convert("RGBA"))
    used = 0
    for level, cave in enumerate(session.game.CAVES):
        placements = dressing_placements(cave.layout, level)
        vents = [p for p in placements if p["sprite"] == "ventilation_grille"]
        assert len(vents) <= 2
        if not vents:
            continue
        assert theme_index(level) == 0
        decoration, _ = dressing(cave.layout, level)
        pygame.image.save(decoration, tmp_path / f"dressing_{level}.png")
        with Image.open(tmp_path / f"dressing_{level}.png") as opened:
            composed = np.asarray(opened.convert("RGBA"))
        pipe_cells = service_pipe_cells(cave.layout)
        for vent in vents:
            cells = placement_cells(vent)
            assert len(cells) == 4
            assert cells.isdisjoint(pipe_cells)
            for col, row in cells:
                assert all(cave.layout[row][col + dx] == "." for dx in (-1, 0, 1))
            assert all(cave.layout[vent["row"] + 2][vent["col"] + dx] == "." for dx in (0, 1))
            for placement in placements:
                if placement is not vent:
                    assert cells.isdisjoint(placement_cells(placement))
            x, y = vent["col"] * 32, vent["row"] * 32
            np.testing.assert_array_equal(composed[y : y + 64, x : x + 64], expected)
            used += 1
    assert used > 0
    assert tuple(cave.layout for cave in session.game.CAVES) == before
