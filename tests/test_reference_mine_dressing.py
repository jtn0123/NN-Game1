"""Main-mine reference fixtures render without occupying its entrances or chains."""

import numpy as np
import pygame
from PIL import Image

from src.unity_bridge import visual_fidelity
from src.unity_bridge.mine import ENTRANCES, MINE_SPEC
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_equipment import barrel, warning_sign
from src.unity_bridge.visual_fidelity import mine_surfaces
from src.unity_bridge.visual_mine_dressing import (
    TORCH_CELLS,
    mine_fixture_cells,
    mine_fixture_placements,
)


def test_main_mine_exports_reference_barrels_and_warning_plates(tmp_path):
    _, _, dressing = mine_surfaces(MINE_SPEC.layout)
    pygame.image.save(dressing, tmp_path / "mine_dressing.png")
    with Image.open(tmp_path / "mine_dressing.png") as opened:
        actual = np.asarray(opened.convert("RGBA"))
    fixtures = (
        (13, 6, barrel()),
        (20, 10, barrel()),
        (29, 14, barrel()),
        (13, 18, barrel()),
        (16, 4, warning_sign()),
        (25, 12, warning_sign()),
    )
    for index, (col, row, image) in enumerate(fixtures):
        path = tmp_path / f"fixture_{index}.png"
        pygame.image.save(image, path)
        with Image.open(path) as opened:
            expected = np.asarray(opened.convert("RGBA"))
        x, y = col * 32, row * 32
        np.testing.assert_array_equal(
            actual[y : y + image.get_height(), x : x + image.get_width()], expected
        )


def test_inventory_keeps_native_supports_and_transport_actor_space_clear(monkeypatch):
    session = CaveSession()
    snapshot = session.handle({"op": "mine"})
    observation = session.game.get_state().copy()
    original = MINE_SPEC.layout
    wall, terrain, current = mine_surfaces(original)
    monkeypatch.setattr(visual_fidelity, "draw_mine_fixtures", lambda layout, image: None)
    old_wall, old_terrain, old_supports = mine_surfaces(original)
    assert pygame.image.tobytes(wall, "RGBA") == pygame.image.tobytes(old_wall, "RGBA")
    assert pygame.image.tobytes(terrain, "RGBA") == pygame.image.tobytes(old_terrain, "RGBA")
    baseline_alpha = pygame.surfarray.array_alpha(old_supports)
    current_alpha = pygame.surfarray.array_alpha(current)
    placements = mine_fixture_placements(original)
    assert sum(p["sprite"] == "barrel" for p in placements) == 4
    assert sum(p["sprite"] == "danger_sign" for p in placements) == 2
    occupied = set()
    for placement in placements:
        footprint = mine_fixture_cells(placement)
        assert occupied.isdisjoint(footprint)
        occupied.update(footprint)
        for col, row in footprint:
            assert not np.any(baseline_alpha[col * 32 : (col + 1) * 32, row * 32 : (row + 1) * 32])
            assert all(original[row][col + dx] == "." for dx in (-1, 0, 1))
    torches = {
        (int(item["x"]) // 32, int(item["y"]) // 32)
        for item in snapshot["entities"]
        if item["id"].startswith("torch_")
    }
    assert torches == set(TORCH_CELLS)
    protected = set(ENTRANCES) | {(col, row - 1) for col, row in ENTRANCES} | torches
    protected |= {
        (col, row)
        for row, line in enumerate(original)
        for col, cell in enumerate(line)
        if cell in "HP"
    }
    assert occupied.isdisjoint(protected)
    for col, row in protected:
        assert not np.any(current_alpha[col * 32 : (col + 1) * 32, row * 32 : (row + 1) * 32])
    np.testing.assert_array_equal(session.game.get_state(), observation)
    assert session.snapshot()["entities"] == snapshot["entities"]
    assert MINE_SPEC.layout == original


def test_partial_obstruction_skips_complete_fixture_and_side_footprints():
    layout = [list(row) for row in MINE_SPEC.layout]
    layout[4][17] = "#"  # The right half of the two-cell plate is occupied.
    layout[6][12] = "H"  # A chain occupies the barrel's required side clearance.
    layout = tuple("".join(row) for row in layout)
    remaining = {(p["col"], p["row"]) for p in mine_fixture_placements(layout)}
    assert (16, 4) not in remaining and (13, 6) not in remaining
    assert len(remaining) == 4


def test_all_sixteen_mine_portals_keep_their_real_entry_behavior():
    session = CaveSession()
    session.handle({"op": "mine"})
    game = session.game
    for level, (col, row) in enumerate(ENTRANCES):
        game.player_x = col * 32 + 5
        game.player_y = (row + 1) * 32 - game.PLAYER_HEIGHT
        game.vx = game.vy = 0
        game.portal_level = -1
        assert session.snapshot()["near_entrance"] == level
        entered = session.handle({"op": "step", "actions": [9]})
        assert entered["portal_level"] == level and not entered["done"] and not entered["won"]
