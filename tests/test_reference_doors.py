"""Reference door silhouettes preserve original bases, locks and collision cells."""

import numpy as np
import pygame
from PIL import Image

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_scenery import placement_cells
from src.unity_bridge.visuals import art_sprites, dressing, dressing_placements


def test_exit_art_uses_clear_headroom_and_keeps_every_authored_base():
    session = CaveSession()
    forms = {32: 0, 64: 0}
    art = art_sprites()
    layouts = tuple(cave.layout for cave in session.game.CAVES)
    for level, layout in enumerate(layouts):
        session.reset(level)
        col, row = session.game.exit_pos
        observation = session.game.get_state().copy()
        expected_height = 64 if row > 0 and layout[row - 1][col] == "." else 32
        for unlocked in (False, True):
            session.game.exit_unlocked = unlocked
            entity = next(item for item in session.snapshot()["entities"] if item["id"] == "exit")
            assert entity["w"] == 32 and entity["h"] == expected_height
            assert entity["x"] == col * 32
            assert entity["y"] + entity["h"] == (row + 1) * 32
            expected_sprite = "exit_open" if unlocked else "exit_locked"
            if expected_height == 64:
                expected_sprite += "_tall"
            assert entity["sprite"] == expected_sprite
            assert art[expected_sprite].get_size() == (32, expected_height)
        session.game.exit_unlocked = False
        np.testing.assert_array_equal(session.game.get_state(), observation)
        forms[expected_height] += 1
    assert forms == {32: 3, 64: 13}
    assert tuple(cave.layout for cave in session.game.CAVES) == layouts


def test_expanded_exit_headroom_stays_clear_in_metadata_and_composited_png(tmp_path):
    session = CaveSession()
    checked = 0
    for level, cave in enumerate(session.game.CAVES):
        session.reset(level)
        col, row = session.game.exit_pos
        if row == 0 or cave.layout[row - 1][col] != ".":
            continue
        headroom = (col, row - 1)
        assert all(
            headroom not in placement_cells(placement)
            for placement in dressing_placements(cave.layout, level)
        ), level
        decoration, _ = dressing(cave.layout, level)
        path = tmp_path / f"dressing_{level}.png"
        pygame.image.save(decoration, path)
        with Image.open(path) as opened:
            pixels = np.asarray(opened.convert("RGBA"))
        assert not np.any(pixels[(row - 1) * 32 : row * 32, col * 32 : (col + 1) * 32, 3]), level
        checked += 1
    assert checked == 13


def test_keyed_gate_art_keeps_each_color_and_its_original_collision_rules():
    session = CaveSession()
    checked = 0
    for level in range(len(session.game.CAVES)):
        session.reset(level)
        game = session.game
        sites = game.doors.copy()
        observation = game.get_state().copy()
        snapshot = session.snapshot()
        for col, row in sites:
            entity = next(
                item for item in snapshot["entities"] if item["id"] == f"door_{col}_{row}"
            )
            assert entity["sprite"] == f"door_{game.door_color[(col, row)]}"
            assert (entity["x"], entity["y"], entity["w"], entity["h"]) == (
                col * 32,
                row * 32,
                32,
                32,
            )
            assert game._solid_at(col, row) and game._solid_at(col, row - 1)
            checked += 1
        np.testing.assert_array_equal(game.get_state(), observation)
        for color in ("red", "blue"):
            game.open_colors = {color}
            remaining = {item["id"] for item in session.snapshot()["entities"]}
            for col, row in sites:
                closed = game.door_color[(col, row)] != color
                assert (f"door_{col}_{row}" in remaining) is closed
                assert game._solid_at(col, row) is closed
                assert game._solid_at(col, row - 1)
        assert game.doors == sites
    assert checked == 13


def test_native_door_pngs_and_legacy_aliases_keep_clear_lock_open_states(tmp_path):
    art = art_sprites()
    for color in ("red", "blue"):
        for tall in (False, True):
            name = f"door_{color}" + ("_tall" if tall else "")
            path = tmp_path / f"{name}.png"
            pygame.image.save(art[name], path)
            with Image.open(path) as opened:
                assert opened.size == (32, 64 if tall else 32)
    for tall in (False, True):
        suffix = "_tall" if tall else ""
        closed, opened = art["exit_locked" + suffix], art["exit_open" + suffix]
        assert closed.get_size() == opened.get_size() == (32, 64 if tall else 32)
        # The real unlocked state opens the lower panel; it does not only tint it.
        divider = 30 if tall else 16
        closed_pixels = pygame.surfarray.array3d(closed)[
            9:24, divider + 4 : closed.get_height() - 5
        ]
        open_pixels = pygame.surfarray.array3d(opened)[9:24, divider + 4 : opened.get_height() - 5]
        assert np.count_nonzero(np.any(closed_pixels != open_pixels, axis=2)) >= 40
    np.testing.assert_array_equal(
        pygame.surfarray.array3d(art["door_locked"]), pygame.surfarray.array3d(art["door_red"])
    )
    np.testing.assert_array_equal(
        pygame.surfarray.array3d(art["door_open"]), pygame.surfarray.array3d(art["exit_open"])
    )


def test_main_mine_numbered_entrances_keep_their_original_compact_metadata():
    session = CaveSession()
    snapshot = session.handle({"op": "mine"})
    entrances = [item for item in snapshot["entities"] if item["id"].startswith("entrance_")]
    assert len(entrances) == 16
    assert all(item["sprite"] == "mine_door" and item["w"] == item["h"] == 32 for item in entrances)
