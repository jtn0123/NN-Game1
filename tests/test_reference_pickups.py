"""Reference pickup art retains existing aliases, sites and authoritative frames."""

import numpy as np
import pygame
import pytest
from PIL import Image

from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_fidelity import pickup_sprites
from src.unity_bridge.visuals import art_sprites


def test_exported_gravity_alias_is_a_native_upward_red_arrow(tmp_path):
    surface = art_sprites()["gravity"]
    path = tmp_path / "gravity.png"
    pygame.image.save(surface, path)
    with Image.open(path) as opened:
        assert opened.size == (32, 32)
        pixels = np.asarray(opened.convert("RGBA"))
    np.testing.assert_array_equal(
        pixels[:, :, :3].transpose(1, 0, 2), pygame.surfarray.array3d(pickup_sprites()["gravity"])
    )
    rgb = pixels[:, :, :3].astype(int)
    red = (
        (rgb[:, :, 0] > 160)
        & (rgb[:, :, 0] > rgb[:, :, 1] * 1.4)
        & (rgb[:, :, 0] > rgb[:, :, 2] * 1.4)
    )
    assert np.count_nonzero(red) >= 30
    upper = np.flatnonzero(np.any(red[:17], axis=0))
    lower = np.flatnonzero(np.any(red[17:], axis=0))
    # A wider red head above its narrow stem reads UP, unlike the old G glyph.
    assert upper[-1] - upper[0] > lower[-1] - lower[0]
    assert np.count_nonzero((rgb[:, :, 0] > 220) & (rgb[:, :, 1] > 220) & (rgb[:, :, 2] < 150)) > 25


def test_gravity_art_keeps_all_authored_sites_without_advancing_a_frame():
    session = CaveSession()
    checked = 0
    for level in range(len(session.game.CAVES)):
        session.reset(level)
        game = session.game
        original = game.powerups.copy()
        observation = game.get_state().copy()
        frame = game.steps
        entities = {item["id"]: item for item in session.snapshot()["entities"]}
        for (col, row), kind in original.items():
            if kind != game.GRAVITY_POWER:
                continue
            entity = entities[f"power_{col}_{row}"]
            assert entity["sprite"] == "gravity"
            assert (entity["x"], entity["y"], entity["w"], entity["h"]) == (
                col * 32,
                row * 32,
                32,
                32,
            )
            checked += 1
        assert game.steps == frame and game.powerups == original
        np.testing.assert_array_equal(game.get_state(), observation)
    assert checked > 0


@pytest.mark.parametrize("classic_controls", [True, False])
def test_collecting_the_arrow_matches_the_authoritative_gravity_frame(classic_controls):
    session = CaveSession(level=1, classic_controls=classic_controls)
    reference = CaveSession(level=1, classic_controls=classic_controls)
    tile = next(tile for tile, kind in session.game.powerups.items() if kind == "g")
    col, row = tile
    for game in (session.game, reference.game):
        game.player_x, game.player_y = col * 32 + 4, row * 32 + 2
        game.vx = game.vy = 0
    before = session.game.steps
    expected_state, expected_reward, expected_done, _ = reference.game.step(0)
    snapshot = session.handle({"op": "step", "actions": [0]})
    np.testing.assert_array_equal(session.game.get_state(), expected_state)
    assert snapshot["last_reward"] == pytest.approx(expected_reward)
    assert snapshot["done"] is expected_done
    assert snapshot["steps"] == before + 1
    assert snapshot["player"]["gravity_dir"] == reference.game.gravity_dir == -1
    assert tile not in session.game.powerups
    assert f"power_{col}_{row}" not in {item["id"] for item in snapshot["entities"]}
    assert "gravity" in snapshot["sounds"]
