"""Marked caches need an intentional head bump before their counted gem appears."""

import pytest

from src.unity_bridge.session import CaveSession

SITES = [(0, (10, 7), (10, 9)), (8, (8, 14), (8, 17))]


@pytest.mark.parametrize("level,block,gem", SITES)
def test_cache_is_clued_but_counted_crystal_cannot_be_collected_early(level, block, gem):
    session = CaveSession(level=level)
    game = session.game
    before = session.snapshot()
    assert gem in game.hidden_crystals and gem in game.crystals
    assert before["crystals"] == game.initial_crystals
    assert not any(e["id"] == f"crystal_{gem[0]}_{gem[1]}" for e in before["entities"])
    cache = next(e for e in before["entities"] if e["id"].startswith("secret_cache_"))
    assert cache["sprite"] == "secret_cache_armed"
    assert (cache["x"], cache["y"]) == (block[0] * 32, block[1] * 32)
    game.player_x, game.player_y = gem[0] * 32 + 4, (gem[1] + 1) * 32 - game.PLAYER_HEIGHT
    # Even the last required gem cannot silently unlock the exit before discovery.
    game.crystals = {gem}
    game.step(0)
    assert gem in game.crystals and not game.exit_unlocked


@pytest.mark.parametrize("level,block,gem", SITES)
def test_real_head_bump_reveals_and_return_landing_collects_existing_gem(level, block, gem):
    session = CaveSession(level=level)
    game = session.game
    count = len(game.crystals)
    game.player_x, game.player_y = gem[0] * 32 + 4, (gem[1] + 1) * 32 - game.PLAYER_HEIGHT
    game.vx = game.vy = 0.0
    game.grounded = True
    game.step(3)
    revealed = False
    for _ in range(65):
        game.step(0)
        revealed |= any(e.kind == "sparkle" and e.text == "SECRET" for e in game.visual_events)
    assert gem not in game.hidden_crystals and gem not in game.crystals
    assert len(game.crystals) == count - 1
    assert game.health == 3 and not game.game_over
    assert game._solid_at(*block)  # Opening the cache preserves its platform.
    assert revealed
    assert any(e["sprite"] == "secret_cache_empty" for e in session.snapshot()["entities"])
    score = game.score
    game.step(3)
    for _ in range(65):
        game.step(0)
    assert game.score == score  # Repeated bumps cannot farm score or extra gems.
    session.reset(level)
    assert gem in game.hidden_crystals and gem in game.crystals
    assert any(e["sprite"] == "secret_cache_armed" for e in session.snapshot()["entities"])


@pytest.mark.parametrize("level,block,gem", SITES)
def test_side_contact_and_shooting_do_not_open_cache(level, block, gem):
    game = CaveSession(level=level).game
    game.player_x, game.player_y = block[0] * 32 - game.PLAYER_WIDTH, block[1] * 32 + 1
    game.vx = 2.0
    game._move_axis(2.0, 0.0)
    assert gem in game.hidden_crystals
    game.facing = 1
    game._try_shoot()
    for _ in range(4):
        game._update_bullets()
    assert gem in game.hidden_crystals and game._solid_at(*block)


def test_training_maps_have_no_hidden_pickups_or_secret_cache_entities():
    session = CaveSession(classic_controls=False)
    assert not getattr(session.game, "hidden_crystals", set())
    assert not any(e["id"].startswith("secret_cache_") for e in session.snapshot()["entities"])


def test_secret_question_marker_has_visible_pixels_in_exported_font():
    from pathlib import Path

    from PIL import Image

    font = Path(__file__).resolve().parents[1] / "unity/Assets/Resources/Interface/pixel_font.png"
    with Image.open(font) as image:
        code = ord("?") - 32
        col, row = code % 16, code // 16
        glyph = image.crop((col * 6, row * 8, col * 6 + 6, row * 8 + 8))
        assert glyph.getchannel("A").getbbox() is not None
