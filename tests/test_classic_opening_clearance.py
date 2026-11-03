"""Ore Shaft's visible opening hazards need room for an ordinary jump."""

import pytest

from src.unity_bridge.human_controls import HumanControls
from src.unity_bridge.session import CaveSession
from src.unity_bridge.visual_scenery import placement_cells
from src.unity_bridge.visuals import dressing_placements


def test_ore_shaft_opening_jump_clears_thorn_and_spike_without_damage():
    game = CaveSession().game
    for _ in range(100):
        if game.player_x >= 7 * 32 + 4:
            break
        game.step_human(HumanControls(1, False, False, False))
    assert game.health == 3 and game.super_timer > 0 and game.ammo == 10
    game.step_human(HumanControls(1, True, False, False))
    for _ in range(90):
        if game.player_x >= 11 * 32 + 4:
            break
        game.step_human(HumanControls(1, False, False, False))
    for _ in range(40):
        if game.grounded:
            break
        game.step_human(HumanControls(0, False, False, False))
    assert game.player_x >= 11 * 32 + 4 and game.grounded
    assert game.health == 3 and not game.game_over
    assert (9, 22) in game.hazards
    assert any((thorn.col, thorn.row) == (8, 22) for thorn in game.thorns)
    assert all(game.level.layout[18][col] == "#" for col in range(6, 11))


@pytest.mark.parametrize("direction", [-1, 1])
def test_walking_into_the_trench_still_hurts_and_allows_escape(direction):
    game = CaveSession().game
    for _ in range(120):
        game.step_human(HumanControls(1, False, False, False))
        if game.health < 3:
            break
    assert game.health == 2 and game.invuln_timer == game.INVULN_FRAMES
    assert game._player_rect().colliderect(game._tile_rect((9, 22)))
    assert game.level.layout[23][8:10] == "##"
    for _ in range(80):
        game.step_human(HumanControls(direction, True, False, False))
        if game.player_x < 7 * 32 or game.player_x > 11 * 32:
            break
    else:
        pytest.fail("a player could not jump out of the shallow trench")
    assert game.health == 2 and not game.game_over


def test_trench_warning_stays_visible_without_overlapping_scenery():
    layout = CaveSession().game.level.layout
    placements = dressing_placements(layout, 0)
    warnings = [
        item
        for item in placements
        if item["sprite"] == "danger_sign"
        and abs(item["col"] - 8) <= 5
        and abs(item["row"] - 22) <= 2
    ]
    assert len(warnings) == 1
    footprint = placement_cells(warnings[0])
    assert all(layout[row][col] == "." for col, row in footprint)
    for item in placements:
        if item is not warnings[0]:
            assert footprint.isdisjoint(placement_cells(item))
