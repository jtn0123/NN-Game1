"""Reject unsafe cave overlays and caches instead of losing authored objects."""

from dataclasses import replace

import pytest

from src.unity_bridge.classic_polish import polish_caves
from src.unity_bridge.session import CaveSession


def change_tile(cave, tile, marker):
    col, row = tile
    layout = list(cave.layout)
    cells = list(layout[row])
    cells[col] = marker
    layout[row] = "".join(cells)
    return replace(cave, layout=tuple(layout))


@pytest.fixture
def unpolished_campaign(monkeypatch):
    # Retain the actual classic traversal/trap repairs, then exercise only the
    # new overlay's authoring boundaries with otherwise valid source caves.
    monkeypatch.setattr("src.unity_bridge.classic_levels.polish_caves", lambda caves: caves)
    return CaveSession().game.CAVES


@pytest.mark.parametrize(
    "level,tile,marker,message",
    [
        (0, (6, 21), "*", "occupied"),
        (0, (8, 19), ".", "not double stone"),
        (0, (8, 22), ".", "lacks solid support"),
        (6, (18, 5), "$", "relocation.*invalid"),
    ],
)
def test_unsafe_polish_edits_preserve_the_input_campaign(
    unpolished_campaign, level, tile, marker, message
):
    caves = list(unpolished_campaign)
    caves[level] = change_tile(caves[level], tile, marker)
    supplied = tuple(caves)
    original_layouts = tuple(cave.layout for cave in supplied)
    with pytest.raises(ValueError, match=message):
        polish_caves(supplied)
    assert tuple(cave.layout for cave in supplied) == original_layouts
    assert supplied[level].layout[tile[1]][tile[0]] == marker


@pytest.mark.parametrize("tile", [(10, 7), (10, 9)])
def test_cache_requires_both_its_solid_block_and_counted_crystal(tile):
    game = CaveSession().game
    assert game.secrets_enabled and (10, 9) in game.crystals
    invalid = change_tile(game.level, tile, ".")
    with pytest.raises(ValueError, match="invalid secret cache in Ore Shaft"):
        game._load_level(invalid)
