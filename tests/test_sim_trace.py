"""Guards for the C# simulation port's golden data.

The C# engine (``unity/Assets/Sim``) is verified against golden traces by
``dotnet test unity/SimTests/Tests``. These tests keep the inputs to that check
honest: the generated cave data and the traces must match the current Python
game, and the layout assumptions the port relies on must keep holding.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.unity_bridge import export_sim_data
from src.unity_bridge.session import CaveSession
from src.unity_bridge.sim_trace import (
    REQUIRED_COVERAGE,
    TRACE_VERSION,
    coverage,
    digest,
    encode,
    expand,
)

TRACE_DIR = Path(__file__).resolve().parents[1] / "unity/SimTests/Traces"
TRACES = sorted(TRACE_DIR.glob("*.json.gz"))
PREFIX_OPS = 300


def _load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _caves() -> list[Any]:
    return list(CaveSession(level=0).cave_game.CAVES)


def test_generated_cave_data_is_current() -> None:
    expected = export_sim_data.source()
    actual = export_sim_data.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    assert actual == expected, "run: python -m src.unity_bridge.export_sim_data"


def test_layouts_use_only_characters_the_port_understands() -> None:
    for cave in _caves():
        used = set("".join(cave.layout))
        assert used <= export_sim_data.LAYOUT_CHARACTERS, (cave.name, used)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.0, "3"),
        (-0.0, "0"),
        (0.5, "d3fe0000000000000"),
        (140 / 60, "d4002aaaaaaaaaaab"),
        (2.0**53, "d4340000000000000"),
    ],
)
def test_numbers_encode_like_the_csharp_encoder(value: float, expected: str) -> None:
    assert encode(value) == expected


def test_objects_encode_like_the_csharp_encoder() -> None:
    value = {"b": [2, None, True], "a": "x", "B": False}
    text = encode(value)
    assert text == '{"B":false,"a":"x","b":[2,null,true]}'
    # Mirrored in unity/SimTests/Tests/CanonicalTests.cs.
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == "c5115fed98b2d1a7"


def _tiles(layout: tuple[str, ...], characters: str) -> list[tuple[int, int]]:
    return [
        (col, row)
        for row, line in enumerate(layout)
        for col, char in enumerate(line)
        if char in characters
    ]


def test_each_cave_has_at_most_one_air_tank() -> None:
    """The port checks air tanks in sorted order; Python iterates a set."""
    for cave in _caves():
        assert len(_tiles(cave.layout, "O")) <= 1, cave.name


def test_no_tile_can_reach_two_switches() -> None:
    """Interact reaches switches within one tile; Python picks from a set."""
    for cave in _caves():
        levers = _tiles(cave.layout, "sS")
        for index, first in enumerate(levers):
            for second in levers[index + 1 :]:
                distance = max(abs(first[0] - second[0]), abs(first[1] - second[1]))
                assert distance > 2, (cave.name, first, second)


def test_golden_traces_exist_and_cover_the_mechanics() -> None:
    assert len(TRACES) >= 40
    counts = coverage(TRACES)
    missing = [key for key in REQUIRED_COVERAGE if not counts.get(key)]
    assert not missing, f"coverage gaps: {missing}"


@pytest.mark.parametrize("path", TRACES, ids=lambda path: path.name.split(".")[0])
def test_golden_trace_prefix_still_reproduces(path: Path) -> None:
    """Stale traces would make the C# parity suite check against an old game."""
    trace = _load(path)
    assert trace["version"] == TRACE_VERSION
    session = CaveSession(level=trace["level"])
    assert digest(session.snapshot()) == trace["initial_digest"]
    for index, operation in enumerate(expand(trace["operations"])):
        if index >= PREFIX_OPS:
            break
        assert (
            digest(session.handle(operation)) == trace["digests"][index]
        ), f"{path.name} diverged at op {index}; run: python -m src.unity_bridge.sim_trace"
