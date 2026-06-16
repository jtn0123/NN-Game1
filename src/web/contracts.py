"""Shared web API and Socket.IO contracts."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, TypedDict


class ControlAck(TypedDict, total=False):
    """Acknowledgement payload returned by dashboard control events."""

    success: bool
    action: str
    error: str


class GameInfoPayload(TypedDict):
    """Game metadata returned by `/api/games`."""

    id: str
    name: str
    description: str
    actions: List[str]
    controls: List[str]
    difficulty: str
    icon: str
    color: Any
    is_current: bool


class PerformanceModePayload(TypedDict):
    """Single performance-mode definition exposed to the frontend."""

    label: str
    learn_every: int
    batch_size: int
    gradient_steps: int
    description: str


CONTROL_ACTIONS: FrozenSet[str] = frozenset(
    {
        "pause",
        "save",
        "save_as",
        "speed",
        "reset",
        "start_fresh",
        "load_model",
        "config_change",
        "performance_mode",
        "save_and_quit",
        "select_game",
        "restart_with_game",
        "go_to_launcher",
    }
)

PerformanceModesResponse = Dict[str, PerformanceModePayload]
