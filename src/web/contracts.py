"""Shared web API and Socket.IO contracts."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, TypedDict


class ApiErrorPayload(TypedDict):
    """Stable error response shape returned by dashboard APIs."""

    error: str


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


class GamesResponse(TypedDict):
    """Game-list API response."""

    games: List[GameInfoPayload]
    current_game: str


class DashboardConfigPayload(TypedDict):
    """Dashboard training config exposed to the browser."""

    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    batch_size: int
    hidden_layers: List[int]
    memory_size: int
    target_update: int
    grad_clip: float
    learn_every: int
    gradient_steps: int
    device: str
    vec_envs: int
    game_name: str


class ModelPayload(TypedDict, total=False):
    """Model metadata returned by `/api/models`."""

    name: str
    id: str
    source: str
    size: int
    modified: float
    modified_str: str
    is_loadable: bool
    has_metadata: bool
    requires_unsafe_load: bool
    security_warning: str
    metadata: Optional[Dict[str, Any]]
    steps: Any
    epsilon: Any
    load_error: str


class ModelsResponse(TypedDict):
    """Model-list API response."""

    models: List[ModelPayload]
    current_game: str


class DeleteModelResponse(TypedDict):
    """Successful model-delete API response."""

    success: bool
    message: str
    filename: str


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
