"""Dashboard callback binding helpers for runtime composition roots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

DashboardCallback = Callable[..., Any]


@dataclass(frozen=True)
class DashboardCallbacks:
    """Callbacks exposed by runtimes to the web dashboard."""

    pause: DashboardCallback
    save: DashboardCallback
    save_as: DashboardCallback
    reset: DashboardCallback
    start_fresh: DashboardCallback
    load_model: DashboardCallback
    config_change: DashboardCallback
    performance_mode: DashboardCallback
    restart_with_game: DashboardCallback
    save_and_quit: DashboardCallback
    speed: Optional[DashboardCallback] = None


def bind_dashboard_callbacks(dashboard: Any, callbacks: DashboardCallbacks) -> None:
    """Attach runtime callbacks to a WebDashboard-compatible object."""
    dashboard.on_pause_callback = callbacks.pause
    dashboard.on_save_callback = callbacks.save
    dashboard.on_save_as_callback = callbacks.save_as
    dashboard.on_reset_callback = callbacks.reset
    dashboard.on_start_fresh_callback = callbacks.start_fresh
    dashboard.on_load_model_callback = callbacks.load_model
    dashboard.on_config_change_callback = callbacks.config_change
    dashboard.on_performance_mode_callback = callbacks.performance_mode
    dashboard.on_restart_with_game_callback = callbacks.restart_with_game
    dashboard.on_save_and_quit_callback = callbacks.save_and_quit
    dashboard.on_speed_callback = callbacks.speed or (lambda _speed: None)
