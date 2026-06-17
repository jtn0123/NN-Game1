"""Unit tests for extracted dashboard socket-control routing."""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import Config
from src.web import socket_controls


class FakePublisher:
    def __init__(self) -> None:
        self.console_logs = []
        self.speed: Optional[float] = None
        self.config: Dict[str, Any] = {}
        self.performance_mode: Optional[str] = None
        self.logs = []

    def log(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None) -> None:
        self.logs.append({"message": message, "level": level, "data": data})

    def set_speed(self, speed: float) -> None:
        self.speed = speed

    def update_config(self, config_data: Dict[str, Any]) -> None:
        self.config.update(config_data)

    def set_performance_mode(self, mode: str) -> None:
        self.performance_mode = mode


class FakeDashboard:
    def __init__(self) -> None:
        self.config = Config()
        self.launcher_mode = False
        self.publisher = FakePublisher()
        self.on_game_selected_callback = None
        self.on_restart_with_game_callback = None
        self.on_pause_callback = None
        self.on_save_callback = None
        self.on_save_as_callback = None
        self.on_speed_callback = None
        self.on_reset_callback = None
        self.on_start_fresh_callback = None
        self.on_load_model_callback = None
        self.on_config_change_callback = None
        self.on_performance_mode_callback = None
        self.on_save_and_quit_callback = None

    def _resolve_model_ref(self, model_ref: str) -> Optional[str]:
        return None


def test_dispatch_control_routes_speed_with_normalized_payload() -> None:
    dashboard = FakeDashboard()
    called = []
    emitted = []
    dashboard.on_speed_callback = lambda speed: called.append(speed)

    ack = socket_controls.dispatch_control(
        dashboard,
        {"action": "speed", "value": 5000},
        lambda event, payload: emitted.append((event, payload)),
    )

    assert ack == {"success": True, "action": "speed"}
    assert called == [1000.0]
    assert dashboard.publisher.speed == 1000.0
    assert emitted == []


def test_dispatch_control_rejects_bad_config_before_callback() -> None:
    dashboard = FakeDashboard()
    called = []
    dashboard.on_config_change_callback = lambda config: called.append(config)

    ack = socket_controls.dispatch_control(
        dashboard,
        {"action": "config_change", "config": {"learning_rate": "bad"}},
        lambda event, payload: None,
    )

    assert ack["success"] is False
    assert ack["action"] == "config_change"
    assert called == []
    assert dashboard.publisher.config == {}
