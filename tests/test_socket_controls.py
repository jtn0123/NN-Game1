"""Unit tests for extracted dashboard socket-control routing."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from config import Config
from src.web import socket_controls
from src.web.contracts import CONTROL_ACTIONS


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


def test_control_handler_table_covers_public_action_contract() -> None:
    assert set(socket_controls.CONTROL_HANDLERS) == set(CONTROL_ACTIONS)


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


def test_callback_ack_translates_callback_results() -> None:
    def raises_error() -> None:
        raise RuntimeError("internal detail")

    assert socket_controls.callback_ack("save", None, failure_message="failed") == {
        "success": True,
        "action": "save",
    }
    assert socket_controls.callback_ack(
        "save", lambda: {"success": True}, failure_message="failed"
    ) == {"success": True, "action": "save"}
    assert socket_controls.callback_ack(
        "save",
        lambda: {"success": False, "error": "bad result"},
        failure_message="failed",
    ) == {"success": False, "action": "save", "error": "bad result"}
    assert socket_controls.callback_ack(
        "save",
        lambda: socket_controls.CommandResult.ok(),
        failure_message="failed",
    ) == {"success": True, "action": "save"}
    assert socket_controls.callback_ack(
        "save",
        lambda: socket_controls.CommandResult.failed("blocked"),
        failure_message="failed",
    ) == {"success": False, "action": "save", "error": "blocked"}
    assert socket_controls.callback_ack("save", lambda: False, failure_message="failed") == {
        "success": False,
        "action": "save",
        "error": "failed",
    }
    assert socket_controls.callback_ack("save", raises_error, failure_message="failed") == {
        "success": False,
        "action": "save",
        "error": "failed",
    }


def test_control_handlers_cover_success_paths(tmp_path, monkeypatch) -> None:
    dashboard = FakeDashboard()
    emitted = []
    selected = []
    restarted = []
    loaded = []
    saved_as = []
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"model")
    dashboard.on_start_fresh_callback = lambda: True
    dashboard.on_game_selected_callback = lambda game, mode: selected.append((game, mode))
    dashboard.on_load_model_callback = loaded.append
    dashboard.on_save_callback = lambda: True
    dashboard.on_save_as_callback = saved_as.append
    dashboard.on_restart_with_game_callback = restarted.append
    dashboard.on_save_and_quit_callback = lambda: True
    dashboard._resolve_model_ref = lambda _model_ref: str(model_path)  # type: ignore[method-assign]
    monkeypatch.setattr(socket_controls, "is_known_game", lambda game: game == "breakout")

    def emit(event, payload):
        emitted.append((event, payload))

    assert socket_controls.handle_start_fresh_control(dashboard, emit) == {
        "success": True,
        "action": "start_fresh",
    }
    assert socket_controls.handle_save_as_control(dashboard, {"filename": "demo.pth"}) == {
        "success": True,
        "action": "save_as",
    }
    assert socket_controls.handle_select_game_control(
        dashboard, {"game": "breakout", "mode": "human"}, emit
    ) == {"success": True, "action": "select_game"}
    assert socket_controls.handle_load_model_control(dashboard, {"id": "breakout:model.pth"}) == {
        "success": True,
        "action": "load_model",
    }
    assert socket_controls.handle_restart_with_game_control(
        dashboard, {"game": "breakout"}, emit
    ) == {"success": True, "action": "restart_with_game"}
    assert socket_controls.handle_go_to_launcher_control(dashboard, emit) == {
        "success": True,
        "action": "go_to_launcher",
    }

    dashboard.publisher.console_logs.append("old")
    assert socket_controls.clear_logs(dashboard, emit) == {"success": True, "action": "clear_logs"}

    assert saved_as == ["demo.pth"]
    assert selected == [("breakout", "human")]
    assert loaded == [str(model_path)]
    assert restarted == ["breakout"]
    assert dashboard.launcher_mode is True
    assert dashboard.publisher.console_logs == []
    assert ("training_reset", {"message": "Training reset - starting fresh"}) in emitted
    assert (
        "game_starting",
        {"game": "breakout", "mode": "human", "message": "Playing breakout..."},
    ) in emitted
    assert ("restarting", {"game": "breakout", "message": "Restarting with breakout..."}) in emitted
    assert ("redirect_to_launcher", {"message": "Returning to game launcher..."}) in emitted
    assert ("console_logs", {"logs": []}) in emitted


def test_control_handlers_reject_bad_inputs_and_failed_callbacks(monkeypatch) -> None:
    dashboard = FakeDashboard()
    emitted = []

    def emit(event, payload):
        emitted.append((event, payload))

    assert socket_controls.handle_save_as_control(dashboard, {"filename": " "}) == {
        "success": False,
        "action": "save_as",
        "error": "Invalid filename",
    }
    assert socket_controls.handle_speed_control(dashboard, {"value": "bad"}) == {
        "success": False,
        "action": "speed",
        "error": "Invalid speed",
    }
    assert socket_controls.handle_config_change_control(dashboard, {"config": []}) == {
        "success": False,
        "action": "config_change",
        "error": "Invalid config",
    }
    assert socket_controls.handle_performance_mode_control(dashboard, {"mode": "missing"}) == {
        "success": False,
        "action": "performance_mode",
        "error": "Invalid performance mode",
    }
    assert socket_controls.handle_select_game_control(
        dashboard, {"game": "breakout", "mode": "invalid"}, emit
    ) == {"success": False, "action": "select_game", "error": "Invalid game"}
    assert socket_controls.handle_load_model_control(dashboard, {}) == {
        "success": False,
        "action": "load_model",
        "error": "Invalid model id",
    }
    assert socket_controls.handle_load_model_control(dashboard, {"id": "missing"}) == {
        "success": False,
        "action": "load_model",
        "error": "Invalid model id",
    }

    dashboard._resolve_model_ref = lambda _model_ref: "/missing/model.pth"  # type: ignore[method-assign]
    assert socket_controls.handle_load_model_control(dashboard, {"id": "missing"}) == {
        "success": False,
        "action": "load_model",
        "error": "Model not found",
    }

    monkeypatch.setattr(socket_controls, "is_known_game", lambda _game: True)
    assert socket_controls.handle_restart_with_game_control(
        dashboard, {"game": "breakout"}, emit
    ) == {"success": False, "action": "restart_with_game", "error": "Invalid game"}

    dashboard.on_restart_with_game_callback = lambda _game: True
    dashboard.on_save_callback = lambda: {"success": False, "error": "save blocked"}
    assert socket_controls.handle_restart_with_game_control(
        dashboard, {"game": "breakout"}, emit
    ) == {"success": False, "action": "restart_with_game", "error": "save blocked"}

    dashboard.on_save_and_quit_callback = lambda: {"success": False, "error": "quit blocked"}
    assert socket_controls.handle_go_to_launcher_control(dashboard, emit) == {
        "success": False,
        "action": "go_to_launcher",
        "error": "save blocked",
    }

    assert socket_controls.dispatch_control(dashboard, {"action": None}, emit) == {
        "success": False,
        "action": "None",
        "error": "Unknown action",
    }
    monkeypatch.setitem(socket_controls.CONTROL_HANDLERS, "save", None)
    assert socket_controls.dispatch_control(dashboard, {"action": "save"}, emit) == {
        "success": False,
        "action": "save",
        "error": "Unknown action",
    }


def test_remaining_socket_control_edges(monkeypatch) -> None:
    dashboard = FakeDashboard()
    emitted = []
    assert socket_controls.is_known_game(123) is False
    monkeypatch.setattr(socket_controls, "is_known_game", lambda _game: True)

    assert socket_controls.handle_performance_mode_control(dashboard, {"mode": "normal"}) == {
        "success": True,
        "action": "performance_mode",
    }
    assert dashboard.publisher.performance_mode == "normal"
    assert socket_controls.handle_select_game_control(
        dashboard,
        {"game": "breakout", "mode": "ai"},
        lambda event, payload: emitted.append((event, payload)),
    ) == {"success": False, "action": "select_game", "error": "Invalid game"}

    ok, normalized, error = socket_controls.normalize_config_change(Config(), {"learning_rate": 0})
    assert ok is False
    assert normalized == {}
    assert error == "Learning rate must be finite and positive"


def test_parse_speed_and_config_normalization_cover_bounds() -> None:
    config = Config()
    assert socket_controls.parse_speed(None) is None
    assert socket_controls.parse_speed(float("inf")) is None
    assert socket_controls.parse_speed(0) is None
    assert socket_controls.parse_speed(0.5) == 1.0
    assert socket_controls.parse_speed(5000) == 1000.0

    ok, normalized, error = socket_controls.normalize_config_change(
        config,
        {
            "learning_rate": "0.001",
            "epsilon": 99,
            "epsilon_decay": 0.99,
            "gamma": 0.95,
            "batch_size": 64,
            "learn_every": 4,
            "gradient_steps": 2,
        },
    )

    assert ok is True
    assert error == ""
    assert normalized["learning_rate"] == 0.001
    assert normalized["epsilon"] == config.EPSILON_START
    assert normalized["epsilon_decay"] == 0.99
    assert normalized["gamma"] == 0.95
    assert normalized["batch_size"] == 64
    assert normalized["learn_every"] == 4
    assert normalized["gradient_steps"] == 2


@pytest.mark.parametrize(
    "config_data",
    [
        {"learning_rate": 11},
        {"learning_rate": 1e-11},
        {"epsilon": float("nan")},
        {"epsilon_decay": 0},
        {"gamma": -1},
        {"batch_size": 0},
        {"batch_size": Config().MEMORY_SIZE + 1},
        {"learn_every": 0},
        {"gradient_steps": 0},
    ],
)
def test_config_normalization_rejects_invalid_values(config_data) -> None:
    ok, normalized, error = socket_controls.normalize_config_change(Config(), config_data)

    assert ok is False
    assert normalized == {}
    assert error
