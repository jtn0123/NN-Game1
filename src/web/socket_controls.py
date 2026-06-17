"""Socket.IO dashboard control routing and acknowledgement helpers."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

from config import Config
from src.app.performance_modes import PERFORMANCE_MODES
from src.utils.logger import get_logger
from src.web.contracts import CONTROL_ACTIONS, ControlAck

_logger = get_logger(__name__)


class DashboardControlContext(Protocol):
    """Dashboard attributes required by socket control actions."""

    config: Config
    launcher_mode: bool
    publisher: Any
    on_game_selected_callback: Optional[Callable[[str, str], Any]]
    on_restart_with_game_callback: Optional[Callable[[str], Any]]
    on_pause_callback: Optional[Callable[[], Any]]
    on_save_callback: Optional[Callable[[], Any]]
    on_save_as_callback: Optional[Callable[[str], Any]]
    on_speed_callback: Optional[Callable[[float], Any]]
    on_reset_callback: Optional[Callable[[], Any]]
    on_start_fresh_callback: Optional[Callable[[], Any]]
    on_load_model_callback: Optional[Callable[[str], Any]]
    on_config_change_callback: Optional[Callable[[Dict[str, Any]], Any]]
    on_performance_mode_callback: Optional[Callable[[str], Any]]
    on_save_and_quit_callback: Optional[Callable[[], Any]]

    def _resolve_model_ref(self, model_ref: str) -> Optional[str]:
        """Resolve a browser model id into a safe local path."""
        ...


EventEmitter = Callable[[str, Dict[str, Any]], None]


def success_ack(action: str) -> ControlAck:
    """Return a successful control acknowledgement."""
    return {"success": True, "action": action}


def error_ack(action: str, error: str) -> ControlAck:
    """Return a failed control acknowledgement."""
    return {"success": False, "action": action, "error": error}


def unauthorized_ack() -> ControlAck:
    """Return the stable acknowledgement for unauthorized control requests."""
    return {"success": False, "error": "Unauthorized"}


def callback_ack(
    action: str,
    callback: Optional[Callable[..., Any]],
    *args: Any,
    failure_message: str,
) -> ControlAck:
    """Run a control callback and translate its return value into an ack."""
    if callback is None:
        return success_ack(action)
    try:
        result = callback(*args)
    except Exception:
        _logger.exception("Dashboard control callback failed for %s", action)
        return error_ack(action, failure_message)

    if isinstance(result, dict):
        success = bool(result.get("success", False))
        if success:
            return success_ack(action)
        return error_ack(action, str(result.get("error", failure_message)))
    if result is False:
        return error_ack(action, failure_message)
    return success_ack(action)


def handle_save_control(context: DashboardControlContext) -> ControlAck:
    """Handle a save control request."""
    return callback_ack("save", context.on_save_callback, failure_message="Save failed")


def handle_save_as_control(context: DashboardControlContext, data: Dict[str, Any]) -> ControlAck:
    """Handle a save-as control request."""
    filename = data.get("filename", "custom_save.pth")
    if not isinstance(filename, str) or not filename.strip():
        return error_ack("save_as", "Invalid filename")
    return callback_ack(
        "save_as",
        context.on_save_as_callback,
        filename,
        failure_message="Save failed",
    )


def handle_start_fresh_control(
    context: DashboardControlContext, emit_event: EventEmitter
) -> ControlAck:
    """Handle a start-fresh request and emit the reset event only after success."""
    ack = callback_ack(
        "start_fresh",
        context.on_start_fresh_callback,
        failure_message="Start fresh failed",
    )
    if ack["success"]:
        emit_event("training_reset", {"message": "Training reset - starting fresh"})
    return ack


def handle_speed_control(context: DashboardControlContext, data: Dict[str, Any]) -> ControlAck:
    """Handle a speed-control request."""
    speed = parse_speed(data.get("value", 1.0))
    if speed is None:
        return error_ack("speed", "Invalid speed")

    ack = callback_ack(
        "speed",
        context.on_speed_callback,
        speed,
        failure_message="Speed change failed",
    )
    if ack["success"]:
        context.publisher.set_speed(speed)
    return ack


def handle_config_change_control(
    context: DashboardControlContext, data: Dict[str, Any]
) -> ControlAck:
    """Handle a training-config change request."""
    config_data = data.get("config", {})
    if not isinstance(config_data, dict):
        return error_ack("config_change", "Invalid config")

    valid_config, normalized_config, error = normalize_config_change(context.config, config_data)
    if not valid_config:
        return error_ack("config_change", error)

    ack = callback_ack(
        "config_change",
        context.on_config_change_callback,
        normalized_config,
        failure_message="Config change failed",
    )
    if ack["success"]:
        context.publisher.update_config(normalized_config)
    return ack


def handle_performance_mode_control(
    context: DashboardControlContext, data: Dict[str, Any]
) -> ControlAck:
    """Handle a dashboard performance-mode request."""
    mode = data.get("mode", "normal")
    if mode not in valid_performance_modes():
        return error_ack("performance_mode", "Invalid performance mode")

    ack = callback_ack(
        "performance_mode",
        context.on_performance_mode_callback,
        mode,
        failure_message="Performance mode failed",
    )
    if ack["success"]:
        context.publisher.set_performance_mode(mode)
    return ack


def handle_save_and_quit_control(context: DashboardControlContext) -> ControlAck:
    """Handle a save-and-quit request."""
    return callback_ack(
        "save_and_quit",
        context.on_save_and_quit_callback,
        failure_message="Save and quit failed",
    )


def handle_select_game_control(
    context: DashboardControlContext, data: Dict[str, Any], emit_event: EventEmitter
) -> ControlAck:
    """Handle launcher game selection."""
    game_name = data.get("game")
    mode = data.get("mode", "ai")
    if not is_known_game(game_name) or mode not in {"ai", "human"}:
        return error_ack("select_game", "Invalid game")
    if not context.on_game_selected_callback:
        return error_ack("select_game", "Invalid game")

    ack = callback_ack(
        "select_game",
        context.on_game_selected_callback,
        game_name,
        mode,
        failure_message="Game selection failed",
    )
    if ack["success"]:
        mode_text = "Playing" if mode == "human" else "Training"
        emit_event(
            "game_starting",
            {
                "game": game_name,
                "mode": mode,
                "message": f"{mode_text} {game_name}...",
            },
        )
    return ack


def handle_load_model_control(context: DashboardControlContext, data: Dict[str, Any]) -> ControlAck:
    """Handle a model-load request."""
    model_ref = data.get("id") or data.get("path")
    if not model_ref:
        return error_ack("load_model", "Invalid model id")

    model_path = context._resolve_model_ref(model_ref)
    if not model_path:
        return error_ack("load_model", "Invalid model id")
    if not os.path.exists(model_path):
        return error_ack("load_model", "Model not found")
    return callback_ack(
        "load_model",
        context.on_load_model_callback,
        model_path,
        failure_message="Load model failed",
    )


def handle_restart_with_game_control(
    context: DashboardControlContext, data: Dict[str, Any], emit_event: EventEmitter
) -> ControlAck:
    """Handle a restart-with-game request."""
    game_name = data.get("game")
    if not game_name or not context.on_restart_with_game_callback:
        return error_ack("restart_with_game", "Invalid game")
    if not is_known_game(game_name):
        return error_ack("restart_with_game", "Invalid game")

    context.publisher.log(f"Switching to {game_name}...", level="warning")
    save_ack = handle_save_control(context)
    if not save_ack["success"]:
        return error_ack("restart_with_game", save_ack["error"])
    ack = callback_ack(
        "restart_with_game",
        context.on_restart_with_game_callback,
        game_name,
        failure_message="Restart failed",
    )
    if not ack["success"]:
        return ack
    emit_event(
        "restarting",
        {"game": game_name, "message": f"Restarting with {game_name}..."},
    )
    return success_ack("restart_with_game")


def handle_go_to_launcher_control(
    context: DashboardControlContext, emit_event: EventEmitter
) -> ControlAck:
    """Handle a return-to-launcher request."""
    context.publisher.log("Returning to launcher...", level="warning")
    save_ack = handle_save_control(context)
    if not save_ack["success"]:
        return error_ack("go_to_launcher", save_ack["error"])

    ack = callback_ack(
        "go_to_launcher",
        context.on_save_and_quit_callback,
        failure_message="Launcher switch failed",
    )
    if not ack["success"]:
        return ack

    context.launcher_mode = True
    emit_event("redirect_to_launcher", {"message": "Returning to game launcher..."})
    return success_ack("go_to_launcher")


def dispatch_control(
    context: DashboardControlContext,
    data: Dict[str, Any],
    emit_event: EventEmitter,
) -> ControlAck:
    """Route a validated control payload to its action handler."""
    action = data.get("action")
    if action not in CONTROL_ACTIONS:
        return error_ack(str(action), "Unknown action")

    if action == "pause":
        return callback_ack("pause", context.on_pause_callback, failure_message="Pause failed")
    if action == "save":
        return handle_save_control(context)
    if action == "save_as":
        return handle_save_as_control(context, data)
    if action == "speed":
        return handle_speed_control(context, data)
    if action == "reset":
        return callback_ack("reset", context.on_reset_callback, failure_message="Reset failed")
    if action == "start_fresh":
        return handle_start_fresh_control(context, emit_event)
    if action == "load_model":
        return handle_load_model_control(context, data)
    if action == "config_change":
        return handle_config_change_control(context, data)
    if action == "performance_mode":
        return handle_performance_mode_control(context, data)
    if action == "save_and_quit":
        return handle_save_and_quit_control(context)
    if action == "select_game":
        return handle_select_game_control(context, data, emit_event)
    if action == "restart_with_game":
        return handle_restart_with_game_control(context, data, emit_event)
    if action == "go_to_launcher":
        return handle_go_to_launcher_control(context, emit_event)
    return success_ack(str(action))


def clear_logs(context: DashboardControlContext, emit_event: EventEmitter) -> ControlAck:
    """Clear dashboard logs and emit the empty log list."""
    context.publisher.console_logs.clear()
    emit_event("console_logs", {"logs": []})
    return success_ack("clear_logs")


def is_known_game(game_name: Any) -> bool:
    """Return whether a value names a registered game."""
    if not isinstance(game_name, str):
        return False
    from src.game import list_games

    return game_name in list_games()


def valid_performance_modes() -> set[str]:
    """Return the accepted dashboard performance mode names."""
    return set(PERFORMANCE_MODES.keys())


def parse_speed(value: Any) -> Optional[float]:
    """Parse and clamp dashboard speed input."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(speed) or speed <= 0:
        return None
    return max(1.0, min(1000.0, speed))


def normalize_config_change(
    config: Config, config_data: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any], str]:
    """Validate dashboard config changes before publishing them back to clients."""
    normalized: Dict[str, Any] = {}
    try:
        if "learning_rate" in config_data:
            learning_rate = float(config_data["learning_rate"])
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("Learning rate must be finite and positive")
            if learning_rate > 10.0:
                raise ValueError("Learning rate is unreasonably large")
            if learning_rate < 1e-10:
                raise ValueError("Learning rate is too small")
            normalized["learning_rate"] = learning_rate

        if "epsilon" in config_data:
            epsilon = float(config_data["epsilon"])
            if not math.isfinite(epsilon):
                raise ValueError("Epsilon must be finite")
            normalized["epsilon"] = max(config.EPSILON_END, min(config.EPSILON_START, epsilon))

        if "epsilon_decay" in config_data:
            epsilon_decay = float(config_data["epsilon_decay"])
            if not math.isfinite(epsilon_decay) or epsilon_decay <= 0 or epsilon_decay > 1:
                raise ValueError("Epsilon decay must be in (0, 1]")
            normalized["epsilon_decay"] = epsilon_decay

        if "gamma" in config_data:
            gamma = float(config_data["gamma"])
            if not math.isfinite(gamma) or gamma < 0 or gamma > 1:
                raise ValueError("Gamma must be in [0, 1]")
            normalized["gamma"] = gamma

        if "batch_size" in config_data:
            batch_size = int(config_data["batch_size"])
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
            if batch_size > config.MEMORY_SIZE:
                raise ValueError("Batch size cannot exceed memory size")
            normalized["batch_size"] = batch_size

        if "learn_every" in config_data:
            learn_every = int(config_data["learn_every"])
            if learn_every <= 0:
                raise ValueError("Learn every must be positive")
            normalized["learn_every"] = learn_every

        if "gradient_steps" in config_data:
            gradient_steps = int(config_data["gradient_steps"])
            if gradient_steps <= 0:
                raise ValueError("Gradient steps must be positive")
            normalized["gradient_steps"] = gradient_steps
    except (TypeError, ValueError) as exc:
        return False, {}, str(exc)

    return True, normalized, ""
