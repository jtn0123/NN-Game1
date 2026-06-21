"""
Web Dashboard Server
====================

Flask + SocketIO server for real-time training visualization.

Features:
    - REST API for model info and current stats
    - WebSocket events for live metrics streaming
    - Runs in background thread alongside training
    - Training controls (pause, save, adjust speed, reset, load model)
    - Real-time console logging with filterable log levels

Usage:
    >>> from src.web import WebDashboard
    >>> dashboard = WebDashboard(port=5000)
    >>> dashboard.start()
    >>> # ... during training ...
    >>> dashboard.emit_metrics(episode, score, loss, ...)
    >>> dashboard.log("Training started", level="info")
    >>> dashboard.stop()
"""

import base64
import os
import secrets
import socket
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np

DASHBOARD_CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "connect-src 'self' ws: wss:; "
    "img-src 'self' data:; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.socket.io; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)

try:
    # Suppress werkzeug logging BEFORE importing Flask
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    logging.getLogger("werkzeug").disabled = True

    from flask import Flask, request
    from flask_socketio import SocketIO, emit

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Web dashboard unavailable.")
    print("Install with: pip install flask flask-socketio eventlet")

from config import Config
from src.app.model_paths import model_id, model_search_dirs
from src.utils.logger import get_logger
from src.web import socket_controls
from src.web.contracts import ControlAck
from src.web.json_utils import make_json_safe
from src.web.metrics_publisher import (
    LayerAnalysisData,
    LogLevel,
    LogMessage,
    MetricsPublisher,
    NeuronInspectionData,
    NNVisualizationData,
    SaveStatus,
    TrainingConfig,
    TrainingState,
)
from src.web.model_service import ModelService

# Module logger
_logger = get_logger(__name__)

__all__ = [
    "FLASK_AVAILABLE",
    "DASHBOARD_CONTENT_SECURITY_POLICY",
    "WebDashboard",
    "create_web_templates",
    "MetricsPublisher",
    "TrainingState",
    "TrainingConfig",
    "SaveStatus",
    "LogLevel",
    "LogMessage",
    "NNVisualizationData",
    "NeuronInspectionData",
    "LayerAnalysisData",
    "_make_json_safe",
]


def _make_json_safe(obj: Any) -> Any:
    """Compatibility wrapper for callers that import from this module."""
    return make_json_safe(obj)


class WebDashboard:
    """
    Flask web dashboard for training visualization.

    Runs a web server in a background thread that serves
    a real-time dashboard with charts, controls, and console logs.

    Example:
        >>> dashboard = WebDashboard(port=5000)
        >>> dashboard.start()
        >>> # Training loop...
        >>> dashboard.publisher.update(episode, score, ...)
        >>> dashboard.log("Episode complete", level="success")
        >>> dashboard.stop()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        port: int = 5000,
        host: str = "127.0.0.1",
        launcher_mode: bool = False,
    ):
        """
        Initialize the web dashboard.

        Args:
            config: Configuration object
            port: Port to run the server on
            host: Host address (0.0.0.0 for all interfaces)
            launcher_mode: If True, show game selection instead of training dashboard
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError(
                "Flask is not installed. Run: pip install flask flask-socketio eventlet"
            )

        self.config = config or Config()
        self.port = port
        self.host = host
        self.launcher_mode = launcher_mode
        self.access_token = os.environ.get("NN_GAME_DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
        self.model_service = ModelService(self._model_search_dirs())
        self.on_game_selected_callback: Optional[Callable[[str, str], None]] = None  # (game, mode)
        self.on_restart_with_game_callback: Optional[Callable[[str], None]] = None

        # Metrics publisher
        self.publisher = MetricsPublisher()
        # Set memory capacity from config
        self.publisher.state.memory_capacity = self.config.MEMORY_SIZE

        # Flask app setup - use absolute paths relative to this module
        base_dir = os.path.dirname(__file__)
        template_dir = os.path.join(base_dir, "templates")
        static_dir = os.path.join(base_dir, "static")

        self.app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
        # Generate secure random secret key (not hardcoded for security)
        self.app.config["SECRET_KEY"] = base64.b64encode(os.urandom(24)).decode("utf-8")

        # SocketIO setup
        allowed_origins = self._socketio_allowed_origins()
        self.socketio = SocketIO(
            self.app, cors_allowed_origins=allowed_origins, async_mode="threading"
        )

        # Register routes
        self._register_routes()
        self._register_socket_events()

        # Server thread
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # Control callbacks
        self.on_pause_callback: Optional[Callable[[], None]] = None
        self.on_save_callback: Optional[Callable[[], None]] = None
        self.on_save_as_callback: Optional[Callable[[str], None]] = None
        self.on_speed_callback: Optional[Callable[[float], None]] = None
        self.on_reset_callback: Optional[Callable[[], None]] = None
        self.on_start_fresh_callback: Optional[Callable[[], None]] = None
        self.on_load_model_callback: Optional[Callable[[str], None]] = None
        self.on_config_change_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_performance_mode_callback: Optional[Callable[[str], None]] = None
        self.on_save_and_quit_callback: Optional[Callable[[], None]] = None

    def _is_authorized_token(self, token: Optional[str]) -> bool:
        """Validate a dashboard mutation token."""
        return bool(token) and secrets.compare_digest(str(token), self.access_token)

    def _is_authorized_request(self) -> bool:
        """Validate mutating HTTP requests from the served dashboard."""
        token = request.headers.get("X-Dashboard-Token") or request.args.get("token")
        return self._is_authorized_token(token)

    def dashboard_url(self) -> str:
        """Return the tokenized URL needed to open the dashboard page."""
        display_host = self.host
        if display_host in {"0.0.0.0", "::"}:
            display_host = "127.0.0.1"
        query = urlencode({"token": self.access_token})
        return f"http://{display_host}:{self.port}/?{query}"

    def dashboard_network_url(self) -> str:
        """Return a best-effort LAN URL for dashboards bound to all interfaces."""
        display_host = self._network_host()
        query = urlencode({"token": self.access_token})
        return f"http://{display_host}:{self.port}/?{query}"

    def _network_host(self) -> str:
        """Return a best-effort LAN host for wildcard dashboard bindings."""
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"

    def _socketio_allowed_origins(self) -> List[str]:
        """Return browser origins allowed to open the live Socket.IO channel."""
        allowed_hosts = {self.host, "localhost", "127.0.0.1"}
        if self.host in {"0.0.0.0", "::"}:
            allowed_hosts.add(self._network_host())
        return [f"http://{host}:{self.port}" for host in sorted(allowed_hosts)]

    def _model_search_dirs(self) -> List[Tuple[str, str]]:
        """Return allowed model directories as (directory, source) pairs."""
        return model_search_dirs(self.config)

    def _refresh_model_service(self) -> None:
        """Keep model browsing/loading aligned with the current game config."""
        current_dirs = self._model_search_dirs()
        if self.model_service.model_dirs != current_dirs:
            self.model_service = ModelService(current_dirs)

    @staticmethod
    def _model_id(source: str, filename: str) -> str:
        """Create a browser-safe model identifier without exposing local paths."""
        return model_id(source, filename)

    def _resolve_model_ref(self, model_ref: str) -> Optional[str]:
        """Resolve a model id, or a legacy absolute path, to an allowed .pth file."""
        self._refresh_model_service()
        return self.model_service.resolve(model_ref)

    @staticmethod
    def _success_ack(action: str) -> ControlAck:
        return socket_controls.success_ack(action)

    @staticmethod
    def _error_ack(action: str, error: str) -> ControlAck:
        return socket_controls.error_ack(action, error)

    @staticmethod
    def _unauthorized_ack() -> ControlAck:
        return socket_controls.unauthorized_ack()

    def _callback_ack(
        self,
        action: str,
        callback: Optional[Callable[..., Any]],
        *args: Any,
        failure_message: str,
    ) -> ControlAck:
        return socket_controls.callback_ack(
            action,
            callback,
            *args,
            failure_message=failure_message,
        )

    def _handle_save_control(self) -> ControlAck:
        return socket_controls.handle_save_control(self)

    def _handle_save_as_control(self, data: Dict[str, Any]) -> ControlAck:
        return socket_controls.handle_save_as_control(self, data)

    def _handle_start_fresh_control(self) -> ControlAck:
        return socket_controls.handle_start_fresh_control(self, emit)

    def _handle_speed_control(self, data: Dict[str, Any]) -> ControlAck:
        return socket_controls.handle_speed_control(self, data)

    def _handle_config_change_control(self, data: Dict[str, Any]) -> ControlAck:
        return socket_controls.handle_config_change_control(self, data)

    def _handle_performance_mode_control(self, data: Dict[str, Any]) -> ControlAck:
        return socket_controls.handle_performance_mode_control(self, data)

    def _handle_save_and_quit_control(self) -> ControlAck:
        return socket_controls.handle_save_and_quit_control(self)

    def _handle_select_game_control(self, data: Dict[str, Any]) -> ControlAck:
        return socket_controls.handle_select_game_control(self, data, emit)

    def _handle_load_model_control(self, data: Dict[str, Any]) -> ControlAck:
        return socket_controls.handle_load_model_control(self, data)

    def _handle_restart_with_game_control(self, data: Dict[str, Any]) -> ControlAck:
        return socket_controls.handle_restart_with_game_control(self, data, emit)

    @staticmethod
    def _is_known_game(game_name: Any) -> bool:
        return socket_controls.is_known_game(game_name)

    @staticmethod
    def _valid_performance_modes() -> set[str]:
        return socket_controls.valid_performance_modes()

    @staticmethod
    def _parse_speed(value: Any) -> Optional[float]:
        return socket_controls.parse_speed(value)

    def _normalize_config_change(
        self, config_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], str]:
        return socket_controls.normalize_config_change(self.config, config_data)

    def _handle_go_to_launcher_control(self) -> ControlAck:
        return socket_controls.handle_go_to_launcher_control(self, emit)

    def _register_routes(self) -> None:
        """Register Flask routes."""
        from src.web.routes import register_dashboard_routes

        register_dashboard_routes(self, DASHBOARD_CONTENT_SECURITY_POLICY)

    def _register_socket_events(self) -> None:
        """Register SocketIO events."""

        @self.socketio.on("connect")
        def handle_connect(auth=None):
            auth = auth or {}
            if not self._is_authorized_token(auth.get("token")):
                return False
            # Send current state on connect (convert NumPy types for JSON serialization)
            emit(
                "state_update",
                _make_json_safe(
                    self.publisher.get_snapshot(
                        history_limit=self.publisher.DEFAULT_SNAPSHOT_HISTORY_LIMIT
                    )
                ),
            )
            # Send recent logs
            emit(
                "console_logs",
                {"logs": _make_json_safe(self.publisher.get_console_logs(100))},
            )
            # Send current NN visualization data if available
            nn_data = self.publisher.get_nn_visualization()
            if nn_data.get("layer_info"):
                emit("nn_update", _make_json_safe(nn_data))

        @self.socketio.on("control")
        def handle_control(data):
            data = data if isinstance(data, dict) else {}
            if not self._is_authorized_token(data.get("token")):
                emit("control_error", {"error": "Unauthorized"})
                return self._unauthorized_ack()

            return socket_controls.dispatch_control(self, data, emit)

        @self.socketio.on("clear_logs")
        def handle_clear_logs(data=None):
            data = data or {}
            if not self._is_authorized_token(data.get("token")):
                emit("control_error", {"error": "Unauthorized"})
                return self._unauthorized_ack()
            return socket_controls.clear_logs(self, emit)

        # Auto-emit on metric updates
        # Bug 65 fix: null/running checks prevent AttributeError during shutdown.
        # These callbacks run from the training thread; a dead/disconnected client can
        # make socketio.emit() raise, which would propagate up and silence ALL future
        # metric emits. Swallow emit errors so one bad client can't black out the run.
        def _safe_emit(event: str, payload: Any) -> None:
            if not (self.socketio and self._running):
                return
            try:
                self.socketio.emit(event, payload)
            except Exception as exc:
                logging.getLogger(__name__).debug("socket emit %s failed: %s", event, exc)

        def broadcast_update(snapshot):
            _safe_emit("state_update", snapshot)

        def broadcast_log(log_entry: LogMessage):
            _safe_emit("console_log", log_entry.to_dict())

        def broadcast_save(save_info: Dict[str, Any]):
            _safe_emit("save_event", save_info)

        def broadcast_nn_update(nn_data: Dict[str, Any]):
            _safe_emit("nn_update", nn_data)

        # Clear and register callbacks atomically (prevents race condition)
        with self.publisher._callback_lock:
            self.publisher._on_update_callbacks.clear()
            self.publisher._on_log_callbacks.clear()
            self.publisher._on_save_callbacks.clear()
            self.publisher._on_nn_update_callbacks.clear()
            # Register new callbacks inside the lock to prevent race condition
            self.publisher._on_update_callbacks.append(broadcast_update)
            self.publisher._on_log_callbacks.append(broadcast_log)
            self.publisher._on_save_callbacks.append(broadcast_save)
            self.publisher._on_nn_update_callbacks.append(broadcast_nn_update)

    def start(self) -> None:
        """Start the web server in a background thread."""
        if self._running:
            return

        self._running = True
        self.publisher.set_running(True)

        # Suppress Flask/werkzeug logging COMPLETELY - do this BEFORE starting server
        import logging

        # Disable werkzeug request logging completely
        werkzeug_log = logging.getLogger("werkzeug")
        werkzeug_log.setLevel(logging.ERROR)
        werkzeug_log.disabled = True

        # Also suppress Flask's internal logger
        flask_log = logging.getLogger("flask.app")
        flask_log.setLevel(logging.ERROR)

        # Suppress socketio and engineio loggers
        logging.getLogger("engineio").setLevel(logging.ERROR)
        logging.getLogger("socketio").setLevel(logging.ERROR)
        logging.getLogger("engineio.server").setLevel(logging.ERROR)
        logging.getLogger("socketio.server").setLevel(logging.ERROR)

        # Redirect werkzeug's output stream to devnull
        # This catches cases where werkzeug bypasses the logging system
        class NullWriter:
            def write(self, *args, **kwargs):
                pass

            def flush(self, *args, **kwargs):
                pass

        # Disable Flask's click CLI echo (used by werkzeug for request logging)
        try:
            import click

            click.echo = lambda *args, **kwargs: None
        except ImportError:
            _logger.debug("click module not available, skipping echo suppression")

        def run_server():
            _logger.info(f"Web Dashboard running at {self.dashboard_url()}")

            try:
                self.socketio.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    log_output=False,
                    allow_unsafe_werkzeug=True,
                )
            # Bug 76 fix: Catch broader exceptions to prevent silent thread crashes
            except (OSError, RuntimeError, ConnectionError, Exception) as e:
                _logger.error(
                    f"Failed to start web dashboard on port {self.port}: {type(e).__name__}: {e}"
                )
                _logger.error(
                    f"Port {self.port} may already be in use. Try a different port with --port"
                )

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

    def stop(self) -> None:
        """Stop the web server and release the port."""
        self._running = False
        self.publisher.set_running(False)

        # Try to stop the SocketIO server (releases the port)
        try:
            if hasattr(self.socketio, "stop"):
                self.socketio.stop()
        except Exception as e:
            # Best effort - daemon thread will die with process anyway
            _logger.debug(f"Server stop (best effort): {e}")

    def emit_metrics(self, **kwargs) -> None:
        """Convenience method to update and emit metrics."""
        self.publisher.update(**kwargs)

    def log(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a log message to the console.

        Args:
            message: Log message text
            level: One of 'debug', 'info', 'success', 'warning', 'error', 'metric', 'action'
            data: Optional dictionary of additional data
        """
        self.publisher.log(message, level, data)

    def capture_screenshot(self, surface) -> None:
        """Capture and store a game screenshot."""
        self.publisher.set_screenshot(surface)

    def emit_nn_visualization(
        self,
        layer_info: List[Dict[str, Any]],
        activations: Dict[str, List[float]],
        q_values: List[float],
        selected_action: int,
        weights: List[List[List[float]]],
        step: int,
        action_labels: Optional[List[str]] = None,
        input_state: Optional[List[float]] = None,
        analysis_activations: Optional[Dict[str, List[float]]] = None,
        analysis_weights: Optional[List[List[List[float]]]] = None,
    ) -> None:
        """
        Emit neural network visualization data to connected clients.

        This is a convenience method that updates the publisher's NN data.
        The update is throttled to ~10 FPS internally.

        Args:
            layer_info: List of layer info dicts with 'name', 'neurons', 'type'
            activations: Dict mapping layer keys to lists of activation values
            q_values: Q-values for each action
            selected_action: Currently selected action index
            weights: Sampled weight matrices as nested lists
            step: Current training step
            action_labels: Labels for each action (e.g., ["LEFT", "STAY", "RIGHT"])
        """
        if not self.publisher.should_update_nn_visualization():
            return

        self._sync_phase2_inspection(
            layer_info=layer_info,
            activations=(analysis_activations if analysis_activations is not None else activations),
            q_values=q_values,
            weights=analysis_weights if analysis_weights is not None else weights,
            input_state=input_state,
            action_labels=action_labels,
        )

        self.publisher.update_nn_visualization(
            layer_info=layer_info,
            activations=activations,
            q_values=q_values,
            selected_action=selected_action,
            weights=weights,
            step=step,
            action_labels=action_labels,
        )

    def _sync_phase2_inspection(
        self,
        layer_info: List[Dict[str, Any]],
        activations: Dict[str, List[float]],
        q_values: List[float],
        weights: List[List[List[float]]],
        input_state: Optional[List[float]],
        action_labels: Optional[List[str]],
    ) -> None:
        """Populate neuron and layer inspection data from live NN snapshots."""
        weight_arrays = [np.asarray(weight) for weight in weights]
        linear_keys = self._sorted_linear_activation_keys(activations)
        hidden_weight_count = sum(1 for info in layer_info if info.get("type") == "hidden")
        hidden_weight_idx = 0
        value_weight_idx = hidden_weight_count
        advantage_weight_idx = hidden_weight_count + 1
        value_output_weight_idx = hidden_weight_count + 2
        advantage_output_weight_idx = hidden_weight_count + 3

        def weight_at(index: int) -> Optional[np.ndarray]:
            if 0 <= index < len(weight_arrays):
                return weight_arrays[index]
            return None

        def output_weights() -> Optional[np.ndarray]:
            value_output = weight_at(value_output_weight_idx)
            advantage_output = weight_at(advantage_output_weight_idx)
            if value_output is None and advantage_output is None:
                return None
            if value_output is None:
                return advantage_output
            if advantage_output is None:
                return value_output
            if value_output.ndim < 2 or advantage_output.ndim < 2:
                return None
            repeated_value = np.repeat(value_output, advantage_output.shape[0], axis=0)
            return np.concatenate([repeated_value, advantage_output], axis=1)

        for layer_idx, info in enumerate(layer_info):
            layer_type = info.get("type", "")
            layer_name = info.get("name", f"Layer {layer_idx}")
            neuron_count = int(info.get("neurons", 0))

            if layer_type == "input":
                layer_acts = np.asarray(input_state if input_state is not None else [])
            elif layer_type == "value_stream":
                layer_acts = np.asarray(activations.get("value_hidden", []))
            elif layer_type == "advantage_stream":
                layer_acts = np.asarray(activations.get("advantage_hidden", []))
            elif hidden_weight_idx < len(linear_keys):
                layer_acts = np.asarray(activations.get(linear_keys[hidden_weight_idx], []))
            elif layer_type == "output":
                layer_acts = np.asarray(q_values)
            else:
                layer_acts = np.asarray([])

            if layer_acts.ndim > 1:
                layer_acts = layer_acts[0]
            layer_acts = layer_acts.reshape(-1)

            incoming_weights = None
            next_weights = None
            if layer_type == "hidden":
                incoming_weights = weight_at(hidden_weight_idx)
                hidden_weight_idx += 1
                if hidden_weight_idx < hidden_weight_count:
                    next_weights = weight_at(hidden_weight_idx)
                else:
                    branch_weights = [
                        candidate
                        for candidate in (
                            weight_at(value_weight_idx),
                            weight_at(advantage_weight_idx),
                        )
                        if candidate is not None
                    ]
                    next_weights = (
                        np.concatenate(branch_weights, axis=0) if branch_weights else None
                    )
            elif layer_type == "value_stream":
                incoming_weights = weight_at(value_weight_idx)
                next_weights = weight_at(value_output_weight_idx)
            elif layer_type == "advantage_stream":
                incoming_weights = weight_at(advantage_weight_idx)
                next_weights = weight_at(advantage_output_weight_idx)
            elif layer_type == "output":
                incoming_weights = (
                    output_weights()
                    if value_output_weight_idx < len(weight_arrays)
                    else weight_at(hidden_weight_idx)
                )

            self.publisher.update_layer_analysis(
                layer_idx=layer_idx,
                layer_name=layer_name,
                neuron_count=neuron_count,
                activations=layer_acts,
                weights=incoming_weights,
            )

            for neuron_idx, activation in enumerate(layer_acts[:neuron_count]):
                neuron_incoming = None
                if (
                    incoming_weights is not None
                    and incoming_weights.ndim >= 2
                    and neuron_idx < incoming_weights.shape[0]
                ):
                    neuron_incoming = incoming_weights[neuron_idx].tolist()

                neuron_outgoing = None
                if (
                    next_weights is not None
                    and next_weights.ndim >= 2
                    and neuron_idx < next_weights.shape[1]
                ):
                    neuron_outgoing = next_weights[:, neuron_idx].tolist()

                q_contributions = None
                if layer_type == "output" and neuron_idx < len(q_values):
                    label = (
                        action_labels[neuron_idx]
                        if action_labels and neuron_idx < len(action_labels)
                        else f"action_{neuron_idx}"
                    )
                    q_contributions = {label: float(q_values[neuron_idx])}

                self.publisher.update_neuron_inspection(
                    layer_idx=layer_idx,
                    neuron_idx=neuron_idx,
                    layer_name=layer_name,
                    current_activation=float(activation),
                    incoming_weights=neuron_incoming,
                    outgoing_weights=neuron_outgoing,
                    q_contributions=q_contributions,
                )

    @staticmethod
    def _sorted_linear_activation_keys(
        activations: Dict[str, List[float]],
    ) -> List[str]:
        """Return valid layer_<n> activation keys in numeric order."""
        parsed_keys: List[Tuple[int, str]] = []
        for key in activations:
            if not key.startswith("layer_"):
                continue
            suffix = key.split("_", 1)[1]
            if not suffix.isdigit():
                continue
            parsed_keys.append((int(suffix), key))
        return [key for _, key in sorted(parsed_keys)]


# Create templates directory structure
def create_web_templates():
    """Helper to create web template files."""
    import os

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    return template_dir, static_dir


if __name__ == "__main__":
    print("Web Dashboard Server")
    print("Import and use with training loop")
