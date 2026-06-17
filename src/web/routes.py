"""Flask route registration for the web dashboard."""

from __future__ import annotations

from typing import Any, List, Protocol, Tuple, cast

from flask import jsonify, make_response, redirect, render_template, request, url_for
from werkzeug import Response

from src.app.performance_modes import performance_mode_payload
from src.utils.logger import get_logger
from src.web.contracts import (
    DashboardConfigPayload,
    GameInfoPayload,
    GamesResponse,
    ModelPayload,
    ModelsResponse,
)
from src.web.game_stats_service import build_game_stats
from src.web.json_utils import make_json_safe

_logger = get_logger(__name__)

RouteResponse = Response | Tuple[Response, int]


class DashboardConfigContext(Protocol):
    """Config fields read by dashboard routes."""

    BATCH_SIZE: int
    DEVICE: Any
    EPSILON_DECAY: float
    EPSILON_END: float
    EPSILON_START: float
    GAMMA: float
    GAME_NAME: str
    GRAD_CLIP: float
    GRADIENT_STEPS: int
    HIDDEN_LAYERS: Any
    LEARN_EVERY: int
    LEARNING_RATE: float
    MEMORY_SIZE: int
    TARGET_UPDATE: int


class DashboardModelServiceContext(Protocol):
    """Model-service methods used by dashboard routes."""

    def list_models(self) -> List[dict[str, Any]]:
        """Return browser-safe model metadata."""
        ...

    def delete(self, model_id: str) -> Tuple[bool, str | None, str | None]:
        """Delete a model by opaque id."""
        ...


class DashboardPublisherStateContext(Protocol):
    """Publisher state fields read directly by route handlers."""

    headless: bool
    num_envs: int


class DashboardPublisherContext(Protocol):
    """Publisher methods used by dashboard routes."""

    DEFAULT_SNAPSHOT_HISTORY_LIMIT: int
    history_length: int
    state: DashboardPublisherStateContext

    def get_snapshot(self, history_limit: int) -> dict[str, Any]:
        """Return a bounded dashboard snapshot."""
        ...

    def get_screenshot(self) -> str | None:
        """Return an encoded screenshot, if available."""
        ...

    def log(self, message: str, level: str = "info", data: dict[str, Any] | None = None) -> None:
        """Publish a dashboard log entry."""
        ...

    def get_save_status(self) -> dict[str, Any]:
        """Return the latest save status."""
        ...

    def get_neuron_details(self, layer_idx: int, neuron_idx: int) -> dict[str, Any]:
        """Return neuron inspection details."""
        ...

    def get_layer_analysis(self, layer_idx: int) -> dict[str, Any]:
        """Return layer analysis details."""
        ...

    def get_all_layer_analysis(self) -> List[dict[str, Any]]:
        """Return all layer analysis details."""
        ...


class DashboardRouteContext(Protocol):
    """Dashboard attributes consumed by Flask route registration."""

    app: Any
    access_token: str
    config: DashboardConfigContext
    launcher_mode: bool
    model_service: DashboardModelServiceContext
    publisher: DashboardPublisherContext

    def _is_authorized_request(self) -> bool:
        """Return whether the current Flask request is authorized."""
        ...

    def _is_authorized_bootstrap_request(self) -> bool:
        """Return whether the current request has a valid bootstrap URL token."""
        ...

    def _set_session_cookie(self, response: Response) -> Response:
        """Attach the dashboard session cookie to a response."""
        ...

    def _control_retry_after(self, action: str) -> float | None:
        """Return retry seconds when a mutating action is throttled."""
        ...


def api_error(message: str, status: int) -> Tuple[Response, int]:
    """Return the dashboard API's stable JSON error shape."""
    return jsonify({"error": message}), status


def model_delete_error_status(error: str | None) -> int:
    """Map model-service delete errors to stable HTTP statuses."""
    if error == "Model not found":
        return 404
    if error == "Invalid file type":
        return 400
    return 403


def parse_history_limit(value: str | None, default: int, maximum: int) -> int:
    """Parse a bounded dashboard history window size."""
    if value is None:
        return default
    try:
        limit = int(value)
    except ValueError:
        return default
    return max(0, min(limit, maximum))


def register_dashboard_routes(
    dashboard: DashboardRouteContext, content_security_policy: str
) -> None:
    """Register Flask routes."""

    @dashboard.app.before_request
    def require_dashboard_token_for_api() -> RouteResponse | None:
        if request.path.startswith("/api/") and not dashboard._is_authorized_request():
            return api_error("Unauthorized", 401)
        return None

    @dashboard.app.after_request
    def apply_security_headers(response: Response) -> Response:
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault(
            "Content-Security-Policy",
            content_security_policy,
        )
        return response

    @dashboard.app.route("/")
    def index() -> Response:
        if request.args.get("token"):
            if not dashboard._is_authorized_bootstrap_request():
                response = make_response("Dashboard token required", 401)
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response

            response = make_response(redirect(url_for("index"), code=303))
            dashboard._set_session_cookie(response)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        if not dashboard._is_authorized_request():
            response = make_response("Dashboard token required", 401)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        if dashboard.launcher_mode:
            response = make_response(render_template("launcher.html"))
        else:
            response = make_response(render_template("dashboard.html"))
        # Prevent browser caching to ensure fresh content
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @dashboard.app.route("/api/status")
    def api_status() -> Response:
        history_limit = parse_history_limit(
            request.args.get("history_limit"),
            dashboard.publisher.DEFAULT_SNAPSHOT_HISTORY_LIMIT,
            dashboard.publisher.history_length,
        )
        snapshot = dashboard.publisher.get_snapshot(history_limit=history_limit)
        snapshot["launcher_mode"] = dashboard.launcher_mode
        return jsonify(snapshot)

    @dashboard.app.route("/api/config")
    def api_config() -> Response:
        payload: DashboardConfigPayload = {
            "learning_rate": dashboard.config.LEARNING_RATE,
            "gamma": dashboard.config.GAMMA,
            "epsilon_start": dashboard.config.EPSILON_START,
            "epsilon_end": dashboard.config.EPSILON_END,
            "epsilon_decay": dashboard.config.EPSILON_DECAY,
            "batch_size": dashboard.config.BATCH_SIZE,
            "hidden_layers": dashboard.config.HIDDEN_LAYERS,
            "memory_size": dashboard.config.MEMORY_SIZE,
            "target_update": dashboard.config.TARGET_UPDATE,
            "grad_clip": dashboard.config.GRAD_CLIP,
            # Performance settings
            "learn_every": dashboard.config.LEARN_EVERY,
            "gradient_steps": dashboard.config.GRADIENT_STEPS,
            "device": str(dashboard.config.DEVICE),
            "vec_envs": dashboard.publisher.state.num_envs,
            # Game settings
            "game_name": dashboard.config.GAME_NAME,
        }
        return jsonify(payload)

    @dashboard.app.route("/api/games")
    def api_games() -> Response:
        """List all available games with their metadata."""
        from src.game import get_game_info, list_games

        games: List[GameInfoPayload] = []
        for game_id in list_games():
            info = get_game_info(game_id)
            if info:
                games.append(
                    {
                        "id": game_id,
                        "name": info.get("name", game_id.title()),
                        "description": info.get("description", ""),
                        "actions": info.get("actions", []),
                        "controls": info.get("controls", []),
                        "difficulty": info.get("difficulty", "Unknown"),
                        "icon": info.get("icon", "🎮"),
                        "color": info.get("color", (100, 100, 100)),
                        "is_current": game_id == dashboard.config.GAME_NAME,
                    }
                )

        payload: GamesResponse = {"games": games, "current_game": dashboard.config.GAME_NAME}
        return jsonify(payload)

    @dashboard.app.route("/api/performance-modes")
    def api_performance_modes() -> Response:
        """List dashboard performance-mode presets."""
        return jsonify({"modes": performance_mode_payload()})

    @dashboard.app.route("/api/screenshot")
    def api_screenshot() -> Response:
        # If headless mode, return early with flag (no screenshots available)
        if dashboard.publisher.state.headless:
            return jsonify({"image": None, "headless": True})
        screenshot = dashboard.publisher.get_screenshot()
        if screenshot:
            return jsonify({"image": screenshot, "headless": False})
        return jsonify({"image": None, "headless": False})

    @dashboard.app.route("/api/models")
    def api_models() -> Response:
        """List available model files with metadata.

        Searches both game-specific directory and legacy models directory.
        """
        payload: ModelsResponse = {
            "models": cast(List[ModelPayload], dashboard.model_service.list_models()),
            "current_game": dashboard.config.GAME_NAME,
        }
        return jsonify(payload)

    @dashboard.app.route("/api/models/<path:model_id>", methods=["DELETE"])
    def api_delete_model(model_id: str) -> RouteResponse:
        """Delete a model file.

        Security: Validates that the path is within the model directory
        to prevent path traversal attacks.
        """
        if dashboard._control_retry_after("delete_model") is not None:
            return api_error("Too many requests", 429)

        try:
            success, filename, error = dashboard.model_service.delete(model_id)
            if not success:
                return api_error(
                    error or "Failed to delete model", model_delete_error_status(error)
                )
            dashboard.publisher.log(f"🗑️ Deleted model: {filename}", level="action")
            return jsonify(
                {
                    "success": True,
                    "message": f"Model {filename} deleted successfully",
                    "filename": filename,
                }
            )
        except Exception:
            _logger.exception("Failed to delete model")
            return api_error("Failed to delete model", 500)

    @dashboard.app.route("/api/save-status")
    def api_save_status() -> Response:
        """Get last save information."""
        return jsonify(dashboard.publisher.get_save_status())

    @dashboard.app.route("/api/game-stats")
    def api_game_stats() -> Response:
        """Get training statistics for all games (for comparison panel)."""
        stats = build_game_stats(dashboard.config)
        return jsonify({"stats": stats, "current_game": dashboard.config.GAME_NAME})

    # ===== Phase 2: Neuron Inspection & Layer Analysis Endpoints =====

    @dashboard.app.route("/api/neuron/<int:layer_idx>/<int:neuron_idx>")
    def api_neuron_details(layer_idx: int, neuron_idx: int) -> Response:
        """Phase 2: Get details for a specific neuron."""
        details = dashboard.publisher.get_neuron_details(layer_idx, neuron_idx)
        return jsonify(make_json_safe(details))

    @dashboard.app.route("/api/layer/<int:layer_idx>")
    def api_layer_analysis(layer_idx: int) -> Response:
        """Phase 2: Get analysis data for a specific layer."""
        analysis = dashboard.publisher.get_layer_analysis(layer_idx)
        return jsonify(make_json_safe(analysis))

    @dashboard.app.route("/api/layers")
    def api_layers_analysis() -> Response:
        """Phase 2: Get analysis data for all layers."""
        analysis = dashboard.publisher.get_all_layer_analysis()
        return jsonify(make_json_safe(analysis))
