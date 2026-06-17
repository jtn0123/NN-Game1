"""Flask route registration for the web dashboard."""

from __future__ import annotations

from typing import Any, List

from flask import jsonify, make_response, render_template, request

from src.app.performance_modes import performance_mode_payload
from src.utils.logger import get_logger
from src.web.contracts import GameInfoPayload
from src.web.game_stats_service import build_game_stats
from src.web.json_utils import make_json_safe

_logger = get_logger(__name__)


def api_error(message: str, status: int):
    """Return the dashboard API's stable JSON error shape."""
    return jsonify({"error": message}), status


def register_dashboard_routes(dashboard: Any, content_security_policy: str) -> None:
    """Register Flask routes."""

    @dashboard.app.before_request
    def require_dashboard_token_for_api():
        if request.path.startswith("/api/") and not dashboard._is_authorized_request():
            return api_error("Unauthorized", 401)

    @dashboard.app.after_request
    def apply_security_headers(response):
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault(
            "Content-Security-Policy",
            content_security_policy,
        )
        return response

    @dashboard.app.route("/")
    def index():
        if not dashboard._is_authorized_request():
            response = make_response("Dashboard token required", 401)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        if dashboard.launcher_mode:
            response = make_response(
                render_template("launcher.html", access_token=dashboard.access_token)
            )
        else:
            response = make_response(
                render_template(
                    "dashboard.html",
                    access_token=dashboard.access_token,
                    control_token=dashboard.access_token,
                )
            )
        # Prevent browser caching to ensure fresh content
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @dashboard.app.route("/api/status")
    def api_status():
        snapshot = dashboard.publisher.get_snapshot()
        snapshot["launcher_mode"] = dashboard.launcher_mode
        return jsonify(snapshot)

    @dashboard.app.route("/api/config")
    def api_config():
        return jsonify(
            {
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
        )

    @dashboard.app.route("/api/games")
    def api_games():
        """List all available games with their metadata."""
        from src.game import list_games, get_game_info

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

        return jsonify({"games": games, "current_game": dashboard.config.GAME_NAME})

    @dashboard.app.route("/api/performance-modes")
    def api_performance_modes():
        """List dashboard performance-mode presets."""
        return jsonify({"modes": performance_mode_payload()})

    @dashboard.app.route("/api/screenshot")
    def api_screenshot():
        # If headless mode, return early with flag (no screenshots available)
        if dashboard.publisher.state.headless:
            return jsonify({"image": None, "headless": True})
        screenshot = dashboard.publisher.get_screenshot()
        if screenshot:
            return jsonify({"image": screenshot, "headless": False})
        return jsonify({"image": None, "headless": False})

    @dashboard.app.route("/api/models")
    def api_models():
        """List available model files with metadata.

        Searches both game-specific directory and legacy models directory.
        """
        return jsonify(
            {
                "models": dashboard.model_service.list_models(),
                "current_game": dashboard.config.GAME_NAME,
            }
        )

    @dashboard.app.route("/api/models/<path:model_id>", methods=["DELETE"])
    def api_delete_model(model_id):
        """Delete a model file.

        Security: Validates that the path is within the model directory
        to prevent path traversal attacks.
        """
        if not dashboard._is_authorized_request():
            return api_error("Unauthorized", 401)

        try:
            success, filename, error = dashboard.model_service.delete(model_id)
            if not success:
                status = 404 if error == "Model not found" else 403
                if error == "Invalid file type":
                    status = 400
                return api_error(error, status)
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
    def api_save_status():
        """Get last save information."""
        return jsonify(dashboard.publisher.get_save_status())

    @dashboard.app.route("/api/game-stats")
    def api_game_stats():
        """Get training statistics for all games (for comparison panel)."""
        stats = build_game_stats(dashboard.config)
        return jsonify({"stats": stats, "current_game": dashboard.config.GAME_NAME})

    # ===== Phase 2: Neuron Inspection & Layer Analysis Endpoints =====

    @dashboard.app.route("/api/neuron/<int:layer_idx>/<int:neuron_idx>")
    def api_neuron_details(layer_idx, neuron_idx):
        """Phase 2: Get details for a specific neuron."""
        details = dashboard.publisher.get_neuron_details(layer_idx, neuron_idx)
        return jsonify(make_json_safe(details))

    @dashboard.app.route("/api/layer/<int:layer_idx>")
    def api_layer_analysis(layer_idx):
        """Phase 2: Get analysis data for a specific layer."""
        analysis = dashboard.publisher.get_layer_analysis(layer_idx)
        return jsonify(make_json_safe(analysis))

    @dashboard.app.route("/api/layers")
    def api_layers_analysis():
        """Phase 2: Get analysis data for all layers."""
        analysis = dashboard.publisher.get_all_layer_analysis()
        return jsonify(make_json_safe(analysis))
