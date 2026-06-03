"""Socket.IO control tests for the web dashboard."""

import json
import os

import numpy as np
import pytest
import torch

try:
    from src.web.server import FLASK_AVAILABLE

    WEB_AVAILABLE = FLASK_AVAILABLE
except ImportError:
    WEB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not WEB_AVAILABLE, reason="Flask/SocketIO not installed")


class TestWebDashboardSocketControls:

    @pytest.fixture
    def web_dashboard(self):
        """Create a WebDashboard instance for testing."""
        try:
            from src.web.server import WebDashboard
            from config import Config

            config = Config()
            config.GAME_NAME = "breakout"

            dashboard = WebDashboard(port=5099, config=config)
            yield dashboard

            # Cleanup
            try:
                dashboard.stop()
            except Exception:
                pass
        except ImportError:
            pytest.skip("WebDashboard not available")

    def test_socket_requires_dashboard_token(self, web_dashboard):
        """Socket.IO clients need the dashboard token to connect."""
        unauthorized = web_dashboard.socketio.test_client(web_dashboard.app)
        authorized = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        assert not unauthorized.is_connected()
        assert authorized.is_connected()
        authorized.disconnect()

    def test_socket_control_requires_dashboard_token(self, web_dashboard):
        """Control messages without the token should not invoke callbacks."""
        called = []
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )
        web_dashboard.on_pause_callback = lambda: called.append("pause")

        client.emit("control", {"action": "pause"})
        client.emit("control", {"action": "pause", "token": web_dashboard.access_token})

        assert called == ["pause"]
        client.disconnect()

    def test_socket_control_acknowledges_destructive_workflows(self, web_dashboard):
        """Dashboard controls should return explicit success acks for UI gating."""
        called = []
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )
        web_dashboard.on_save_callback = lambda: called.append("save")
        web_dashboard.on_start_fresh_callback = lambda: called.append("start_fresh")

        save_ack = client.emit(
            "control",
            {"action": "save", "token": web_dashboard.access_token},
            callback=True,
        )
        fresh_ack = client.emit(
            "control",
            {"action": "start_fresh", "token": web_dashboard.access_token},
            callback=True,
        )

        assert save_ack == {"success": True, "action": "save"}
        assert fresh_ack == {"success": True, "action": "start_fresh"}
        assert called == ["save", "start_fresh"]
        assert any(event["name"] == "training_reset" for event in client.get_received())
        client.disconnect()

    def test_socket_control_reports_failed_save_callback(self, web_dashboard):
        """Save ack should fail when the app save callback reports failure."""
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )
        web_dashboard.on_save_callback = lambda: False

        save_ack = client.emit(
            "control",
            {"action": "save", "token": web_dashboard.access_token},
            callback=True,
        )

        assert save_ack == {
            "success": False,
            "action": "save",
            "error": "Save failed",
        }
        client.disconnect()

    def test_socket_control_rejects_unknown_or_invalid_model_actions(self, web_dashboard):
        """Bad control actions should fail through the same ack path as success."""
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        unknown_ack = client.emit(
            "control",
            {"action": "not_real", "token": web_dashboard.access_token},
            callback=True,
        )
        load_ack = client.emit(
            "control",
            {
                "action": "load_model",
                "id": "breakout:../bad.pth",
                "token": web_dashboard.access_token,
            },
            callback=True,
        )

        assert unknown_ack == {
            "success": False,
            "action": "not_real",
            "error": "Unknown action",
        }
        assert load_ack == {
            "success": False,
            "action": "load_model",
            "error": "Invalid model id",
        }
        client.disconnect()

    def test_socket_control_rejects_empty_payload(self, web_dashboard):
        """Malformed control payloads should fail instead of crashing the handler."""
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        ack = client.emit("control", None, callback=True)

        assert ack == {"success": False, "error": "Unauthorized"}
        client.disconnect()

    def test_socket_control_rejects_missing_load_model_id(self, web_dashboard):
        """A load-model request without a model id should not report success."""
        called = []
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )
        web_dashboard.on_load_model_callback = lambda path: called.append(path)

        ack = client.emit(
            "control",
            {"action": "load_model", "token": web_dashboard.access_token},
            callback=True,
        )

        assert ack == {
            "success": False,
            "action": "load_model",
            "error": "Invalid model id",
        }
        assert called == []
        client.disconnect()

    def test_socket_control_rejects_missing_load_model_file(self, tmp_path):
        """A stale allowed model id should fail before invoking load callbacks."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        config.GAME_NAME = "breakout"
        config.MODEL_DIR = str(tmp_path / "models")
        os.makedirs(config.GAME_MODEL_DIR)
        dashboard = WebDashboard(port=5104, config=config)
        called = []
        client = dashboard.socketio.test_client(
            dashboard.app,
            auth={"token": dashboard.access_token},
        )
        dashboard.on_load_model_callback = lambda path: called.append(path)

        ack = client.emit(
            "control",
            {
                "action": "load_model",
                "id": "breakout:missing.pth",
                "token": dashboard.access_token,
            },
            callback=True,
        )

        assert ack == {
            "success": False,
            "action": "load_model",
            "error": "Model not found",
        }
        assert called == []
        client.disconnect()

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                {"action": "speed", "value": "fast"},
                {"success": False, "action": "speed", "error": "Invalid speed"},
            ),
            (
                {"action": "config_change", "config": ["bad"]},
                {"success": False, "action": "config_change", "error": "Invalid config"},
            ),
            (
                {"action": "performance_mode", "mode": "warp"},
                {
                    "success": False,
                    "action": "performance_mode",
                    "error": "Invalid performance mode",
                },
            ),
            (
                {"action": "select_game", "game": "not_a_game"},
                {"success": False, "action": "select_game", "error": "Invalid game"},
            ),
            (
                {"action": "save_as", "filename": {"bad": "name"}},
                {"success": False, "action": "save_as", "error": "Invalid filename"},
            ),
        ],
    )
    def test_socket_control_rejects_malformed_control_payloads(
        self, web_dashboard, payload, expected
    ):
        """Malformed control payloads should fail without invoking app callbacks."""
        called = []
        web_dashboard.on_game_selected_callback = lambda game, mode: called.append((game, mode))
        web_dashboard.on_save_as_callback = lambda filename: called.append(filename)
        web_dashboard.on_speed_callback = lambda speed: called.append(speed)
        web_dashboard.on_config_change_callback = lambda config: called.append(config)
        web_dashboard.on_performance_mode_callback = lambda mode: called.append(mode)
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )
        payload = {**payload, "token": web_dashboard.access_token}

        ack = client.emit("control", payload, callback=True)

        assert ack == expected
        assert called == []
        client.disconnect()

    def test_socket_control_rejects_invalid_restart_game_before_callback(self, web_dashboard):
        """Invalid restart targets should not reach the restart callback."""
        called = []
        web_dashboard.on_restart_with_game_callback = lambda game: called.append(game)
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        ack = client.emit(
            "control",
            {
                "action": "restart_with_game",
                "game": "not_a_game",
                "token": web_dashboard.access_token,
            },
            callback=True,
        )

        assert ack == {
            "success": False,
            "action": "restart_with_game",
            "error": "Invalid game",
        }
        assert called == []
        client.disconnect()
