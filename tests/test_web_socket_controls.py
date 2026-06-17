"""Socket.IO control tests for the web dashboard."""

import os

import pytest

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
            from config import Config
            from src.web.server import WebDashboard

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

    @pytest.mark.parametrize(
        ("action", "payload", "callback_attr", "expected_error", "blocked_events"),
        [
            ("pause", {"action": "pause"}, "on_pause_callback", "Pause failed", []),
            ("reset", {"action": "reset"}, "on_reset_callback", "Reset failed", []),
            (
                "speed",
                {"action": "speed", "value": 2},
                "on_speed_callback",
                "Speed change failed",
                [],
            ),
            (
                "config_change",
                {"action": "config_change", "config": {"batch_size": 64}},
                "on_config_change_callback",
                "Config change failed",
                [],
            ),
            (
                "performance_mode",
                {"action": "performance_mode", "mode": "fast"},
                "on_performance_mode_callback",
                "Performance mode failed",
                [],
            ),
            (
                "save_and_quit",
                {"action": "save_and_quit"},
                "on_save_and_quit_callback",
                "Save and quit failed",
                [],
            ),
            (
                "select_game",
                {"action": "select_game", "game": "breakout", "mode": "ai"},
                "on_game_selected_callback",
                "Game selection failed",
                ["game_starting"],
            ),
            (
                "restart_with_game",
                {"action": "restart_with_game", "game": "snake"},
                "on_restart_with_game_callback",
                "Restart failed",
                ["restarting"],
            ),
            (
                "go_to_launcher",
                {"action": "go_to_launcher"},
                "on_save_and_quit_callback",
                "Launcher switch failed",
                ["redirect_to_launcher"],
            ),
        ],
    )
    def test_socket_control_callback_failures_return_acks_without_side_effect_events(
        self, web_dashboard, action, payload, callback_attr, expected_error, blocked_events
    ):
        """Callback crashes should not tear down the socket handler or emit success events."""
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        def fail_callback(*_args):
            raise RuntimeError("boom")

        setattr(web_dashboard, callback_attr, fail_callback)
        payload = {**payload, "token": web_dashboard.access_token}

        ack = client.emit("control", payload, callback=True)
        received_names = {event["name"] for event in client.get_received()}

        assert ack == {"success": False, "action": action, "error": expected_error}
        assert not received_names.intersection(blocked_events)
        client.disconnect()

    def test_control_ack_helpers_have_stable_shapes(self, web_dashboard):
        """Control responses should keep a predictable browser contract."""
        assert web_dashboard._success_ack("pause") == {"success": True, "action": "pause"}
        assert web_dashboard._error_ack("save", "Save failed") == {
            "success": False,
            "action": "save",
            "error": "Save failed",
        }
        assert web_dashboard._unauthorized_ack() == {
            "success": False,
            "error": "Unauthorized",
        }

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
        from config import Config
        from src.web.server import WebDashboard

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

    def test_socket_speed_uses_clamped_value_for_callback_and_publisher(self, web_dashboard):
        """Speed controls should publish the same clamped value the runtime receives."""
        called = []
        web_dashboard.on_speed_callback = lambda speed: called.append(speed)
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        ack = client.emit(
            "control",
            {
                "action": "speed",
                "value": 5000,
                "token": web_dashboard.access_token,
            },
            callback=True,
        )

        assert ack == {"success": True, "action": "speed"}
        assert called == [1000.0]
        assert web_dashboard.publisher.state.game_speed == 1000.0
        client.disconnect()

    def test_socket_config_change_publishes_only_normalized_values(self, web_dashboard):
        """Dashboard config state should mirror accepted runtime config values."""
        called = []
        web_dashboard.on_config_change_callback = lambda config: called.append(config)
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        ack = client.emit(
            "control",
            {
                "action": "config_change",
                "config": {
                    "learning_rate": "0.002",
                    "batch_size": "64",
                    "learn_every": "2",
                    "gradient_steps": "3",
                },
                "token": web_dashboard.access_token,
            },
            callback=True,
        )

        assert ack == {"success": True, "action": "config_change"}
        assert called == [
            {
                "learning_rate": 0.002,
                "batch_size": 64,
                "learn_every": 2,
                "gradient_steps": 3,
            }
        ]
        assert web_dashboard.publisher.state.learning_rate == 0.002
        assert web_dashboard.publisher.state.batch_size == 64
        assert web_dashboard.publisher.state.learn_every == 2
        assert web_dashboard.publisher.state.gradient_steps == 3
        client.disconnect()

    def test_socket_config_change_rejects_invalid_values_without_publishing(self, web_dashboard):
        """Rejected config changes should not leak into dashboard state."""
        called = []
        original_lr = web_dashboard.publisher.state.learning_rate
        web_dashboard.on_config_change_callback = lambda config: called.append(config)
        client = web_dashboard.socketio.test_client(
            web_dashboard.app,
            auth={"token": web_dashboard.access_token},
        )

        ack = client.emit(
            "control",
            {
                "action": "config_change",
                "config": {"learning_rate": "not-a-number"},
                "token": web_dashboard.access_token,
            },
            callback=True,
        )

        assert ack["success"] is False
        assert ack["action"] == "config_change"
        assert called == []
        assert web_dashboard.publisher.state.learning_rate == original_lr
        client.disconnect()
