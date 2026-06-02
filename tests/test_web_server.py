"""
Tests for the web dashboard server.

Tests cover:
- Utility functions
- Data structures
- MetricsPublisher functionality
- Training state management
"""

import pytest
import json
import re
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Try to import web server components
try:
    from src.web.server import (
        MetricsPublisher,
        TrainingState,
        LogMessage,
        _make_json_safe,
        FLASK_AVAILABLE,
    )

    WEB_AVAILABLE = FLASK_AVAILABLE
except ImportError:
    WEB_AVAILABLE = False


# Skip all tests if Flask is not available
pytestmark = pytest.mark.skipif(
    not WEB_AVAILABLE, reason="Flask/SocketIO not installed"
)


class TestMakeJsonSafe:
    """Tests for the _make_json_safe utility function."""

    def test_native_types_unchanged(self):
        """Native Python types should pass through unchanged."""
        assert _make_json_safe(42) == 42
        assert _make_json_safe(3.14) == 3.14
        assert _make_json_safe("hello") == "hello"
        assert _make_json_safe(True) is True
        assert _make_json_safe(None) is None

    def test_numpy_int_converted(self):
        """NumPy integer types should be converted to native Python int."""
        import numpy as np

        assert _make_json_safe(np.int64(42)) == 42
        assert isinstance(_make_json_safe(np.int64(42)), int)

    def test_numpy_float_converted(self):
        """NumPy float types should be converted to native Python float."""
        import numpy as np

        assert _make_json_safe(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_make_json_safe(np.float64(3.14)), float)

    def test_numpy_arrays_converted(self):
        """NumPy arrays should be converted to lists."""
        import numpy as np

        arr = np.array([1, 2, 3])
        result = _make_json_safe(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_nested_dict_converted(self):
        """Nested dictionaries with NumPy values should be converted."""
        import numpy as np

        data = {"score": np.float64(100.0), "nested": {"values": np.array([1, 2, 3])}}
        result = _make_json_safe(data)
        assert result["score"] == 100.0
        assert result["nested"]["values"] == [1, 2, 3]

    def test_list_with_numpy_converted(self):
        """Lists containing NumPy values should be converted."""
        import numpy as np

        data = [np.int64(1), np.float64(2.0), "three"]
        result = _make_json_safe(data)
        assert result[0] == 1
        assert result[1] == 2.0
        assert result[2] == "three"


class TestTrainingState:
    """Tests for the TrainingState dataclass."""

    def test_default_values(self):
        """TrainingState should have sensible defaults."""
        state = TrainingState()
        assert state.episode == 0
        assert state.score == 0
        assert state.best_score == 0
        assert state.epsilon == 1.0
        assert state.is_paused is False
        assert state.is_running is False

    def test_update_from_kwargs(self):
        """TrainingState fields should be assignable."""
        state = TrainingState()
        state.episode = 100
        state.score = 50
        state.epsilon = 0.5
        assert state.episode == 100
        assert state.score == 50
        assert state.epsilon == 0.5

    def test_device_default(self):
        """TrainingState should have cpu as default device."""
        state = TrainingState()
        assert state.device == "cpu"

    def test_performance_fields(self):
        """TrainingState should have performance tracking fields."""
        state = TrainingState()
        assert state.learn_every == 1
        assert state.gradient_steps == 1
        assert state.steps_per_second == 0.0


class TestLogMessage:
    """Tests for the LogMessage class."""

    def test_creation(self):
        """LogMessage should store message and level."""
        msg = LogMessage(
            timestamp=datetime.now().isoformat(), message="Test message", level="info"
        )
        assert msg.message == "Test message"
        assert msg.level == "info"

    def test_to_dict(self):
        """LogMessage should convert to dictionary correctly."""
        msg = LogMessage(
            timestamp="2024-01-01T00:00:00", message="Test", level="warning"
        )
        data = msg.to_dict()
        assert data["message"] == "Test"
        assert data["level"] == "warning"
        assert data["timestamp"] == "2024-01-01T00:00:00"

    def test_data_field(self):
        """LogMessage should support optional data field."""
        msg = LogMessage(
            timestamp="2024-01-01T00:00:00",
            message="With data",
            level="info",
            data={"key": "value"},
        )
        data = msg.to_dict()
        assert data["data"] == {"key": "value"}


class TestMetricsPublisher:
    """Tests for the MetricsPublisher class."""

    def test_initialization(self):
        """MetricsPublisher should initialize with default state."""
        publisher = MetricsPublisher(history_length=100)
        assert publisher.state.episode == 0
        assert len(publisher.scores) == 0
        assert len(publisher.console_logs) == 0

    def test_update_state(self):
        """MetricsPublisher.update should update state correctly."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(episode=10, score=100, epsilon=0.5, loss=0.01)

        assert publisher.state.episode == 10
        assert publisher.state.score == 100
        assert publisher.state.epsilon == 0.5
        assert publisher.state.loss == 0.01

    def test_history_tracking(self):
        """MetricsPublisher should track history."""
        publisher = MetricsPublisher(history_length=100)

        for i in range(5):
            publisher.update(
                episode=i, score=i * 10, epsilon=1.0 - i * 0.1, loss=1.0 / (i + 1)
            )

        assert len(publisher.scores) == 5
        assert len(publisher.losses) == 5
        assert len(publisher.epsilons) == 5

    def test_best_score_tracking(self):
        """MetricsPublisher should track best score."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(episode=1, score=50, epsilon=0.9, loss=0.1)
        assert publisher.state.best_score == 50

        publisher.update(episode=2, score=30, epsilon=0.8, loss=0.1)
        assert publisher.state.best_score == 50  # Should not decrease

        publisher.update(episode=3, score=75, epsilon=0.7, loss=0.1)
        assert publisher.state.best_score == 75

    def test_log_message(self):
        """MetricsPublisher.log should add messages to console."""
        publisher = MetricsPublisher(history_length=100)

        publisher.log("Test message", level="info")

        assert len(publisher.console_logs) == 1
        assert publisher.console_logs[0].message == "Test message"

    def test_log_timestamp_uses_milliseconds(self):
        """Log timestamps should keep three fractional-second digits."""
        publisher = MetricsPublisher(history_length=100)

        publisher.log("Test message", level="info")

        assert re.match(
            r"^\d{2}:\d{2}:\d{2}\.\d{3}$", publisher.console_logs[0].timestamp
        )

    def test_log_level_parsing(self):
        """MetricsPublisher.log should handle different log levels."""
        publisher = MetricsPublisher(history_length=100)

        publisher.log("Debug", level="debug")
        publisher.log("Info", level="info")
        publisher.log("Warning", level="warning")
        publisher.log("Error", level="error")

        assert publisher.console_logs[0].level == "debug"
        assert publisher.console_logs[1].level == "info"
        assert publisher.console_logs[2].level == "warning"
        assert publisher.console_logs[3].level == "error"

    def test_console_log_limit(self):
        """MetricsPublisher should limit console log size."""
        publisher = MetricsPublisher(history_length=100)

        # Add more than the limit (500)
        for i in range(600):
            publisher.log(f"Message {i}")

        # Should be capped at 500
        assert len(publisher.console_logs) <= 500

    def test_get_snapshot(self):
        """MetricsPublisher.get_snapshot should return current state."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update(episode=5, score=25, epsilon=0.8, loss=0.05)
        snapshot = publisher.get_snapshot()

        assert "state" in snapshot
        assert snapshot["state"]["episode"] == 5
        assert snapshot["state"]["score"] == 25

    def test_screenshot_handling(self):
        """MetricsPublisher should handle screenshot retrieval."""
        publisher = MetricsPublisher(history_length=100)

        # get_screenshot should return None initially
        assert publisher.get_screenshot() is None

    def test_set_running(self):
        """MetricsPublisher.set_running should update is_running flag."""
        publisher = MetricsPublisher(history_length=100)

        assert publisher.state.is_running is False
        publisher.set_running(True)
        assert publisher.state.is_running is True
        publisher.set_running(False)
        assert publisher.state.is_running is False

    def test_action_frequency_tracking(self):
        """MetricsPublisher should track action frequencies."""
        publisher = MetricsPublisher(history_length=100)

        # Action frequency dict should exist
        assert "left" in publisher.action_frequency
        assert "stay" in publisher.action_frequency
        assert "right" in publisher.action_frequency

    def test_layer_analysis_handles_empty_arrays(self):
        """Layer analysis should not crash on empty activation, weight, or gradient arrays."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="empty",
            neuron_count=0,
            activations=np.array([], dtype=np.float32),
            weights=np.array([], dtype=np.float32),
            gradients=np.array([], dtype=np.float32),
        )

        analysis = publisher.get_layer_analysis(0)
        assert analysis["avg_activation"] == 0.0
        assert analysis["activation_histogram"] == [0] * 20
        assert analysis["weight_histogram"] == [0] * 20
        assert analysis["gradient_max_magnitude"] == 0.0

    def test_all_layer_analysis_is_sorted(self):
        """All layer analysis should return layers in index order."""
        publisher = MetricsPublisher(history_length=100)

        publisher.update_layer_analysis(
            2, "layer_2", 1, np.array([0.2], dtype=np.float32)
        )
        publisher.update_layer_analysis(
            1, "layer_1", 1, np.array([0.1], dtype=np.float32)
        )

        layers = publisher.get_all_layer_analysis()
        assert [layer["layer_idx"] for layer in layers] == [1, 2]


class TestWebDashboardIntegration:
    """Integration tests requiring WebDashboard instantiation."""

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

    def test_initialization(self, web_dashboard):
        """WebDashboard should initialize with correct config."""
        assert web_dashboard.port == 5099
        assert web_dashboard.host == "127.0.0.1"
        assert web_dashboard.access_token
        assert web_dashboard.config.GAME_NAME == "breakout"
        assert web_dashboard.publisher is not None

    def test_emit_metrics(self, web_dashboard):
        """WebDashboard.emit_metrics should update publisher."""
        web_dashboard.emit_metrics(episode=10, score=100, epsilon=0.5, loss=0.01)

        assert web_dashboard.publisher.state.episode == 10
        assert web_dashboard.publisher.state.score == 100

    def test_log_method(self, web_dashboard):
        """WebDashboard.log should add to console log."""
        web_dashboard.log("Test message", level="info")

        assert len(web_dashboard.publisher.console_logs) >= 1
        messages = [m.message for m in web_dashboard.publisher.console_logs]
        assert "Test message" in messages

    def test_full_training_cycle(self, web_dashboard):
        """Simulate a full training cycle with metrics updates."""
        for episode in range(1, 11):
            web_dashboard.emit_metrics(
                episode=episode,
                score=episode * 10,
                epsilon=1.0 - (episode * 0.09),
                loss=1.0 / episode,
            )

        state = web_dashboard.publisher.state
        assert state.episode == 10
        assert state.score == 100
        assert state.best_score == 100

    def test_api_status_endpoint(self, web_dashboard):
        """GET /api/status should return current training state."""
        web_dashboard.emit_metrics(episode=5, score=50, epsilon=0.8, loss=0.05)

        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get("/api/status")
            assert response.status_code == 200

            data = json.loads(response.data)
            # Status endpoint returns nested state
            assert "state" in data or "episode" in data
            if "state" in data:
                assert data["state"]["episode"] == 5
            else:
                assert data["episode"] == 5

    def test_api_models_endpoint(self, web_dashboard):
        """GET /api/models should return available models."""
        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get("/api/models")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert "models" in data
            assert "current_game" in data

    def test_api_models_uses_opaque_ids(self, tmp_path):
        """Model list should not expose absolute local filesystem paths."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        config.GAME_NAME = "breakout"
        config.MODEL_DIR = str(tmp_path / "models")
        os.makedirs(config.GAME_MODEL_DIR)
        model_path = os.path.join(config.GAME_MODEL_DIR, "demo.pth")
        torch.save({"steps": 1, "epsilon": 0.5}, model_path)

        dashboard = WebDashboard(port=5101, config=config)
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            response = client.get("/api/models")

        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["models"][0]["id"] == "breakout:demo.pth"
        assert "path" not in data["models"][0]

    def test_delete_model_requires_dashboard_token(self, tmp_path):
        """Mutating model routes should reject requests without the session token."""
        from src.web.server import WebDashboard
        from config import Config

        config = Config()
        config.GAME_NAME = "breakout"
        config.MODEL_DIR = str(tmp_path / "models")
        os.makedirs(config.GAME_MODEL_DIR)
        model_path = os.path.join(config.GAME_MODEL_DIR, "demo.pth")
        torch.save({"steps": 1}, model_path)

        dashboard = WebDashboard(port=5102, config=config)
        dashboard.app.config["TESTING"] = True
        with dashboard.app.test_client() as client:
            unauthorized = client.delete("/api/models/breakout:demo.pth")
            authorized = client.delete(
                "/api/models/breakout:demo.pth",
                headers={"X-Dashboard-Token": dashboard.access_token},
            )

        assert unauthorized.status_code == 401
        assert authorized.status_code == 200
        assert not os.path.exists(model_path)

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

    def test_api_config_endpoint(self, web_dashboard):
        """GET /api/config should return configuration."""
        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get("/api/config")
            assert response.status_code == 200

            data = json.loads(response.data)
            # Config returns training hyperparameters
            assert "batch_size" in data or "learning_rate" in data
            assert "device" in data

    def test_api_layers_endpoint(self, web_dashboard):
        """GET /api/layers should return all layer analysis data."""
        web_dashboard.publisher.update_layer_analysis(
            layer_idx=0,
            layer_name="input",
            neuron_count=2,
            activations=np.array([0.1, 0.2], dtype=np.float32),
        )

        web_dashboard.app.config["TESTING"] = True
        with web_dashboard.app.test_client() as client:
            response = client.get("/api/layers")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert isinstance(data, list)
            assert data[0]["layer_idx"] == 0
            assert data[0]["layer_name"] == "input"

    def test_emit_nn_visualization_updates_phase2_inspection(self, web_dashboard):
        """Live NN snapshots should populate layer and neuron inspection endpoints."""
        layer_info = [
            {"name": "Input", "neurons": 2, "type": "input"},
            {"name": "Hidden 1", "neurons": 2, "type": "hidden"},
            {"name": "Output", "neurons": 2, "type": "output"},
        ]

        web_dashboard.emit_nn_visualization(
            layer_info=layer_info,
            activations={"layer_0": [0.5, 0.6], "layer_1": [0.7, 0.8]},
            q_values=[0.7, 0.8],
            selected_action=1,
            weights=[
                [[0.1, 0.2], [0.3, 0.4]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            step=10,
            action_labels=["LEFT", "RIGHT"],
            input_state=[0.9, 0.1],
        )

        hidden = web_dashboard.publisher.get_layer_analysis(1)
        assert hidden["layer_name"] == "Hidden 1"
        assert hidden["avg_activation"] == pytest.approx(0.55)

        hidden_neuron = web_dashboard.publisher.get_neuron_details(1, 0)
        assert hidden_neuron["current_activation"] == pytest.approx(0.5)
        assert hidden_neuron["incoming_weights"] == [0.1, 0.2]
        assert hidden_neuron["outgoing_weights"] == [1.0, 3.0]

        output_neuron = web_dashboard.publisher.get_neuron_details(2, 0)
        assert output_neuron["q_value_contributions"] == {"LEFT": 0.7}

    def test_emit_nn_visualization_maps_dueling_stream_weights(self, web_dashboard):
        """Dueling stream output weights should map to Phase 2 inspection data."""
        layer_info = [
            {"name": "Input", "neurons": 2, "type": "input"},
            {"name": "Shared 1", "neurons": 2, "type": "hidden"},
            {"name": "Value", "neurons": 2, "type": "value_stream"},
            {"name": "Advantage", "neurons": 2, "type": "advantage_stream"},
            {"name": "Output (Q)", "neurons": 2, "type": "output"},
        ]

        web_dashboard.emit_nn_visualization(
            layer_info=layer_info,
            activations={
                "layer_0": [0.5, 0.6],
                "value_hidden": [0.7, 0.8],
                "advantage_hidden": [0.9, 1.0],
                "layer_output": [123.0],
            },
            q_values=[1.1, 1.2],
            selected_action=1,
            weights=[
                [[0.1, 0.2], [0.3, 0.4]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0]],
                [[11.0, 12.0], [13.0, 14.0]],
            ],
            step=20,
            action_labels=["LEFT", "RIGHT"],
            input_state=[0.1, 0.2],
        )

        value_neuron = web_dashboard.publisher.get_neuron_details(2, 0)
        advantage_neuron = web_dashboard.publisher.get_neuron_details(3, 0)
        output_neuron = web_dashboard.publisher.get_neuron_details(4, 1)

        assert value_neuron["incoming_weights"] == [1.0, 2.0]
        assert value_neuron["outgoing_weights"] == [9.0]
        assert advantage_neuron["incoming_weights"] == [5.0, 6.0]
        assert advantage_neuron["outgoing_weights"] == [11.0, 13.0]
        assert output_neuron["incoming_weights"] == [9.0, 10.0, 13.0, 14.0]
        assert output_neuron["q_value_contributions"] == {"RIGHT": 1.2}

    def test_emit_nn_visualization_skips_phase2_work_when_throttled(
        self, web_dashboard, monkeypatch
    ):
        """Phase 2 inspection should not run when NN visualization is throttled."""
        calls = []
        original_sync = web_dashboard._sync_phase2_inspection

        def counting_sync(*args, **kwargs):
            calls.append(1)
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(web_dashboard, "_sync_phase2_inspection", counting_sync)
        web_dashboard.publisher._nn_update_interval = 10.0

        payload = {
            "layer_info": [
                {"name": "Input", "neurons": 1, "type": "input"},
                {"name": "Output", "neurons": 1, "type": "output"},
            ],
            "activations": {"layer_0": [0.5]},
            "q_values": [0.5],
            "selected_action": 0,
            "weights": [[[0.1]]],
            "action_labels": ["STAY"],
            "input_state": [0.25],
        }

        web_dashboard.emit_nn_visualization(step=1, **payload)
        web_dashboard.emit_nn_visualization(step=2, **payload)

        assert len(calls) == 1

    def test_logging_during_training(self, web_dashboard):
        """Logging should work during training simulation."""
        web_dashboard.log("Training started", level="info")
        web_dashboard.emit_metrics(episode=1, score=10, epsilon=0.95, loss=0.1)
        web_dashboard.log("Episode 1 complete", level="success")

        logs = web_dashboard.publisher.console_logs
        messages = [log.message for log in logs]

        assert "Training started" in messages
        assert "Episode 1 complete" in messages
