"""
Tests for the web dashboard server.

Tests cover:
- REST API endpoints
- WebSocket events
- MetricsPublisher functionality
- Training state management
"""

import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock, patch
from dataclasses import asdict

# Try to import web server components
try:
    from src.web.server import (
        WebDashboard,
        MetricsPublisher,
        TrainingState,
        LogMessage,
        LogLevel,
        NNVisualizationData,
        _make_json_safe,
        FLASK_AVAILABLE,
    )
    WEB_AVAILABLE = FLASK_AVAILABLE
except ImportError:
    WEB_AVAILABLE = False


# Skip all tests if Flask is not available
pytestmark = pytest.mark.skipif(
    not WEB_AVAILABLE,
    reason="Flask/SocketIO not installed"
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

    def test_numpy_types_converted(self):
        """NumPy types should be converted to native Python types."""
        import numpy as np

        assert _make_json_safe(np.int64(42)) == 42
        assert _make_json_safe(np.float64(3.14)) == 3.14
        assert _make_json_safe(np.bool_(True)) is True

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

        data = {
            'score': np.float64(100.0),
            'nested': {
                'values': np.array([1, 2, 3])
            }
        }
        result = _make_json_safe(data)
        assert result['score'] == 100.0
        assert result['nested']['values'] == [1, 2, 3]

    def test_list_with_numpy_converted(self):
        """Lists containing NumPy values should be converted."""
        import numpy as np

        data = [np.int64(1), np.float64(2.0), "three"]
        result = _make_json_safe(data)
        assert result == [1, 2.0, "three"]


class TestTrainingState:
    """Tests for the TrainingState dataclass."""

    def test_default_values(self):
        """TrainingState should have sensible defaults."""
        state = TrainingState()
        assert state.episode == 0
        assert state.score == 0.0
        assert state.best_score == 0.0
        assert state.epsilon == 1.0
        assert state.paused is False

    def test_update_from_kwargs(self):
        """TrainingState should update from keyword arguments."""
        state = TrainingState()
        state.episode = 100
        state.score = 50.0
        assert state.episode == 100
        assert state.score == 50.0


class TestLogMessage:
    """Tests for the LogMessage dataclass."""

    def test_creation(self):
        """LogMessage should store message and level."""
        msg = LogMessage(
            message="Test message",
            level=LogLevel.INFO
        )
        assert msg.message == "Test message"
        assert msg.level == LogLevel.INFO

    def test_to_dict(self):
        """LogMessage should convert to dictionary correctly."""
        msg = LogMessage(
            message="Test",
            level=LogLevel.WARNING,
            timestamp="2024-01-01T00:00:00"
        )
        data = asdict(msg)
        assert data['message'] == "Test"
        assert data['level'] == LogLevel.WARNING


class TestMetricsPublisher:
    """Tests for the MetricsPublisher class."""

    def test_initialization(self):
        """MetricsPublisher should initialize with default state."""
        config = MagicMock()
        config.GAME_NAME = 'breakout'
        publisher = MetricsPublisher(config)

        assert publisher.state.episode == 0
        assert publisher.config == config

    def test_update_state(self):
        """MetricsPublisher.update should update state correctly."""
        config = MagicMock()
        config.GAME_NAME = 'breakout'
        publisher = MetricsPublisher(config)

        publisher.update(
            episode=10,
            score=100.0,
            epsilon=0.5
        )

        assert publisher.state.episode == 10
        assert publisher.state.score == 100.0
        assert publisher.state.epsilon == 0.5

    def test_best_score_tracking(self):
        """MetricsPublisher should track best score."""
        config = MagicMock()
        config.GAME_NAME = 'breakout'
        publisher = MetricsPublisher(config)

        publisher.update(score=50.0)
        assert publisher.state.best_score == 50.0

        publisher.update(score=30.0)
        assert publisher.state.best_score == 50.0  # Should not decrease

        publisher.update(score=75.0)
        assert publisher.state.best_score == 75.0

    def test_log_message(self):
        """MetricsPublisher.log should add messages to console."""
        config = MagicMock()
        config.GAME_NAME = 'breakout'
        publisher = MetricsPublisher(config)

        publisher.log("Test message", level="info")

        assert len(publisher._console_log) == 1
        assert publisher._console_log[0].message == "Test message"

    def test_log_level_parsing(self):
        """MetricsPublisher.log should parse string log levels."""
        config = MagicMock()
        config.GAME_NAME = 'breakout'
        publisher = MetricsPublisher(config)

        publisher.log("Debug", level="debug")
        publisher.log("Info", level="info")
        publisher.log("Warning", level="warning")
        publisher.log("Error", level="error")

        assert publisher._console_log[0].level == LogLevel.DEBUG
        assert publisher._console_log[1].level == LogLevel.INFO
        assert publisher._console_log[2].level == LogLevel.WARNING
        assert publisher._console_log[3].level == LogLevel.ERROR

    def test_console_log_limit(self):
        """MetricsPublisher should limit console log size."""
        config = MagicMock()
        config.GAME_NAME = 'breakout'
        publisher = MetricsPublisher(config)

        # Add more than the limit
        for i in range(600):
            publisher.log(f"Message {i}")

        # Should be capped at 500
        assert len(publisher._console_log) <= 500

    def test_get_snapshot(self):
        """MetricsPublisher.get_snapshot should return current state."""
        config = MagicMock()
        config.GAME_NAME = 'breakout'
        publisher = MetricsPublisher(config)

        publisher.update(episode=5, score=25.0)
        snapshot = publisher.get_snapshot()

        assert snapshot['state']['episode'] == 5
        assert snapshot['state']['score'] == 25.0


class TestNNVisualizationData:
    """Tests for neural network visualization data structures."""

    def test_creation(self):
        """NNVisualizationData should store layer information."""
        data = NNVisualizationData(
            layer_sizes=[55, 128, 64, 3],
            activations=[[0.5] * 55, [0.3] * 128, [0.1] * 64, [0.8, 0.1, 0.1]],
            weights=[],
            q_values=[1.0, 0.5, 0.3],
            selected_action=0
        )
        assert data.layer_sizes == [55, 128, 64, 3]
        assert len(data.activations) == 4
        assert data.selected_action == 0


@pytest.fixture
def web_dashboard():
    """Create a WebDashboard instance for testing."""
    from config import Config
    config = Config()
    config.GAME_NAME = 'breakout'

    dashboard = WebDashboard(port=5099, config=config)
    yield dashboard

    # Cleanup
    try:
        dashboard.stop()
    except Exception:
        pass


class TestWebDashboard:
    """Tests for the WebDashboard class."""

    def test_initialization(self, web_dashboard):
        """WebDashboard should initialize with correct config."""
        assert web_dashboard.port == 5099
        assert web_dashboard.config.GAME_NAME == 'breakout'
        assert web_dashboard.publisher is not None

    def test_emit_metrics(self, web_dashboard):
        """WebDashboard.emit_metrics should update publisher."""
        web_dashboard.emit_metrics(
            episode=10,
            score=100.0,
            epsilon=0.5
        )

        assert web_dashboard.publisher.state.episode == 10
        assert web_dashboard.publisher.state.score == 100.0

    def test_log_method(self, web_dashboard):
        """WebDashboard.log should add to console log."""
        web_dashboard.log("Test message", level="info")

        assert len(web_dashboard.publisher._console_log) >= 1
        # Find our message (there may be initialization messages)
        messages = [m.message for m in web_dashboard.publisher._console_log]
        assert "Test message" in messages


class TestWebDashboardAPI:
    """Tests for the REST API endpoints."""

    @pytest.fixture
    def client(self, web_dashboard):
        """Create a test client for the Flask app."""
        web_dashboard.app.config['TESTING'] = True
        with web_dashboard.app.test_client() as client:
            yield client

    def test_api_status(self, client, web_dashboard):
        """GET /api/status should return current training state."""
        web_dashboard.emit_metrics(episode=5, score=50.0)

        response = client.get('/api/status')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['episode'] == 5
        assert data['score'] == 50.0

    def test_api_model_info(self, client):
        """GET /api/model should return model information."""
        response = client.get('/api/model')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'state_size' in data
        assert 'action_size' in data
        assert 'hidden_layers' in data

    def test_api_console(self, client, web_dashboard):
        """GET /api/console should return console log."""
        web_dashboard.log("Test log entry", level="info")

        response = client.get('/api/console')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'logs' in data
        assert len(data['logs']) >= 1

    def test_api_models_list(self, client):
        """GET /api/models should return available models."""
        response = client.get('/api/models')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'models' in data
        assert 'current_game' in data


class TestWebDashboardControls:
    """Tests for training control functionality."""

    @pytest.fixture
    def client(self, web_dashboard):
        """Create a test client for the Flask app."""
        web_dashboard.app.config['TESTING'] = True
        with web_dashboard.app.test_client() as client:
            yield client

    def test_pause_toggle(self, web_dashboard, client):
        """POST /api/control/pause should toggle pause state."""
        # Set up a pause callback
        pause_called = [False]
        def on_pause(paused):
            pause_called[0] = True

        web_dashboard.set_control_callback('pause', on_pause)

        response = client.post('/api/control/pause')
        assert response.status_code == 200

        # Callback should have been triggered
        assert pause_called[0]

    def test_speed_control(self, web_dashboard, client):
        """POST /api/control/speed should update game speed."""
        speed_value = [None]
        def on_speed(speed):
            speed_value[0] = speed

        web_dashboard.set_control_callback('speed', on_speed)

        response = client.post(
            '/api/control/speed',
            json={'speed': 2.0}
        )
        assert response.status_code == 200
        assert speed_value[0] == 2.0

    def test_reset_control(self, web_dashboard, client):
        """POST /api/control/reset should trigger reset."""
        reset_called = [False]
        def on_reset():
            reset_called[0] = True

        web_dashboard.set_control_callback('reset', on_reset)

        response = client.post('/api/control/reset')
        assert response.status_code == 200
        assert reset_called[0]


class TestWebDashboardScreenshot:
    """Tests for screenshot capture functionality."""

    def test_capture_screenshot_without_surface(self, web_dashboard):
        """capture_screenshot should handle missing surface gracefully."""
        # Should not raise
        web_dashboard.publisher.capture_screenshot(None)
        assert web_dashboard.publisher.get_screenshot() is None

    def test_get_screenshot_empty(self, web_dashboard):
        """get_screenshot should return None when no screenshot captured."""
        screenshot = web_dashboard.publisher.get_screenshot()
        assert screenshot is None


class TestIntegration:
    """Integration tests for the web dashboard."""

    def test_full_training_cycle(self, web_dashboard):
        """Simulate a full training cycle with metrics updates."""
        # Simulate training
        for episode in range(1, 11):
            web_dashboard.emit_metrics(
                episode=episode,
                score=episode * 10.0,
                epsilon=1.0 - (episode * 0.1),
                loss=1.0 / episode
            )

        # Verify final state
        state = web_dashboard.publisher.state
        assert state.episode == 10
        assert state.score == 100.0
        assert state.best_score == 100.0
        assert state.epsilon == pytest.approx(0.0, abs=0.01)

    def test_logging_during_training(self, web_dashboard):
        """Logging should work during training simulation."""
        web_dashboard.log("Training started", level="info")
        web_dashboard.emit_metrics(episode=1, score=10.0)
        web_dashboard.log("Episode 1 complete", level="success")

        logs = web_dashboard.publisher._console_log
        messages = [log.message for log in logs]

        assert "Training started" in messages
        assert "Episode 1 complete" in messages
