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
from datetime import datetime
from unittest.mock import MagicMock, patch

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
            timestamp=datetime.now().isoformat(),
            message="Test message",
            level="info"
        )
        assert msg.message == "Test message"
        assert msg.level == "info"

    def test_to_dict(self):
        """LogMessage should convert to dictionary correctly."""
        msg = LogMessage(
            timestamp="2024-01-01T00:00:00",
            message="Test",
            level="warning"
        )
        data = msg.to_dict()
        assert data['message'] == "Test"
        assert data['level'] == "warning"
        assert data['timestamp'] == "2024-01-01T00:00:00"

    def test_data_field(self):
        """LogMessage should support optional data field."""
        msg = LogMessage(
            timestamp="2024-01-01T00:00:00",
            message="With data",
            level="info",
            data={'key': 'value'}
        )
        data = msg.to_dict()
        assert data['data'] == {'key': 'value'}


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

        publisher.update(
            episode=10,
            score=100,
            epsilon=0.5,
            loss=0.01
        )

        assert publisher.state.episode == 10
        assert publisher.state.score == 100
        assert publisher.state.epsilon == 0.5
        assert publisher.state.loss == 0.01

    def test_history_tracking(self):
        """MetricsPublisher should track history."""
        publisher = MetricsPublisher(history_length=100)

        for i in range(5):
            publisher.update(
                episode=i,
                score=i * 10,
                epsilon=1.0 - i * 0.1,
                loss=1.0 / (i + 1)
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

        assert 'state' in snapshot
        assert snapshot['state']['episode'] == 5
        assert snapshot['state']['score'] == 25

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
        assert 'left' in publisher.action_frequency
        assert 'stay' in publisher.action_frequency
        assert 'right' in publisher.action_frequency


class TestWebDashboardIntegration:
    """Integration tests requiring WebDashboard instantiation."""

    @pytest.fixture
    def web_dashboard(self):
        """Create a WebDashboard instance for testing."""
        try:
            from src.web.server import WebDashboard
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
        except ImportError:
            pytest.skip("WebDashboard not available")

    def test_initialization(self, web_dashboard):
        """WebDashboard should initialize with correct config."""
        assert web_dashboard.port == 5099
        assert web_dashboard.config.GAME_NAME == 'breakout'
        assert web_dashboard.publisher is not None

    def test_emit_metrics(self, web_dashboard):
        """WebDashboard.emit_metrics should update publisher."""
        web_dashboard.emit_metrics(
            episode=10,
            score=100,
            epsilon=0.5,
            loss=0.01
        )

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
                loss=1.0 / episode
            )

        state = web_dashboard.publisher.state
        assert state.episode == 10
        assert state.score == 100
        assert state.best_score == 100

    def test_api_status_endpoint(self, web_dashboard):
        """GET /api/status should return current training state."""
        web_dashboard.emit_metrics(episode=5, score=50, epsilon=0.8, loss=0.05)

        web_dashboard.app.config['TESTING'] = True
        with web_dashboard.app.test_client() as client:
            response = client.get('/api/status')
            assert response.status_code == 200

            data = json.loads(response.data)
            # Status endpoint returns nested state
            assert 'state' in data or 'episode' in data
            if 'state' in data:
                assert data['state']['episode'] == 5
            else:
                assert data['episode'] == 5

    def test_api_models_endpoint(self, web_dashboard):
        """GET /api/models should return available models."""
        web_dashboard.app.config['TESTING'] = True
        with web_dashboard.app.test_client() as client:
            response = client.get('/api/models')
            assert response.status_code == 200

            data = json.loads(response.data)
            assert 'models' in data
            assert 'current_game' in data

    def test_api_config_endpoint(self, web_dashboard):
        """GET /api/config should return configuration."""
        web_dashboard.app.config['TESTING'] = True
        with web_dashboard.app.test_client() as client:
            response = client.get('/api/config')
            assert response.status_code == 200

            data = json.loads(response.data)
            # Config returns training hyperparameters
            assert 'batch_size' in data or 'learning_rate' in data
            assert 'device' in data

    def test_logging_during_training(self, web_dashboard):
        """Logging should work during training simulation."""
        web_dashboard.log("Training started", level="info")
        web_dashboard.emit_metrics(episode=1, score=10, epsilon=0.95, loss=0.1)
        web_dashboard.log("Episode 1 complete", level="success")

        logs = web_dashboard.publisher.console_logs
        messages = [log.message for log in logs]

        assert "Training started" in messages
        assert "Episode 1 complete" in messages
